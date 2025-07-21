import logging
import uuid
from traceback import format_exc
from typing import List, Optional

from fastapi import HTTPException
from google.cloud import bigquery

from app.models.payment import (
    CreditBalance,
    CreditTransaction,
    CreditTransactionType,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CreditManager:
    """Manages user credit operations including balance tracking and transactions."""

    def __init__(self):
        self.bq_client = bigquery.Client()

    async def get_credits(self, user_id: str) -> Optional[CreditBalance]:
        """
        Get user's current credit balance.

        Args:
            user_id: The user ID to get credits for

        Returns:
            CreditBalance object or None if user has no credit record
        """
        try:
            query = """
            SELECT user_id, balance, total_earned, total_spent, last_updated, created_at
            FROM `bai-buchai-p.users.credits`
            WHERE user_id = @user_id
            """

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("user_id", "STRING", user_id)
                ]
            )

            query_job = self.bq_client.query(query, job_config=job_config)
            results = list(query_job.result())

            if results:
                row = results[0]
                return CreditBalance(
                    user_id=row.user_id,
                    balance=row.balance,
                    total_earned=row.total_earned,
                    total_spent=row.total_spent,
                    last_updated=row.last_updated,
                    created_at=row.created_at,
                )
            return None

        except Exception as e:
            logger.error(
                f"Failed to get user credits for {user_id}: {str(e)}\n{format_exc()}"
            )
            raise HTTPException(
                status_code=500, detail=f"Failed to retrieve user credits: {str(e)}"
            )

    async def add_credits(
        self,
        user_id: str,
        amount: int,
        transaction_type: CreditTransactionType,
        description: str,
        reference_id: Optional[str] = None,
    ) -> bool:
        """
        Add credits to user account.

        Args:
            user_id: The user ID to add credits to
            amount: Number of credits to add
            transaction_type: Type of credit transaction
            description: Description of the transaction
            reference_id: Reference ID (e.g., subscription_id, payment_id)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure user has a credit record
            await self._ensure_user_credit_record(user_id)

            # Record the transaction
            transaction_id = str(uuid.uuid4())
            await self._record_credit_transaction(
                transaction_id,
                user_id,
                transaction_type,
                amount,
                description,
                reference_id,
            )

            # Update user's credit balance
            update_query = """
            UPDATE `bai-buchai-p.users.credits`
            SET 
                balance = balance + @amount,
                total_earned = total_earned + @amount,
                last_updated = CURRENT_TIMESTAMP()
            WHERE user_id = @user_id
            """

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("amount", "INT64", amount),
                    bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
                ]
            )

            query_job = self.bq_client.query(update_query, job_config=job_config)
            query_job.result()

            logger.info(
                f"Added {amount} credits to user {user_id} ({transaction_type.value})"
            )
            return True

        except Exception as e:
            logger.error(
                f"Failed to add credits to user {user_id}: {str(e)}\n{format_exc()}"
            )
            return False

    async def spend_credits(
        self,
        user_id: str,
        amount: int,
        description: str,
        reference_id: Optional[str] = None,
    ) -> bool:
        """
        Spend credits from user account.

        Args:
            user_id: The user ID to spend credits from
            amount: Number of credits to spend
            description: Description of the transaction
            reference_id: Reference ID (e.g., creation_id, task_id)

        Returns:
            True if successful

        Raises:
            HTTPException: If user has insufficient credits
        """
        try:
            # Check if user has enough credits
            current_credits = await self.get_credits(user_id)
            if not current_credits or current_credits.balance < amount:
                raise HTTPException(
                    status_code=400,
                    detail=f"Insufficient credits. Required: {amount}, Available: {current_credits.balance if current_credits else 0}",
                )

            # Record the transaction
            transaction_id = str(uuid.uuid4())
            await self._record_credit_transaction(
                transaction_id,
                user_id,
                CreditTransactionType.SPENT,
                amount,
                description,
                reference_id,
            )

            # Update user's credit balance
            update_query = """
            UPDATE `bai-buchai-p.users.credits`
            SET 
                balance = balance - @amount,
                total_spent = total_spent + @amount,
                last_updated = CURRENT_TIMESTAMP()
            WHERE user_id = @user_id
            """

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("amount", "INT64", amount),
                    bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
                ]
            )

            query_job = self.bq_client.query(update_query, job_config=job_config)
            query_job.result()

            logger.info(f"Spent {amount} credits for user {user_id}")
            return True

        except HTTPException:
            raise
        except Exception as e:
            logger.error(
                f"Failed to spend credits for user {user_id}: {str(e)}\n{format_exc()}"
            )
            raise HTTPException(
                status_code=500, detail=f"Failed to spend credits: {str(e)}"
            )

    async def get_credit_transactions(
        self, user_id: str, limit: int = 50, offset: int = 0
    ) -> List[CreditTransaction]:
        """
        Get user's credit transaction history.

        Args:
            user_id: The user ID to get transactions for
            limit: Maximum number of transactions to return
            offset: Number of transactions to skip

        Returns:
            List of CreditTransaction objects
        """
        try:
            query = """
            SELECT transaction_id, user_id, type, amount, description, reference_id, created_at
            FROM `bai-buchai-p.credits.transactions`
            WHERE user_id = @user_id
            ORDER BY created_at DESC
            LIMIT @limit OFFSET @offset
            """

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
                    bigquery.ScalarQueryParameter("limit", "INT64", limit),
                    bigquery.ScalarQueryParameter("offset", "INT64", offset),
                ]
            )

            query_job = self.bq_client.query(query, job_config=job_config)
            results = query_job.result()

            transactions = []
            for row in results:
                transactions.append(
                    CreditTransaction(
                        transaction_id=row.transaction_id,
                        user_id=row.user_id,
                        type=CreditTransactionType(row.type),
                        amount=row.amount,
                        description=row.description,
                        reference_id=row.reference_id,
                        created_at=row.created_at,
                    )
                )

            return transactions

        except Exception as e:
            logger.error(
                f"Failed to get credit transactions for {user_id}: {str(e)}\n{format_exc()}"
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve credit transactions: {str(e)}",
            )

    async def _ensure_user_credit_record(self, user_id: str) -> None:
        """
        Ensure user has a credit record, create if not exists.

        Args:
            user_id: The user ID to ensure has a credit record
        """
        existing = await self.get_credits(user_id)
        if not existing:
            query = """
            INSERT INTO `bai-buchai-p.users.credits` 
            (user_id, balance, total_earned, total_spent, last_updated, created_at)
            VALUES (@user_id, 0, 0, 0, CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP())
            """

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("user_id", "STRING", user_id)
                ]
            )

            query_job = self.bq_client.query(query, job_config=job_config)
            query_job.result()

            logger.info(f"Created credit record for user {user_id}")

    async def _record_credit_transaction(
        self,
        transaction_id: str,
        user_id: str,
        transaction_type: CreditTransactionType,
        amount: int,
        description: str,
        reference_id: Optional[str] = None,
    ) -> None:
        """
        Record a credit transaction.

        Args:
            transaction_id: Unique transaction ID
            user_id: The user ID
            transaction_type: Type of transaction
            amount: Number of credits
            description: Description of the transaction
            reference_id: Optional reference ID
        """
        query = """
        INSERT INTO `bai-buchai-p.credits.transactions`
        (transaction_id, user_id, type, amount, description, reference_id, created_at)
        VALUES (@transaction_id, @user_id, @type, @amount, @description, @reference_id, CURRENT_TIMESTAMP())
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter(
                    "transaction_id", "STRING", transaction_id
                ),
                bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
                bigquery.ScalarQueryParameter("type", "STRING", transaction_type.value),
                bigquery.ScalarQueryParameter("amount", "INT64", amount),
                bigquery.ScalarQueryParameter("description", "STRING", description),
                bigquery.ScalarQueryParameter("reference_id", "STRING", reference_id),
            ]
        )

        query_job = self.bq_client.query(query, job_config=job_config)
        query_job.result()
