"""
Credit Manager

This module manages user credit operations including balance tracking and transactions,
using Firestore instead of BigQuery for improved performance and real-time capabilities.
"""

import logging
import uuid
from datetime import datetime
from traceback import format_exc
from typing import List, Optional

from fastapi import HTTPException

from app.models.credits import (
    CreditPool,
    CreditTransaction,
    CreditTransactionType,
)
from app.models.shared import CreditBalance
from app.services.firestore import get_firestore_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CreditManager:
    """Manages user credit operations including balance tracking and transactions."""

    def __init__(self):
        self.firestore_service = get_firestore_service()

    async def get_credits(self, user_id: str) -> Optional[CreditBalance]:
        """
        Get user's current credit balance.

        Args:
            user_id: The user ID to get credits for

        Returns:
            CreditBalance object or None if user has no credit record
        """
        try:
            # Get user credits from Firestore with automatic conversion
            # Pass the API model class directly - automatic conversion!
            credits = await self.firestore_service.get_document(
                collection_name="users_credits",
                document_id=user_id,
                model_class=CreditBalance,  # API model class - automatic conversion!
            )

            return credits  # Already converted to CreditBalance

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
        pool: Optional[CreditPool] = None,
    ) -> bool:
        """
        Add credits to user account.

        Args:
            user_id: The user ID to add credits to
            amount: Number of credits to add
            transaction_type: Type of credit transaction
            description: Description of the transaction
            reference_id: Reference ID (e.g., subscription_id, payment_id)
            pool: Which credit pool to add to (defaults based on transaction type)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure user has a credit record
            await self._ensure_user_credit_record(user_id)

            # Determine credit pool based on transaction type if not explicitly provided
            if pool is None:
                if transaction_type == CreditTransactionType.EARNED_SUBSCRIPTION:
                    pool = CreditPool.MONTHLY
                elif transaction_type == CreditTransactionType.EARNED_BONUS:
                    pool = CreditPool.PERMANENT
                else:
                    pool = CreditPool.PERMANENT  # Default to permanent for other types

            # Get current credits using automatic conversion
            credits = await self.firestore_service.get_document(
                collection_name="users_credits",
                document_id=user_id,
                model_class=CreditBalance,  # API model class - automatic conversion!
            )

            if credits:
                new_monthly = credits.credits_monthly
                new_permanent = credits.credits_permanent
                new_total_earned = credits.total_earned + amount
            else:
                new_monthly = 0
                new_permanent = 0
                new_total_earned = amount

            # Add credits to the appropriate pool
            if pool == CreditPool.MONTHLY:
                new_monthly += amount
            else:  # CreditPool.PERMANENT
                new_permanent += amount

            # Update user credits
            update_data = {
                "credits_monthly": new_monthly,
                "credits_permanent": new_permanent,
                "total_earned": new_total_earned,
                "last_updated": datetime.utcnow(),
            }

            await self.firestore_service.update_document(
                collection_name="users_credits",
                document_id=user_id,
                update_data=update_data,
            )

            # Record the transaction
            await self._record_credit_transaction(
                user_id, transaction_type, amount, description, reference_id, pool
            )

            logger.info(
                f"Added {amount} credits to {pool.value} pool for user {user_id}"
            )
            return True

        except Exception as e:
            logger.error(
                f"Failed to add credits for user {user_id}: {str(e)}\n{format_exc()}"
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
        Spend/deduct credits from user account.
        Prioritizes spending from permanent pool first, then monthly pool.

        Args:
            user_id: The user ID to spend credits from
            amount: Number of credits to spend
            description: Description of the transaction
            reference_id: Reference ID for the transaction

        Returns:
            True if successful, False if insufficient credits or error
        """
        try:
            # Get current credits using automatic conversion
            credits = await self.firestore_service.get_document(
                collection_name="users_credits",
                document_id=user_id,
                model_class=CreditBalance,  # API model class - automatic conversion!
            )

            if not credits:
                logger.warning(f"No credit record found for user {user_id}")
                return False

            total_available = credits.balance
            if total_available < amount:
                logger.warning(
                    f"Insufficient credits for user {user_id}. "
                    f"Available: {total_available}, Requested: {amount}"
                )
                return False

            # Calculate how to spend from each pool (permanent first)
            remaining_to_spend = amount
            from_permanent = min(remaining_to_spend, credits.credits_permanent)
            remaining_to_spend -= from_permanent
            from_monthly = min(remaining_to_spend, credits.credits_monthly)

            # Update balances
            new_permanent = credits.credits_permanent - from_permanent
            new_monthly = credits.credits_monthly - from_monthly
            new_total_spent = credits.total_spent + amount

            update_data = {
                "credits_monthly": new_monthly,
                "credits_permanent": new_permanent,
                "total_spent": new_total_spent,
                "last_updated": datetime.utcnow(),
            }

            await self.firestore_service.update_document(
                collection_name="users_credits",
                document_id=user_id,
                update_data=update_data,
            )

            # Record transactions for each pool spent from
            if from_permanent > 0:
                await self._record_credit_transaction(
                    user_id,
                    CreditTransactionType.SPENT,
                    -from_permanent,
                    f"{description} (from permanent pool)",
                    reference_id,
                    CreditPool.PERMANENT,
                )

            if from_monthly > 0:
                await self._record_credit_transaction(
                    user_id,
                    CreditTransactionType.SPENT,
                    -from_monthly,
                    f"{description} (from monthly pool)",
                    reference_id,
                    CreditPool.MONTHLY,
                )

            logger.info(
                f"Spent {amount} credits for user {user_id} "
                f"(permanent: {from_permanent}, monthly: {from_monthly})"
            )
            return True

        except Exception as e:
            logger.error(
                f"Failed to spend credits for user {user_id}: {str(e)}\n{format_exc()}"
            )
            return False

    async def get_credit_transactions(
        self, user_id: str, limit: int = 50, offset: int = 0
    ) -> List[CreditTransaction]:
        """
        Get credit transaction history for a user.

        Args:
            user_id: The user ID to get transactions for
            limit: Maximum number of transactions to return
            offset: Number of transactions to skip

        Returns:
            List of CreditTransaction objects
        """
        try:
            # Query credit transactions from Firestore with automatic conversion
            # Pass the API model class directly - the service will handle conversion automatically
            transactions = await self.firestore_service.query_collection(
                collection_name="credits_transactions",
                filters=[("user_id", "==", user_id)],
                order_by="created_at",
                limit=limit,
                offset=offset,
                model_class=CreditTransaction,  # API model class - automatic conversion!
            )

            # Sort by created_at descending (most recent first)
            transactions.sort(key=lambda x: x.created_at, reverse=True)

            return transactions

        except Exception as e:
            logger.error(
                f"Failed to get credit transactions for user {user_id}: {str(e)}\n{format_exc()}"
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve credit transactions: {str(e)}",
            )

    async def _ensure_user_credit_record(self, user_id: str) -> None:
        """
        Ensure user has a credit record, create one if it doesn't exist.

        Args:
            user_id: The user ID to ensure has a credit record
        """
        # Check if user credit record exists using automatic conversion
        credits = await self.firestore_service.get_document(
            collection_name="users_credits",
            document_id=user_id,
            model_class=CreditBalance,  # API model class - automatic conversion!
        )

        if not credits:
            # Create initial credit record
            credit_data = {
                "user_id": user_id,
                "credits_monthly": 0,
                "credits_permanent": 0,
                "total_earned": 0,
                "total_spent": 0,
                "last_updated": datetime.utcnow(),
                "created_at": datetime.utcnow(),
            }

            await self.firestore_service.create_document(
                collection_name="users_credits",
                document_data=credit_data,
                document_id=user_id,
            )

            logger.info(f"Created initial credit record for user {user_id}")

    async def _record_credit_transaction(
        self,
        user_id: str,
        transaction_type: CreditTransactionType,
        amount: int,
        description: str,
        reference_id: Optional[str] = None,
        pool: Optional[CreditPool] = None,
    ) -> None:
        """
        Record a credit transaction.

        Args:
            user_id: The user ID for the transaction
            transaction_type: Type of credit transaction
            amount: Amount of credits (positive for earned, negative for spent)
            description: Description of the transaction
            reference_id: Reference ID for the transaction
            pool: Which credit pool was affected
        """
        transaction_id = str(uuid.uuid4())

        transaction_data = {
            "transaction_id": transaction_id,
            "user_id": user_id,
            "type": transaction_type.value,
            "pool": pool.value if pool else None,
            "amount": amount,
            "description": description,
            "reference_id": reference_id,
            "created_at": datetime.utcnow(),
        }

        await self.firestore_service.create_document(
            collection_name="credits_transactions",
            document_data=transaction_data,
            document_id=transaction_id,
        )
