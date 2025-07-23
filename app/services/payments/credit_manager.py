"""
Credit Manager - Firestore Version

This module manages user credit operations including balance tracking and transactions,
using Firestore instead of BigQuery for improved performance and real-time capabilities.
"""

import logging
import uuid
from datetime import datetime
from traceback import format_exc
from typing import List, Optional

from fastapi import HTTPException

from app.models.firestore import CreditTransaction as FirestoreCreditTransaction
from app.models.firestore import UserCredits as FirestoreUserCredits
from app.models.payment import (
    CreditBalance,
    CreditTransaction,
    CreditTransactionType,
)
from app.services.firestore_service import get_firestore_service

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
            # Get user credits from Firestore
            credits = await self.firestore_service.get_document(
                collection_name="users_credits",
                document_id=user_id,
                model_class=FirestoreUserCredits,
            )

            if credits:
                return CreditBalance(
                    user_id=credits.user_id,
                    balance=credits.balance,
                    total_earned=credits.total_earned,
                    total_spent=credits.total_spent,
                    last_updated=credits.last_updated,
                    created_at=credits.created_at,
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

            # Get current credits
            credits = await self.firestore_service.get_document(
                collection_name="users_credits",
                document_id=user_id,
                model_class=FirestoreUserCredits,
            )

            if credits:
                new_balance = credits.balance + amount
                new_total_earned = credits.total_earned + amount
            else:
                new_balance = amount
                new_total_earned = amount

            # Update user credits
            update_data = {
                "balance": new_balance,
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
                user_id, transaction_type, amount, description, reference_id
            )

            logger.info(f"Added {amount} credits to user {user_id}")
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

        Args:
            user_id: The user ID to spend credits from
            amount: Number of credits to spend
            description: Description of the transaction
            reference_id: Reference ID for the transaction

        Returns:
            True if successful, False if insufficient credits or error
        """
        try:
            # Get current credits
            credits = await self.firestore_service.get_document(
                collection_name="users_credits",
                document_id=user_id,
                model_class=FirestoreUserCredits,
            )

            if not credits:
                logger.warning(f"No credit record found for user {user_id}")
                return False

            if credits.balance < amount:
                logger.warning(
                    f"Insufficient credits for user {user_id}. "
                    f"Available: {credits.balance}, Requested: {amount}"
                )
                return False

            # Update user credits
            new_balance = credits.balance - amount
            new_total_spent = credits.total_spent + amount

            update_data = {
                "balance": new_balance,
                "total_spent": new_total_spent,
                "last_updated": datetime.utcnow(),
            }

            await self.firestore_service.update_document(
                collection_name="users_credits",
                document_id=user_id,
                update_data=update_data,
            )

            # Record the transaction
            await self._record_credit_transaction(
                user_id, CreditTransactionType.SPENT, -amount, description, reference_id
            )

            logger.info(f"Spent {amount} credits for user {user_id}")
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
            # Query credit transactions from Firestore
            firestore_transactions = await self.firestore_service.query_collection(
                collection_name="credits_transactions",
                filters=[("user_id", "==", user_id)],
                order_by="created_at",
                limit=limit,
                offset=offset,
                model_class=FirestoreCreditTransaction,
            )

            # Convert Firestore models to API models
            transactions = []
            for ft in firestore_transactions:
                transaction = CreditTransaction(
                    transaction_id=ft.transaction_id,
                    user_id=ft.user_id,
                    type=CreditTransactionType(ft.type),
                    amount=ft.amount,
                    description=ft.description,
                    reference_id=ft.reference_id,
                    created_at=ft.created_at,
                )
                transactions.append(transaction)

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
        # Check if user credit record exists
        credits = await self.firestore_service.get_document(
            collection_name="users_credits",
            document_id=user_id,
            model_class=FirestoreUserCredits,
        )

        if not credits:
            # Create initial credit record
            credit_data = {
                "user_id": user_id,
                "balance": 0,
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
    ) -> None:
        """
        Record a credit transaction.

        Args:
            user_id: The user ID for the transaction
            transaction_type: Type of credit transaction
            amount: Amount of credits (positive for earned, negative for spent)
            description: Description of the transaction
            reference_id: Reference ID for the transaction
        """
        transaction_id = str(uuid.uuid4())

        transaction_data = {
            "transaction_id": transaction_id,
            "user_id": user_id,
            "type": transaction_type.value,
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
