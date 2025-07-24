"""
Credit Data Models

This module contains models related to credit transactions.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from app.models.shared import FirestoreBaseModel


class CreditTransactionType(str, Enum):
    """Credit transaction type enumeration."""

    EARNED_SUBSCRIPTION = "earned_subscription"
    EARNED_BONUS = "earned_bonus"
    SPENT = "spent"


class CreditPool(str, Enum):
    """Credit pool enumeration."""

    MONTHLY = "monthly"
    PERMANENT = "permanent"


class BaseCreditTransaction(BaseModel):
    """Base credit transaction model shared between Firestore and API."""

    transaction_id: str = Field(..., description="Unique transaction identifier")
    user_id: str = Field(..., description="Associated user ID")
    type: str = Field(..., description="Transaction type")
    pool: Optional[str] = Field(
        None, description="Credit pool affected (monthly/permanent)"
    )
    amount: int = Field(
        ..., description="Credit amount (positive for earned, negative for spent)"
    )
    description: Optional[str] = Field(None, description="Transaction description")
    reference_id: Optional[str] = Field(None, description="Reference to related entity")
    created_at: datetime = Field(..., description="Transaction timestamp")


class CreditTransaction(BaseCreditTransaction, FirestoreBaseModel):
    """Credit transaction document model for credits_transactions collection."""

    pass
