"""
Payment Data Models

This module contains models related to payments and products.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from app.models.shared import FirestoreBaseModel


class PaymentStatus(str, Enum):
    """Payment status enumeration."""

    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BasePaymentRecord(BaseModel):
    """Base payment record model shared between Firestore and API."""

    payment_id: str = Field(..., description="Unique payment identifier")
    user_id: str = Field(..., description="Associated user ID")
    stripe_payment_intent_id: str = Field(..., description="Stripe payment intent ID")
    amount: int = Field(..., gt=0, description="Payment amount in cents")
    currency: str = Field(..., description="Payment currency code")
    status: str = Field(..., description="Payment status")
    product_type: str = Field(..., description="Type of product purchased")
    product_id: str = Field(..., description="Product identifier")
    quantity: int = Field(..., gt=0, description="Quantity purchased")
    description: Optional[str] = Field(None, description="Payment description")
    created_at: datetime = Field(..., description="Payment creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    completed_at: Optional[datetime] = Field(
        None, description="Payment completion timestamp"
    )


class PaymentRecord(BasePaymentRecord, FirestoreBaseModel):
    """Payment record document model for payments_records collection."""

    pass
