from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel


class PaymentStatus(str, Enum):
    """Payment status enumeration."""

    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PaymentType(str, Enum):
    """Payment type enumeration."""

    # TODO: Payment!
    ONE_TIME = "one_time"
    CREDIT_PURCHASE = "credit_purchase"
    FEATURE_UNLOCK = "feature_unlock"


class PaymentRecord(BaseModel):
    """Payment record model for database storage."""

    payment_id: str
    user_id: str
    stripe_payment_intent_id: str
    amount: int
    currency: str
    status: PaymentStatus
    payment_type: PaymentType
    product_id: str
    quantity: int
    description: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None


class PaymentHistoryResponse(BaseModel):
    """Response model for payment history."""

    payments: list[PaymentRecord]
    total_count: int


class ProductInfo(BaseModel):
    """Product information model."""

    product_id: str
    name: str
    description: str
    price: int  # Price in cents
    currency: str = "usd"
    type: PaymentType
