from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel

from app.models.shared import (
    BaseCreditTransaction,
    BasePaymentRecord,
    BaseUserCredits,
    BaseUserSubscription,
)


class PaymentStatus(str, Enum):
    """Payment status enumeration."""

    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProductType(str, Enum):
    """Product type enumeration."""

    BONUS = "bonus"  # One-time Stripe product
    SUBSCRIPTION = "subscription"  # Recurring Stripe product


class SubscriptionStatus(str, Enum):
    """Subscription status enumeration."""

    ACTIVE = "active"
    CANCELED = "canceled"
    PAST_DUE = "past_due"
    UNPAID = "unpaid"
    INCOMPLETE = "incomplete"
    INCOMPLETE_EXPIRED = "incomplete_expired"
    TRIALING = "trialing"


class CreditTransactionType(str, Enum):
    """Credit transaction type enumeration."""

    EARNED_SUBSCRIPTION = "earned_subscription"
    EARNED_BONUS = "earned_bonus"
    SPENT = "spent"


# Use the shared base model for API responses
PaymentRecord = BasePaymentRecord


class PaymentHistoryResponse(BaseModel):
    """Response model for payment history."""

    payments: list[PaymentRecord]
    total_count: int


class ProductInfo(BaseModel):
    """Product information model from Stripe."""

    product_id: str
    name: str
    description: str
    price: int  # Price in cents
    currency: str = "usd"
    type: ProductType


# Use the shared base model for API responses
CreditBalance = BaseUserCredits


# Use the shared base model for API responses
CreditTransaction = BaseCreditTransaction


# Use the shared base model for API responses
SubscriptionRecord = BaseUserSubscription


class CreditBalanceResponse(BaseModel):
    """Response model for user credit balance."""

    user_id: str
    balance: int
    total_earned: int
    total_spent: int
    last_updated: datetime


class UserSubscriptionResponse(BaseModel):
    """Response model for user subscription information."""

    subscriptions: list[SubscriptionRecord]
    active_subscription: Optional[SubscriptionRecord] = None
