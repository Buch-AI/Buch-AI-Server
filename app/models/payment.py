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


class PaymentRecord(BaseModel):
    """
    Payment record model for database storage.

    Corresponds to BigQuery table: `bai-buchai-p.payments.records`
    """

    payment_id: str
    user_id: str
    stripe_payment_intent_id: str
    amount: int
    currency: str
    status: PaymentStatus
    product_type: ProductType
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
    """Product information model from Stripe."""

    product_id: str
    name: str
    description: str
    price: int  # Price in cents
    currency: str = "usd"
    type: ProductType


class CreditBalance(BaseModel):
    """
    User credit balance model.

    Corresponds to BigQuery table: `bai-buchai-p.users.credits`
    """

    user_id: str
    balance: int
    total_earned: int
    total_spent: int
    last_updated: datetime
    created_at: datetime


class CreditTransaction(BaseModel):
    """
    Credit transaction model.

    Corresponds to BigQuery table: `bai-buchai-p.credits.transactions`
    """

    transaction_id: str
    user_id: str
    type: CreditTransactionType
    amount: int
    description: Optional[str] = None
    reference_id: Optional[str] = None
    created_at: datetime


class SubscriptionRecord(BaseModel):
    """
    Subscription record model.

    Corresponds to BigQuery table: `bai-buchai-p.users.subscriptions`
    """

    subscription_id: str
    user_id: str
    stripe_subscription_id: str
    plan_name: str
    status: SubscriptionStatus
    credits_monthly: int
    current_period_start: datetime
    current_period_end: datetime
    cancel_at_period_end: bool = False
    created_at: datetime
    updated_at: datetime


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
