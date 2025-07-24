"""
Shared Data Models

This module contains shared Pydantic models, base classes, and embedded models
that are used across multiple database schemas or for API responses.
"""

from datetime import datetime
from enum import Enum
from typing import Any, List, Optional

from google.cloud.firestore import DocumentReference
from pydantic import BaseModel, ConfigDict, Field


class FirestoreBaseModel(BaseModel):
    """Base model for all Firestore documents with common configuration."""

    model_config = ConfigDict(
        # Allow population by field name or alias
        populate_by_name=True,
        # Convert datetime objects to timestamps for Firestore
        json_encoders={
            datetime: lambda dt: dt,  # Firestore handles datetime conversion
            DocumentReference: lambda ref: ref.path,  # Convert refs to paths
        },
        # Validate assignments
        validate_assignment=True,
        # Use enum values instead of names
        use_enum_values=True,
    )


# Enums
class SubscriptionStatus(str, Enum):
    """Subscription status enumeration."""

    ACTIVE = "active"
    CANCELED = "canceled"
    PAST_DUE = "past_due"
    UNPAID = "unpaid"
    INCOMPLETE = "incomplete"
    INCOMPLETE_EXPIRED = "incomplete_expired"
    TRIALING = "trialing"


class ProductType(str, Enum):
    """Product type enumeration."""

    BONUS = "bonus"  # One-time Stripe product
    SUBSCRIPTION = "subscription"  # Recurring Stripe product


# Embedded Models
class SocialLink(BaseModel):
    """Social media link embedded in user profiles."""

    platform: str = Field(..., description="Social media platform name")
    url: str = Field(..., description="Profile URL on the platform")


# API Response Models
class CreditBalance(BaseModel):
    """Credit balance model for API responses."""

    user_id: str = Field(..., description="Associated user ID")
    credits_monthly: int = Field(
        ..., ge=0, description="Monthly credit balance (from subscriptions)"
    )
    credits_permanent: int = Field(
        ..., ge=0, description="Permanent credit balance (from bonuses/add-ons)"
    )
    total_earned: int = Field(..., ge=0, description="Total credits earned")
    total_spent: int = Field(..., ge=0, description="Total credits spent")
    last_updated: datetime = Field(..., description="Last balance update timestamp")
    created_at: datetime = Field(..., description="Record creation timestamp")

    @property
    def balance(self) -> int:
        """Total available credits (sum of monthly and permanent)."""
        return self.credits_monthly + self.credits_permanent


class CreditBalanceResponse(BaseModel):
    """Response model for user credit balance."""

    user_id: str
    credits_monthly: int
    credits_permanent: int
    balance: int  # Total of monthly + permanent
    total_earned: int
    total_spent: int
    last_updated: datetime


class SubscriptionRecord(BaseModel):
    """Subscription record model for API responses."""

    subscription_id: str = Field(..., description="Unique subscription identifier")
    user_id: str = Field(..., description="Associated user ID")
    stripe_subscription_id: str = Field(..., description="Stripe subscription ID")
    plan_name: str = Field(..., description="Subscription plan name")
    status: str = Field(..., description="Subscription status")
    credits_monthly: int = Field(..., ge=0, description="Monthly credit allocation")
    current_period_start: datetime = Field(
        ..., description="Current billing period start"
    )
    current_period_end: datetime = Field(..., description="Current billing period end")
    cancel_at_period_end: bool = Field(
        False, description="Whether to cancel at period end"
    )
    created_at: datetime = Field(..., description="Subscription creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class UserSubscriptionResponse(BaseModel):
    """Response model for user subscription information."""

    subscriptions: List[SubscriptionRecord]
    active_subscription: Optional[SubscriptionRecord] = None


class PaymentHistoryResponse(BaseModel):
    """Response model for payment history."""

    payments: List[Any]  # BasePaymentRecord - avoiding circular import
    total_count: int


class ProductInfo(BaseModel):
    """Product information model from Stripe."""

    product_id: str
    name: str
    description: str
    price: int  # Price in cents
    currency: str = "usd"
    type: ProductType
