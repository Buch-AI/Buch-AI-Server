"""
Shared Data Models

This module contains shared Pydantic models that serve as the single source of truth
for data structures. These models can be used for both Firestore storage and API responses,
eliminating field duplication between Firestore and API models.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class BaseCreationProfile(BaseModel):
    """Base creation profile model shared between Firestore and API."""

    creation_id: str = Field(..., description="Unique creation identifier")
    title: str = Field(..., min_length=1, description="Creation title")
    description: Optional[str] = Field(None, description="Creation description")
    creator_id: str = Field(..., description="ID of the user who created this")
    user_id: str = Field(..., description="Current owner user ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    status: str = Field(..., description="Creation status (draft, published, etc.)")
    visibility: str = Field(
        ..., description="Visibility setting (public, private, etc.)"
    )
    tags: List[str] = Field(
        default_factory=list, description="Tags associated with creation"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional metadata"
    )
    is_active: bool = Field(True, description="Whether the creation is active")


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


class BaseCreditTransaction(BaseModel):
    """Base credit transaction model shared between Firestore and API."""

    transaction_id: str = Field(..., description="Unique transaction identifier")
    user_id: str = Field(..., description="Associated user ID")
    type: str = Field(..., description="Transaction type")
    amount: int = Field(
        ..., description="Credit amount (positive for earned, negative for spent)"
    )
    description: Optional[str] = Field(None, description="Transaction description")
    reference_id: Optional[str] = Field(None, description="Reference to related entity")
    created_at: datetime = Field(..., description="Transaction timestamp")


class BaseUserCredits(BaseModel):
    """Base user credits model shared between Firestore and API."""

    user_id: str = Field(..., description="Associated user ID")
    balance: int = Field(..., ge=0, description="Current credit balance")
    total_earned: int = Field(..., ge=0, description="Total credits earned")
    total_spent: int = Field(..., ge=0, description="Total credits spent")
    last_updated: datetime = Field(..., description="Last balance update timestamp")
    created_at: datetime = Field(..., description="Record creation timestamp")


class BaseUserSubscription(BaseModel):
    """Base user subscription model shared between Firestore and API."""

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


class BaseCostCentre(BaseModel):
    """Base cost centre model shared between Firestore and API."""

    cost_centre_id: str = Field(..., description="Unique cost centre identifier")
    creation_id: str = Field(..., description="Associated creation ID")
    user_id: str = Field(..., description="User who owns this cost centre")
    created_at: datetime = Field(..., description="When the cost centre was created")
    cost: float = Field(..., ge=0, description="Total cost tracked by this centre")


class BaseVideoGeneratorTask(BaseModel):
    """Base video generator task model shared between Firestore and API."""

    creation_id: str = Field(..., description="Associated creation ID")
    execution_id: str = Field(..., description="Unique execution identifier")
    created_at: datetime = Field(..., description="Task creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    status: str = Field(
        ..., description="Task status (pending, running, completed, failed)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Task metadata"
    )
