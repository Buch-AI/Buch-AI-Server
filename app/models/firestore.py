"""
Firestore Data Models

This module contains Pydantic models that correspond to Firestore collections.
These models ensure schema consistency between the application and Firestore,
replacing the previous BigQuery table schemas.

Collection Mapping:
- BigQuery datasets.tables -> Firestore collections
- creations.cost_centres -> creations_cost_centres
- creations.profiles -> creations_profiles
- tasks.video_generator -> tasks_video_generator
- users.auth -> users_auth
- users.profiles -> users_profiles
- users.geolocation -> users_geolocation
- users.credits -> users_credits
- users.subscriptions -> users_subscriptions
- payments.records -> payments_records
- credits.transactions -> credits_transactions
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

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


# Creation-related models
class CostCentre(FirestoreBaseModel):
    """Cost centre document model for creations_cost_centres collection."""

    cost_centre_id: str = Field(..., description="Unique cost centre identifier")
    creation_id: str = Field(..., description="Associated creation ID")
    user_id: str = Field(..., description="User who owns this cost centre")
    created_at: datetime = Field(..., description="When the cost centre was created")
    cost: float = Field(..., ge=0, description="Total cost tracked by this centre")


class CreationProfile(FirestoreBaseModel):
    """Creation profile document model for creations_profiles collection."""

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


# Task-related models
class VideoGeneratorTask(FirestoreBaseModel):
    """Video generator task document model for tasks_video_generator collection."""

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


# User-related models
class UserAuth(FirestoreBaseModel):
    """User authentication document model for users_auth collection."""

    user_id: str = Field(..., description="Unique user identifier")
    username: str = Field(..., min_length=1, description="Username")
    email: str = Field(..., description="User email address")
    password_hash: str = Field(..., description="Hashed password")
    created_at: datetime = Field(..., description="Account creation timestamp")
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")
    is_active: bool = Field(True, description="Whether the account is active")
    roles: List[str] = Field(default_factory=list, description="User roles")


class SocialLink(BaseModel):
    """Social media link embedded in user profiles."""

    platform: str = Field(..., description="Social media platform name")
    url: str = Field(..., description="Profile URL on the platform")


class UserProfile(FirestoreBaseModel):
    """User profile document model for users_profiles collection."""

    user_id: str = Field(..., description="Unique user identifier")
    display_name: str = Field(..., min_length=1, description="Display name")
    email: str = Field(..., description="User email address")
    bio: Optional[str] = Field(None, description="User bio/description")
    profile_picture_url: Optional[str] = Field(None, description="Profile picture URL")
    created_at: datetime = Field(..., description="Profile creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    is_active: bool = Field(True, description="Whether the profile is active")
    preferences: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="User preferences"
    )
    social_links: List[SocialLink] = Field(
        default_factory=list, description="Social media links"
    )


class UserGeolocation(FirestoreBaseModel):
    """User geolocation document model for users_geolocation collection."""

    user_id: str = Field(..., description="Associated user ID")
    time: datetime = Field(..., description="Timestamp of the geolocation record")
    ipv4: Optional[str] = Field(None, description="IPv4 address")
    geolocation: Optional[str] = Field(None, description="Geolocation string")
    coord_lat: Optional[float] = Field(
        None, ge=-90, le=90, description="Latitude coordinate"
    )
    coord_lon: Optional[float] = Field(
        None, ge=-180, le=180, description="Longitude coordinate"
    )
    country_code: Optional[str] = Field(None, description="ISO country code")
    is_vpn: Optional[bool] = Field(None, description="Whether using VPN")


class UserCredits(FirestoreBaseModel):
    """User credits document model for users_credits collection."""

    user_id: str = Field(..., description="Associated user ID")
    balance: int = Field(..., ge=0, description="Current credit balance")
    total_earned: int = Field(..., ge=0, description="Total credits earned")
    total_spent: int = Field(..., ge=0, description="Total credits spent")
    last_updated: datetime = Field(..., description="Last balance update timestamp")
    created_at: datetime = Field(..., description="Record creation timestamp")


class UserSubscription(FirestoreBaseModel):
    """User subscription document model for users_subscriptions collection."""

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


# Payment-related models
class PaymentRecord(FirestoreBaseModel):
    """Payment record document model for payments_records collection."""

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


class CreditTransaction(FirestoreBaseModel):
    """Credit transaction document model for credits_transactions collection."""

    transaction_id: str = Field(..., description="Unique transaction identifier")
    user_id: str = Field(..., description="Associated user ID")
    type: str = Field(..., description="Transaction type")
    amount: int = Field(
        ..., description="Credit amount (positive for earned, negative for spent)"
    )
    description: Optional[str] = Field(None, description="Transaction description")
    reference_id: Optional[str] = Field(None, description="Reference to related entity")
    created_at: datetime = Field(..., description="Transaction timestamp")


# Collection name mappings for easy reference
COLLECTION_NAMES = {
    "cost_centres": "creations_cost_centres",
    "creation_profiles": "creations_profiles",
    "video_generator_tasks": "tasks_video_generator",
    "user_auth": "users_auth",
    "user_profiles": "users_profiles",
    "user_geolocation": "users_geolocation",
    "user_credits": "users_credits",
    "user_subscriptions": "users_subscriptions",
    "payment_records": "payments_records",
    "credit_transactions": "credits_transactions",
}

# Model mappings for easy reference
COLLECTION_MODELS = {
    "creations_cost_centres": CostCentre,
    "creations_profiles": CreationProfile,
    "tasks_video_generator": VideoGeneratorTask,
    "users_auth": UserAuth,
    "users_profiles": UserProfile,
    "users_geolocation": UserGeolocation,
    "users_credits": UserCredits,
    "users_subscriptions": UserSubscription,
    "payments_records": PaymentRecord,
    "credits_transactions": CreditTransaction,
}
