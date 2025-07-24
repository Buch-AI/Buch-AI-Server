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

from app.models.shared import (
    BaseCostCentre,
    BaseCreationProfile,
    BaseCreditTransaction,
    BasePaymentRecord,
    BaseUserCredits,
    BaseUserSubscription,
    BaseVideoGeneratorTask,
)


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
class CostCentre(BaseCostCentre, FirestoreBaseModel):
    """Cost centre document model for creations_cost_centres collection."""

    pass


class CreationProfile(BaseCreationProfile, FirestoreBaseModel):
    """Creation profile document model for creations_profiles collection."""

    pass


# Task-related models
class VideoGeneratorTask(BaseVideoGeneratorTask, FirestoreBaseModel):
    """Video generator task document model for tasks_video_generator collection."""

    pass


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


class UserCredits(BaseUserCredits, FirestoreBaseModel):
    """User credits document model for users_credits collection."""

    pass


class UserSubscription(BaseUserSubscription, FirestoreBaseModel):
    """User subscription document model for users_subscriptions collection."""

    pass


# Payment-related models
class PaymentRecord(BasePaymentRecord, FirestoreBaseModel):
    """Payment record document model for payments_records collection."""

    pass


class CreditTransaction(BaseCreditTransaction, FirestoreBaseModel):
    """Credit transaction document model for credits_transactions collection."""

    pass


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
