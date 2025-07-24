"""
User Data Models

This module contains models related to users, authentication, profiles, and subscriptions.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.models.shared import FirestoreBaseModel, SocialLink


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
