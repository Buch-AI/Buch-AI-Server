"""
Creation Data Models

This module contains models related to user creations and associated cost centres.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.models.shared import FirestoreBaseModel


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


class BaseCostCentre(BaseModel):
    """Base cost centre model shared between Firestore and API."""

    cost_centre_id: str = Field(..., description="Unique cost centre identifier")
    creation_id: str = Field(..., description="Associated creation ID")
    user_id: str = Field(..., description="User who owns this cost centre")
    created_at: datetime = Field(..., description="When the cost centre was created")
    cost: float = Field(..., ge=0, description="Total cost tracked by this centre")


class CreationProfile(BaseCreationProfile, FirestoreBaseModel):
    """Creation profile document model for creations_profiles collection."""

    pass


class CostCentre(BaseCostCentre, FirestoreBaseModel):
    """Cost centre document model for creations_cost_centres collection."""

    pass
