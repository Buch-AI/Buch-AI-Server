"""
Task Data Models

This module contains models related to background tasks and job processing.
"""

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from app.models.shared import FirestoreBaseModel


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


class VideoGeneratorTask(BaseVideoGeneratorTask, FirestoreBaseModel):
    """Video generator task document model for tasks_video_generator collection."""

    pass
