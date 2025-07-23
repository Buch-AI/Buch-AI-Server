import logging
from datetime import datetime
from traceback import format_exc
from typing import Annotated, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.models.firestore import CreationProfile as FirestoreCreationProfile
from app.server.routers.auth_routes import User, get_current_user
from app.server.routers.creation_routes import CreationProfile
from app.services.firestore_service import get_firestore_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a router for user-specific operations
me_router = APIRouter()


class CreationsResponse(BaseModel):
    data: List[CreationProfile]


@me_router.get("/creations", response_model=CreationsResponse)
async def get_user_creations(
    current_user: Annotated[User, Depends(get_current_user)],
) -> CreationsResponse:
    """Get all creations for the current user."""
    # Initialize Firestore service
    firestore_service = get_firestore_service()

    try:
        # Query user's creations from Firestore
        # NOTE: Removed order_by to avoid index requirement, sorting in Python instead
        firestore_creations = await firestore_service.query_collection(
            collection_name="creations_profiles",
            filters=[("user_id", "==", current_user.username)],
            model_class=FirestoreCreationProfile,
        )

        # Sort by created_at in Python (temporary solution until index is created)
        firestore_creations.sort(
            key=lambda x: x.created_at or datetime.min, reverse=True
        )

        # Since both models now inherit from the same base, we can use the Firestore models directly
        creations = firestore_creations

        return CreationsResponse(data=creations)

    except Exception as e:
        logger.error(f"Failed to fetch user creations: {str(e)}\n{format_exc()}")
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch user creations: {str(e)}"
        )
