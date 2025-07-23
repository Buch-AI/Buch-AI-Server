import logging
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
        firestore_creations = await firestore_service.query_collection(
            collection_name="creations_profiles",
            filters=[("user_id", "==", current_user.username)],
            order_by="created_at",
            model_class=FirestoreCreationProfile,
        )

        # Convert Firestore models to API models
        creations = []
        for fc in firestore_creations:
            creation = CreationProfile(
                creation_id=fc.creation_id,
                title=fc.title,
                description=fc.description,
                creator_id=fc.creator_id,
                user_id=fc.user_id,
                created_at=fc.created_at,
                updated_at=fc.updated_at,
                status=fc.status,
                visibility=fc.visibility,
                tags=fc.tags,
                metadata=fc.metadata,
                is_active=fc.is_active,
            )
            creations.append(creation)

        return CreationsResponse(data=creations)

    except Exception as e:
        logger.error(f"Failed to fetch user creations: {str(e)}\n{format_exc()}")
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch user creations: {str(e)}"
        )
