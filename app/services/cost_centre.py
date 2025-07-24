import logging
import uuid
from datetime import datetime
from traceback import format_exc

from fastapi import HTTPException

from app.models.firestore import CostCentre as FirestoreCostCentre
from app.models.shared import BaseCreationProfile
from app.services.firestore import get_firestore_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CostCentreManager:
    async def create_cost_centre(
        self,
        creation_id: str,
        user_id: str,
    ) -> str:
        """Generate a new cost centre for a specific creation."""
        # Initialize Firestore service
        firestore_service = get_firestore_service()

        # First, verify the creation belongs to the user using automatic conversion
        creation = await firestore_service.get_document(
            collection_name="creations_profiles",
            document_id=creation_id,
            model_class=BaseCreationProfile,  # API model class - automatic conversion!
        )

        if not creation or creation.user_id != user_id:
            logger.error(
                f"Creation {creation_id} not found or unauthorized\n{format_exc()}"
            )
            raise HTTPException(
                status_code=404,
                detail="Creation not found or you don't have permission to create a cost centre for it",
            )

        # Generate a unique cost centre ID
        cost_centre_id = str(uuid.uuid4())

        # Create the cost centre data
        cost_centre_data = {
            "cost_centre_id": cost_centre_id,
            "creation_id": creation_id,
            "user_id": user_id,
            "created_at": datetime.utcnow(),
            "cost": 0.0,
        }

        # Insert the cost centre into Firestore
        await firestore_service.create_document(
            collection_name="creations_cost_centres",
            document_data=cost_centre_data,
            document_id=cost_centre_id,
        )

        return cost_centre_id

    async def update_cost_centre(self, cost_centre_id: str, cost: float) -> None:
        """Update the cost for a cost centre."""
        if not cost_centre_id:
            return

        try:
            firestore_service = get_firestore_service()

            # Get the current cost centre
            cost_centre = await firestore_service.get_document(
                collection_name="creations_cost_centres",
                document_id=cost_centre_id,
                model_class=FirestoreCostCentre,
            )

            if cost_centre:
                # Update the cost by adding the additional cost
                new_cost = cost_centre.cost + cost
                await firestore_service.update_document(
                    collection_name="creations_cost_centres",
                    document_id=cost_centre_id,
                    update_data={"cost": new_cost},
                )
        except Exception as e:
            logging.error(f"Failed to update cost centre: {e}")
            # Don't raise exception to prevent disrupting the main flow

    async def delete_cost_centre(self, cost_centre_id: str, user_id: str) -> bool:
        """Delete a cost centre if it exists and belongs to the user."""
        if not cost_centre_id:
            return False

        try:
            firestore_service = get_firestore_service()

            # First verify the cost centre belongs to the user
            cost_centre = await firestore_service.get_document(
                collection_name="creations_cost_centres",
                document_id=cost_centre_id,
                model_class=FirestoreCostCentre,
            )

            if not cost_centre or cost_centre.user_id != user_id:
                logger.error(
                    f"Cost centre {cost_centre_id} not found or unauthorized\n{format_exc()}"
                )
                return False

            # Delete the cost centre from Firestore
            await firestore_service.delete_document(
                collection_name="creations_cost_centres", document_id=cost_centre_id
            )

            return True
        except Exception as e:
            logger.error(f"Failed to delete cost centre: {e}\n{format_exc()}")
            return False
