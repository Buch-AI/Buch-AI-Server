import logging
from traceback import format_exc
from typing import Annotated, List

from fastapi import APIRouter, Depends, HTTPException
from google.cloud import bigquery
from pydantic import BaseModel

from app.server.routers.auth_routes import User, get_current_active_user
from app.server.routers.creation_routes import CreationProfile

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Create a router for user-specific operations
me_router = APIRouter()


class CreationsResponse(BaseModel):
    data: List[CreationProfile]


@me_router.get("/creations", response_model=CreationsResponse)
async def get_user_creations(
    current_user: Annotated[User, Depends(get_current_active_user)],
) -> CreationsResponse:
    """Get all creations for the current user."""
    # Initialize BigQuery client
    client = bigquery.Client()

    try:
        query = """
        SELECT *
        FROM `bai-buchai-p.creations.profiles`
        WHERE user_id = @user_id
        ORDER BY created_at DESC
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter(
                    "user_id", "STRING", current_user.username
                )
            ]
        )

        query_job = client.query(query, job_config=job_config)
        results = query_job.result()

        creations = []
        for row in results:
            creation = CreationProfile(
                creation_id=row.creation_id,
                title=row.title,
                description=row.description,
                creator_id=row.creator_id,
                user_id=row.user_id,
                created_at=row.created_at,
                updated_at=row.updated_at,
                status=row.status,
                visibility=row.visibility,
                tags=row.tags,
                metadata=row.metadata,
                is_active=row.is_active,
            )
            creations.append(creation)

        return CreationsResponse(data=creations)

    except Exception as e:
        logger.error(f"Failed to fetch user creations: {str(e)}\n{format_exc()}")
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch user creations: {str(e)}"
        )
