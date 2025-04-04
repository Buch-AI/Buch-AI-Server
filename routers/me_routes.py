import base64
import re
import uuid
from datetime import datetime
from io import BytesIO
from typing import Annotated, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from google.cloud import bigquery, storage
from PIL import Image
from pydantic import BaseModel, HttpUrl

from .auth_routes import User, get_current_active_user

# Create a router for user-specific operations
me_router = APIRouter()

# Constants
BUCKET_NAME = "bai-buchai-p-stb-usea1-creations"


class ImageData(BaseModel):
    data: str


class CreationProfile(BaseModel):
    """Model representing a creation profile."""

    creation_id: str
    title: str
    description: Optional[str] = None
    creator_id: str
    user_id: str
    created_at: datetime
    updated_at: datetime
    status: str  # draft, published, archived
    visibility: str  # public, private
    tags: List[str]
    metadata: Optional[Dict] = None
    is_active: bool


class CreationResponse(BaseModel):
    data: str


class CreationsResponse(BaseModel):
    data: List[CreationProfile]


class StoryPartsResponse(BaseModel):
    data: List[str]


class ImagesResponse(BaseModel):
    data: List[str]  # Base64 encoded PNG images


class VideoResponse(BaseModel):
    data: HttpUrl


class GenerateResponse(BaseModel):
    data: HttpUrl


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
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch user creations: {str(e)}"
        )


@me_router.post(
    "/creation/{creation_id}/set_story_parts", response_model=StoryPartsResponse
)
async def set_story_parts(
    creation_id: str,
    story_parts: List[str],
    current_user: Annotated[User, Depends(get_current_active_user)],
) -> StoryPartsResponse:
    """Set story parts for a specific creation."""
    try:
        # Initialize client
        storage_client = storage.Client()

        # Get bucket reference
        bucket = storage_client.bucket(BUCKET_NAME)

        # Upload each story part
        for index, story_part in enumerate(story_parts, start=1):
            blob_path = f"{creation_id}/assets/{index}/story.txt"
            blob = bucket.blob(blob_path)
            blob.upload_from_string(story_part)

        return StoryPartsResponse(data=story_parts)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to set story parts: {str(e)}"
        )


@me_router.get(
    "/creation/{creation_id}/get_story_parts", response_model=StoryPartsResponse
)
async def get_story_parts(
    creation_id: str, current_user: Annotated[User, Depends(get_current_active_user)]
) -> StoryPartsResponse:
    """Get story parts for a specific creation."""
    try:
        # Initialize client
        storage_client = storage.Client()

        # Get bucket reference
        bucket = storage_client.bucket(BUCKET_NAME)
        story_parts = []
        part_number = 1

        # Keep reading parts until we don't find the next one
        while True:
            blob_path = f"{creation_id}/assets/{part_number}/story.txt"
            blob = bucket.blob(blob_path)

            if not blob.exists():
                break

            story_parts.append(blob.download_as_text())
            part_number += 1

        if not story_parts:
            raise HTTPException(
                status_code=404, detail="No story parts found for this creation"
            )

        return StoryPartsResponse(data=story_parts)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get story parts: {str(e)}"
        )


@me_router.post("/creation/{creation_id}/set_images", response_model=ImagesResponse)
async def set_images(
    creation_id: str,
    images: List[ImageData],
    current_user: Annotated[User, Depends(get_current_active_user)],
) -> ImagesResponse:
    """Set images for a specific creation."""
    try:
        # Initialize client
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        processed_images = []

        # Upload each image
        for index, image in enumerate(images, start=1):
            try:
                # Check if we have valid image data
                if len(image.data) == 0:
                    raise ValueError(f"Image {index} is empty")

                # Convert data URL to bytes if needed
                img_bytes = BytesIO()
                if isinstance(image.data, bytes):
                    data_str = image.data.decode("utf-8")
                else:
                    data_str = image.data

                if data_str.startswith("data:"):
                    # Extract the base64 data from the data URL
                    pattern = r"data:image/[^;]+;base64,(.+)"
                    match = re.match(pattern, data_str)
                    if not match:
                        raise ValueError(f"Image {index} has invalid data URL format")

                    # Decode base64 data
                    img_data = base64.b64decode(match.group(1))
                    img_bytes = BytesIO(img_data)
                else:
                    # Handle raw base64 string
                    try:
                        img_data = base64.b64decode(data_str)
                        img_bytes = BytesIO(img_data)
                    except Exception:
                        raise ValueError(f"Image {index} has invalid base64 encoding")

                # Open and validate the image
                try:
                    img = Image.open(img_bytes)
                    img.verify()  # Verify it's a valid image
                    img_bytes.seek(0)  # Reset buffer position
                    img = Image.open(img_bytes)  # Reopen after verify
                except Exception as e:
                    raise ValueError(f"Image {index} is not a valid image: {str(e)}")

                # Convert to PNG format
                png_buffer = BytesIO()
                img.convert("RGBA" if img.mode == "RGBA" else "RGB").save(
                    png_buffer, format="PNG"
                )
                png_buffer.seek(0)

                # Convert PNG to base64 for response
                png_base64 = base64.b64encode(png_buffer.getvalue()).decode("utf-8")
                processed_images.append(f"data:image/png;base64,{png_base64}")

                # Upload to Google Cloud Storage
                png_buffer.seek(0)
                blob_path = f"{creation_id}/assets/{index}/image.png"
                blob = bucket.blob(blob_path)
                blob.upload_from_file(png_buffer, content_type="image/png")

            except Exception as img_error:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to process image {index}: {str(img_error)}",
                )

        return ImagesResponse(data=processed_images)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to set images: {str(e)}")


@me_router.get("/creation/{creation_id}/get_images", response_model=ImagesResponse)
async def get_images(
    creation_id: str, current_user: Annotated[User, Depends(get_current_active_user)]
) -> ImagesResponse:
    """Get images for a specific creation."""
    try:
        # Initialize client
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        images = []
        part_number = 1

        # Keep reading parts until we don't find the next one
        while True:
            blob_path = f"{creation_id}/assets/{part_number}/image.png"
            blob = bucket.blob(blob_path)

            if not blob.exists():
                break

            # Download image bytes
            img_bytes = blob.download_as_bytes()

            # Convert to base64 data URL
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")
            images.append(f"data:image/png;base64,{img_base64}")

            part_number += 1

        if not images:
            raise HTTPException(
                status_code=404, detail="No images found for this creation"
            )

        return ImagesResponse(data=images)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get images: {str(e)}")


@me_router.get("/creation/{creation_id}/get_video", response_model=VideoResponse)
async def get_video(
    creation_id: str, current_user: Annotated[User, Depends(get_current_active_user)]
) -> VideoResponse:
    """Get video URL for a specific creation."""
    raise HTTPException(status_code=500, detail="This endpoint is not implemented yet!")


@me_router.post("/creation/generate", response_model=CreationResponse)
async def generate_creation(
    current_user: Annotated[User, Depends(get_current_active_user)],
) -> CreationResponse:
    """Generate a new creation."""
    try:
        # Initialize clients
        bigquery_client = bigquery.Client()

        # Generate a unique creation ID
        creation_id = str(uuid.uuid4())

        # Insert the creation profile
        query = """
        INSERT INTO `bai-buchai-p.creations.profiles` (
            creation_id, title, description, creator_id, user_id, 
            created_at, updated_at, status, visibility, tags, metadata, is_active
        )
        VALUES (
            @creation_id, 
            @creation_id,
            'No description provided',
            @user_id, 
            @user_id, 
            CURRENT_TIMESTAMP(), 
            CURRENT_TIMESTAMP(), 
            'draft', 
            'private', 
            [], 
            JSON_OBJECT(), 
            TRUE
        )
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("creation_id", "STRING", creation_id),
                bigquery.ScalarQueryParameter(
                    "user_id", "STRING", current_user.username
                ),
            ]
        )

        query_job = bigquery_client.query(query, job_config=job_config)
        query_job.result()  # Wait for the query to complete

        return CreationResponse(data=creation_id)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to generate creation: {str(e)}"
        )


@me_router.delete("/creation/{creation_id}/delete", response_model=CreationResponse)
async def delete_creation(
    creation_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
) -> CreationResponse:
    """Delete a creation and all its associated data."""
    try:
        # Initialize clients
        bigquery_client = bigquery.Client()
        storage_client = storage.Client()

        # First, verify the creation belongs to the user
        verify_query = """
        SELECT creation_id
        FROM `bai-buchai-p.creations.profiles`
        WHERE creation_id = @creation_id
        AND user_id = @user_id
        """

        verify_job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("creation_id", "STRING", creation_id),
                bigquery.ScalarQueryParameter(
                    "user_id", "STRING", current_user.username
                ),
            ]
        )

        verify_job = bigquery_client.query(verify_query, verify_job_config)
        if not list(verify_job.result()):
            raise HTTPException(
                status_code=404,
                detail="Creation not found or you don't have permission to delete it",
            )

        # Delete from BigQuery
        delete_query = """
        DELETE FROM `bai-buchai-p.creations.profiles`
        WHERE creation_id = @creation_id
        AND user_id = @user_id
        """

        delete_job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("creation_id", "STRING", creation_id),
                bigquery.ScalarQueryParameter(
                    "user_id", "STRING", current_user.username
                ),
            ]
        )

        delete_job = bigquery_client.query(delete_query, delete_job_config)
        delete_job.result()  # Wait for the query to complete

        # Delete from Google Cloud Storage
        bucket = storage_client.bucket(BUCKET_NAME)
        blobs = bucket.list_blobs(prefix=f"{creation_id}/")
        for blob in blobs:
            blob.delete()

        return CreationResponse(data=creation_id)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete creation: {str(e)}",
        )
