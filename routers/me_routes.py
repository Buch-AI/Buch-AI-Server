import uuid
from io import BytesIO
from typing import Annotated, List

from fastapi import APIRouter, Depends, HTTPException
from google.cloud import bigquery, storage
from pydantic import BaseModel, HttpUrl

from .auth_routes import User, get_current_active_user
from .image_routes import ImageGenerationRequest, generate_image
from .llm_routes import GenerateStoryRequest, generate_story_string, split_story

# Create a router for user-specific operations
me_router = APIRouter()

# Constants
BUCKET_NAME = "bai-buchai-p-stb-usea1-creations"


class ImageData(BaseModel):
    data: bytes


class CreationResponse(BaseModel):
    data: str


class StoryPartsResponse(BaseModel):
    data: List[str]


class ImagesResponse(BaseModel):
    data: List[bytes]


class VideoResponse(BaseModel):
    data: HttpUrl


class GenerateResponse(BaseModel):
    data: HttpUrl


@me_router.get("/me/creations", response_model=List[str])
async def get_user_creations(
    current_user: Annotated[User, Depends(get_current_active_user)],
) -> List[str]:
    """Get all creation IDs for the current user."""
    # Initialize BigQuery client
    client = bigquery.Client()

    try:
        query = """
        SELECT creation_id
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

        return [row.creation_id for row in results]

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch user creations: {str(e)}"
        )


@me_router.post(
    "/me/creation/{creation_id}/set_story_parts", response_model=StoryPartsResponse
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
    "/me/creation/{creation_id}/get_story_parts", response_model=StoryPartsResponse
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


@me_router.post("/me/creation/{creation_id}/set_images", response_model=ImagesResponse)
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

        # Upload each image
        for index, image in enumerate(images, start=1):
            # Upload to Google Cloud Storage
            blob_path = f"{creation_id}/assets/{index}/image.png"
            blob = bucket.blob(blob_path)
            blob.upload_from_file(BytesIO(image.data))

        return ImagesResponse(data=[img.data for img in images])

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to set images: {str(e)}")


@me_router.get("/me/creation/{creation_id}/get_images", response_model=ImagesResponse)
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

            images.append(blob.download_as_bytes())
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


@me_router.get("/me/creation/{creation_id}/get_video", response_model=VideoResponse)
async def get_video(
    creation_id: str, current_user: Annotated[User, Depends(get_current_active_user)]
) -> VideoResponse:
    """Get video URL for a specific creation."""
    raise HTTPException(status_code=500, detail="This endpoint is not implemented yet!")


@me_router.post("/me/creation/generate", response_model=GenerateResponse)
async def generate_creation(
    request: GenerateStoryRequest,
    current_user: Annotated[User, Depends(get_current_active_user)],
) -> GenerateResponse:
    """Generate a new creation."""
    creation_id: str = str(uuid.uuid4())

    story: str = await generate_story_string(request)

    story_parts: List[str] = await split_story(
        GenerateStoryRequest(prompt=story, model_type=request.model_type)
    )
    await set_story_parts(
        creation_id=creation_id, story_parts=story_parts, current_user=current_user
    )

    images: List[bytes] = [
        (await generate_image(ImageGenerationRequest(prompt=story_part))).data
        for story_part in story_parts
    ]
    await set_images(creation_id=creation_id, images=images, current_user=current_user)

    raise HTTPException(status_code=500, detail="This endpoint is not implemented yet!")
