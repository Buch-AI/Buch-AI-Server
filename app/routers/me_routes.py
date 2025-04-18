import base64
import logging
import re
import uuid
from datetime import datetime
from io import BytesIO
from traceback import format_exc
from typing import Annotated, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from google.cloud import bigquery, run_v2, storage
from google.cloud.run_v2.types.condition import Condition
from PIL import Image
from pydantic import BaseModel

from app.routers.auth_routes import User, get_current_active_user
from app.tasks.video_generator.main import VideoGenerator
from config import ENV, GCLOUD_STB_CREATIONS_NAME

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

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
    data: List[List[str]]


class ImagesResponse(BaseModel):
    data: List[str]  # Base64 encoded PNG images


class VideoResponse(BaseModel):
    data: str  # Base64 encoded video


class GenerateResponse(BaseModel):
    data: str  # Base64 encoded video


class TaskStatus(BaseModel):
    """Model representing a Google Cloud Run Job status."""

    status: str  # pending, running, completed, failed
    message: Optional[str] = None


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


@me_router.post(
    "/creation/{creation_id}/set_story_parts", response_model=StoryPartsResponse
)
async def set_story_parts(
    creation_id: str,
    story_parts: List[List[str]],
    current_user: Annotated[User, Depends(get_current_active_user)],
) -> StoryPartsResponse:
    """Set story parts for a specific creation.

    Each story part will be saved as numbered text files (1.txt, 2.txt, etc.) in its respective
    numbered directory to maintain compatibility with VideoGenerator's expected structure.
    Each inner list represents the sub-parts for a part, which will be saved as separate files.
    """
    try:
        # Initialize client
        storage_client = storage.Client()

        # Get bucket reference
        bucket = storage_client.bucket(GCLOUD_STB_CREATIONS_NAME)

        # Upload each part and its sub-parts
        for part_index, sub_parts in enumerate(story_parts, start=1):
            # Upload each sub-part as a numbered text file
            for sub_part_index, sub_part in enumerate(sub_parts, start=1):
                if not sub_part.strip():  # Skip empty sub-parts
                    continue

                blob_path = f"{creation_id}/assets/{part_index}/{sub_part_index}.txt"
                blob = bucket.blob(blob_path)
                blob.upload_from_string(sub_part)

        return StoryPartsResponse(data=story_parts)

    except Exception as e:
        logger.error(f"Failed to set story parts: {str(e)}\n{format_exc()}")
        raise HTTPException(
            status_code=500, detail=f"Failed to set story parts: {str(e)}"
        )


@me_router.get(
    "/creation/{creation_id}/get_story_parts", response_model=StoryPartsResponse
)
async def get_story_parts(
    creation_id: str, current_user: Annotated[User, Depends(get_current_active_user)]
) -> StoryPartsResponse:
    """Get story parts for a specific creation.

    Reads all TXT files from each numbered directory to maintain compatibility with
    VideoGenerator's expected structure. Returns a list of lists where each inner list
    contains all the text parts from a single directory.
    """
    try:
        # Initialize client
        storage_client = storage.Client()

        # Get bucket reference
        bucket = storage_client.bucket(GCLOUD_STB_CREATIONS_NAME)
        story_parts = []
        part_number = 1

        # Keep reading parts until we don't find the next directory
        while True:
            part_texts = []
            base_path = f"{creation_id}/assets/{part_number}/"

            # List all blobs in this directory
            blobs = list(bucket.list_blobs(prefix=base_path))
            txt_blobs = [blob for blob in blobs if blob.name.endswith(".txt")]

            if not txt_blobs:
                break

            # Sort the text files numerically
            txt_blobs.sort(key=lambda x: int(x.name.split("/")[-1].split(".")[0]))

            # Read each text file in order
            for blob in txt_blobs:
                text = blob.download_as_text().strip()
                if text:  # Only add non-empty texts
                    part_texts.append(text)

            if part_texts:  # Only add parts that have text
                story_parts.append(part_texts)

            part_number += 1

        if not story_parts:
            logger.error(
                f"No story parts found for creation {creation_id}\n{format_exc()}"
            )
            raise HTTPException(
                status_code=404, detail="No story parts found for this creation"
            )

        return StoryPartsResponse(data=story_parts)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get story parts: {str(e)}\n{format_exc()}")
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
        bucket = storage_client.bucket(GCLOUD_STB_CREATIONS_NAME)
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
                logger.error(
                    f"Failed to process image {index}: {str(img_error)}\n{format_exc()}"
                )
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to process image {index}: {str(img_error)}",
                )

        return ImagesResponse(data=processed_images)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to set images: {str(e)}\n{format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to set images: {str(e)}")


@me_router.get("/creation/{creation_id}/get_images", response_model=ImagesResponse)
async def get_images(
    creation_id: str, current_user: Annotated[User, Depends(get_current_active_user)]
) -> ImagesResponse:
    """Get images for a specific creation."""
    try:
        # Initialize client
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCLOUD_STB_CREATIONS_NAME)
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
            logger.error(f"No images found for creation {creation_id}\n{format_exc()}")
            raise HTTPException(
                status_code=404, detail="No images found for this creation"
            )

        return ImagesResponse(data=images)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get images: {str(e)}\n{format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to get images: {str(e)}")


@me_router.get("/creation/{creation_id}/get_video", response_model=VideoResponse)
async def get_video(
    creation_id: str, current_user: Annotated[User, Depends(get_current_active_user)]
) -> VideoResponse:
    """Get video for a specific creation."""
    try:
        # Initialize client
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCLOUD_STB_CREATIONS_NAME)

        # Get video blob
        video_blob_path = f"{creation_id}/assets/video.mp4"
        video_blob = bucket.blob(video_blob_path)

        if not video_blob.exists():
            logger.error(f"No video found for creation {creation_id}\n{format_exc()}")
            raise HTTPException(
                status_code=404, detail="No video found for this creation"
            )

        # Download video bytes
        video_bytes = video_blob.download_as_bytes()

        # Convert to base64
        video_base64 = base64.b64encode(video_bytes).decode("utf-8")

        return VideoResponse(data=f"data:video/mp4;base64,{video_base64}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get video: {str(e)}\n{format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to get video: {str(e)}")


@me_router.get(
    "/creation/{creation_id}/generate_video_status", response_model=TaskStatus
)
async def generate_video_status(
    creation_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
) -> TaskStatus:
    """Get the status of video generation for a creation."""
    try:
        if ENV == "p":
            # Initialize BigQuery client
            bq_client = bigquery.Client()

            # First, get the task record from BigQuery
            query = """
            SELECT execution_id, status
            FROM `bai-buchai-p.tasks.video_generator`
            WHERE creation_id = @creation_id
            ORDER BY created_at DESC
            LIMIT 1
            """

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("creation_id", "STRING", creation_id),
                ]
            )

            query_job = bq_client.query(query, job_config=job_config)
            results = list(query_job.result())

            if not results:
                return TaskStatus(
                    status="not_found", message="No task found for this creation"
                )

            task = results[0]
            execution_name = task.execution_id
            current_status = task.status

            # Initialize Cloud Run Executions client
            client = run_v2.ExecutionsClient()

            try:
                # Get the execution status using the full execution name
                execution = client.get_execution(name=execution_name)

                if not execution.conditions:
                    return TaskStatus(
                        status=current_status,
                        message="Job execution status not available",
                    )

                # Get the latest condition
                latest_condition = execution.conditions[-1]

                # Map condition to status
                if latest_condition.state == Condition.State.CONDITION_FAILED:
                    new_status = "failed"
                elif (
                    latest_condition.state == Condition.State.CONDITION_SUCCEEDED
                    and latest_condition.type_ == "ResourcesAvailable"
                ):
                    new_status = "completed"
                elif (
                    latest_condition.state == Condition.State.CONDITION_SUCCEEDED
                    and latest_condition.type_ == "Retry"
                ):
                    new_status = "running"
                else:
                    # Unknown state
                    new_status = current_status

                message = latest_condition.reason if new_status == "failed" else None

                # Update the task status in BigQuery if it has changed
                if new_status != current_status:
                    update_query = """
                    UPDATE `bai-buchai-p.tasks.video_generator`
                    SET status = @new_status,
                        updated_at = CURRENT_TIMESTAMP()
                    WHERE creation_id = @creation_id
                    AND execution_id = @execution_id
                    """

                    update_config = bigquery.QueryJobConfig(
                        query_parameters=[
                            bigquery.ScalarQueryParameter(
                                "new_status", "STRING", new_status
                            ),
                            bigquery.ScalarQueryParameter(
                                "creation_id", "STRING", creation_id
                            ),
                            bigquery.ScalarQueryParameter(
                                "execution_id", "STRING", execution_name
                            ),
                        ]
                    )

                    bq_client.query(update_query, update_config).result()

                return TaskStatus(status=new_status, message=message)

            except Exception as e:
                if "NOT_FOUND" in str(e):
                    return TaskStatus(
                        status="not_found",
                        message="No job execution found for this creation",
                    )
                raise

        else:
            # In development, check if video exists
            storage_client = storage.Client()
            bucket = storage_client.bucket(GCLOUD_STB_CREATIONS_NAME)
            video_blob = bucket.blob(f"{creation_id}/assets/video.mp4")

            if video_blob.exists():
                return TaskStatus(status="completed")
            return TaskStatus(status="running")

    except Exception as e:
        logger.error(f"Failed to get video status: {str(e)}\n{format_exc()}")
        return TaskStatus(status="failed", message=str(e))


@me_router.get("/creation/{creation_id}/generate_video", response_model=TaskStatus)
async def generate_video(
    creation_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
) -> TaskStatus:
    """Generate a video for a specific creation.

    In production (ENV='p'), this triggers a Cloud Run Job.
    In development (ENV='d'), this runs the video generation locally.
    """
    try:
        if ENV == "p":
            # Initialize Cloud Run client
            client = run_v2.JobsClient()

            parent = "projects/bai-buchai-p/locations/us-east1"
            name = f"{parent}/jobs/bai-buchai-p-crj-usea1-vidgen"

            # Execute the existing job
            operation = client.run_job(
                run_v2.RunJobRequest(
                    name=name,
                    overrides={
                        "container_overrides": [
                            {
                                "args": [creation_id],
                                "env": [{"name": "ENV", "value": f"{ENV}"}],
                            }
                        ],
                        "task_count": 1,
                    },
                )
            )
            # Access the metadata (which contains the execution info)
            metadata = operation.metadata

            # This is a google.cloud.run_v2.types.RunJobMetadata object
            execution_name = metadata.name  # Full resource name

            # Initialize BigQuery client
            bq_client = bigquery.Client()

            # Insert initial task record
            query = """
            INSERT INTO `bai-buchai-p.tasks.video_generator` (
                creation_id, execution_id, created_at, updated_at, status, metadata
            )
            VALUES (
                @creation_id,
                @execution_id,
                CURRENT_TIMESTAMP(),
                CURRENT_TIMESTAMP(),
                'pending',
                JSON_OBJECT()
            )
            """

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("creation_id", "STRING", creation_id),
                    bigquery.ScalarQueryParameter(
                        "execution_id", "STRING", execution_name
                    ),
                ]
            )

            bq_client.query(query, job_config=job_config).result()

            # Return immediate response
            return TaskStatus(
                status="pending",
                message=f"Job started. Check status at /me/creation/{creation_id}/generate_video_status.",
            )

        else:
            # Development: Run locally
            # Use VideoGenerator to load assets directly from GCS
            gcs_path = f"gs://{GCLOUD_STB_CREATIONS_NAME}/{creation_id}/assets"
            slides = VideoGenerator.load_assets_from_directory(gcs_path)

            if not slides:
                logger.error(
                    f"No valid slides found for creation {creation_id}\n{format_exc()}"
                )
                raise HTTPException(
                    status_code=404,
                    detail="No valid slides found for this creation",
                )

            # Generate video using VideoGenerator
            video_bytes = VideoGenerator.create_video_from_slides(slides=slides)

            # Upload video to Google Cloud Storage
            storage_client = storage.Client()
            bucket = storage_client.bucket(GCLOUD_STB_CREATIONS_NAME)
            video_blob_path = f"{creation_id}/assets/video.mp4"
            video_blob = bucket.blob(video_blob_path)
            video_blob.upload_from_string(video_bytes, content_type="video/mp4")

            return TaskStatus(status="completed")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate video: {str(e)}\n{format_exc()}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate video: {str(e)}"
        )


# TODO: Rename this.
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
        logger.error(f"Failed to generate creation: {str(e)}\n{format_exc()}")
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
            logger.error(
                f"Creation {creation_id} not found or unauthorized\n{format_exc()}"
            )
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
        bucket = storage_client.bucket(GCLOUD_STB_CREATIONS_NAME)
        blobs = bucket.list_blobs(prefix=f"{creation_id}/")
        for blob in blobs:
            blob.delete()

        return CreationResponse(data=creation_id)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete creation: {str(e)}\n{format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete creation: {str(e)}",
        )
