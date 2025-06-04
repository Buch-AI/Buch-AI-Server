import inspect
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, List, Optional

from google.cloud import storage
from pydantic import BaseModel

from app.models.llm import ModelType
from config import ENV, GCLOUD_STB_CREATIONS_NAME, PROJECT_ROOT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GenerateStoryRequest(BaseModel):
    prompt: str
    model_type: ModelType = ModelType.LITE
    cost_centre_id: Optional[str] = None


class SummariseStoryRequest(BaseModel):
    story: str
    model_type: ModelType = ModelType.LITE
    cost_centre_id: Optional[str] = None


class GenerateImagePromptsRequest(BaseModel):
    story: str
    story_parts: List[str]
    model_type: ModelType = ModelType.LITE
    cost_centre_id: Optional[str] = None


class CostUsage(BaseModel):
    """Model for tracking token usage in LLM calls."""

    embedding_tokens: int = 0
    generation_prompt_tokens: int = 0
    generation_completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0


class TextResponse(BaseModel):
    """Response model for endpoints that return text."""

    text: str
    usage: CostUsage


class SplitStoryResponse(BaseModel):
    """Response model for split_story endpoint."""

    data: List[List[str]]
    usage: CostUsage


class ImagePromptsResponse(BaseModel):
    """Response model for generate_image_prompts endpoint."""

    data: List[str]
    usage: CostUsage


class LlmLogger:
    """Utility class for logging LLM prompts and uploading logs to Cloud Storage."""

    @staticmethod
    def log_prompt(cost_centre_id: str, prompt: str) -> None:
        """Log a prompt to a local file using cost_centre_id as filename.

        Args:
            cost_centre_id: The cost centre ID to use as filename
            prompt: The prompt text to log
        """
        # TODO: Remove this once we have a proper logging system.
        if ENV != "d":
            return

        if not cost_centre_id or not cost_centre_id.strip():
            logger.error("cost_centre_id cannot be empty or None")
            return

        # Get calling function information
        caller_frame = inspect.currentframe().f_back
        caller_function = caller_frame.f_code.co_name
        caller_filename = caller_frame.f_code.co_filename.split("/")[
            -1
        ]  # Get just the filename
        caller_line = caller_frame.f_lineno
        caller_info = f"{caller_filename}:{caller_function}:{caller_line}"

        # Create logs directory if it doesn't exist
        logs_dir = Path(PROJECT_ROOT) / "logs"
        logs_dir.mkdir(exist_ok=True)

        # Create log file path using cost_centre_id as filename
        log_file_path = logs_dir / f"{cost_centre_id.strip()}.log"

        # Prepare log entry with timestamp and caller info
        timestamp = datetime.utcnow().isoformat()
        log_entry = f"[{timestamp}] [{caller_info}]\n\n{prompt}\n\n\n\n"

        try:
            # Append to log file (create if doesn't exist)
            with open(log_file_path, "a", encoding="utf-8") as log_file:
                log_file.write(log_entry)
        except OSError as e:
            logger.error(f"Failed to write to log file {log_file_path}: {str(e)}")

    @staticmethod
    def submit_logs(cost_centre_id: str) -> None:
        """Upload a log file to Google Cloud Storage.

        TODO: This is not being used anywhere yet.

        Args:
            cost_centre_id: The cost centre ID (used as filename)
        """
        if not cost_centre_id or not cost_centre_id.strip():
            logger.error("cost_centre_id cannot be empty or None")
            return

        # Construct local log file path
        logs_dir = Path(PROJECT_ROOT) / "logs"
        log_file_path = logs_dir / f"{cost_centre_id.strip()}.log"

        if not log_file_path.exists():
            logger.error(f"Log file does not exist: {log_file_path}")
            return

        try:
            # Initialize Google Cloud Storage client
            storage_client = storage.Client()
            bucket = storage_client.bucket(GCLOUD_STB_CREATIONS_NAME)

            # Define GCS blob path
            blob_path = f"logs/{cost_centre_id.strip()}.log"
            blob = bucket.blob(blob_path)

            # Upload the file
            with open(log_file_path, "rb") as log_file:
                blob.upload_from_file(log_file, content_type="text/plain")

            gcs_path = f"gs://{GCLOUD_STB_CREATIONS_NAME}/{blob_path}"
            logger.info(f"Successfully uploaded log file to Cloud Storage: {gcs_path}")

        except Exception as e:
            logger.error(f"Failed to upload log file to Cloud Storage: {str(e)}")


class LlmRouterService(ABC):
    """Abstract base class for LLM router services."""

    @abstractmethod
    async def generate_story_string(
        self, request: GenerateStoryRequest
    ) -> TextResponse:
        """Generate a story as a single string."""
        pass

    @abstractmethod
    async def generate_story_stream(
        self, request: GenerateStoryRequest
    ) -> AsyncGenerator[str, None]:
        """Generate a story as a stream of text chunks."""
        pass

    @abstractmethod
    async def split_story(self, request: GenerateStoryRequest) -> SplitStoryResponse:
        """Generate a story split into parts and sub-parts."""
        pass

    @abstractmethod
    async def summarise_story(self, request: SummariseStoryRequest) -> TextResponse:
        """Summarise a story."""
        pass

    @abstractmethod
    async def generate_image_prompts(
        self, request: GenerateImagePromptsRequest
    ) -> ImagePromptsResponse:
        """Generate image prompts for each section of the story."""
        pass
