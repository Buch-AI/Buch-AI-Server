import inspect
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import AsyncGenerator, Awaitable, Callable, List, Optional, TypeVar

from google.cloud import storage
from pydantic import BaseModel

from app.models.llm import ModelType
from config import ENV, GCLOUD_STB_CREATIONS_NAME, PROJECT_ROOT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type variable for the return type of decorated functions
T = TypeVar("T")


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

    @staticmethod
    def validation_with_retries(
        validation_func: Callable[[T], bool], max_retries: int = 3
    ) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
        """
        Decorator that retries an async function if it throws an exception or validation fails.

        Args:
            validation_func: Function that takes the result and returns True if valid
            max_retries: Maximum number of retry attempts (default: 3)

        Returns:
            Decorated async function that will retry on failure or invalid results
        """

        def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
            @wraps(func)
            async def wrapper(*args, **kwargs) -> T:
                last_exception = None

                for attempt in range(max_retries + 1):  # +1 for initial attempt
                    try:
                        result: T = await func(*args, **kwargs)

                        # Check validation
                        if validation_func(result):
                            if attempt > 0:
                                logger.warning(
                                    f"Function {func.__name__} succeeded on attempt {attempt + 1}"
                                )
                            return result
                        else:
                            logger.warning(
                                f"Validation failed for {func.__name__}, attempt {attempt + 1}/{max_retries + 1}"
                            )
                            if attempt == max_retries:
                                raise ValueError(
                                    f"Validation failed for {func.__name__} after {max_retries + 1} attempts"
                                )

                    except Exception as e:
                        last_exception = e
                        logger.warning(
                            f"Exception in {func.__name__}, attempt {attempt + 1}/{max_retries + 1}: {str(e)}"
                        )
                        if attempt == max_retries:
                            logger.error(
                                f"Function {func.__name__} failed after {max_retries + 1} attempts"
                            )
                            raise last_exception

                # This shouldn't be reached, but just in case
                if last_exception:
                    raise last_exception
                raise ValueError(
                    f"Function {func.__name__} failed after {max_retries + 1} attempts"
                )

            return wrapper

        return decorator

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
