import logging
from base64 import b64encode
from io import BytesIO
from traceback import format_exc

import requests
from fastapi import HTTPException, status
from google import genai
from google.auth.exceptions import DefaultCredentialsError
from google.genai.types import GenerateImagesConfig
from PIL import Image

from app.models.cost_centre import CostCentreManager
from app.services.image.common import (
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImageRouterService,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VertexAiRouterService(ImageRouterService):
    """Vertex AI Imagen implementation of the image router service."""

    # Default model - using Imagen 3.0 Fast Generate
    DEFAULT_MODEL = "imagen-3.0-fast-generate-001"

    # Google Cloud pricing varies by region and usage tier
    # Refer to https://cloud.google.com/vertex-ai/generative-ai/pricing#imagen-models.
    DEFAULT_COST_PER_IMAGE = 0.02

    def __init__(self):
        """
        Initialize the Vertex AI router service.

        Args:
            model_name: Name of the Imagen model to use. Defaults to Imagen 3.0 Fast.
            project_id: Google Cloud project ID. If not provided, uses GOOGLE_CLOUD_PROJECT env var.
            location: Google Cloud location. If not provided, uses GOOGLE_CLOUD_LOCATION env var or defaults to us-central1.
        """
        self.model_name = self.DEFAULT_MODEL
        self.project_id = "bai-buchai-p"
        self.location = "us-east1"
        self.cost_centre_manager = CostCentreManager()
        self._client = None

        # Validate required configuration
        if not self.project_id:
            raise ValueError(
                "Google Cloud project ID is required. Set GOOGLE_CLOUD_PROJECT environment variable or pass project_id parameter."
            )

    def _get_client(self) -> genai.Client:
        """Get or create the GenAI client configured for Vertex AI."""
        if self._client is None:
            try:
                logger.info(
                    f"Initializing Vertex AI client for project: {self.project_id}, location: {self.location}"
                )
                self._client = genai.Client(
                    vertexai=True, project=self.project_id, location=self.location
                )
            except DefaultCredentialsError as e:
                logger.error(f"Google Cloud authentication failed: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Google Cloud authentication not configured properly. Please set up Application Default Credentials.",
                )
            except Exception as e:
                logger.error(f"Failed to initialize GenAI client: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to initialize Vertex AI client: {str(e)}",
                )
        return self._client

    def _resize_image(self, image_bytes: bytes, width: int, height: int) -> bytes:
        """
        Resize image to the specified dimensions using Pillow.

        Args:
            image_bytes: Original image bytes
            width: Target width in pixels
            height: Target height in pixels

        Returns:
            bytes: Resized image bytes in PNG format
        """
        try:
            # Open the image from bytes
            image = Image.open(BytesIO(image_bytes))

            # Resize the image with high-quality resampling
            resized_image = image.resize((width, height), Image.LANCZOS)

            # Convert to RGB if it has an alpha channel for consistent output
            if resized_image.mode in ("RGBA", "LA", "P"):
                resized_image = resized_image.convert("RGB")

            # Save to bytes buffer
            output_buffer = BytesIO()
            resized_image.save(output_buffer, format="PNG")

            return output_buffer.getvalue()

        except Exception as e:
            logger.error(f"Failed to resize image: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to resize image: {str(e)}",
            )

    def calculate_cost(self) -> float:
        """
        Calculate cost for Vertex AI Imagen generation.
        Returns the cost per image generation.
        """
        return self.DEFAULT_COST_PER_IMAGE

    async def generate_image(
        self, request: ImageGenerationRequest
    ) -> ImageGenerationResponse:
        """Generate an image based on the provided prompt using Vertex AI Imagen."""
        try:
            client = self._get_client()

            response = client.models.generate_images(
                model=self.model_name,
                prompt=request.prompt,
                config=GenerateImagesConfig(
                    number_of_images=1,
                    aspect_ratio="1:1",
                    safety_filter_level="BLOCK_LOW_AND_ABOVE",
                    person_generation="ALLOW_ALL",
                    language="en",
                ),
            )

            if not response.generated_images:
                logger.error("No images generated from Vertex AI")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="No images were generated",
                )

            # Get the first generated image
            generated_image = response.generated_images[0]

            # Get image bytes
            image_bytes = generated_image.image.image_bytes
            if not image_bytes:
                logger.error("Generated image has no bytes")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Generated image is empty",
                )

            # Resize image if requested dimensions differ from default
            # Vertex AI Imagen generates square images (1:1 aspect ratio)
            # We'll resize to match the requested width and height
            # Refer to https://cloud.google.com/vertex-ai/generative-ai/docs/models/imagen/3-0-fast-generate-001.
            if request.width != 1024 or request.height != 1024:
                logger.info(
                    f"Resizing image from default size to {request.width}x{request.height}..."
                )
                image_bytes = self._resize_image(
                    image_bytes, request.width, request.height
                )

            # Convert to base64
            base64_data = b64encode(image_bytes)

            # Determine content type (Imagen typically generates PNG images)
            content_type = "image/png"

            # Calculate cost and update cost centre if provided
            cost = self.calculate_cost()
            if request.cost_centre_id:
                await self.cost_centre_manager.update_cost_centre(
                    request.cost_centre_id, cost
                )

            return ImageGenerationResponse(data=base64_data, content_type=content_type)

        except requests.RequestException as e:
            logger.error(f"Error downloading image: {str(e)}\n{format_exc()}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to download image: {str(e)}",
            )
        except Exception as e:
            logger.error(f"Error generating image: {str(e)}\n{format_exc()}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
            )
