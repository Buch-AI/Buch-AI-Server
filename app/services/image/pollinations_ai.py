import logging
from base64 import b64encode
from traceback import format_exc
from urllib.parse import quote

import requests
from fastapi import HTTPException, status

from app.models.cost_centre import CostCentreManager
from app.services.image.common import (
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImageRouterService,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PollinationsAiRouterService(ImageRouterService):
    """Pollinations AI implementation of the image router service."""

    API_URL = "https://image.pollinations.ai/prompt/{prompt}?width={width}&height={height}&nologo={nologo}&private={private}&safe={safe}"

    def __init__(self):
        self.cost_centre_manager = CostCentreManager()

    def calculate_cost(self) -> float:
        """
        Calculate cost for Pollinations AI image generation.
        Currently free service, so cost is set to 0.
        """
        return 0.0

    async def generate_image(
        self, request: ImageGenerationRequest
    ) -> ImageGenerationResponse:
        """Generate an image based on the provided prompt using Pollinations AI."""
        # NOTE: https://github.com/pollinations/pollinations/blob/master/APIDOCS.md#generate-image-api-%EF%B8%8F
        # NOTE: "Rate Limit (per IP): 1 concurrent request / 5 sec interval."

        try:
            # URL-encode the prompt
            encoded_prompt = quote(request.prompt)
            image_url = self.API_URL.format(
                prompt=encoded_prompt,
                width=request.width,
                height=request.height,
                nologo="true",
                private="false",
                safe="true",
            )

            # Download the image
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()

            # Verify content type is an image
            content_type = response.headers.get("content-type", "")
            if not content_type.startswith("image/"):
                logger.error(
                    f"Invalid content type received: {content_type}\n{format_exc()}"
                )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Response from image service was not an image",
                )

            # Convert binary data to base64 string
            base64_data = b64encode(response.content)

            # Calculate cost (currently 0) and update cost centre if provided
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
