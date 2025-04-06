import logging
from base64 import b64encode
from traceback import format_exc
from urllib.parse import quote

import requests
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Create a router for image generation
image_router = APIRouter()

POLLINATIONS_IMAGE_URL = (
    "https://image.pollinations.ai/prompt/{prompt}?width=512&height=512"
)


class ImageGenerationRequest(BaseModel):
    prompt: str


class ImageGenerationResponse(BaseModel):
    data: bytes
    content_type: str


# TODO: Refactor this into PollinationsRouterService.
@image_router.post("/generate", response_model=ImageGenerationResponse)
async def generate_image(request: ImageGenerationRequest) -> ImageGenerationResponse:
    """Generate an image based on the provided prompt."""
    # NOTE: https://github.com/pollinations/pollinations/blob/master/APIDOCS.md#generate-image-api-%EF%B8%8F
    # NOTE: "Rate Limit (per IP): 1 concurrent request / 5 sec interval."

    try:
        # URL-encode the prompt
        encoded_prompt = quote(request.prompt)
        image_url = POLLINATIONS_IMAGE_URL.format(prompt=encoded_prompt)

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
