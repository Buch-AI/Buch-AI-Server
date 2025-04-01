import logging
import time
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

# Create a router for image generation
image_router = APIRouter()

POLLINATIONS_IMAGE_URL = "https://image.pollinations.ai/prompt/{prompt}"


class ImageGenerationRequest(BaseModel):
    """Request model for image generation."""

    prompt: str


# TODO: Refactor this into PollinationsRouterService.
@image_router.post("/generate")
async def generate_image(request: ImageGenerationRequest):
    """Generate an image based on the provided prompt."""
    # NOTE: https://github.com/pollinations/pollinations/blob/master/APIDOCS.md#generate-image-api-%EF%B8%8F
    # NOTE: "Rate Limit (per IP): 1 concurrent request / 5 sec interval."
    time.sleep(5)

    try:
        # URL-encode the prompt
        encoded_prompt = quote(request.prompt)
        image_url = POLLINATIONS_IMAGE_URL.format(prompt=encoded_prompt)

        # TODO: Refactor this into ImageGenerationResponse.
        return {"image_url": image_url, "status": "success", "prompt": request.prompt}

    except Exception as e:
        logging.error(f"Error generating image URL: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
