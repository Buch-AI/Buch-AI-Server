from typing import Literal

from fastapi import APIRouter

from app.services.image.common import ImageGenerationRequest, ImageGenerationResponse
from app.services.image.pollinations_ai import PollinationsAiRouterService
from app.services.image.vertex_ai import VertexAiRouterService

# Create a router for image generation
image_router = APIRouter()

# Global image provider setting
IMAGE_PROVIDER: Literal["pollinations_ai", "vertex_ai"] = "vertex_ai"

# Initialize the appropriate service based on provider
if IMAGE_PROVIDER == "pollinations_ai":
    image_service = PollinationsAiRouterService()
if IMAGE_PROVIDER == "vertex_ai":
    image_service = VertexAiRouterService()


@image_router.post("/generate", response_model=ImageGenerationResponse)
async def generate_image(request: ImageGenerationRequest) -> ImageGenerationResponse:
    """Generate an image based on the provided prompt."""
    return await image_service.generate_image(request)
