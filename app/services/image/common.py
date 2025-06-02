from abc import ABC, abstractmethod
from typing import Optional

from pydantic import BaseModel


class ImageGenerationRequest(BaseModel):
    prompt: str
    width: int = 480
    height: int = 480
    cost_centre_id: Optional[str] = None


class ImageGenerationResponse(BaseModel):
    data: bytes
    content_type: str


class ImageRouterService(ABC):
    """Abstract base class for image generation router services."""

    @abstractmethod
    async def generate_image(
        self, request: ImageGenerationRequest
    ) -> ImageGenerationResponse:
        """Generate an image based on the provided prompt."""
        pass

    @abstractmethod
    def calculate_cost(self) -> float:
        """Calculate the cost for image generation."""
        pass
