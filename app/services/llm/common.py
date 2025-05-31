from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Optional

from pydantic import BaseModel

from app.models.llm import ModelType


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
