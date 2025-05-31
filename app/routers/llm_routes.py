from typing import List, Literal

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.services.llm.common import (
    GenerateImagePromptsRequest,
    GenerateStoryRequest,
    SummariseStoryRequest,
)
from app.services.llm.hugging_face import HuggingFaceRouterService
from app.services.llm.vertex_ai import VertexAiRouterService

# TODO: Depending on which provider is available, switch.
# TODO: Make this an environment variable.
# Global LLM provider setting
LLM_PROVIDER: Literal["huggingface", "vertexai"] = "vertexai"

# Initialize the appropriate service based on provider
if LLM_PROVIDER == "huggingface":
    llm_service = HuggingFaceRouterService()
if LLM_PROVIDER == "vertexai":
    llm_service = VertexAiRouterService()


# Create a router for calls to the LLM API
llm_router = APIRouter()


@llm_router.post("/generate_story_string", response_model=str)
async def generate_story_string(request: GenerateStoryRequest) -> str:
    response_data = await llm_service.generate_story_string(request)
    return response_data.text


@llm_router.post("/generate_story_stream")
async def generate_story_stream(request: GenerateStoryRequest) -> StreamingResponse:
    return StreamingResponse(
        llm_service.generate_story_stream(request), media_type="text/plain"
    )


@llm_router.post("/split_story", response_model=List[List[str]])
async def split_story(request: GenerateStoryRequest) -> List[List[str]]:
    response_data = await llm_service.split_story(request)
    return response_data.data


@llm_router.post("/summarise_story", response_model=str)
async def summarise_story(request: SummariseStoryRequest) -> str:
    response_data = await llm_service.summarise_story(request)
    return response_data.text


@llm_router.post("/generate_image_prompts", response_model=List[str])
async def generate_image_prompts(request: GenerateImagePromptsRequest) -> List[str]:
    response_data = await llm_service.generate_image_prompts(request)
    return response_data.data
