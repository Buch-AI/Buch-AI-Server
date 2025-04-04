import logging
import os
from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Literal

import vertexai
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import Response, StreamingResponse
from huggingface_hub import AsyncInferenceClient, InferenceClient
from pydantic import BaseModel
from vertexai.generative_models import Content, GenerationConfig, GenerativeModel, Part

from models.llm_config import (
    HuggingFaceConfigManager,
    ModelType,
    VertexAiConfigManager,
)

load_dotenv()

# TODO: Depending on which provider is avaiable, switch.
# Global LLM provider setting
LLM_PROVIDER: Literal["huggingface", "vertexai"] = "vertexai"


class GenerateStoryRequest(BaseModel):
    prompt: str
    model_type: ModelType = ModelType.LITE


class LlmRouterService(ABC):
    """Abstract base class for LLM router services."""

    @abstractmethod
    async def generate_story_string(self, request: GenerateStoryRequest) -> str:
        """Generate a story as a single string."""
        pass

    @abstractmethod
    async def generate_story_stream(
        self, request: GenerateStoryRequest
    ) -> AsyncGenerator[str, None]:
        """Generate a story as a stream of text chunks."""
        pass

    @abstractmethod
    async def split_story(self, request: GenerateStoryRequest) -> List[str]:
        """Generate a story split into title and content."""
        pass


class HuggingFaceRouterService(LlmRouterService):
    """HuggingFace implementation of the LLM router service."""

    def __init__(self):
        HF_API_KEY = os.getenv("HF_API_KEY")
        if not HF_API_KEY:
            raise ValueError(
                "HF_API_KEY environment variable is not set. Please set it in your .env file."
            )

        self.api_key = HF_API_KEY
        self.client = InferenceClient(
            base_url="https://api-inference.huggingface.co", token=self.api_key
        )
        self.async_client = AsyncInferenceClient(
            base_url="https://api-inference.huggingface.co", token=self.api_key
        )

    async def generate_story_string(self, request: GenerateStoryRequest) -> str:
        try:
            model_config = HuggingFaceConfigManager.get_model_config(request.model_type)
            messages = HuggingFaceConfigManager.get_prompt_story_generate(
                request.model_type, request.prompt
            )

            response = self.client.chat.completions.create(
                messages=messages,
                model=model_config.model_id,
                max_tokens=model_config.max_tokens,
                temperature=model_config.temperature,
                top_p=model_config.top_p,
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error generating story with HuggingFace: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
            )

    async def generate_story_stream(
        self, request: GenerateStoryRequest
    ) -> AsyncGenerator[str, None]:
        try:
            model_config = HuggingFaceConfigManager.get_model_config(request.model_type)
            messages = HuggingFaceConfigManager.get_prompt_story_generate(
                request.model_type, request.prompt
            )

            stream = await self.async_client.chat.completions.create(
                messages=messages,
                model=model_config.model_id,
                stream=True,
                max_tokens=model_config.max_tokens,
                temperature=model_config.temperature,
                top_p=model_config.top_p,
            )
            async for chunk in stream:
                if "choices" in chunk and chunk.choices:
                    yield chunk.choices[0].delta.content or ""
        except Exception as e:
            logging.error(f"Error streaming story with HuggingFace: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def split_story(self, request: GenerateStoryRequest) -> List[str]:
        try:
            model_config = HuggingFaceConfigManager.get_model_config(request.model_type)
            messages = HuggingFaceConfigManager.get_prompt_story_split(
                request.model_type, request.prompt
            )

            response = self.client.chat.completions.create(
                messages=messages,
                model=model_config.model_id,
                max_tokens=model_config.max_tokens,
                temperature=model_config.temperature,
                top_p=model_config.top_p,
            )

            # Parse the response into title and content
            text = response.choices[0].message.content
            parts = text.split("[SPLIT]")

            # TODO: Set the upper limit of parts to 4
            return parts[:4]
        except Exception as e:
            logging.error(f"Error splitting story with HuggingFace: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
            )


class VertexAiRouterService(LlmRouterService):
    """Vertex AI implementation of the LLM router service."""

    def __init__(self):
        vertexai.init(project="bai-buchai-p", location="us-east1")

    async def generate_story_string(self, request: GenerateStoryRequest) -> str:
        try:
            model_config = VertexAiConfigManager.get_model_config(request.model_type)
            prompt = VertexAiConfigManager.get_prompt_story_generate(
                request.model_type, request.prompt
            )

            model = GenerativeModel(model_config.model_id)
            generation_config = GenerationConfig(
                max_output_tokens=model_config.max_tokens,
                temperature=model_config.temperature,
                top_p=model_config.top_p,
            )

            prompt_content = Content(role="user", parts=[Part.from_text(prompt)])

            response = model.generate_content(
                prompt_content,
                generation_config=generation_config,
            )
            return response.text
        except Exception as e:
            logging.error(f"Error generating story with Vertex AI: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
            )

    async def generate_story_stream(
        self, request: GenerateStoryRequest
    ) -> AsyncGenerator[str, None]:
        try:
            model_config = VertexAiConfigManager.get_model_config(request.model_type)
            prompt = VertexAiConfigManager.get_prompt_story_generate(
                request.model_type, request.prompt
            )

            model = GenerativeModel(model_config.model_id)
            generation_config = GenerationConfig(
                max_output_tokens=model_config.max_tokens,
                temperature=model_config.temperature,
                top_p=model_config.top_p,
            )

            prompt_content = Content(role="user", parts=[Part.from_text(prompt)])

            response = model.generate_content(
                prompt_content,
                generation_config=generation_config,
                stream=True,
            )
            for chunk in response:
                yield chunk.text
        except Exception as e:
            logging.error(f"Error streaming story with Vertex AI: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def split_story(self, request: GenerateStoryRequest) -> List[str]:
        try:
            model_config = VertexAiConfigManager.get_model_config(request.model_type)
            prompt = VertexAiConfigManager.get_prompt_story_split(
                request.model_type, request.prompt
            )

            model = GenerativeModel(model_config.model_id)
            generation_config = GenerationConfig(
                max_output_tokens=model_config.max_tokens,
                temperature=model_config.temperature,
                top_p=model_config.top_p,
            )

            prompt_content = Content(role="user", parts=[Part.from_text(prompt)])

            response = model.generate_content(
                prompt_content,
                generation_config=generation_config,
            )

            # Parse the response into title and content
            text = response.text
            parts = text.split("[SPLIT]")

            # TODO: Set the upper limit of parts to 4
            return parts[:4]
        except Exception as e:
            logging.error(f"Error splitting story with Vertex AI: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
            )


# Initialize the appropriate service based on provider
if LLM_PROVIDER == "huggingface":
    llm_service = HuggingFaceRouterService()
if LLM_PROVIDER == "vertexai":
    llm_service = VertexAiRouterService()


# Create a router for calls to the LLM API
llm_router = APIRouter()


@llm_router.post("/generate_story_string")
async def generate_story_string(request: GenerateStoryRequest) -> Response:
    return await llm_service.generate_story_string(request)


@llm_router.post("/generate_story_stream")
async def generate_story_stream(request: GenerateStoryRequest) -> StreamingResponse:
    return StreamingResponse(
        llm_service.generate_story_stream(request), media_type="text/plain"
    )


@llm_router.post("/split_story")
async def split_story(request: GenerateStoryRequest) -> Response:
    return await llm_service.split_story(request)
