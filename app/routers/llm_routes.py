import logging
from abc import ABC, abstractmethod
from traceback import format_exc
from typing import AsyncGenerator, List, Literal, Optional

import vertexai
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from google.cloud import bigquery
from huggingface_hub import AsyncInferenceClient, InferenceClient
from pydantic import BaseModel
from vertexai.generative_models import Content, GenerationConfig, GenerativeModel, Part

from app.models.llm import (
    HuggingFaceConfigManager,
    ModelType,
    VertexAiConfigManager,
)
from config import HF_API_KEY

# TODO: Depending on which provider is avaiable, switch.
# TODO: Make this an environment variable.
# Global LLM provider setting
LLM_PROVIDER: Literal["huggingface", "vertexai"] = "vertexai"

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


class GenerateStoryRequest(BaseModel):
    prompt: str
    model_type: ModelType = ModelType.LITE
    cost_centre_id: Optional[str] = None


class SummariseStoryRequest(BaseModel):
    story: str
    model_type: ModelType = ModelType.LITE
    cost_centre_id: Optional[str] = None


class GenerateImagePromptsRequest(BaseModel):
    story_summary: str
    story_parts: List[str]
    model_type: ModelType = ModelType.LITE
    cost_centre_id: Optional[str] = None


class TokenUsage(BaseModel):
    """Model for tracking token usage in LLM calls."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float = 0.0


class TextResponse(BaseModel):
    """Response model for endpoints that return text."""

    text: str
    usage: TokenUsage


class SplitStoryResponse(BaseModel):
    """Response model for split_story endpoint."""

    data: List[List[str]]
    usage: TokenUsage


class ImagePromptsResponse(BaseModel):
    """Response model for generate_image_prompts endpoint."""

    data: List[str]
    usage: TokenUsage


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

    @abstractmethod
    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate the cost based on token usage."""
        pass

    async def update_cost_center(self, cost_centre_id: str, cost: float) -> None:
        """Update the cost for a cost centre."""
        if not cost_centre_id:
            return

        try:
            bigquery_client = bigquery.Client()
            query = """
            UPDATE `bai-buchai-p.creations.cost_centres`
            SET cost = cost + @additional_cost
            WHERE cost_centre_id = @cost_centre_id
            """

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("additional_cost", "FLOAT", cost),
                    bigquery.ScalarQueryParameter(
                        "cost_centre_id", "STRING", cost_centre_id
                    ),
                ]
            )

            await bigquery_client.query_async(query, job_config=job_config)
        except Exception as e:
            logging.error(f"Failed to update cost center: {e}")
            # Don't raise exception to prevent disrupting the main flow


class HuggingFaceRouterService(LlmRouterService):
    """HuggingFace implementation of the LLM router service."""

    def __init__(self):
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

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Calculate cost for Hugging Face models.

        For Llama 2 models on Hugging Face Inference API:
        - Input: $0.0002 per 1K tokens ($0.0000002 per token)
        - Output: $0.0002 per 1K tokens ($0.0000002 per token)

        These are approximate rates for Llama 2 models on the Inference API.
        """
        input_cost_per_token = 0.0000002  # $0.0002 per 1K tokens
        output_cost_per_token = 0.0000002  # $0.0002 per 1K tokens

        return (prompt_tokens * input_cost_per_token) + (
            completion_tokens * output_cost_per_token
        )

    async def generate_story_string(
        self, request: GenerateStoryRequest
    ) -> TextResponse:
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

            # Extract usage data for cost calculation
            usage = response.usage
            cost = self.calculate_cost(usage.prompt_tokens, usage.completion_tokens)

            # Update cost centre if provided
            if request.cost_centre_id:
                await self.update_cost_center(request.cost_centre_id, cost)

            return TextResponse(
                text=response.choices[0].message.content,
                usage=TokenUsage(
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    total_tokens=usage.total_tokens,
                    cost=cost,
                ),
            )
        except Exception as e:
            logger.error(
                f"Error generating story with HuggingFace: {str(e)}\n{format_exc()}"
            )
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

            # For streaming, we'll track tokens for cost calculation
            prompt_tokens = sum(len(msg.get("content", "").split()) for msg in messages)
            completion_tokens = 0

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
                    content = chunk.choices[0].delta.content or ""
                    # Roughly estimate tokens from content for cost tracking
                    if content:
                        completion_tokens += max(1, len(content.split()))
                    yield content

            # After streaming completes, calculate cost and update
            if request.cost_centre_id:
                cost = self.calculate_cost(prompt_tokens, completion_tokens)
                await self.update_cost_center(request.cost_centre_id, cost)

        except Exception as e:
            logger.error(
                f"Error streaming story with HuggingFace: {str(e)}\n{format_exc()}"
            )
            raise HTTPException(status_code=500, detail=str(e))

    async def split_story(self, request: GenerateStoryRequest) -> SplitStoryResponse:
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

            # Extract usage data for cost calculation
            usage = response.usage
            cost = self.calculate_cost(usage.prompt_tokens, usage.completion_tokens)

            # Update cost centre if provided
            if request.cost_centre_id:
                await self.update_cost_center(request.cost_centre_id, cost)

            # Parse the response into parts and sub-parts
            text = response.choices[0].message.content
            parts = text.split("[PART]")

            # Process each part to extract sub-parts and clean the text
            result = []
            for part in parts:
                if part.strip():
                    sub_parts = part.split("[SUBPART]")
                    # Clean each sub-part
                    cleaned_sub_parts = [
                        sub_part.strip() for sub_part in sub_parts if sub_part.strip()
                    ]
                    if cleaned_sub_parts:
                        result.append(cleaned_sub_parts)

            return SplitStoryResponse(
                data=result,
                usage=TokenUsage(
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    total_tokens=usage.total_tokens,
                    cost=cost,
                ),
            )
        except Exception as e:
            logger.error(
                f"Error splitting story with HuggingFace: {str(e)}\n{format_exc()}"
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
            )

    async def summarise_story(self, request: SummariseStoryRequest) -> TextResponse:
        try:
            model_config = HuggingFaceConfigManager.get_model_config(request.model_type)
            messages = HuggingFaceConfigManager.get_prompt_story_summarise(
                request.model_type, request.story
            )

            response = self.client.chat.completions.create(
                messages=messages,
                model=model_config.model_id,
                max_tokens=model_config.max_tokens,
                temperature=model_config.temperature,
                top_p=model_config.top_p,
            )

            # Extract usage data for cost calculation
            usage = response.usage
            cost = self.calculate_cost(usage.prompt_tokens, usage.completion_tokens)

            # Update cost centre if provided
            if request.cost_centre_id:
                await self.update_cost_center(request.cost_centre_id, cost)

            return TextResponse(
                text=response.choices[0].message.content,
                usage=TokenUsage(
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    total_tokens=usage.total_tokens,
                    cost=cost,
                ),
            )
        except Exception as e:
            logger.error(
                f"Error summarising story with HuggingFace: {str(e)}\n{format_exc()}"
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
            )

    async def generate_image_prompts(
        self, request: GenerateImagePromptsRequest
    ) -> ImagePromptsResponse:
        try:
            model_config = HuggingFaceConfigManager.get_model_config(request.model_type)

            # Create a list to store the generated prompts
            image_prompts = []
            total_prompt_tokens = 0
            total_completion_tokens = 0
            total_cost = 0.0

            # For each story part
            for part in request.story_parts:
                part_messages = HuggingFaceConfigManager.get_prompt_image_generate(
                    request.model_type, request.story_summary, part
                )

                part_response = self.client.chat.completions.create(
                    messages=part_messages,
                    model=model_config.model_id,
                    max_tokens=model_config.max_tokens,
                    temperature=model_config.temperature,
                    top_p=model_config.top_p,
                )

                # Accumulate token usage
                usage = part_response.usage
                part_cost = self.calculate_cost(
                    usage.prompt_tokens, usage.completion_tokens
                )

                total_prompt_tokens += usage.prompt_tokens
                total_completion_tokens += usage.completion_tokens
                total_cost += part_cost

                image_prompts.append(part_response.choices[0].message.content)

            # Update cost centre if provided
            if request.cost_centre_id:
                await self.update_cost_center(request.cost_centre_id, total_cost)

            return ImagePromptsResponse(
                data=image_prompts,
                usage=TokenUsage(
                    prompt_tokens=total_prompt_tokens,
                    completion_tokens=total_completion_tokens,
                    total_tokens=total_prompt_tokens + total_completion_tokens,
                    cost=total_cost,
                ),
            )

        except Exception as e:
            logger.error(
                f"Error generating image prompts with HuggingFace: {str(e)}\n{format_exc()}"
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
            )


class VertexAiRouterService(LlmRouterService):
    """Vertex AI implementation of the LLM router service."""

    def __init__(self):
        vertexai.init(project="bai-buchai-p", location="us-east1")

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Calculate cost for Vertex AI models (Gemini Pro).

        For Gemini Pro pricing:
        - Input: $0.00025 per 1K tokens ($0.00000025 per token)
        - Output: $0.0005 per 1K tokens ($0.0000005 per token)

        Rates are based on Google Cloud documentation.
        """
        input_cost_per_token = 0.00000025  # $0.00025 per 1K tokens
        output_cost_per_token = 0.0000005  # $0.0005 per 1K tokens

        return (prompt_tokens * input_cost_per_token) + (
            completion_tokens * output_cost_per_token
        )

    async def generate_story_string(
        self, request: GenerateStoryRequest
    ) -> TextResponse:
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

            # Extract accurate token usage from Vertex AI response
            prompt_tokens = response.usage_metadata.prompt_token_count
            completion_tokens = response.usage_metadata.candidates_token_count
            total_tokens = prompt_tokens + completion_tokens

            # Calculate cost based on actual token usage
            cost = self.calculate_cost(prompt_tokens, completion_tokens)

            # Update cost centre if provided
            if request.cost_centre_id:
                await self.update_cost_center(request.cost_centre_id, cost)

            return TextResponse(
                text=response.text,
                usage=TokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cost=cost,
                ),
            )
        except Exception as e:
            logger.error(
                f"Error generating story with Vertex AI: {str(e)}\n{format_exc()}"
            )
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

            # We'll need to track tokens throughout streaming
            prompt_tokens = 0
            completion_tokens = 0

            for chunk in response:
                # Some chunks may contain usage metadata
                if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                    if hasattr(chunk.usage_metadata, "prompt_token_count"):
                        prompt_tokens = chunk.usage_metadata.prompt_token_count
                    if hasattr(chunk.usage_metadata, "candidates_token_count"):
                        completion_tokens = max(
                            completion_tokens,
                            chunk.usage_metadata.candidates_token_count,
                        )

                if chunk.text:
                    yield chunk.text

            # After streaming completes, calculate cost and update
            if request.cost_centre_id:
                # If we didn't get token counts from the API, estimate them
                if prompt_tokens == 0:
                    prompt_tokens = len(prompt.split())
                if completion_tokens == 0:
                    completion_tokens = 500  # Conservative estimate

                cost = self.calculate_cost(prompt_tokens, completion_tokens)
                await self.update_cost_center(request.cost_centre_id, cost)

        except Exception as e:
            logger.error(
                f"Error streaming story with Vertex AI: {str(e)}\n{format_exc()}"
            )
            raise HTTPException(status_code=500, detail=str(e))

    async def split_story(self, request: GenerateStoryRequest) -> SplitStoryResponse:
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

            # Extract accurate token usage from Vertex AI response
            prompt_tokens = response.usage_metadata.prompt_token_count
            completion_tokens = response.usage_metadata.candidates_token_count
            total_tokens = prompt_tokens + completion_tokens

            # Calculate cost based on actual token usage
            cost = self.calculate_cost(prompt_tokens, completion_tokens)

            # Update cost centre if provided
            if request.cost_centre_id:
                await self.update_cost_center(request.cost_centre_id, cost)

            # Parse the response into parts and sub-parts
            text = response.text
            parts = text.split("[PART]")

            # Process each part to extract sub-parts and clean the text
            result = []
            for part in parts:
                if part.strip():
                    sub_parts = part.split("[SUBPART]")
                    # Clean each sub-part
                    cleaned_sub_parts = [
                        sub_part.strip() for sub_part in sub_parts if sub_part.strip()
                    ]
                    if cleaned_sub_parts:
                        result.append(cleaned_sub_parts)

            return SplitStoryResponse(
                data=result,
                usage=TokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cost=cost,
                ),
            )
        except Exception as e:
            logger.error(
                f"Error splitting story with Vertex AI: {str(e)}\n{format_exc()}"
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
            )

    async def summarise_story(self, request: SummariseStoryRequest) -> TextResponse:
        try:
            model_config = VertexAiConfigManager.get_model_config(request.model_type)
            prompt = VertexAiConfigManager.get_prompt_story_summarise(
                request.model_type, request.story
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

            # Extract accurate token usage from Vertex AI response
            prompt_tokens = response.usage_metadata.prompt_token_count
            completion_tokens = response.usage_metadata.candidates_token_count
            total_tokens = prompt_tokens + completion_tokens

            # Calculate cost based on actual token usage
            cost = self.calculate_cost(prompt_tokens, completion_tokens)

            # Update cost centre if provided
            if request.cost_centre_id:
                await self.update_cost_center(request.cost_centre_id, cost)

            return TextResponse(
                text=response.text,
                usage=TokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cost=cost,
                ),
            )
        except Exception as e:
            logger.error(
                f"Error summarising story with Vertex AI: {str(e)}\n{format_exc()}"
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
            )

    async def generate_image_prompts(
        self, request: GenerateImagePromptsRequest
    ) -> ImagePromptsResponse:
        try:
            model_config = VertexAiConfigManager.get_model_config(request.model_type)

            # Create a list to store the generated prompts
            image_prompts = []
            total_prompt_tokens = 0
            total_completion_tokens = 0
            total_cost = 0.0

            model = GenerativeModel(model_config.model_id)
            generation_config = GenerationConfig(
                max_output_tokens=model_config.max_tokens,
                temperature=model_config.temperature,
                top_p=model_config.top_p,
            )

            # For each story part
            for part in request.story_parts:
                part_prompt = VertexAiConfigManager.get_prompt_image_generate(
                    request.model_type, request.story_summary, part
                )

                part_content = Content(role="user", parts=[Part.from_text(part_prompt)])
                part_response = model.generate_content(
                    part_content,
                    generation_config=generation_config,
                )

                # Extract accurate token usage
                prompt_tokens = part_response.usage_metadata.prompt_token_count
                completion_tokens = part_response.usage_metadata.candidates_token_count

                # Calculate cost for this part
                part_cost = self.calculate_cost(prompt_tokens, completion_tokens)

                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                total_cost += part_cost

                image_prompts.append(part_response.text)

            # Update cost centre if provided
            if request.cost_centre_id:
                await self.update_cost_center(request.cost_centre_id, total_cost)

            return ImagePromptsResponse(
                data=image_prompts,
                usage=TokenUsage(
                    prompt_tokens=total_prompt_tokens,
                    completion_tokens=total_completion_tokens,
                    total_tokens=total_prompt_tokens + total_completion_tokens,
                    cost=total_cost,
                ),
            )

        except Exception as e:
            logger.error(
                f"Error generating image prompts with Vertex AI: {str(e)}\n{format_exc()}"
            )
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
