import json
import logging
from abc import ABC, abstractmethod
from traceback import format_exc
from typing import AsyncGenerator, Dict, List, Literal, Optional

import faiss
import numpy as np
import vertexai
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from huggingface_hub import AsyncInferenceClient, InferenceClient
from pydantic import BaseModel
from vertexai.generative_models import Content, GenerationConfig, GenerativeModel, Part
from vertexai.language_models import TextEmbeddingModel

from app.models.cost_centre import CostCentreManager
from app.models.llm import (
    HuggingFaceConfigManager,
    ModelType,
    VertexAiConfigManager,
)
from config import HF_API_KEY

# TODO: Depending on which provider is available, switch.
# TODO: Make this an environment variable.
# Global LLM provider setting
LLM_PROVIDER: Literal["huggingface", "vertexai"] = "vertexai"

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Initialize cost centre manager
cost_centre_manager = CostCentreManager()


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

    @abstractmethod
    def calculate_embedding_cost(self, token_count: int) -> float:
        """Calculate the cost based on embedding token usage."""
        pass

    @abstractmethod
    def calculate_generation_cost(
        self, prompt_tokens: int, completion_tokens: int
    ) -> float:
        """Calculate the cost based on token usage."""
        pass


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

    def calculate_embedding_cost(self, token_count: int) -> float:
        """
        Calculate cost for text embedding models in Hugging Face.

        For sentence-transformers models on Hugging Face Inference API:
        - Approximately $0.0001 per 1K tokens ($0.0000001 per token)

        These are approximate rates for embedding models on the Inference API.
        """
        embedding_cost_per_token = 0.0000001  # $0.0001 per 1K tokens

        return token_count * embedding_cost_per_token

    def calculate_generation_cost(
        self, prompt_tokens: int, completion_tokens: int
    ) -> float:
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
            text_generation_model_config = (
                HuggingFaceConfigManager.get_text_generation_model_config(
                    request.model_type
                )
            )
            messages = HuggingFaceConfigManager.get_prompt_story_generate(
                request.model_type, request.prompt
            )

            response = self.client.chat.completions.create(
                messages=messages,
                model=text_generation_model_config.model_id,
                max_tokens=text_generation_model_config.max_tokens,
                temperature=text_generation_model_config.temperature,
                top_p=text_generation_model_config.top_p,
            )

            # Extract usage data for cost calculation
            usage = response.usage
            cost = self.calculate_generation_cost(
                usage.prompt_tokens, usage.completion_tokens
            )

            # Update cost centre if provided
            if request.cost_centre_id:
                await cost_centre_manager.update_cost_centre(
                    request.cost_centre_id, cost
                )

            return TextResponse(
                text=response.choices[0].message.content,
                usage=CostUsage(
                    embedding_tokens=0,  # No embeddings used in this method
                    generation_prompt_tokens=usage.prompt_tokens,
                    generation_completion_tokens=usage.completion_tokens,
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
            text_generation_model_config = (
                HuggingFaceConfigManager.get_text_generation_model_config(
                    request.model_type
                )
            )
            messages = HuggingFaceConfigManager.get_prompt_story_generate(
                request.model_type, request.prompt
            )

            # For streaming, we'll track tokens for cost calculation
            prompt_tokens = sum(len(msg.get("content", "").split()) for msg in messages)
            completion_tokens = 0

            stream = await self.async_client.chat.completions.create(
                messages=messages,
                model=text_generation_model_config.model_id,
                stream=True,
                max_tokens=text_generation_model_config.max_tokens,
                temperature=text_generation_model_config.temperature,
                top_p=text_generation_model_config.top_p,
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
                cost = self.calculate_generation_cost(prompt_tokens, completion_tokens)
                await cost_centre_manager.update_cost_centre(
                    request.cost_centre_id, cost
                )

        except Exception as e:
            logger.error(
                f"Error streaming story with HuggingFace: {str(e)}\n{format_exc()}"
            )
            raise HTTPException(status_code=500, detail=str(e))

    async def split_story(self, request: GenerateStoryRequest) -> SplitStoryResponse:
        try:
            text_generation_model_config = (
                HuggingFaceConfigManager.get_text_generation_model_config(
                    request.model_type
                )
            )
            messages = HuggingFaceConfigManager.get_prompt_story_split(
                request.model_type, request.prompt
            )

            response = self.client.chat.completions.create(
                messages=messages,
                model=text_generation_model_config.model_id,
                max_tokens=text_generation_model_config.max_tokens,
                temperature=text_generation_model_config.temperature,
                top_p=text_generation_model_config.top_p,
            )

            # Extract usage data for cost calculation
            usage = response.usage
            cost = self.calculate_generation_cost(
                usage.prompt_tokens, usage.completion_tokens
            )

            # Update cost centre if provided
            if request.cost_centre_id:
                await cost_centre_manager.update_cost_centre(
                    request.cost_centre_id, cost
                )

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
                usage=CostUsage(
                    embedding_tokens=0,  # No embeddings used in this method
                    generation_prompt_tokens=usage.prompt_tokens,
                    generation_completion_tokens=usage.completion_tokens,
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
            text_generation_model_config = (
                HuggingFaceConfigManager.get_text_generation_model_config(
                    request.model_type
                )
            )
            messages = HuggingFaceConfigManager.get_prompt_story_summarise(
                request.model_type, request.story
            )

            response = self.client.chat.completions.create(
                messages=messages,
                model=text_generation_model_config.model_id,
                max_tokens=text_generation_model_config.max_tokens,
                temperature=text_generation_model_config.temperature,
                top_p=text_generation_model_config.top_p,
            )

            # Extract usage data for cost calculation
            usage = response.usage
            cost = self.calculate_generation_cost(
                usage.prompt_tokens, usage.completion_tokens
            )

            # Update cost centre if provided
            if request.cost_centre_id:
                await cost_centre_manager.update_cost_centre(
                    request.cost_centre_id, cost
                )

            return TextResponse(
                text=response.choices[0].message.content,
                usage=CostUsage(
                    embedding_tokens=0,  # No embeddings used in this method
                    generation_prompt_tokens=usage.prompt_tokens,
                    generation_completion_tokens=usage.completion_tokens,
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
            text_generation_model_config = (
                HuggingFaceConfigManager.get_text_generation_model_config(
                    request.model_type
                )
            )

            # Step 1: Extract entities from the story
            entities = await self._extract_story_entities(
                request.model_type, request.story
            )

            # Step 2: Create vector index from entities and track embedding costs
            entity_index, entity_map, embedding_token_count = self._create_entity_index(
                entities
            )
            embedding_cost = self.calculate_embedding_cost(embedding_token_count)

            # Track total costs
            total_embedding_tokens = embedding_token_count
            total_prompt_tokens = 0
            total_completion_tokens = 0
            total_cost = embedding_cost

            # Step 3: Generate enhanced image prompts with entity context
            image_prompts = []

            # For each story part
            for part in request.story_parts:
                # Retrieve relevant entities for this part using vector search
                relevant_entities, query_embedding_tokens = (
                    self._retrieve_relevant_entities(
                        entity_index, entity_map, part, top_k=3
                    )
                )
                entity_description = "\n".join(
                    [
                        f"{entity['type'].upper()}: {entity['name']} - {entity['description']}"
                        for entity in relevant_entities
                        if entity
                    ]
                )

                # Add embedding cost for query
                query_embedding_cost = self.calculate_embedding_cost(
                    query_embedding_tokens
                )
                total_embedding_tokens += query_embedding_tokens
                total_cost += query_embedding_cost

                # Use the updated get_prompt_image_generate method with entity_description
                messages = HuggingFaceConfigManager.get_prompt_image_generate(
                    request.model_type,
                    request.story,
                    part,
                    entity_description=entity_description,
                )

                part_response = self.client.chat.completions.create(
                    messages=messages,
                    model=text_generation_model_config.model_id,
                    max_tokens=text_generation_model_config.max_tokens,
                    temperature=text_generation_model_config.temperature,
                    top_p=text_generation_model_config.top_p,
                )

                # Accumulate token usage
                usage = part_response.usage
                part_cost = self.calculate_generation_cost(
                    usage.prompt_tokens, usage.completion_tokens
                )

                total_prompt_tokens += usage.prompt_tokens
                total_completion_tokens += usage.completion_tokens
                total_cost += part_cost

                image_prompts.append(part_response.choices[0].message.content)

            # Update cost centre if provided
            if request.cost_centre_id:
                await cost_centre_manager.update_cost_centre(
                    request.cost_centre_id, total_cost
                )

            return ImagePromptsResponse(
                data=image_prompts,
                usage=CostUsage(
                    embedding_tokens=total_embedding_tokens,
                    generation_prompt_tokens=total_prompt_tokens,
                    generation_completion_tokens=total_completion_tokens,
                    total_tokens=total_prompt_tokens
                    + total_completion_tokens
                    + total_embedding_tokens,
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

    async def _extract_story_entities(
        self, model_type: ModelType, story: str
    ) -> List[Dict]:
        """
        Extract entities (characters, locations) from story text using LLM.

        Args:
            model_type: The model type to use for entity extraction
            story: The complete story text

        Returns:
            List of entity dictionaries containing type, name, and description
        """
        text_generation_model_config = (
            HuggingFaceConfigManager.get_text_generation_model_config(model_type)
        )

        messages = HuggingFaceConfigManager.get_prompt_story_entities(model_type, story)

        response = self.client.chat.completions.create(
            messages=messages,
            model=text_generation_model_config.model_id,
            max_tokens=text_generation_model_config.max_tokens,
            temperature=0.2,  # Lower temperature for more deterministic output
            top_p=0.9,
        )

        # Parse the JSON response to extract entities
        try:
            result = response.choices[0].message.content
            # Clean up the JSON string if needed
            cleaned_json = result.strip()
            if cleaned_json.startswith("```json"):
                cleaned_json = cleaned_json[7:]
            if cleaned_json.endswith("```"):
                cleaned_json = cleaned_json[:-3]

            entities = json.loads(cleaned_json)
            if not isinstance(entities, list):
                entities = [entities]
            return entities
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            logger.error(f"Failed to parse entity JSON response: {result}")
            return []

    def _create_entity_index(self, entities: List[Dict]) -> tuple:
        """
        Create a FAISS index from entity descriptions for vector search.

        Args:
            entities: List of entity dictionaries containing name, type, and description

        Returns:
            tuple: (
                index: FAISS index for vector similarity search,
                entity_map: Dictionary mapping index positions to entity dictionaries,
                token_count: Estimated number of tokens used for embeddings
            )
        """
        if not entities:
            return None, {}, 0

        # Extract descriptions for embedding
        descriptions = [entity.get("description", "") for entity in entities]

        # Estimate token count (roughly 4 chars per token)
        token_count = sum(len(desc) // 4 for desc in descriptions)

        # Get embedding model config from the manager
        embedding_model_config = (
            HuggingFaceConfigManager.get_text_embedding_model_config(ModelType.LITE)
        )

        # Create embeddings using the HuggingFace client
        embedding_response = self.client.feature_extraction(
            text=descriptions, model=embedding_model_config.model_id
        )

        # Convert to numpy array
        embeddings = np.array(embedding_response)

        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        # Create mapping from index to entity
        entity_map = {i: entities[i] for i in range(len(entities))}

        return index, entity_map, token_count

    def _retrieve_relevant_entities(self, index, entity_map, query_text, top_k=3):
        """
        Retrieve relevant entities based on query text using vector similarity.

        Args:
            index: FAISS index for vector similarity search
            entity_map: Dictionary mapping index positions to entity dictionaries
            query_text: Text to find relevant entities for
            top_k: Number of entities to retrieve

        Returns:
            tuple: (
                entities: List of entity dictionaries most relevant to the query,
                token_count: Estimated number of tokens used for query embedding
            )
        """
        if index is None:
            return [], 0

        # Estimate token count (roughly 4 chars per token)
        token_count = len(query_text) // 4

        # Get embedding model config from the manager
        embedding_model_config = (
            HuggingFaceConfigManager.get_text_embedding_model_config(ModelType.LITE)
        )

        # Get embedding for query
        query_emb = self.client.feature_extraction(
            text=[query_text], model=embedding_model_config.model_id
        )

        # Search in FAISS index
        distances, indices = index.search(np.array(query_emb), top_k)

        # Return relevant entities
        return [entity_map.get(i) for i in indices[0]], token_count


class VertexAiRouterService(LlmRouterService):
    """Vertex AI implementation of the LLM router service."""

    def __init__(self):
        vertexai.init(project="bai-buchai-p", location="us-east1")

    def calculate_embedding_cost(self, token_count: int) -> float:
        """
        Calculate cost for text embedding models in Vertex AI.

        For Vertex AI text-embedding models:
        - Approximately $0.0001 per 1K tokens ($0.0000001 per token)

        Rates are based on Google Cloud documentation.
        """
        embedding_cost_per_token = 0.0000001  # $0.0001 per 1K tokens

        return token_count * embedding_cost_per_token

    def calculate_generation_cost(
        self, prompt_tokens: int, completion_tokens: int
    ) -> float:
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
            text_generation_model_config = (
                VertexAiConfigManager.get_text_generation_model_config(
                    request.model_type
                )
            )
            prompt = VertexAiConfigManager.get_prompt_story_generate(
                request.model_type, request.prompt
            )

            model = GenerativeModel(text_generation_model_config.model_id)
            generation_config = GenerationConfig(
                max_output_tokens=text_generation_model_config.max_tokens,
                temperature=text_generation_model_config.temperature,
                top_p=text_generation_model_config.top_p,
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
            cost = self.calculate_generation_cost(prompt_tokens, completion_tokens)

            # Update cost centre if provided
            if request.cost_centre_id:
                await cost_centre_manager.update_cost_centre(
                    request.cost_centre_id, cost
                )

            return TextResponse(
                text=response.text,
                usage=CostUsage(
                    embedding_tokens=0,  # No embeddings used in this method
                    generation_prompt_tokens=prompt_tokens,
                    generation_completion_tokens=completion_tokens,
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
            text_generation_model_config = (
                VertexAiConfigManager.get_text_generation_model_config(
                    request.model_type
                )
            )
            prompt = VertexAiConfigManager.get_prompt_story_generate(
                request.model_type, request.prompt
            )

            model = GenerativeModel(text_generation_model_config.model_id)
            generation_config = GenerationConfig(
                max_output_tokens=text_generation_model_config.max_tokens,
                temperature=text_generation_model_config.temperature,
                top_p=text_generation_model_config.top_p,
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

                cost = self.calculate_generation_cost(prompt_tokens, completion_tokens)
                await cost_centre_manager.update_cost_centre(
                    request.cost_centre_id, cost
                )

        except Exception as e:
            logger.error(
                f"Error streaming story with Vertex AI: {str(e)}\n{format_exc()}"
            )
            raise HTTPException(status_code=500, detail=str(e))

    async def split_story(self, request: GenerateStoryRequest) -> SplitStoryResponse:
        try:
            text_generation_model_config = (
                VertexAiConfigManager.get_text_generation_model_config(
                    request.model_type
                )
            )
            prompt = VertexAiConfigManager.get_prompt_story_split(
                request.model_type, request.prompt
            )

            model = GenerativeModel(text_generation_model_config.model_id)
            generation_config = GenerationConfig(
                max_output_tokens=text_generation_model_config.max_tokens,
                temperature=text_generation_model_config.temperature,
                top_p=text_generation_model_config.top_p,
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
            cost = self.calculate_generation_cost(prompt_tokens, completion_tokens)

            # Update cost centre if provided
            if request.cost_centre_id:
                await cost_centre_manager.update_cost_centre(
                    request.cost_centre_id, cost
                )

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
                usage=CostUsage(
                    embedding_tokens=0,  # No embeddings used in this method
                    generation_prompt_tokens=prompt_tokens,
                    generation_completion_tokens=completion_tokens,
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
            text_generation_model_config = (
                VertexAiConfigManager.get_text_generation_model_config(
                    request.model_type
                )
            )
            prompt = VertexAiConfigManager.get_prompt_story_summarise(
                request.model_type, request.story
            )

            model = GenerativeModel(text_generation_model_config.model_id)
            generation_config = GenerationConfig(
                max_output_tokens=text_generation_model_config.max_tokens,
                temperature=text_generation_model_config.temperature,
                top_p=text_generation_model_config.top_p,
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
            cost = self.calculate_generation_cost(prompt_tokens, completion_tokens)

            # Update cost centre if provided
            if request.cost_centre_id:
                await cost_centre_manager.update_cost_centre(
                    request.cost_centre_id, cost
                )

            return TextResponse(
                text=response.text,
                usage=CostUsage(
                    embedding_tokens=0,  # No embeddings used in this method
                    generation_prompt_tokens=prompt_tokens,
                    generation_completion_tokens=completion_tokens,
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
            text_generation_model_config = (
                VertexAiConfigManager.get_text_generation_model_config(
                    request.model_type
                )
            )

            # Step 1: Extract entities from the story
            entities = await self._extract_story_entities(
                request.model_type, request.story
            )

            # Step 2: Create vector index from entities and track embedding costs
            entity_index, entity_map, embedding_token_count = self._create_entity_index(
                entities
            )
            embedding_cost = self.calculate_embedding_cost(embedding_token_count)

            # Track total costs
            total_prompt_tokens = 0
            total_completion_tokens = 0
            total_embedding_tokens = embedding_token_count
            total_cost = embedding_cost

            model = GenerativeModel(text_generation_model_config.model_id)
            generation_config = GenerationConfig(
                max_output_tokens=text_generation_model_config.max_tokens,
                temperature=text_generation_model_config.temperature,
                top_p=text_generation_model_config.top_p,
            )

            # For each story part
            image_prompts = []
            for part in request.story_parts:
                # Retrieve relevant entities for this part using vector search
                relevant_entities, query_embedding_tokens = (
                    self._retrieve_relevant_entities(
                        entity_index, entity_map, part, top_k=3
                    )
                )
                entity_description = "\n".join(
                    [
                        f"{entity['type'].upper()}: {entity['name']} - {entity['description']}"
                        for entity in relevant_entities
                        if entity
                    ]
                )

                # Add embedding cost for query
                query_embedding_cost = self.calculate_embedding_cost(
                    query_embedding_tokens
                )
                total_embedding_tokens += query_embedding_tokens
                total_cost += query_embedding_cost

                # Use the updated get_prompt_image_generate method with entity_description
                prompt = VertexAiConfigManager.get_prompt_image_generate(
                    request.model_type,
                    request.story,
                    part,
                    entity_description=entity_description,
                )

                prompt_content = Content(role="user", parts=[Part.from_text(prompt)])
                part_response = model.generate_content(
                    prompt_content,
                    generation_config=generation_config,
                )

                # Extract accurate token usage
                prompt_tokens = part_response.usage_metadata.prompt_token_count
                completion_tokens = part_response.usage_metadata.candidates_token_count

                # Calculate cost for this part
                part_cost = self.calculate_generation_cost(
                    prompt_tokens, completion_tokens
                )

                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                total_cost += part_cost

                image_prompts.append(part_response.text)

            # Update cost centre if provided
            if request.cost_centre_id:
                await cost_centre_manager.update_cost_centre(
                    request.cost_centre_id, total_cost
                )

            return ImagePromptsResponse(
                data=image_prompts,
                usage=CostUsage(
                    embedding_tokens=total_embedding_tokens,
                    generation_prompt_tokens=total_prompt_tokens,
                    generation_completion_tokens=total_completion_tokens,
                    total_tokens=total_prompt_tokens
                    + total_completion_tokens
                    + total_embedding_tokens,
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

    async def _extract_story_entities(
        self, model_type: ModelType, story: str
    ) -> List[Dict]:
        """
        Extract entities (characters, locations) from story text using LLM.

        Args:
            model_type: The model type to use for entity extraction
            story: The complete story text

        Returns:
            List of entity dictionaries containing type, name, and description
        """
        text_generation_model_config = (
            VertexAiConfigManager.get_text_generation_model_config(model_type)
        )

        prompt = VertexAiConfigManager.get_prompt_story_entities(model_type, story)

        model = GenerativeModel(text_generation_model_config.model_id)
        generation_config = GenerationConfig(
            max_output_tokens=text_generation_model_config.max_tokens,
            temperature=0.2,  # Lower temperature for more deterministic output
            top_p=0.9,
        )

        prompt_content = Content(role="user", parts=[Part.from_text(prompt)])
        response = model.generate_content(
            prompt_content,
            generation_config=generation_config,
        )

        # Parse the JSON response to extract entities
        try:
            result = response.text
            # Clean up the JSON string if needed
            cleaned_json = result.strip()
            if cleaned_json.startswith("```json"):
                cleaned_json = cleaned_json[7:]
            if cleaned_json.endswith("```"):
                cleaned_json = cleaned_json[:-3]

            entities = json.loads(cleaned_json)
            if not isinstance(entities, list):
                entities = [entities]
            return entities
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            logger.error(f"Failed to parse entity JSON response: {result}")
            return []

    def _create_entity_index(self, entities: List[Dict]) -> tuple:
        """
        Create a FAISS index from entity descriptions for vector search.

        Args:
            entities: List of entity dictionaries containing name, type, and description

        Returns:
            tuple: (
                index: FAISS index for vector similarity search,
                entity_map: Dictionary mapping index positions to entity dictionaries,
                token_count: Estimated number of tokens used for embeddings
            )
        """
        if not entities:
            return None, {}, 0

        # Extract descriptions for embedding
        descriptions = [entity.get("description", "") for entity in entities]

        # Estimate token count (roughly 4 chars per token)
        token_count = sum(len(desc) // 4 for desc in descriptions)

        # Get embedding model config from the manager
        embedding_model_config = VertexAiConfigManager.get_text_embedding_model_config(
            ModelType.LITE
        )

        # Get embeddings using Vertex AI embedding model
        embedding_model = TextEmbeddingModel.from_pretrained(
            embedding_model_config.model_id
        )
        embeddings_response = embedding_model.get_embeddings(descriptions)
        embeddings = np.array(
            [emb.values for emb in embeddings_response], dtype=np.float32
        )

        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        # Create mapping from index to entity
        entity_map = {i: entities[i] for i in range(len(entities))}

        return index, entity_map, token_count

    def _retrieve_relevant_entities(self, index, entity_map, query_text, top_k=3):
        """
        Retrieve relevant entities based on query text using vector similarity.

        Args:
            index: FAISS index for vector similarity search
            entity_map: Dictionary mapping index positions to entity dictionaries
            query_text: Text to find relevant entities for
            top_k: Number of entities to retrieve

        Returns:
            tuple: (
                entities: List of entity dictionaries most relevant to the query,
                token_count: Estimated number of tokens used for query embedding
            )
        """
        if index is None:
            return [], 0

        # Estimate token count (roughly 4 chars per token)
        token_count = len(query_text) // 4

        # Get embedding model config from the manager
        embedding_model_config = VertexAiConfigManager.get_text_embedding_model_config(
            ModelType.LITE
        )

        # Get embedding for query using Vertex AI embedding model
        embedding_model = TextEmbeddingModel.from_pretrained(
            embedding_model_config.model_id
        )
        query_emb_response = embedding_model.get_embeddings([query_text])
        query_emb = np.array([query_emb_response[0].values], dtype=np.float32)

        # Search in FAISS index
        distances, indices = index.search(query_emb, top_k)

        # Return relevant entities
        return [entity_map.get(i) for i in indices[0]], token_count


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
