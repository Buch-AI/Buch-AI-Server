import json
import logging
from traceback import format_exc
from typing import AsyncGenerator, Dict, List

import faiss
import numpy as np
import vertexai
from fastapi import HTTPException, status
from vertexai.generative_models import Content, GenerationConfig, GenerativeModel, Part
from vertexai.language_models import TextEmbeddingModel

from app.models.cost_centre import CostCentreManager
from app.models.llm import ModelType, VertexAiConfigManager
from app.services.llm.common import (
    CostUsage,
    GenerateImagePromptsRequest,
    GenerateStoryRequest,
    ImagePromptsResponse,
    LlmLogger,
    LlmRouterService,
    SplitStoryResponse,
    SummariseStoryRequest,
    TextResponse,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize cost centre manager
cost_centre_manager = CostCentreManager()


class VertexAiRouterService(LlmRouterService):
    """Vertex AI implementation of the LLM router service."""

    def __init__(self):
        vertexai.init(project="bai-buchai-p", location="us-east1")

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
            LlmLogger.log_prompt(request.cost_centre_id, prompt)

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
            cost = VertexAiConfigManager.calculate_generation_cost(
                request.model_type, prompt_tokens, completion_tokens
            )

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
            LlmLogger.log_prompt(request.cost_centre_id, prompt)

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

                cost = VertexAiConfigManager.calculate_generation_cost(
                    request.model_type, prompt_tokens, completion_tokens
                )
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
            LlmLogger.log_prompt(request.cost_centre_id, prompt)

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
            cost = VertexAiConfigManager.calculate_generation_cost(
                request.model_type, prompt_tokens, completion_tokens
            )

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
            LlmLogger.log_prompt(request.cost_centre_id, prompt)

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
            cost = VertexAiConfigManager.calculate_generation_cost(
                request.model_type, prompt_tokens, completion_tokens
            )

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
                request.model_type, request.story, request.cost_centre_id
            )

            # Step 2: Create vector index from entities and track embedding costs
            entity_index, entity_map, embedding_token_count = self._create_entity_index(
                request.model_type, entities
            )
            embedding_cost = VertexAiConfigManager.calculate_embedding_cost(
                request.model_type, embedding_token_count
            )

            # Track total costs
            total_embedding_tokens = embedding_token_count
            total_prompt_tokens = 0
            total_completion_tokens = 0
            total_cost = embedding_cost

            model = GenerativeModel(text_generation_model_config.model_id)
            generation_config = GenerationConfig(
                max_output_tokens=text_generation_model_config.max_tokens,
                temperature=text_generation_model_config.temperature,
                top_p=text_generation_model_config.top_p,
            )

            image_prompts = []

            # For each story part
            for part in request.story_parts:
                # Retrieve relevant entities for this part using vector search
                relevant_entities, query_embedding_tokens = (
                    self._retrieve_relevant_entities(
                        request.model_type, entity_index, entity_map, part, top_k=5
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
                query_embedding_cost = VertexAiConfigManager.calculate_embedding_cost(
                    request.model_type, query_embedding_tokens
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
                LlmLogger.log_prompt(request.cost_centre_id, prompt)

                prompt_content = Content(role="user", parts=[Part.from_text(prompt)])
                part_response = model.generate_content(
                    prompt_content,
                    generation_config=generation_config,
                )

                # Extract accurate token usage
                prompt_tokens = part_response.usage_metadata.prompt_token_count
                completion_tokens = part_response.usage_metadata.candidates_token_count

                # Calculate cost for this part
                part_cost = VertexAiConfigManager.calculate_generation_cost(
                    request.model_type, prompt_tokens, completion_tokens
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
        self, model_type: ModelType, story: str, cost_centre_id: str
    ) -> List[Dict]:
        """
        Extract entities (characters, locations) from story text using LLM.

        Args:
            model_type: The model type to use for entity extraction
            story: The complete story text
            cost_centre_id: The cost centre ID to use for logging

        Returns:
            List of entity dictionaries containing type, name, and description
        """
        text_generation_model_config = (
            VertexAiConfigManager.get_text_generation_model_config(model_type)
        )

        prompt = VertexAiConfigManager.get_prompt_story_entities(model_type, story)
        LlmLogger.log_prompt(cost_centre_id, prompt)

        model = GenerativeModel(text_generation_model_config.model_id)
        generation_config = GenerationConfig(
            max_output_tokens=text_generation_model_config.max_tokens,
            temperature=0.2,  # Lower temperature for more deterministic output
            top_p=0.9,
        )
        # TODO: Is this cost being tracked?

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

    def _create_entity_index(
        self, model_type: ModelType, entities: List[Dict]
    ) -> tuple:
        """
        Create a FAISS index from entity descriptions for vector search.

        Args:
            model_type: The model type to use for embedding
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
            model_type
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

    def _retrieve_relevant_entities(
        self, model_type: ModelType, index, entity_map, query_text, top_k
    ):
        """
        Retrieve relevant entities based on query text using vector similarity.

        Args:
            model_type: The model type to use for embedding
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
            model_type
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
