from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ModelType(str, Enum):
    """Available model configurations."""

    LITE = "lite"
    STANDARD = "standard"
    PRO = "pro"
    MAX = "max"


class TextEmbeddingModelConfig(BaseModel):
    """Configuration for a language model."""

    model_id: str


class TextGenerationModelConfig(BaseModel):
    """Configuration for a language model."""

    model_id: str
    max_tokens: int = Field(gt=0)
    temperature: float = Field(ge=0.0, le=1.0)
    top_p: float = Field(ge=0.0, le=1.0)


class ConfigManager(ABC):
    """Abstract base class for LLM configuration managers."""

    text_embedding_model_configs: Dict[ModelType, TextEmbeddingModelConfig] = {}
    text_generation_model_configs: Dict[ModelType, TextGenerationModelConfig] = {}

    @staticmethod
    @abstractmethod
    def get_text_embedding_model_config(
        model_type: ModelType,
    ) -> TextEmbeddingModelConfig:
        """Get the configuration for a specific model type."""
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def get_text_generation_model_config(
        model_type: ModelType,
    ) -> TextGenerationModelConfig:
        """Get the configuration for a specific model type."""
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def get_prompt_story_generate(model_type: ModelType, user_prompt: str) -> Any:
        """Get the formatted prompt for story generation."""
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def get_prompt_story_split(model_type: ModelType, user_prompt: str) -> Any:
        """Get the formatted prompt for story splitting."""
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def get_prompt_story_summarise(model_type: ModelType, story: str) -> Any:
        """Get the formatted prompt for story summarisation."""
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def get_prompt_story_entities(model_type: ModelType, story: str) -> Any:
        """Get the formatted prompt for listing out story entities."""
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def get_prompt_image_generate(
        model_type: ModelType,
        story: str,
        story_part: str,
        entity_description: Optional[str] = None,
    ) -> Any:
        """Get the formatted prompt for image generation.

        Args:
            model_type: The model type to use.
            story: The complete story text.
            story_part: The specific part of the story to generate an image for.
            entity_description: Optional context about entities (characters, locations) to maintain consistency.
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def calculate_embedding_cost(model_type: ModelType, token_count: int) -> float:
        """Calculate the cost based on embedding token usage.

        Args:
            model_type: The model type used for the embedding.
            token_count: Number of tokens processed.

        Returns:
            Cost in USD.
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def calculate_generation_cost(
        model_type: ModelType, prompt_tokens: int, completion_tokens: int
    ) -> float:
        """Calculate the cost based on token usage.

        Args:
            model_type: The model type used for generation.
            prompt_tokens: Number of tokens in the prompt.
            completion_tokens: Number of tokens in the completion.

        Returns:
            Cost in USD.
        """
        raise NotImplementedError()
