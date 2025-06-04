from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Optional

from pydantic import BaseModel, Field


class ModelType(str, Enum):
    """Available model configurations."""

    LITE = "lite"
    STANDARD = "standard"
    PRO = "pro"
    MAX = "max"


class Prompt(BaseModel):
    """Standardized prompt structure for all LLM providers."""

    system_message: str
    user_message: str

    def __str__(self) -> str:
        """Convert the prompt to a string representation with system and user messages separated by double newlines."""
        return f"{self.system_message}\n\n{self.user_message}"


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
    def get_prompt_story_generate(model_type: ModelType, user_prompt: str) -> Prompt:
        """Get the formatted prompt for story generation."""
        return Prompt(
            system_message="You are a creative story writer.",
            user_message=f"Write a story based on this prompt: {user_prompt}",
        )

    @staticmethod
    def get_prompt_story_split(model_type: ModelType, user_prompt: str) -> Prompt:
        """Get the formatted prompt for story splitting."""
        _user_prompt = user_prompt.replace("\n", " ")
        return Prompt(
            system_message="You are a story analyzer. Your job is to re-format and split story text into a pre-defined number of parts and sub-parts, inserting a [PART] tag between each part and a [SUBPART] tag between each sub-part. Remove empty lines if the text belongs to the same part. Do not include any other text in your response. Do not include section titles.",
            user_message=f"Split the following story into 4 parts and 3 sub-parts per part: {_user_prompt}",
        )

    @staticmethod
    def get_prompt_story_summarise(model_type: ModelType, story: str) -> Prompt:
        """Get the formatted prompt for story summarisation."""
        return Prompt(
            system_message="You are an expert at summarizing stories and text. Create a concise summary that captures the key elements, main plot points, and core themes of the provided text within 3 sentences.",
            user_message=f"Summarize the following story: {story}",
        )

    @staticmethod
    def get_prompt_story_entities(model_type: ModelType, story: str) -> Prompt:
        """Get the formatted prompt for listing out story entities."""
        return Prompt(
            system_message="You are an assistant helping convert story chapters into image prompts.",
            user_message=(
                "Extract the key visual elements that should appear in a scene illustration for the following story in the following JSON format:\n"
                '"""\n'
                "{"
                '    "type": (One of the following options: "character", "location"),'
                '    "name": (The name of the entity),'
                '    "description": (The visual description of the entity that will be used in the image prompt),'
                "}"
                '"""\n'
                "\n"
                "Refer to the following story:\n"
                '"""\n'
                f"{story}"
                '"""\n'
            ),
        )

    @staticmethod
    def get_prompt_image_generate(
        model_type: ModelType,
        story: str,
        story_part: str,
        entity_description: Optional[str] = None,
    ) -> Prompt:
        """Get the formatted prompt for image generation.

        Args:
            model_type: The model type to use.
            story: The complete story text.
            story_part: The specific part of the story to generate an image for.
            entity_description: Optional context about entities (characters, locations) to maintain consistency.
        """
        system_message = (
            "You are an expert at creating image generation prompts. "
            "Given a story or text passage, create a descriptive prompt that captures the essence of the text for an image generation AI. "
            "Be specific, detailed, and concise. Keep the prompt within 3 sentences. "
            "Do not include visual elements that are not present in this section of the story. "
        )

        user_message = ""
        if entity_description:
            user_message += f"\n\nVisual elements that appear throughout the story and can be selectively referenced in this section:\n{entity_description}"

        user_message += f"\n\nCreate an image generation prompt based on the following excerpt of the story:\n'{story_part}'."

        return Prompt(system_message=system_message, user_message=user_message)

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
