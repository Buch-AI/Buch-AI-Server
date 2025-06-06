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

    @classmethod
    def get_prompt_story_generate(
        cls, model_type: ModelType, user_prompt: str
    ) -> Prompt:
        """Get the formatted prompt for story generation."""
        word_limit = int(
            cls.get_text_generation_model_config(model_type).max_tokens * 0.75
        )

        return Prompt(
            system_message=(
                "You are a skilled creative writer specializing in engaging storytelling. "
                "Create compelling narratives with well-developed characters, vivid settings, and clear story arcs. "
                "Focus on showing rather than telling, use descriptive language, and maintain consistent tone throughout."
            ),
            user_message=(
                f"Write a complete story based on this prompt. "
                f"The story should be {word_limit} words or less. "
                f"Include character development, setting details, dialogue where appropriate, and a satisfying resolution:\n"
                f'"""\n{user_prompt}\n"""'
            ),
        )

    @classmethod
    def get_prompt_story_split(cls, model_type: ModelType, user_prompt: str) -> Prompt:
        """Get the formatted prompt for story splitting."""
        _user_prompt = user_prompt.replace("\n", " ")
        return Prompt(
            system_message=(
                "You are a story structure analyzer. "
                "Your task is to reorganize story text into exactly 4 main parts with 3 sub-parts each. "
                "Use [PART] tags to separate main parts and [SUBPART] tags to separate sub-parts within each part. "
                "Ensure each section has meaningful content and natural narrative flow. "
                "Remove excessive line breaks but preserve paragraph structure. "
                "Output ONLY the formatted story text with tags - no explanations or section titles."
            ),
            user_message=(
                f"Restructure the following story into 4 parts with 3 sub-parts each, "
                f"ensuring balanced content distribution and logical narrative breaks:\n"
                f'"""\n{_user_prompt}\n"""'
            ),
        )

    @classmethod
    def get_prompt_story_summarise(cls, model_type: ModelType, story: str) -> Prompt:
        """Get the formatted prompt for story summarisation."""
        return Prompt(
            system_message=(
                "You are an expert literary analyst specializing in concise story summarization. "
                "Create summaries that capture the essential plot, main characters, central conflict, and resolution. "
                "Focus on the most important narrative elements while maintaining the story's tone and impact. "
                "Write exactly 3 sentences that flow naturally together."
            ),
            user_message=(
                f"Provide a 3-sentence summary of this story that captures its main plot points, "
                f"key characters, and central themes:\n"
                f'"""\n{story}\n"""'
            ),
        )

    @classmethod
    def get_prompt_story_entities(cls, model_type: ModelType, story: str) -> Prompt:
        """Get the formatted prompt for listing out story entities."""
        return Prompt(
            system_message=(
                "You are a visual content specialist who extracts key visual elements from stories for illustration purposes. "
                "Focus on characters and locations that are central to the story and would be important for visual consistency across illustrations. "
                "Provide detailed descriptions suitable for image generation, including physical appearance, clothing, architectural details, and atmospheric elements."
            ),
            user_message=(
                "Extract the key visual entities from the following story that should appear in scene illustrations. "
                "Return a JSON array where each entity follows this exact format:\n"
                '"""\n'
                "{\n"
                '    "type": "character", "object", "location", "background" or "other",\n'
                '    "name": "Entity name",\n'
                '    "description": "Detailed visual description for image generation"\n'
                "}\n"
                '"""\n'
                "\n"
                "Only include entities that are visually significant and mentioned multiple times or are central to key scenes. "
                "Make descriptions specific enough for consistent visual representation across multiple images.\n\n"
                "Story to analyze:\n"
                '"""\n'
                f"{story}\n"
                '"""\n'
            ),
        )

    @classmethod
    def get_prompt_image_generate(
        cls,
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
            "You are an expert at crafting detailed image generation prompts for AI art systems. "
            "Create vivid, specific prompts that capture the mood, setting, characters, and action of story scenes. "
            "Include artistic style guidance, lighting, composition, and atmosphere details. "
            "Focus on visual elements that enhance storytelling and emotional impact. "
            "Keep prompts concise but richly descriptive, within 2-3 sentences."
        )

        user_message = ""
        if entity_description:
            user_message += (
                f"Visual reference for consistent character/location representation:\n"
                f"{entity_description}\n\n"
                f"Use these references selectively - only include elements that appear in this specific scene.\n\n"
            )

        user_message += (
            f"Create a detailed image generation prompt for this story excerpt. "
            f"Focus on the specific scene, mood, and visual elements present in this section. "
            f"Include artistic style, lighting, and composition suggestions:\n"
            f'"""\n{story_part}\n"""'
        )

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
