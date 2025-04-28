from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Configuration for a language model."""

    model_id: str
    max_tokens: int = Field(gt=0)
    temperature: float = Field(ge=0.0, le=1.0)
    top_p: float = Field(ge=0.0, le=1.0)


class ModelType(str, Enum):
    """Available model configurations."""

    LITE = "lite"
    STANDARD = "standard"
    PRO = "pro"
    MAX = "max"


class ConfigManager(ABC):
    """Abstract base class for LLM configuration managers."""

    model_configs: Dict[ModelType, ModelConfig] = {}

    @staticmethod
    @abstractmethod
    def get_model_config(model_type: ModelType) -> ModelConfig:
        """Get the configuration for a specific model type."""
        pass

    @staticmethod
    @abstractmethod
    def get_prompt_story_generate(model_type: ModelType, user_prompt: str) -> Any:
        """Get the formatted prompt for story generation."""
        pass

    @staticmethod
    @abstractmethod
    def get_prompt_story_split(model_type: ModelType, user_prompt: str) -> Any:
        """Get the formatted prompt for story splitting."""
        pass

    @staticmethod
    @abstractmethod
    def get_prompt_story_summarise(model_type: ModelType, story: str) -> Any:
        """Get the formatted prompt for story summarisation."""
        pass

    @staticmethod
    @abstractmethod
    def get_prompt_image_generate(
        model_type: ModelType, story_summary: str, story_part: str
    ) -> Any:
        """Get the formatted prompt for image generation."""
        pass


class HuggingFaceConfigManager(ConfigManager):
    # Define preset configurations for different model types
    model_configs: Dict[ModelType, ModelConfig] = {
        ModelType.LITE: ModelConfig(
            model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            max_tokens=1024,
            temperature=0.6,
            top_p=0.9,
        ),
        ModelType.STANDARD: ModelConfig(
            model_id="mistralai/Mistral-7B-Instruct-v0.2",
            max_tokens=1024,
            temperature=0.6,
            top_p=0.9,
        ),
        ModelType.PRO: ModelConfig(
            model_id="google/gemma-3-27b-it",
            max_tokens=2048,
            temperature=0.6,
            top_p=0.9,
        ),
    }

    @staticmethod
    def get_model_config(model_type: ModelType) -> ModelConfig:
        """Get the configuration for a specific model type."""
        if model_type not in HuggingFaceConfigManager.model_configs:
            raise ValueError(f"Model type {model_type} not found in configurations")
        return HuggingFaceConfigManager.model_configs[model_type]

    @staticmethod
    def get_prompt_story_generate(
        model_type: ModelType, user_prompt: str
    ) -> list[dict]:
        model_config = HuggingFaceConfigManager.get_model_config(model_type)
        return [
            {
                "role": "system",
                "content": f"You are a creative story writer. Your story should be at most {model_config.max_tokens} tokens long.",
            },
            {
                "role": "user",
                "content": f"Write a story based on this prompt: {user_prompt}",
            },
        ]

    @staticmethod
    def get_prompt_story_split(model_type: ModelType, user_prompt: str) -> list[dict]:
        _user_prompt = user_prompt.replace("\n", " ")
        return [
            {
                "role": "system",
                "content": "You are a story analyzer. Your job is to re-format and split story text into a pre-defined number of parts and sub-parts, inserting a [PART] tag between each part and a [SUBPART] tag between each sub-part. Remove empty lines if the text belongs to the same part. Do not include any other text in your response. Do not include section titles.",
            },
            {
                "role": "user",
                "content": f"Split the following story into 4 parts and 3 sub-parts per part: {_user_prompt}",
            },
        ]

    @staticmethod
    def get_prompt_story_summarise(model_type: ModelType, story: str) -> list[dict]:
        """Get the formatted prompt for story summarisation."""
        return [
            {
                "role": "system",
                "content": "You are an expert at summarizing stories and text. Create a concise summary that captures the key elements, main plot points, and core themes of the provided text within 3 sentences.",
            },
            {"role": "user", "content": f"Summarize the following story: {story}"},
        ]

    @staticmethod
    def get_prompt_image_generate(
        model_type: ModelType, story_summary: str, story_part: str
    ) -> list[dict]:
        """Get the formatted prompt for image generation."""
        return [
            {
                "role": "system",
                "content": "You are an expert at creating image generation prompts. Given a story or text passage, create a descriptive prompt that captures the essence of the text for an image generation AI. Focus on visual elements, setting, characters, mood, and style. Be specific, detailed, and concise. Keep the prompt within 3 sentences.",
            },
            {
                "role": "user",
                "content": (
                    f"The following is a summary of the story: '{story_summary}'."
                    f"Create an image generation prompt based the following section of the story: '{story_part}'."
                ),
            },
        ]


class VertexAiConfigManager(ConfigManager):
    # Define preset configurations for different model types
    model_configs: Dict[ModelType, ModelConfig] = {
        ModelType.LITE: ModelConfig(
            model_id="gemini-1.5-flash",
            max_tokens=1024,
            temperature=0.6,
            top_p=0.9,
        ),
        # Add more configurations as needed
        # ModelType.STANDARD: ModelConfig(...),
        # ModelType.PRO: ModelConfig(...),
    }

    @staticmethod
    def get_model_config(model_type: ModelType) -> ModelConfig:
        """Get the configuration for a specific model type."""
        if model_type not in VertexAiConfigManager.model_configs:
            raise ValueError(f"Model type {model_type} not found in configurations")
        return VertexAiConfigManager.model_configs[model_type]

    @staticmethod
    def get_prompt_story_generate(model_type: ModelType, user_prompt: str) -> str:
        """Get the formatted prompt for story generation."""
        model_config = VertexAiConfigManager.get_model_config(model_type)
        return (
            f"You are a creative story writer. Your story should be at most {model_config.max_tokens} tokens long."
            f"Write a story based on this prompt: {user_prompt}"
        )

    @staticmethod
    def get_prompt_story_split(model_type: ModelType, user_prompt: str) -> str:
        _user_prompt = user_prompt.replace("\n", " ")
        return (
            f"You are a story analyzer. Your job is to re-format and split story text into a pre-defined number of parts and sub-parts, inserting a [PART] tag between each part and a [SUBPART] tag between each sub-part. Remove empty lines if the text belongs to the same part. Do not include any other text in your response. Do not include section titles."
            f"Split the following story into 4 parts and 3 sub-parts per part: {_user_prompt}"
        )

    @staticmethod
    def get_prompt_story_summarise(model_type: ModelType, story: str) -> str:
        """Get the formatted prompt for story summarisation."""
        return (
            "You are an expert at summarizing stories and text. Create a concise summary that captures the key elements, main plot points, and core themes of the provided text within 3 sentences. "
            f"Summarize the following story: {story}"
        )

    @staticmethod
    def get_prompt_image_generate(
        model_type: ModelType, story_summary: str, story_part: str
    ) -> str:
        """Get the formatted prompt for image generation."""
        return (
            "You are an expert at creating image generation prompts. Given a story or text passage, create a descriptive prompt that captures the essence of the text for an image generation AI. "
            "Focus on visual elements, setting, characters, mood, and style. Be specific, detailed, and concise. "
            "Keep the prompt within 3 sentences. "
            f"The following is a summary of the story: '{story_summary}'."
            f"Create an image generation prompt based on the following section of the story: '{story_part}'."
        )
