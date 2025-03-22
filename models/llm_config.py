from enum import Enum
from typing import Dict

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


# Define preset configurations for different model types
MODEL_CONFIGS: Dict[ModelType, ModelConfig] = {
    ModelType.LITE: ModelConfig(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_tokens=2048,
        temperature=0.8,
        top_p=0.9,
    ),
    ModelType.STANDARD: ModelConfig(
        model_id="mistralai/Mistral-7B-Instruct-v0.2",
        max_tokens=2048,
        temperature=0.8,
        top_p=0.9,
    ),
    ModelType.PRO: ModelConfig(
        model_id="google/gemma-3-27b-it",
        max_tokens=4096,
        temperature=0.8,
        top_p=0.9,
    ),
    # Add more configurations as needed
    # ModelType.PRO: ModelConfig(...),
    # ModelType.MAX: ModelConfig(...),
}


def get_model_config(model_type: ModelType) -> ModelConfig:
    """Get the configuration for a specific model type."""
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Model type {model_type} not found in configurations")
    return MODEL_CONFIGS[model_type]


def get_prompt_messages_story(model_type: ModelType, user_prompt: str) -> list[dict]:
    model_config = get_model_config(model_type)
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
