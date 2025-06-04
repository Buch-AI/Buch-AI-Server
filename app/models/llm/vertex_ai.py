from typing import Dict

from app.models.llm.common import (
    ConfigManager,
    ModelType,
    TextEmbeddingModelConfig,
    TextGenerationModelConfig,
)


class VertexAiConfigManager(ConfigManager):
    text_embedding_model_configs: Dict[ModelType, TextEmbeddingModelConfig] = {
        ModelType.LITE: TextEmbeddingModelConfig(
            model_id="text-embedding-005",
        )
    }

    text_generation_model_configs: Dict[ModelType, TextGenerationModelConfig] = {
        ModelType.LITE: TextGenerationModelConfig(
            model_id="gemini-2.0-flash-lite",
            max_tokens=1024,
            temperature=0.6,
            top_p=0.9,
        ),
        ModelType.STANDARD: TextGenerationModelConfig(
            model_id="gemini-2.0-flash",
            max_tokens=1024,
            temperature=0.6,
            top_p=0.9,
        ),
        # Add more configurations as needed
        # ModelType.STANDARD: ModelConfig(...),
        # ModelType.PRO: ModelConfig(...),
    }

    @staticmethod
    def get_text_embedding_model_config(
        model_type: ModelType,
    ) -> TextEmbeddingModelConfig:
        """Get the configuration for a specific model type."""
        if model_type not in VertexAiConfigManager.text_embedding_model_configs:
            raise ValueError(f"Model type {model_type} not found in configurations")
        return VertexAiConfigManager.text_embedding_model_configs[model_type]

    @staticmethod
    def get_text_generation_model_config(
        model_type: ModelType,
    ) -> TextGenerationModelConfig:
        """Get the configuration for a specific model type."""
        if model_type not in VertexAiConfigManager.text_generation_model_configs:
            raise ValueError(f"Model type {model_type} not found in configurations")
        return VertexAiConfigManager.text_generation_model_configs[model_type]

    @staticmethod
    def calculate_embedding_cost(model_type: ModelType, token_count: int) -> float:
        """
        Calculate cost for text embedding models in Vertex AI.
        """
        if model_type == ModelType.LITE:
            # https://ai.google.dev/gemini-api/docs/pricing#text-embedding-004
            embedding_cost_per_token = 0.0
        if model_type == ModelType.STANDARD:
            embedding_cost_per_token = 0.0
        if model_type == ModelType.PRO:
            embedding_cost_per_token = 0.0
        if model_type == ModelType.MAX:
            embedding_cost_per_token = 0.0

        return token_count * embedding_cost_per_token

    @staticmethod
    def calculate_generation_cost(
        model_type: ModelType, prompt_tokens: int, completion_tokens: int
    ) -> float:
        """
        Calculate cost for Vertex AI models.
        """
        if model_type == ModelType.LITE:
            # https://cloud.google.com/vertex-ai/generative-ai/pricing#gemini-models
            input_cost_per_token = 0.075e-6
            output_cost_per_token = 0.300e-6
        if model_type == ModelType.STANDARD:
            input_cost_per_token = 0.150e-6
            output_cost_per_token = 0.600e-6
        if model_type == ModelType.PRO:
            input_cost_per_token = 0.150e-6
            output_cost_per_token = 0.600e-6
        if model_type == ModelType.MAX:
            input_cost_per_token = 0.150e-6
            output_cost_per_token = 0.600e-6

        return (prompt_tokens * input_cost_per_token) + (
            completion_tokens * output_cost_per_token
        )
