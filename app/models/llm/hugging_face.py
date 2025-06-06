from typing import Dict

from app.models.llm.common import (
    ConfigManager,
    ModelType,
    TextEmbeddingModelConfig,
    TextGenerationModelConfig,
)


class HuggingFaceConfigManager(ConfigManager):
    text_embedding_model_configs: Dict[ModelType, TextEmbeddingModelConfig] = {
        ModelType.LITE: TextEmbeddingModelConfig(
            model_id="sentence-transformers/all-MiniLM-L6-v2",
        )
    }

    text_generation_model_configs: Dict[ModelType, TextGenerationModelConfig] = {
        ModelType.LITE: TextGenerationModelConfig(
            model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            max_tokens=512,
            temperature=0.6,
            top_p=0.9,
        ),
        ModelType.STANDARD: TextGenerationModelConfig(
            model_id="mistralai/Mistral-7B-Instruct-v0.2",
            max_tokens=512,
            temperature=0.6,
            top_p=0.9,
        ),
        ModelType.PRO: TextGenerationModelConfig(
            model_id="google/gemma-3-27b-it",
            max_tokens=1024,
            temperature=0.6,
            top_p=0.9,
        ),
    }

    @staticmethod
    def get_text_embedding_model_config(
        model_type: ModelType,
    ) -> TextEmbeddingModelConfig:
        """Get the configuration for a specific model type."""
        if model_type not in HuggingFaceConfigManager.text_embedding_model_configs:
            raise ValueError(f"Model type {model_type} not found in configurations")
        return HuggingFaceConfigManager.text_embedding_model_configs[model_type]

    @staticmethod
    def get_text_generation_model_config(
        model_type: ModelType,
    ) -> TextGenerationModelConfig:
        """Get the configuration for a specific model type."""
        if model_type not in HuggingFaceConfigManager.text_generation_model_configs:
            raise ValueError(f"Model type {model_type} not found in configurations")
        return HuggingFaceConfigManager.text_generation_model_configs[model_type]

    @staticmethod
    def calculate_embedding_cost(model_type: ModelType, token_count: int) -> float:
        """
        Calculate cost for text embedding models in Hugging Face.

        For sentence-transformers models on Hugging Face Inference API:
        - Approximately $0.0001 per 1K tokens ($0.0000001 per token)

        These are approximate rates for embedding models on the Inference API.
        """
        # TODO: Add pricing for Hugging Face models.
        embedding_cost_per_token = 0.0000001  # $0.0001 per 1K tokens

        return token_count * embedding_cost_per_token

    @staticmethod
    def calculate_generation_cost(
        model_type: ModelType, prompt_tokens: int, completion_tokens: int
    ) -> float:
        """
        Calculate cost for Hugging Face models.

        For Llama 2 models on Hugging Face Inference API:
        - Input: $0.0002 per 1K tokens ($0.0000002 per token)
        - Output: $0.0002 per 1K tokens ($0.0000002 per token)

        These are approximate rates for Llama 2 models on the Inference API.
        """
        # TODO: Add pricing for Hugging Face models.
        input_cost_per_token = 0.0000002  # $0.0002 per 1K tokens
        output_cost_per_token = 0.0000002  # $0.0002 per 1K tokens

        return (prompt_tokens * input_cost_per_token) + (
            completion_tokens * output_cost_per_token
        )
