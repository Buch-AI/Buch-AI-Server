from typing import Dict, Optional

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
            max_tokens=1024,
            temperature=0.6,
            top_p=0.9,
        ),
        ModelType.STANDARD: TextGenerationModelConfig(
            model_id="mistralai/Mistral-7B-Instruct-v0.2",
            max_tokens=1024,
            temperature=0.6,
            top_p=0.9,
        ),
        ModelType.PRO: TextGenerationModelConfig(
            model_id="google/gemma-3-27b-it",
            max_tokens=2048,
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
    def get_prompt_story_generate(
        model_type: ModelType, user_prompt: str
    ) -> list[dict]:
        model_config = HuggingFaceConfigManager.get_text_generation_model_config(
            model_type
        )
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
    def get_prompt_story_entities(model_type: ModelType, story: str) -> list[dict]:
        return [
            {
                "role": "system",
                "content": "You are an assistant helping convert story chapters into image prompts.",
            },
            {
                "role": "user",
                "content": (
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
            },
        ]

    @staticmethod
    def get_prompt_image_generate(
        model_type: ModelType,
        story: str,
        story_part: str,
        entity_description: Optional[str] = None,
    ) -> list[dict]:
        """Get the formatted prompt for image generation."""
        system_content = (
            "You are an expert at creating image generation prompts. "
            "Given a story or text passage, create a descriptive prompt that captures the essence of the text for an image generation AI. "
            "Be specific, detailed, and concise. Keep the prompt within 3 sentences. "
            "Do not include visual elements that are not present in this section of the story. "
        )

        user_content = ""

        if entity_description:
            user_content += f"\n\nVisual elements that appear throughout the story and can be selectively referenced in this section:\n{entity_description}"

        user_content += f"\n\nCreate an image generation prompt based on the following excerpt of the story:\n'{story_part}'."

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

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
