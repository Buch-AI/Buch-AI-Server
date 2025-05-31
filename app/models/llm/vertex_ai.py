from typing import Dict, Optional

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
    def get_prompt_story_generate(model_type: ModelType, user_prompt: str) -> str:
        """Get the formatted prompt for story generation."""
        model_config = VertexAiConfigManager.get_text_generation_model_config(
            model_type
        )
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
    def get_prompt_story_entities(model_type: ModelType, story: str) -> str:
        return (
            "You are an assistant helping convert story chapters into image prompts.\n"
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
        )

    @staticmethod
    def get_prompt_image_generate(
        model_type: ModelType,
        story: str,
        story_part: str,
        entity_description: Optional[str] = None,
    ) -> str:
        """Get the formatted prompt for image generation."""
        system_prompt = (
            "You are an expert at creating image generation prompts. "
            "Given a story or text passage, create a descriptive prompt that captures the essence of the text for an image generation AI. "
            "Be specific, detailed, and concise. Keep the prompt within 3 sentences. "
            "Do not include visual elements that are not present in this section of the story. "
        )

        user_prompt = ""

        if entity_description:
            user_prompt += f"\n\nVisual elements that appear throughout the story and can be selectively referenced in this section:\n{entity_description}"

        user_prompt += f"\n\nCreate an image generation prompt based on the following excerpt of the story:\n'{story_part}'."

        return f"{system_prompt}\n{user_prompt}"

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
