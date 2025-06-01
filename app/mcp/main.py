import asyncio
import base64
import json
from typing import List

from mcp.server.fastmcp import FastMCP

# Import the config generator
from app.mcp.config_generator import write_config_file
from app.services.image.common import ImageGenerationRequest
from app.services.image.pollinations_ai import PollinationsAiRouterService
from app.services.llm.common import (
    GenerateImagePromptsRequest,
    GenerateStoryRequest,
    SummariseStoryRequest,
)
from app.services.llm.vertex_ai import VertexAiRouterService

# Initialize services
llm_service = VertexAiRouterService()
image_service = PollinationsAiRouterService()

# Create FastMCP server
mcp = FastMCP("buch-ai-mcp-server", host="127.0.0.1", port=8050)


@mcp.tool()
def generate_story_string(
    prompt: str, model_type: str = "lite", cost_centre_id: str = None
) -> str:
    """Generate a complete story as a single string.

    Args:
        prompt: The story prompt or theme
        model_type: The model type to use (lite, standard, pro, max)
        cost_centre_id: Optional cost centre ID for tracking

    Returns:
        JSON string containing the story and usage information
    """

    async def _generate():
        request = GenerateStoryRequest(
            prompt=prompt, model_type=model_type, cost_centre_id=cost_centre_id
        )
        response = await llm_service.generate_story_string(request)
        return json.dumps(
            {
                "story": response.text,
                "usage": {
                    "embedding_tokens": response.usage.embedding_tokens,
                    "generation_prompt_tokens": response.usage.generation_prompt_tokens,
                    "generation_completion_tokens": response.usage.generation_completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                    "cost": response.usage.cost,
                },
            },
            indent=2,
        )

    return asyncio.run(_generate())


@mcp.tool()
def split_story(
    prompt: str, model_type: str = "lite", cost_centre_id: str = None
) -> str:
    """Generate a story split into parts and sub-parts.

    Args:
        prompt: The story prompt or theme
        model_type: The model type to use (lite, standard, pro, max)
        cost_centre_id: Optional cost centre ID for tracking

    Returns:
        JSON string containing the story parts and usage information
    """

    async def _split():
        request = GenerateStoryRequest(
            prompt=prompt, model_type=model_type, cost_centre_id=cost_centre_id
        )
        response = await llm_service.split_story(request)
        return json.dumps(
            {
                "story_parts": response.data,
                "usage": {
                    "embedding_tokens": response.usage.embedding_tokens,
                    "generation_prompt_tokens": response.usage.generation_prompt_tokens,
                    "generation_completion_tokens": response.usage.generation_completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                    "cost": response.usage.cost,
                },
            },
            indent=2,
        )

    return asyncio.run(_split())


@mcp.tool()
def summarise_story(
    story: str, model_type: str = "lite", cost_centre_id: str = None
) -> str:
    """Summarise an existing story.

    Args:
        story: The story text to summarise
        model_type: The model type to use (lite, standard, pro, max)
        cost_centre_id: Optional cost centre ID for tracking

    Returns:
        JSON string containing the summary and usage information
    """

    async def _summarise():
        request = SummariseStoryRequest(
            story=story, model_type=model_type, cost_centre_id=cost_centre_id
        )
        response = await llm_service.summarise_story(request)
        return json.dumps(
            {
                "summary": response.text,
                "usage": {
                    "embedding_tokens": response.usage.embedding_tokens,
                    "generation_prompt_tokens": response.usage.generation_prompt_tokens,
                    "generation_completion_tokens": response.usage.generation_completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                    "cost": response.usage.cost,
                },
            },
            indent=2,
        )

    return asyncio.run(_summarise())


@mcp.tool()
def generate_image_prompts(
    story: str,
    story_parts: List[str],
    model_type: str = "lite",
    cost_centre_id: str = None,
) -> str:
    """Generate image prompts for each section of a story.

    Args:
        story: The complete story text
        story_parts: List of story parts/sections
        model_type: The model type to use (lite, standard, pro, max)
        cost_centre_id: Optional cost centre ID for tracking

    Returns:
        JSON string containing the image prompts and usage information
    """

    async def _generate_prompts():
        request = GenerateImagePromptsRequest(
            story=story,
            story_parts=story_parts,
            model_type=model_type,
            cost_centre_id=cost_centre_id,
        )
        response = await llm_service.generate_image_prompts(request)
        return json.dumps(
            {
                "image_prompts": response.data,
                "usage": {
                    "embedding_tokens": response.usage.embedding_tokens,
                    "generation_prompt_tokens": response.usage.generation_prompt_tokens,
                    "generation_completion_tokens": response.usage.generation_completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                    "cost": response.usage.cost,
                },
            },
            indent=2,
        )

    return asyncio.run(_generate_prompts())


@mcp.tool()
def generate_image(
    prompt: str, width: int = 720, height: int = 720, cost_centre_id: str = None
) -> str:
    """Generate an image based on a text prompt.

    Args:
        prompt: The image generation prompt
        width: Image width in pixels
        height: Image height in pixels
        cost_centre_id: Optional cost centre ID for tracking

    Returns:
        JSON string containing base64-encoded image data and content type
    """

    async def _generate_image():
        request = ImageGenerationRequest(
            prompt=prompt, width=width, height=height, cost_centre_id=cost_centre_id
        )
        response = await image_service.generate_image(request)

        # Convert bytes to base64 for JSON serialization
        image_data_b64 = base64.b64encode(response.data).decode("utf-8")

        return json.dumps(
            {
                "image_data": image_data_b64,
                "content_type": response.content_type,
                "note": "Image data is base64 encoded. Decode to get the actual image bytes.",
            },
            indent=2,
        )

    return asyncio.run(_generate_image())


if __name__ == "__main__":
    # Generate the config file dynamically
    print("Generating MCP configuration...")
    write_config_file()

    TRANSPORT = "stdio"
    print(f"Running MCP server with {TRANSPORT} transport...")
    if TRANSPORT == "stdio":
        mcp.run(transport="stdio")
    elif TRANSPORT == "sse":
        mcp.run(transport="sse")
    else:
        raise ValueError(f"Invalid transport: {TRANSPORT}")
