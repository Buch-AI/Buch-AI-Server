import asyncio
import base64
import json
import logging
from typing import List, Optional

from mcp.server.fastmcp import FastMCP

from app.mcp.config_generator import write_config_file
from app.services.image.common import ImageGenerationRequest
from app.services.image.pollinations_ai import PollinationsAiRouterService
from app.services.llm.common import (
    GenerateImagePromptsRequest,
    GenerateStoryRequest,
    SummariseStoryRequest,
)
from app.services.llm.vertex_ai import VertexAiRouterService

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize services
llm_service = VertexAiRouterService()
image_service = PollinationsAiRouterService()

# Create FastMCP server
mcp = FastMCP("buch-ai-mcp-server", host="127.0.0.1", port=8050)


@mcp.tool()
def generate_story(
    prompt: str, model_type: str = "lite", cost_centre_id: Optional[str] = None
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

        # Restructure the response to match the expected format
        return json.dumps(
            {"story": response.text, "usage": response.usage.model_dump()}, indent=2
        )

    return asyncio.run(_generate())


@mcp.tool()
def split_story(
    prompt: str, model_type: str = "lite", cost_centre_id: Optional[str] = None
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

        # Restructure the response to match the expected format
        return json.dumps(
            {"story_parts": response.data, "usage": response.usage.model_dump()},
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

        # Restructure the response to match the expected format
        return json.dumps(
            {"summary": response.text, "usage": response.usage.model_dump()}, indent=2
        )

    return asyncio.run(_summarise())


@mcp.tool()
def generate_image_prompts(
    story: str,
    story_parts: List[str],
    model_type: str = "lite",
    cost_centre_id: Optional[str] = None,
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

        # Restructure the response to match the expected format
        return json.dumps(
            {"image_prompts": response.data, "usage": response.usage.model_dump()},
            indent=2,
        )

    return asyncio.run(_generate_prompts())


@mcp.tool()
def generate_image(
    prompt: str,
    width: int = 720,
    height: int = 720,
    cost_centre_id: Optional[str] = None,
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

        # Handle the bytes field by converting to base64
        response_dict = response.model_dump()
        response_dict["image_data"] = base64.b64encode(response_dict["data"]).decode(
            "utf-8"
        )
        response_dict["note"] = (
            "Image data is base64 encoded. Decode to get the actual image bytes."
        )

        # Remove the original bytes field
        del response_dict["data"]

        return json.dumps(response_dict, indent=2)

    return asyncio.run(_generate_image())


@mcp.tool()
def generate_video(
    creation_id: str,
    cost_centre_id: Optional[str] = None,
) -> str:
    """Generate a video for a specific creation.

    Args:
        creation_id: ID of the creation to generate video for
        cost_centre_id: Optional cost centre ID for tracking
    """

    async def _generate_video():
        raise NotImplementedError("Video generation is not implemented yet!")

    return asyncio.run(_generate_video())


if __name__ == "__main__":
    # Generate the config file dynamically
    logger.info("Generating MCP configuration...")
    write_config_file()

    TRANSPORT = "stdio"

    logger.info(f"Running MCP server with {TRANSPORT} transport...")
    if TRANSPORT == "stdio":
        mcp.run(transport="stdio")
    elif TRANSPORT == "sse":
        mcp.run(transport="sse")
    else:
        raise ValueError(f"Invalid transport: {TRANSPORT}")
