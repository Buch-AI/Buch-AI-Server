import logging
import os

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, StreamingResponse
from huggingface_hub import AsyncInferenceClient, InferenceClient
from pydantic import BaseModel

from models.llm_config import ModelType, get_model_config, get_prompt_messages_story

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
if not HF_API_KEY:
    raise ValueError(
        "HF_API_KEY environment variable is not set. Please set it in your .env file."
    )


class GenerateStoryRequest(BaseModel):
    prompt: str
    model_type: ModelType = ModelType.LITE  # Default to LITE if not specified


# Create a router for calls to the LLM API
llm_router = APIRouter()


@llm_router.post("/generate_story_string")
async def generate_story_string(request: GenerateStoryRequest) -> Response:
    client = InferenceClient(
        base_url="https://api-inference.huggingface.co", token=HF_API_KEY
    )

    try:
        model_config = get_model_config(request.model_type)

        response = client.chat.completions.create(
            messages=get_prompt_messages_story(request.model_type, request.prompt),
            model=model_config.model_id,
            max_tokens=model_config.max_tokens,
            temperature=model_config.temperature,
            top_p=model_config.top_p,
        )
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@llm_router.post("/generate_story_stream")
async def generate_story_stream(request: GenerateStoryRequest) -> StreamingResponse:
    async_client = AsyncInferenceClient(
        base_url="https://api-inference.huggingface.co", token=HF_API_KEY
    )

    async def _generate_story_stream():
        try:
            model_config = get_model_config(request.model_type)

            # Use the async client to create a streaming response
            stream = await async_client.chat.completions.create(
                messages=get_prompt_messages_story(request.model_type, request.prompt),
                model=model_config.model_id,
                stream=True,
                max_tokens=model_config.max_tokens,
                temperature=model_config.temperature,
                top_p=model_config.top_p,
            )
            async for chunk in stream:
                # Log the chunk for debugging
                logging.debug(f"Received chunk: {chunk}")
                if "choices" in chunk and chunk.choices:
                    yield chunk.choices[0].delta.content or ""
                else:
                    logging.warning("Chunk does not contain expected data structure.")
        except Exception as e:
            logging.error(f"Error during streaming: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return StreamingResponse(_generate_story_stream(), media_type="text/plain")
