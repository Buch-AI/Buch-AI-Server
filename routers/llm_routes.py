import logging
import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, StreamingResponse
from huggingface_hub import AsyncInferenceClient, InferenceClient
from pydantic import BaseModel

HF_API_KEY = os.getenv("HF_API_KEY")
HF_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


class StoryRequest(BaseModel):
    prompt: str


# Create a router for calls to the LLM API
llm_router = APIRouter()


@llm_router.post("/generate_story_string")
async def generate_story_string(request: StoryRequest) -> Response:
    client = InferenceClient(
        base_url="https://api-inference.huggingface.co", token=HF_API_KEY
    )

    messages = [
        {"role": "system", "content": "You are a creative story writer."},
        {
            "role": "user",
            "content": f"Write a story based on this prompt: {request.prompt}",
        },
    ]

    try:
        response = client.chat.completions.create(
            model=HF_MODEL_ID,
            messages=messages,
            max_tokens=300,
            temperature=0.8,
            top_p=0.9,
        )
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@llm_router.post("/generate_story_stream")
async def generate_story_stream(request: StoryRequest) -> StreamingResponse:
    async_client = AsyncInferenceClient(
        base_url="https://api-inference.huggingface.co", token=HF_API_KEY
    )

    messages = [
        {"role": "system", "content": "You are a creative story writer."},
        {
            "role": "user",
            "content": f"Write a story based on this prompt: {request.prompt}",
        },
    ]

    async def _generate_story_stream():
        try:
            # Use the async client to create a streaming response
            stream = await async_client.chat.completions.create(
                model=HF_MODEL_ID,
                messages=messages,
                stream=True,
                max_tokens=300,
                temperature=0.8,
                top_p=0.9,
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
