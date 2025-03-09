import asyncio
import logging
import os

from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from google.cloud import bigquery
from huggingface_hub import AsyncInferenceClient, InferenceClient
from pydantic import BaseModel

load_dotenv()

app = FastAPI()

# Define the allowed origins
origins = [
    "http://localhost:8080",
    "http://localhost:8081",
    # Add other origins as needed
    "https://buch-ai.github.io/Buch-AI-App"
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows requests from these origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Define a Pydantic model for the request
class DatabaseQueryRequest(BaseModel):
    query: str

@app.get("/", tags=["root"])
def root():
    return {"message": "success"}

# Create a router for database operations
database_router = APIRouter()

@database_router.post("/query")
async def database_query(request: DatabaseQueryRequest):
    client = bigquery.Client()
    
    try:
        query_job = client.query(request.query)
        results = query_job.result()  # Waits for job to complete.
        
        # Convert results to a list of dictionaries
        rows = [dict(row) for row in results]
        return {"data": rows}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Create a router for calls to the LLM API
llm_router = APIRouter()

HF_API_KEY = os.getenv('HF_API_KEY')
HF_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

class StoryRequest(BaseModel):
    prompt: str

@llm_router.post("/generate_story_string")
async def generate_story_string(request: StoryRequest) -> Response:
    client = InferenceClient(base_url="https://api-inference.huggingface.co", token=HF_API_KEY)

    messages = [
        {"role": "system", "content": "You are a creative story writer."},
        {"role": "user", "content": f"Write a story based on this prompt: {request.prompt}"},
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
    async_client = AsyncInferenceClient(base_url="https://api-inference.huggingface.co", token=HF_API_KEY)

    messages = [
        {"role": "system", "content": "You are a creative story writer."},
        {"role": "user", "content": f"Write a story based on this prompt: {request.prompt}"},
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
                if 'choices' in chunk and chunk.choices:
                    yield chunk.choices[0].delta.content or ""
                else:
                    logging.warning("Chunk does not contain expected data structure.")
        except Exception as e:
            logging.error(f"Error during streaming: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return StreamingResponse(_generate_story_stream(), media_type="text/plain")

# Include the router in the main app with a prefix
app.include_router(database_router, prefix="/database")
app.include_router(llm_router, prefix="/llm")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)