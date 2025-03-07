from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import bigquery
from pydantic import BaseModel

app = FastAPI()

# Define the allowed origins
origins = [
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

# Include the router in the main app with a prefix
app.include_router(database_router, prefix="/database")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)