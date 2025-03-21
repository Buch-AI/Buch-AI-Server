from fastapi import APIRouter, HTTPException
from google.cloud import bigquery
from pydantic import BaseModel


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
