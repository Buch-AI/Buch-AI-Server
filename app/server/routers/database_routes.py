import logging
from traceback import format_exc
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.firestore_service import get_firestore_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define Pydantic models for requests
class FirestoreQueryRequest(BaseModel):
    collection_name: str
    filters: Optional[List[List]] = None  # List of [field, operator, value]
    order_by: Optional[str] = None
    limit: Optional[int] = None
    offset: Optional[int] = None


class FirestoreDocumentRequest(BaseModel):
    collection_name: str
    document_id: str


# Create a router for database operations
database_router = APIRouter()


@database_router.post("/query")
async def database_query(request: FirestoreQueryRequest):
    """
    Query Firestore collections with filters, ordering, and pagination.
    Replaces the old BigQuery SQL interface with Firestore operations.
    """
    firestore_service = get_firestore_service()

    try:
        # Convert filters from list format to tuples
        filters = None
        if request.filters:
            filters = [tuple(filter_list) for filter_list in request.filters]

        # Query the collection
        results = await firestore_service.query_collection(
            collection_name=request.collection_name,
            filters=filters,
            order_by=request.order_by,
            limit=request.limit,
            offset=request.offset,
        )

        # Convert Pydantic models to dictionaries for response
        rows = []
        for result in results:
            if hasattr(result, "model_dump"):
                rows.append(result.model_dump())
            else:
                rows.append(result)

        return {"data": rows}

    except Exception as e:
        logger.error(f"Firestore query error: {str(e)}\n{format_exc()}")
        raise HTTPException(status_code=400, detail=str(e))


@database_router.get("/document/{collection_name}/{document_id}")
async def get_document(collection_name: str, document_id: str):
    """Get a single document by ID."""
    firestore_service = get_firestore_service()

    try:
        document = await firestore_service.get_document(
            collection_name=collection_name, document_id=document_id
        )

        if document is None:
            raise HTTPException(status_code=404, detail="Document not found")

        if hasattr(document, "model_dump"):
            return {"data": document.model_dump()}
        else:
            return {"data": document}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get document error: {str(e)}\n{format_exc()}")
        raise HTTPException(status_code=400, detail=str(e))
