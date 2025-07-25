"""
Firestore Service Layer with Direct API Model Usage

This module provides a simplified service layer for Firestore operations where developers
work directly with API models. Since API models and Firestore models share the same fields
(they inherit from the same base models), we can use API models directly with Firestore data.

Simply pass your API model class and get back instances of that class with Firestore data.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, TypeVar

from firebase_admin import firestore, initialize_app
from google.cloud.firestore import Client, DocumentReference, FieldFilter
from pydantic import BaseModel

from app.models import COLLECTION_MODELS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type variable for generic model operations
T = TypeVar("T", bound=BaseModel)


class FirestoreService:
    """
    Service class for Firestore operations with direct API model usage.

    Since API models and Firestore models share identical fields (both inherit from the same base models),
    we can use API models directly with Firestore data without any conversion overhead.
    """

    def __init__(self):
        """
        Initialize the Firestore service.
        """
        self._client: Optional[Client] = None

    @property
    def client(self) -> Client:
        """Get or create the Firestore client."""
        if self._client is None:
            # Initialize Firebase Admin SDK if not already initialized
            try:
                # Try to get the default app
                app = initialize_app()
            except ValueError:
                # App already exists, get it
                import firebase_admin

                app = firebase_admin.get_app()

            # Get Firestore client (using default database)
            self._client = firestore.client(app)

        return self._client

    def get_collection_ref(self, collection_name: str):
        """Get a reference to a Firestore collection."""
        return self.client.collection(collection_name)

    def get_document_ref(
        self, collection_name: str, document_id: str
    ) -> DocumentReference:
        """Get a reference to a specific document."""
        return self.client.collection(collection_name).document(document_id)

    # Generic CRUD operations
    async def create_document(
        self,
        collection_name: str,
        document_data: Dict[str, Any],
        document_id: Optional[str] = None,
    ) -> str:
        """
        Create a new document in the specified collection.

        Args:
            collection_name: Name of the collection
            document_data: Data to store in the document
            document_id: Optional document ID, will generate UUID if not provided

        Returns:
            The document ID of the created document
        """
        try:
            # Generate document ID if not provided
            if document_id is None:
                document_id = str(uuid.uuid4())

            # Add timestamps
            now = datetime.utcnow()
            document_data.setdefault("created_at", now)
            document_data.setdefault("updated_at", now)

            # Create the document
            doc_ref = self.get_document_ref(collection_name, document_id)
            doc_ref.set(document_data)

            logger.info(f"Created document {document_id} in {collection_name}")
            return document_id

        except Exception as e:
            logger.error(f"Failed to create document in {collection_name}: {str(e)}")
            raise

    async def get_document(
        self,
        collection_name: str,
        document_id: str,
        model_class: Optional[Type[T]] = None,
    ) -> Optional[T]:
        """
        Get a document by ID and return as API model instance.

        Args:
            collection_name: Name of the collection
            document_id: ID of the document to retrieve
            model_class: API model class to instantiate with Firestore data

        Returns:
            Document data as API model instance or None if not found
        """
        try:
            doc_ref = self.get_document_ref(collection_name, document_id)
            doc = doc_ref.get()

            if not doc.exists:
                return None

            data = doc.to_dict()
            data["id"] = doc.id  # Add document ID to data

            if model_class:
                # Create the API model instance directly with Firestore data
                return model_class(**data)

            # Fallback to collection mapping if no model class provided
            if collection_name in COLLECTION_MODELS:
                firestore_model = COLLECTION_MODELS[collection_name]
                return firestore_model(**data)

            return data

        except Exception as e:
            logger.error(
                f"Failed to get document {document_id} from {collection_name}: {str(e)}"
            )
            raise

    async def update_document(
        self, collection_name: str, document_id: str, update_data: Dict[str, Any]
    ) -> bool:
        """
        Update a document.

        Args:
            collection_name: Name of the collection
            document_id: ID of the document to update
            update_data: Data to update

        Returns:
            True if successful
        """
        try:
            # Add update timestamp
            update_data["updated_at"] = datetime.utcnow()

            doc_ref = self.get_document_ref(collection_name, document_id)
            doc_ref.update(update_data)

            logger.info(f"Updated document {document_id} in {collection_name}")
            return True

        except Exception as e:
            logger.error(
                f"Failed to update document {document_id} in {collection_name}: {str(e)}"
            )
            raise

    async def delete_document(self, collection_name: str, document_id: str) -> bool:
        """
        Delete a document.

        Args:
            collection_name: Name of the collection
            document_id: ID of the document to delete

        Returns:
            True if successful
        """
        try:
            doc_ref = self.get_document_ref(collection_name, document_id)
            doc_ref.delete()

            logger.info(f"Deleted document {document_id} from {collection_name}")
            return True

        except Exception as e:
            logger.error(
                f"Failed to delete document {document_id} from {collection_name}: {str(e)}"
            )
            raise

    async def query_collection(
        self,
        collection_name: str,
        filters: Optional[List[tuple]] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        model_class: Optional[Type[T]] = None,
    ) -> List[T]:
        """
        Query a collection and return API model instances.

        Args:
            collection_name: Name of the collection to query
            filters: List of filter tuples (field, operator, value)
            order_by: Field to order by
            limit: Maximum number of results
            offset: Number of results to skip
            model_class: API model class to instantiate with Firestore data

        Returns:
            List of documents as API model instances
        """
        try:
            query = self.get_collection_ref(collection_name)

            # Apply filters using the newer filter syntax to avoid deprecation warnings
            if filters:
                for field, operator, value in filters:
                    query = query.where(filter=FieldFilter(field, operator, value))

            # Apply ordering
            if order_by:
                query = query.order_by(order_by)

            # Apply pagination
            if offset:
                query = query.offset(offset)
            if limit:
                query = query.limit(limit)

            # Execute query
            docs = query.stream()

            # Convert to model instances
            results = []
            for doc in docs:
                data = doc.to_dict()
                data["id"] = doc.id

                if model_class:
                    # Create the API model instance directly with Firestore data
                    results.append(model_class(**data))
                elif collection_name in COLLECTION_MODELS:
                    # Fallback to collection mapping
                    firestore_model = COLLECTION_MODELS[collection_name]
                    results.append(firestore_model(**data))
                else:
                    results.append(data)

            return results

        except Exception as e:
            error_message = str(e)
            if "query requires an index" in error_message:
                logger.error(
                    f"Firestore index required for query on {collection_name}. "
                    f"Create the index using the link in the error message: {error_message}"
                )
                # Re-raise with a more user-friendly message
                raise Exception(
                    "Database query requires an index to be created. "
                    "Please contact system administrator."
                ) from e
            else:
                logger.error(
                    f"Failed to query collection {collection_name}: {error_message}"
                )
                raise

    async def get_user_documents(
        self,
        collection_name: str,
        user_id: str,
        limit: int = 50,
        model_class: Optional[Type[T]] = None,
    ) -> List[T]:
        """
        Get documents belonging to a specific user as API model instances.

        Args:
            collection_name: Name of the collection
            user_id: ID of the user
            limit: Maximum number of results
            model_class: API model class to instantiate with Firestore data

        Returns:
            List of user's documents as API model instances
        """
        return await self.query_collection(
            collection_name=collection_name,
            filters=[("user_id", "==", user_id)],
            order_by="created_at",
            limit=limit,
            model_class=model_class,
        )

    async def count_documents(
        self, collection_name: str, filters: Optional[List[tuple]] = None
    ) -> int:
        """
        Count documents in a collection with optional filters.

        Args:
            collection_name: Name of the collection
            filters: Optional list of filter tuples

        Returns:
            Number of matching documents
        """
        try:
            query = self.get_collection_ref(collection_name)

            # Apply filters using the newer filter syntax to avoid deprecation warnings
            if filters:
                for field, operator, value in filters:
                    query = query.where(filter=FieldFilter(field, operator, value))

            # Execute count query
            return len(list(query.stream()))

        except Exception as e:
            logger.error(f"Failed to count documents in {collection_name}: {str(e)}")
            raise

    # Transaction support
    def transaction(self):
        """Create a Firestore transaction context manager."""
        return self.client.transaction()

    # Batch operations
    def batch(self):
        """Create a Firestore batch operation context manager."""
        return self.client.batch()


# Global service instance
_firestore_service = None


def get_firestore_service() -> FirestoreService:
    """
    Get a singleton Firestore service instance.

    Returns:
        FirestoreService instance
    """
    global _firestore_service
    if _firestore_service is None:
        _firestore_service = FirestoreService()
    return _firestore_service
