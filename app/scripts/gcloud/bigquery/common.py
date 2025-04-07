import logging
from typing import List, Optional

from google.api_core import exceptions
from google.cloud import bigquery

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler with formatting if no handlers exist
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def create_dataset(dataset_id: str, location: str = "us-east1") -> str:
    """Create a BigQuery dataset if it doesn't exist.

    Args:
        dataset_id: The ID of the dataset to create
        location: The location for the dataset (default: us-east1)

    Returns:
        The ID of the created/existing dataset
    """
    client = bigquery.Client()
    dataset_ref = client.dataset(dataset_id)
    dataset = bigquery.Dataset(dataset_ref)
    dataset.location = location
    dataset = client.create_dataset(dataset, exists_ok=True)
    logger.info("Dataset %s created", dataset.dataset_id)
    return dataset.dataset_id


def create_table(
    dataset_id: str, table_id: str, schema: List[bigquery.SchemaField]
) -> None:
    """Create a BigQuery table if it doesn't exist.

    Args:
        dataset_id: The ID of the dataset to create the table in
        table_id: The ID of the table to create
        schema: The schema for the table
    """
    client = bigquery.Client()
    table_ref = client.dataset(dataset_id).table(table_id)
    table = bigquery.Table(table_ref, schema=schema)
    table = client.create_table(table, exists_ok=True)
    logger.info("Table %s created", table.table_id)


def update_table_schema(
    dataset_id: str,
    table_id: str,
    schema: List[bigquery.SchemaField],
    location: Optional[str] = None,
) -> None:
    """Update an existing table's schema or create the table if it doesn't exist.

    Args:
        dataset_id: The ID of the dataset containing the table
        table_id: The ID of the table to update
        schema: The new schema to apply
        location: Optional location for dataset creation if needed
    """
    client = bigquery.Client()
    table_ref = client.dataset(dataset_id).table(table_id)

    try:
        # Get the existing table
        table = client.get_table(table_ref)
        logger.info("Current schema has %d fields", len(table.schema))

        # Update the table with new schema
        table.schema = schema

        # Update the table
        updated_table = client.update_table(table, ["schema"])
        logger.info(
            "Schema updated successfully. Now has %d fields", len(updated_table.schema)
        )

        # Log the new schema for verification
        logger.info("New schema fields:")
        for field in updated_table.schema:
            if field.fields:  # Handle nested fields
                logger.info("- %s (%s, %s)", field.name, field.field_type, field.mode)
                for nested_field in field.fields:
                    logger.info(
                        "  - %s (%s, %s)",
                        nested_field.name,
                        nested_field.field_type,
                        nested_field.mode,
                    )
            else:
                logger.info("- %s (%s, %s)", field.name, field.field_type, field.mode)

    except exceptions.NotFound:
        logger.info("Table %s not found. Creating new table...", table_id)
        # Ensure dataset exists before creating table
        if location:
            create_dataset(dataset_id, location)
        create_table(dataset_id, table_id, schema)
    except exceptions.BadRequest as e:
        logger.error("Error updating schema: %s", e)
        if "incompatible" in str(e).lower():
            logger.warning("Some fields might be incompatible with existing data.")
            logger.warning(
                "Consider using a temporary table for data migration if needed."
            )
    except Exception as e:
        logger.error("Unexpected error: %s", e)
