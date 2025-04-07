from google.cloud import bigquery

from app.scripts.gcloud.bigquery import common

dataset_id = "creations"
table_id = "profiles"
location = "us-east1"


def get_schema():
    """Define the schema for the creations profiles table."""
    return [
        bigquery.SchemaField("creation_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("title", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("description", "STRING", mode="NULLABLE"),
        bigquery.SchemaField(
            "creator_id", "STRING", mode="REQUIRED"
        ),  # Original creator, never changes
        bigquery.SchemaField(
            "user_id", "STRING", mode="REQUIRED"
        ),  # Current owner, can change
        bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("updated_at", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField(
            "status", "STRING", mode="REQUIRED"
        ),  # draft, published, archived
        bigquery.SchemaField(
            "visibility", "STRING", mode="REQUIRED"
        ),  # public, private
        bigquery.SchemaField("tags", "STRING", mode="REPEATED"),
        bigquery.SchemaField("metadata", "JSON", mode="NULLABLE"),
        bigquery.SchemaField("is_active", "BOOLEAN", mode="REQUIRED"),
    ]


def main():
    """Main function that handles both creation and updates."""
    common.update_table_schema(dataset_id, table_id, get_schema(), location)


if __name__ == "__main__":
    main()
