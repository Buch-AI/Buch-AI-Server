from google.cloud import bigquery

from app.scripts.gcloud.bigquery import common

dataset_id = "tasks"
table_id = "video_generator"
location = "us-east1"


def get_schema():
    """Define the schema for the video generator tasks table."""
    return [
        bigquery.SchemaField("creation_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("execution_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("updated_at", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField(
            "status", "STRING", mode="REQUIRED"
        ),  # pending, processing, completed, failed
        bigquery.SchemaField("metadata", "JSON", mode="NULLABLE"),
    ]


def main():
    """Main function that handles both creation and updates."""
    common.update_table_schema(dataset_id, table_id, get_schema(), location)


if __name__ == "__main__":
    main()
