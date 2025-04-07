from google.cloud import bigquery

from app.scripts.gcloud.bigquery import common

dataset_id = "users"
table_id = "profiles"
location = "us-east1"


def get_schema():
    """Define the schema for the profiles table."""
    return [
        bigquery.SchemaField("user_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("display_name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("email", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("bio", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("profile_picture_url", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("updated_at", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("is_active", "BOOLEAN", mode="REQUIRED"),
        bigquery.SchemaField("preferences", "JSON", mode="NULLABLE"),
        bigquery.SchemaField(
            "social_links",
            "RECORD",
            mode="REPEATED",
            fields=[
                bigquery.SchemaField("platform", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("url", "STRING", mode="REQUIRED"),
            ],
        ),
    ]


def main():
    """Main function that handles both creation and updates."""
    common.update_table_schema(dataset_id, table_id, get_schema(), location)


if __name__ == "__main__":
    main()
