from google.cloud import bigquery
from scripts.gcloud.bigquery import common

dataset_id = "users"
table_id = "auth"
location = "us-east1"


def get_schema():
    """Define the schema for the auth table."""
    return [
        bigquery.SchemaField("user_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("username", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("email", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("password_hash", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("last_login", "TIMESTAMP", mode="NULLABLE"),
        bigquery.SchemaField("is_active", "BOOLEAN", mode="REQUIRED"),
        bigquery.SchemaField("roles", "STRING", mode="REPEATED"),
    ]


def main():
    """Main function that handles both creation and updates."""
    common.update_table_schema(dataset_id, table_id, get_schema(), location)


if __name__ == "__main__":
    main()
