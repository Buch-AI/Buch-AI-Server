from google.cloud import bigquery

dataset_id = "users"
table_id = "auth"
location = "europe-west2"

def create_dataset(dataset_id):
    client = bigquery.Client()
    dataset_ref = client.dataset(dataset_id)
    dataset = bigquery.Dataset(dataset_ref)
    dataset.location = location
    dataset = client.create_dataset(dataset, exists_ok=True)
    print(f"Dataset {dataset.dataset_id} created.")
    return dataset.dataset_id

def create_table(dataset_id, table_id):
    client = bigquery.Client()
    table_ref = client.dataset(dataset_id).table(table_id)
    schema = [
        bigquery.SchemaField("user_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("username", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("email", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("password_hash", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("last_login", "TIMESTAMP", mode="NULLABLE"),
        bigquery.SchemaField("is_active", "BOOLEAN", mode="REQUIRED"),
        bigquery.SchemaField("roles", "STRING", mode="REPEATED"),
    ]
    table = bigquery.Table(table_ref, schema=schema)
    table = client.create_table(table, exists_ok=True)
    print(f"Table {table.table_id} created.")

def main():
    create_dataset(dataset_id)
    create_table(dataset_id, table_id)

if __name__ == "__main__":
    main()
