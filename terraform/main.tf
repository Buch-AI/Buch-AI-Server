# Google Cloud Run

resource "google_cloud_run_service" "server" {
  name     = "bai-buchai-p-run-usea1-server"
  location = var.gcp_region

  template {
    metadata {
      annotations = {
        "custom/revision-suffix" = substr(md5(timestamp()), 0, 4)
      }
    }

    spec {
      containers {
        image = var.server_image_tag

        resources {
          limits = {
            memory = "2Gi"
            cpu    = "1000m"
          }
        }

        env {
          name  = "AUTH_JWT_KEY"
          value = var.auth_jwt_key
        }

        env {
          name  = "HF_API_KEY"
          value = var.hf_api_key
        }
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }
}

resource "google_cloud_run_service_iam_member" "server_public" {
  service  = google_cloud_run_service.server.name
  location = google_cloud_run_service.server.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}

resource "google_cloud_run_v2_job" "video_generator" {
  name     = "bai-buchai-p-crj-usea1-vidgen"
  location = var.gcp_region

  template {
    template {
      containers {
        image = var.vidgen_image_tag

        resources {
          limits = {
            memory = "2Gi"
            cpu    = "1000m"
          }
        }
      }
    }
  }

  annotations = {
    "custom/revision-suffix" = substr(md5(timestamp()), 0, 4)
  }
}

# Google Cloud BigQuery

resource "google_bigquery_dataset" "creations" {
  dataset_id  = "creations"
  location    = "us-east1"
  description = "Dataset for storing creation-related data"
}

resource "google_bigquery_table" "creations_cost_centres" {
  dataset_id  = google_bigquery_dataset.creations.dataset_id
  table_id    = "cost_centres"
  description = "Table for storing creation cost centres"

  schema = jsonencode([
    {
      name = "cost_centre_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "creation_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "user_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "created_at"
      type = "TIMESTAMP"
      mode = "REQUIRED"
    },
    {
      name = "cost"
      type = "NUMERIC"
      mode = "REQUIRED"
    }
  ])
}

resource "google_bigquery_table" "creations_profiles" {
  dataset_id  = google_bigquery_dataset.creations.dataset_id
  table_id    = "profiles"
  description = "Table for storing creation profiles"

  schema = jsonencode([
    {
      name = "creation_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "title"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "description"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "creator_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "user_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "created_at"
      type = "TIMESTAMP"
      mode = "REQUIRED"
    },
    {
      name = "updated_at"
      type = "TIMESTAMP"
      mode = "REQUIRED"
    },
    {
      name = "status"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "visibility"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "tags"
      type = "STRING"
      mode = "REPEATED"
    },
    {
      name = "metadata"
      type = "JSON"
      mode = "NULLABLE"
    },
    {
      name = "is_active"
      type = "BOOLEAN"
      mode = "REQUIRED"
    }
  ])
}

resource "google_bigquery_dataset" "tasks" {
  dataset_id  = "tasks"
  location    = "us-east1"
  description = "Dataset for storing background task execution data"
}

resource "google_bigquery_table" "tasks_video_generator" {
  dataset_id  = google_bigquery_dataset.tasks.dataset_id
  table_id    = "video_generator"
  description = "Table for storing Video Generator tasks"

  schema = jsonencode([
    {
      name = "creation_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "execution_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "created_at"
      type = "TIMESTAMP"
      mode = "REQUIRED"
    },
    {
      name = "updated_at"
      type = "TIMESTAMP"
      mode = "REQUIRED"
    },
    {
      name = "status"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "metadata"
      type = "JSON"
      mode = "NULLABLE"
    }
  ])
}

resource "google_bigquery_dataset" "users" {
  dataset_id  = "users"
  location    = "us-east1"
  description = "Dataset for storing user-related data"
}

resource "google_bigquery_table" "users_auth" {
  dataset_id  = google_bigquery_dataset.users.dataset_id
  table_id    = "auth"
  description = "Table for storing user authentication data"

  schema = jsonencode([
    {
      name = "user_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "username"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "email"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "password_hash"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "created_at"
      type = "TIMESTAMP"
      mode = "REQUIRED"
    },
    {
      name = "last_login"
      type = "TIMESTAMP"
      mode = "NULLABLE"
    },
    {
      name = "is_active"
      type = "BOOLEAN"
      mode = "REQUIRED"
    },
    {
      name = "roles"
      type = "STRING"
      mode = "REPEATED"
    }
  ])
}

resource "google_bigquery_table" "users_profiles" {
  dataset_id  = google_bigquery_dataset.users.dataset_id
  table_id    = "profiles"
  description = "Table for storing user profile data"

  schema = jsonencode([
    {
      name = "user_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "display_name"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "email"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "bio"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "profile_picture_url"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "created_at"
      type = "TIMESTAMP"
      mode = "REQUIRED"
    },
    {
      name = "updated_at"
      type = "TIMESTAMP"
      mode = "REQUIRED"
    },
    {
      name = "is_active"
      type = "BOOLEAN"
      mode = "REQUIRED"
    },
    {
      name = "preferences"
      type = "JSON"
      mode = "NULLABLE"
    },
    {
      name = "social_links"
      type = "RECORD"
      mode = "REPEATED"
      fields = [
        {
          name = "platform"
          type = "STRING"
          mode = "REQUIRED"
        },
        {
          name = "url"
          type = "STRING"
          mode = "REQUIRED"
        }
      ]
    }
  ])
}

# Google Cloud Storage

resource "google_storage_bucket" "creations" {
  name     = "bai-buchai-p-stb-usea1-creations"
  location = "us-east1"

  storage_class               = "STANDARD"
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
}
