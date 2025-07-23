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
          value = var.buchai_env
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

# Import Firestore configuration
# Note: BigQuery resources have been replaced with Firestore
# See firestore.tf for the new database configuration

# Google Cloud Firestore (see firestore.tf)
# Collections will be created automatically when documents are first written

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

resource "google_bigquery_table" "users_geolocation" {
  dataset_id  = google_bigquery_dataset.users.dataset_id
  table_id    = "geolocation"
  description = "Table for storing user geolocation data"

  schema = jsonencode([
    {
      name = "user_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "time"
      type = "TIMESTAMP"
      mode = "REQUIRED"
    },
    {
      name = "ipv4"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "geolocation"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "coord_lat"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "coord_lon"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "country_code"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "is_vpn"
      type = "BOOLEAN"
      mode = "NULLABLE"
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

# Payments Dataset and Tables

resource "google_bigquery_dataset" "payments" {
  dataset_id  = "payments"
  location    = "us-east1"
  description = "Dataset for storing payment and transaction data"
}

resource "google_bigquery_table" "payments_records" {
  dataset_id  = google_bigquery_dataset.payments.dataset_id
  table_id    = "records"
  description = "Table for storing payment transaction records"

  schema = jsonencode([
    {
      name = "payment_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "user_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "stripe_payment_intent_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "amount"
      type = "INT64"
      mode = "REQUIRED"
    },
    {
      name = "currency"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "status"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "product_type"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "product_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "quantity"
      type = "INT64"
      mode = "REQUIRED"
    },
    {
      name = "description"
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
      name = "completed_at"
      type = "TIMESTAMP"
      mode = "NULLABLE"
    }
  ])
}

# User Credits and Subscriptions Tables

resource "google_bigquery_table" "users_credits" {
  dataset_id  = google_bigquery_dataset.users.dataset_id
  table_id    = "credits"
  description = "Table for storing user credit balances"

  schema = jsonencode([
    {
      name = "user_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "balance"
      type = "INT64"
      mode = "REQUIRED"
    },
    {
      name = "total_earned"
      type = "INT64"
      mode = "REQUIRED"
    },
    {
      name = "total_spent"
      type = "INT64"
      mode = "REQUIRED"
    },
    {
      name = "last_updated"
      type = "TIMESTAMP"
      mode = "REQUIRED"
    },
    {
      name = "created_at"
      type = "TIMESTAMP"
      mode = "REQUIRED"
    }
  ])
}

resource "google_bigquery_table" "users_subscriptions" {
  dataset_id  = google_bigquery_dataset.users.dataset_id
  table_id    = "subscriptions"
  description = "Table for storing user subscription data"

  schema = jsonencode([
    {
      name = "subscription_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "user_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "stripe_subscription_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "plan_name"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "status"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "credits_monthly"
      type = "INT64"
      mode = "REQUIRED"
    },
    {
      name = "current_period_start"
      type = "TIMESTAMP"
      mode = "REQUIRED"
    },
    {
      name = "current_period_end"
      type = "TIMESTAMP"
      mode = "REQUIRED"
    },
    {
      name = "cancel_at_period_end"
      type = "BOOLEAN"
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
    }
  ])
}

# Credits Dataset and Tables

resource "google_bigquery_dataset" "credits" {
  dataset_id  = "credits"
  location    = "us-east1"
  description = "Dataset for storing credit transaction data"
}

resource "google_bigquery_table" "credits_transactions" {
  dataset_id  = google_bigquery_dataset.credits.dataset_id
  table_id    = "transactions"
  description = "Table for storing credit transaction records"

  schema = jsonencode([
    {
      name = "transaction_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "user_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "type"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "amount"
      type = "INT64"
      mode = "REQUIRED"
    },
    {
      name = "description"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "reference_id"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "created_at"
      type = "TIMESTAMP"
      mode = "REQUIRED"
    }
  ])
}
