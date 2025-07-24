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
# NOTE: BigQuery resources have been replaced with Firestore
# See firestore.tf for the new database configuration

# Google Cloud Firestore (see firestore.tf)
# Collections will be created automatically when documents are first written

# Google Cloud Storage

resource "google_storage_bucket" "creations" {
  name     = "bai-buchai-p-stb-usea1-creations"
  location = "us-east1"

  storage_class               = "STANDARD"
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
}
