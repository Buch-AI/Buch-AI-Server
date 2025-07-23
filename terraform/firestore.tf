# Google Cloud Firestore Database
# Replace BigQuery datasets with a single Firestore database
# Note: Firestore uses collections (not datasets) and documents (not tables)

resource "google_firestore_database" "main" {
  project     = "bai-buchai-p"
  name        = "(default)"
  location_id = "us-east1"
  type        = "FIRESTORE_NATIVE"

  # Enable deletion protection to prevent accidental deletion
  delete_protection_state = "DELETE_PROTECTION_ENABLED"
}

# Optional: Create a separate database for development/testing
resource "google_firestore_database" "development" {
  project     = "bai-buchai-p"
  name        = "development"
  location_id = "us-east1"
  type        = "FIRESTORE_NATIVE"

  delete_protection_state = "DELETE_PROTECTION_DISABLED"
}

# Note: Collections in Firestore are created automatically when documents are added
# The equivalent collections for your BigQuery tables will be:
#
# BigQuery Datasets/Tables -> Firestore Collections:
# - creations.cost_centres -> creations_cost_centres (collection)
# - creations.profiles -> creations_profiles (collection)  
# - tasks.video_generator -> tasks_video_generator (collection)
# - users.auth -> users_auth (collection)
# - users.profiles -> users_profiles (collection)
# - users.geolocation -> users_geolocation (collection)
# - users.credits -> users_credits (collection)
# - users.subscriptions -> users_subscriptions (collection)
# - payments.records -> payments_records (collection)
# - credits.transactions -> credits_transactions (collection)

# Firestore Security Rules (basic structure)
resource "google_firestore_document" "security_rules" {
  project     = "bai-buchai-p"
  database    = google_firestore_database.main.name
  collection  = "_firestore"
  document_id = "rules"

  fields = jsonencode({
    rules = {
      stringValue = file("${path.module}/firestore.rules")
    }
  })
} 