variable "gcp_svc_key_file" {
  type        = string
  description = "Path to the service account key file"
}


variable "gcp_project_id" {
  type        = string
  description = "The ID of the Google Cloud project"
}


variable "gcp_region" {
  type        = string
  description = "The region of the Google Cloud project"
}

variable "server_image_tag" {
  type        = string
  description = "The Docker image tag for the Buch AI Server"
}

variable "vidgen_image_tag" {
  type        = string
  description = "The Docker image tag for the Buch AI Video Generator"
}

variable "auth_jwt_key" {
  type        = string
  description = "The JWT key for authentication"
  sensitive   = true
}

variable "hf_api_key" {
  type        = string
  description = "The Hugging Face API key"
  sensitive   = true
}
