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

variable "buchai_env" {
  type        = string
  description = "The environment override for the Buch AI Server"
  sensitive   = true
}
