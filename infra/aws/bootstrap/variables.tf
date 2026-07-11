variable "region" {
  description = "AWS region for the state backend + all resources (Sydney, to match the GCP Cloud Run region)."
  type        = string
  default     = "ap-southeast-2"
}

variable "name_prefix" {
  description = "Prefix for globally- and account-scoped resource names."
  type        = string
  default     = "fantasy-coach"
}

variable "github_org" {
  description = "GitHub org/owner that hosts the repo allowed to assume deploy roles via OIDC."
  type        = string
  default     = "lopeztech"
}
