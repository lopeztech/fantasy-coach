variable "account_id" {
  description = "AWS account id these resources belong to (used as an apply guardrail)."
  type        = string
}

variable "region" {
  description = "AWS region (Sydney, to match the GCP Cloud Run region australia-southeast1)."
  type        = string
  default     = "ap-southeast-2"
}

variable "environment" {
  description = "Deployment environment name (tag + naming)."
  type        = string
  default     = "prod"
}

variable "name_prefix" {
  description = "Prefix for resource names."
  type        = string
  default     = "fantasy-coach"
}

variable "github_repo" {
  description = "owner/name of the repo whose GitHub Actions may assume the deploy role via OIDC."
  type        = string
  default     = "lopeztech/fantasy-coach"
}
