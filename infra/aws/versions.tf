terraform {
  required_version = ">= 1.6"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Remote state in the S3 bucket + DynamoDB lock table created by ./bootstrap.
  # Values are supplied at `terraform init` time via -backend-config (see
  # README) so the bucket name — which embeds the account id — isn't hard-coded.
  backend "s3" {
    key     = "fantasy-coach/prod/terraform.tfstate"
    encrypt = true
  }
}
