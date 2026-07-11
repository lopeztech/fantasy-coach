provider "aws" {
  region = var.region

  default_tags {
    tags = {
      project   = "fantasy-coach"
      managedby = "terraform"
      component = "bootstrap"
    }
  }
}

data "aws_caller_identity" "current" {}

# ---------------------------------------------------------------------------
# Remote-state backend: S3 bucket (state) + DynamoDB table (lock)
# ---------------------------------------------------------------------------

resource "aws_s3_bucket" "tf_state" {
  # Account id keeps the name globally unique without leaking anything.
  bucket = "${var.name_prefix}-tf-state-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket_versioning" "tf_state" {
  bucket = aws_s3_bucket.tf_state.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "tf_state" {
  bucket = aws_s3_bucket.tf_state.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "aws:kms"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "tf_state" {
  bucket                  = aws_s3_bucket.tf_state.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_dynamodb_table" "tf_lock" {
  name         = "${var.name_prefix}-tf-lock"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "LockID"

  attribute {
    name = "LockID"
    type = "S"
  }
}

# ---------------------------------------------------------------------------
# GitHub Actions OIDC identity provider (account-level singleton)
# ---------------------------------------------------------------------------
# Lets GitHub Actions mint short-lived AWS credentials by assuming an IAM role
# (see ../iam_github_oidc.tf) instead of storing long-lived access keys.
# The thumbprint list is ignored by AWS for this provider since Aug 2023 (AWS
# validates GitHub's cert chain against its trust store), but the field is
# still required; this is GitHub's documented value.

resource "aws_iam_openid_connect_provider" "github" {
  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"]
}
