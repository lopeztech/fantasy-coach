provider "aws" {
  region = var.region

  # Guardrail: refuse to apply against the wrong account.
  allowed_account_ids = [var.account_id]

  default_tags {
    tags = {
      project   = "fantasy-coach"
      env       = var.environment
      managedby = "terraform"
    }
  }
}
