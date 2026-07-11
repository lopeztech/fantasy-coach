output "state_bucket" {
  description = "S3 bucket name for the main config's remote state backend."
  value       = aws_s3_bucket.tf_state.id
}

output "lock_table" {
  description = "DynamoDB table name for state locking."
  value       = aws_dynamodb_table.tf_lock.name
}

output "github_oidc_provider_arn" {
  description = "ARN of the GitHub Actions OIDC provider (consumed by the deploy role's trust policy)."
  value       = aws_iam_openid_connect_provider.github.arn
}
