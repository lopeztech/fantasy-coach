output "ecr_repository_url" {
  description = "Push target for the API image (docker push <url>:<sha>)."
  value       = aws_ecr_repository.api.repository_url
}

output "github_actions_role_arn" {
  description = "Role ARN for GitHub Actions to assume via OIDC (set as a repo variable for the deploy workflow)."
  value       = aws_iam_role.github_actions_deploy.arn
}
