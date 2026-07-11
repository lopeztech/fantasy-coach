# Keyless deploy identity for GitHub Actions.
#
# Mirrors the GCP Workload Identity Federation setup in
# .github/workflows/deploy.yml: the workflow requests an OIDC token
# (`permissions: id-token: write`) and assumes this role. No static access keys.
#
# The OIDC provider itself is an account-level singleton created in ./bootstrap;
# we look it up here rather than re-declaring it.

data "aws_iam_openid_connect_provider" "github" {
  url = "https://token.actions.githubusercontent.com"
}

data "aws_iam_policy_document" "github_actions_trust" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    effect  = "Allow"

    principals {
      type        = "Federated"
      identifiers = [data.aws_iam_openid_connect_provider.github.arn]
    }

    # Audience must be the AWS STS audience.
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    # Only this repo's `main` branch may assume the role. Widen per phase
    # (e.g. add `repo:${var.github_repo}:pull_request`) when a plan-on-PR
    # workflow is introduced.
    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${var.github_repo}:ref:refs/heads/main"]
    }
  }
}

resource "aws_iam_role" "github_actions_deploy" {
  name                 = "${var.name_prefix}-github-actions-deploy"
  assume_role_policy   = data.aws_iam_policy_document.github_actions_trust.json
  max_session_duration = 3600
  description          = "Assumed by ${var.github_repo} GitHub Actions (OIDC) to push images + deploy."
}

# Phase 0 grant: push/pull to the ECR repo. Additional policies (ECS/Lambda
# deploy, etc.) attach as later phases land — one policy per capability so the
# blast radius stays legible.
data "aws_iam_policy_document" "ecr_push" {
  statement {
    sid       = "EcrAuthToken"
    effect    = "Allow"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"] # GetAuthorizationToken is not resource-scopable
  }

  statement {
    sid    = "EcrPushPull"
    effect = "Allow"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:BatchGetImage",
      "ecr:CompleteLayerUpload",
      "ecr:GetDownloadUrlForLayer",
      "ecr:InitiateLayerUpload",
      "ecr:PutImage",
      "ecr:UploadLayerPart",
    ]
    resources = [aws_ecr_repository.api.arn]
  }
}

resource "aws_iam_role_policy" "ecr_push" {
  name   = "ecr-push"
  role   = aws_iam_role.github_actions_deploy.id
  policy = data.aws_iam_policy_document.ecr_push.json
}
