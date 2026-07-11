# AWS Terraform (`infra/aws/`)

Terraform for the **AWS** footprint of Fantasy Coach. Part of the GCP → AWS
migration tracked in [#292](https://github.com/lopeztech/fantasy-coach/issues/292).

> **Scope:** this directory manages *new AWS* resources only. The **GCP**
> infrastructure still lives in
> [`lopeztech/platform-infra`](https://github.com/lopeztech/platform-infra)
> under `projects/fantasy-coach/` and is retired phase-by-phase as the migration
> cuts over. During the transition, infra intentionally lives in two repos.

## Layout

```
infra/aws/
  bootstrap/          # one-time, LOCAL state — creates the remote-state backend
                      # + the GitHub OIDC provider (chicken-and-egg breaker)
  versions.tf         # providers + S3 remote backend (uses what bootstrap created)
  providers.tf        # aws provider: region + default tags
  variables.tf        # account id, region, github repo, name prefix
  ecr.tf              # container image registry (api)
  iam_github_oidc.tf  # IAM role GitHub Actions assumes (keyless, via OIDC)
  outputs.tf
  environments/
    prod.tfvars       # per-env variable values
```

## Bootstrap → main sequence (first-time only)

The main config stores state in S3, but that bucket doesn't exist yet — the
classic chicken-and-egg. `bootstrap/` breaks it with a small **local-state**
config that creates the state bucket, the DynamoDB lock table, and the
account-level GitHub OIDC provider.

```bash
# 0. Authenticate to AWS (SSO or access keys) — `aws sts get-caller-identity`
#    must succeed first.

# 1. Bootstrap the backend + OIDC provider (local state, committed).
cd infra/aws/bootstrap
terraform init
terraform apply -var-file=../environments/prod.tfvars

# 2. Point the main config at the bucket bootstrap just made, then init/apply.
cd ..
terraform init \
  -backend-config="bucket=$(terraform -chdir=bootstrap output -raw state_bucket)" \
  -backend-config="key=fantasy-coach/prod/terraform.tfstate" \
  -backend-config="region=ap-southeast-2" \
  -backend-config="dynamodb_table=$(terraform -chdir=bootstrap output -raw lock_table)"
terraform apply -var-file=environments/prod.tfvars
```

After the first bootstrap, day-to-day work is just `terraform` in `infra/aws/`;
you don't touch `bootstrap/` again unless the backend or OIDC provider changes.

> The DynamoDB lock table is the classic locking mechanism. Terraform ≥1.10
> also supports S3-native locking (`use_lockfile = true`), which removes the
> table — noted as a future simplification once we're on a new enough CLI.

## CI / auth model

GitHub Actions authenticates to AWS **keyless** via OIDC (mirrors the existing
GCP WIF pattern in `.github/workflows/deploy.yml`): the workflow requests an
OIDC token (`permissions: id-token: write`) and assumes the role defined in
`iam_github_oidc.tf`, whose trust policy is scoped to this repo. No long-lived
access keys in GitHub secrets.

## Not applied yet

This is the Phase 0 skeleton. Applying is gated on:
- an AWS account + working local credentials (`aws sts get-caller-identity`),
- the account id + region wired into `environments/prod.tfvars`.
