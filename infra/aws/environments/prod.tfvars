# Fill in once the AWS account exists. `account_id` has no default on purpose —
# it's the apply guardrail, so a missing value fails loudly rather than
# targeting the wrong account.
account_id  = "" # TODO: 12-digit AWS account id
region      = "ap-southeast-2"
environment = "prod"
github_repo = "lopeztech/fantasy-coach"
