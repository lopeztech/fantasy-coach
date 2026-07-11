terraform {
  required_version = ">= 1.6"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Bootstrap uses LOCAL state on purpose: it creates the very bucket the rest
  # of the config stores its state in, so it can't store its own state there.
  # The resulting terraform.tfstate stays LOCAL and gitignored (this repo is
  # public). It rarely changes; if the local file is lost, re-import the six
  # resources rather than committing state.
}
