terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# Uses the same credentials already configured via `aws configure` --
# no secrets are stored in this repo.
provider "aws" {
  region = var.aws_region
}
