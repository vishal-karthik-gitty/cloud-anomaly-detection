# -----------------------------------------------------------------------
# This project deliberately uses Terraform DATA SOURCES, not resources.
#
# WHY: the IAM user used for this project has intentionally scoped
# READ-ONLY permissions (CloudWatch, CloudTrail, EC2 describe) -- no
# create/modify/delete rights. This matches least-privilege security
# practice for a monitoring/detection tool, which should never need
# write access to the infrastructure it's watching.
#
# Rather than have Terraform provision new infrastructure (which this
# IAM user couldn't do anyway, and which isn't needed since the EC2
# instance already exists), this configuration uses Terraform to
# formally DOCUMENT and VALIDATE the existing infrastructure as code --
# a legitimate, common IaC pattern for environments Terraform didn't
# originally create.
# -----------------------------------------------------------------------

# Who Terraform is currently authenticated as -- also gives us the IAM
# user's ARN without needing iam:GetUser permission (which this
# intentionally read-scoped user doesn't have).
data "aws_caller_identity" "current" {}

# The real EC2 instance being monitored by the anomaly detection system
data "aws_instance" "monitored_instance" {
  instance_id = var.instance_id
}
