output "account_id" {
  description = "AWS account ID Terraform is operating against"
  value       = data.aws_caller_identity.current.account_id
}

output "monitored_instance_id" {
  value = data.aws_instance.monitored_instance.id
}

output "monitored_instance_public_ip" {
  value = data.aws_instance.monitored_instance.public_ip
}

output "monitored_instance_state" {
  value = data.aws_instance.monitored_instance.instance_state
}

output "monitored_instance_type" {
  value = data.aws_instance.monitored_instance.instance_type
}

output "collector_iam_user_arn" {
  description = "IAM identity used by cloud_collector.py to pull CloudWatch/CloudTrail data"
  value       = data.aws_caller_identity.current.arn
}
