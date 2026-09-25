variable "aws_region" {
  description = "AWS region where the anomaly-detection EC2 instance runs"
  type        = string
  default     = "ap-south-1"
}

variable "instance_id" {
  description = "EC2 instance ID being monitored by the anomaly detection system"
  type        = string
  default     = "i-057eb97384bca83d2"
}

variable "iam_username" {
  description = "IAM user used for CloudWatch/CloudTrail read access"
  type        = string
  default     = "secure-file-storage-app"
}
