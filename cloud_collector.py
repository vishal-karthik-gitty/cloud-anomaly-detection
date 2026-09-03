"""
Real AWS Cloud Telemetry Collector
--------------------------------------
Replaces the synthetic live_simulator.py with REAL data pulled from
your own AWS account via boto3, PLUS real SSH auth log data pulled
directly from the EC2 instance via paramiko.

Data sources:
    cpu_utilization      -> CloudWatch: AWS/EC2 CPUUtilization metric
    network_bytes_out    -> CloudWatch: AWS/EC2 NetworkOut metric
    api_calls_per_min    -> CloudTrail: count of management events in window
    unique_ips_per_user  -> CloudTrail: distinct sourceIPAddress values in window
    failed_login_count   -> SSH into the instance, count "Failed password" /
                             "Failed publickey" lines in /var/log/auth.log
                             within the polling window (real OS-level data)
    login_hour           -> current server hour

Requires:
    - AWS credentials already configured via `aws configure`
    - SSH key (.pem) for the EC2 instance, readable by this script
"""

import argparse
import time
from datetime import datetime, timedelta, timezone

import boto3
import paramiko
import requests


def get_cpu_and_network(cw_client, instance_id: str, window_minutes: int = 5):
    """Pull latest CPUUtilization and NetworkOut averages from CloudWatch."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=window_minutes)

    def fetch(metric_name, stat="Average"):
        resp = cw_client.get_metric_statistics(
            Namespace="AWS/EC2",
            MetricName=metric_name,
            Dimensions=[{"Name": "InstanceId", "Value": instance_id}],
            StartTime=start,
            EndTime=end,
            Period=300,
            Statistics=[stat],
        )
        points = resp.get("Datapoints", [])
        if not points:
            return 0.0
        latest = sorted(points, key=lambda p: p["Timestamp"])[-1]
        return latest[stat]

    cpu = fetch("CPUUtilization")
    network_bytes = fetch("NetworkOut")
    network_mb = network_bytes / (1024 * 1024)

    return round(cpu, 2), round(network_mb, 2)


def get_api_activity(ct_client, window_minutes: int = 5):
    """Pull recent CloudTrail events to approximate API call rate and unique source IPs."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=window_minutes)

    resp = ct_client.lookup_events(StartTime=start, EndTime=end, MaxResults=50)
    events = resp.get("Events", [])

    call_count = len(events)
    api_calls_per_min = round(call_count / max(window_minutes, 1), 2)

    unique_ips = set()
    for ev in events:
        raw = ev.get("CloudTrailEvent", "")
        if '"sourceIPAddress":"' in raw:
            ip = raw.split('"sourceIPAddress":"')[1].split('"')[0]
            unique_ips.add(ip)

    return api_calls_per_min, max(1, len(unique_ips))


def get_failed_logins(ssh_host: str, ssh_user: str, ssh_key_path: str, window_minutes: int = 5) -> int:
    """
    SSH into the instance and count failed SSH auth attempts in the last
    `window_minutes`, by grepping /var/log/auth.log. Returns 0 on any
    connection failure (fails safe -- never blocks the pipeline).
    """
    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(hostname=ssh_host, username=ssh_user, key_filename=ssh_key_path, timeout=8)

        # journalctl is more reliable than grepping auth.log directly on modern
        # Ubuntu (auth.log may be rotated/compressed); fall back to auth.log.
        cmd = (
            f"journalctl -u ssh --since '-{window_minutes}min' 2>/dev/null "
            f"| grep -Ec 'Failed password|Failed publickey|Connection closed by authenticating user|AuthorizedKeysCommand.*failed' "
            f"|| grep -Ec 'Failed password|Failed publickey|Connection closed by authenticating user|AuthorizedKeysCommand.*failed' /var/log/auth.log 2>/dev/null "
            f"|| echo 0"
        )
        stdin, stdout, stderr = client.exec_command(cmd, timeout=8)
        result = stdout.read().decode().strip()
        client.close()

        return int(result) if result.isdigit() else 0
    except Exception as e:
        print(f"  (auth log check skipped: {e})")
        return 0


def build_snapshot(cw_client, ct_client, instance_id: str, ssh_host: str, ssh_user: str, ssh_key_path: str):
    cpu, network_mb = get_cpu_and_network(cw_client, instance_id)
    api_calls, unique_ips = get_api_activity(ct_client)
    failed_logins = get_failed_logins(ssh_host, ssh_user, ssh_key_path)

    return {
        "cpu_utilization": cpu,
        "network_bytes_out": network_mb,
        "failed_login_count": failed_logins,
        "api_calls_per_min": api_calls,
        "login_hour": datetime.now().hour,
        "unique_ips_per_user": unique_ips,
    }


def run(url: str, instance_id: str, region: str, interval: float, ssh_host: str, ssh_user: str, ssh_key_path: str):
    session = boto3.Session(region_name=region)
    cw_client = session.client("cloudwatch")
    ct_client = session.client("cloudtrail")

    print(f"Real AWS collector started -> instance={instance_id} region={region}")
    print(f"SSH auth-log source: {ssh_user}@{ssh_host}")
    print(f"Streaming to {url}/predict every {interval}s. Press Ctrl+C to stop.\n")

    count = 0
    while True:
        count += 1
        try:
            snapshot = build_snapshot(cw_client, ct_client, instance_id, ssh_host, ssh_user, ssh_key_path)

            resp = requests.post(f"{url}/predict", params={"source": "aws_live"}, json=snapshot, timeout=10)
            resp.raise_for_status()
            result = resp.json()

            tag = result["severity"].upper() if result["is_anomaly"] else "normal"
            marker = "\U0001F534" if result["is_anomaly"] else "\U0001F7E2"
            print(f"[{count:04d}] {marker} cpu={snapshot['cpu_utilization']:>5.1f}% "
                  f"net={snapshot['network_bytes_out']:>7.2f}MB "
                  f"api={snapshot['api_calls_per_min']:>5.1f}/min "
                  f"logins={snapshot['failed_login_count']:>3} "
                  f"-> {tag:<8} score={result['anomaly_score']}")

        except Exception as e:
            print(f"[{count:04d}] ERROR: {e}")

        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real AWS telemetry collector")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the FastAPI service")
    parser.add_argument("--instance-id", required=True, help="EC2 instance ID to monitor")
    parser.add_argument("--region", default="ap-south-1", help="AWS region")
    parser.add_argument("--interval", type=float, default=30.0, help="Seconds between pulls")
    parser.add_argument("--ssh-host", required=True, help="Public IP of the EC2 instance")
    parser.add_argument("--ssh-user", default="ubuntu", help="SSH username (ubuntu / ec2-user)")
    parser.add_argument("--ssh-key", required=True, help="Path to the .pem SSH key")
    args = parser.parse_args()

    try:
        run(args.url, args.instance_id, args.region, args.interval, args.ssh_host, args.ssh_user, args.ssh_key)
    except KeyboardInterrupt:
        print("\nCollector stopped.")
