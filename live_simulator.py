"""
Live Cloud Telemetry Simulator
----------------------------------
Stands in for a real monitoring agent (CloudWatch agent, Prometheus
node-exporter, etc). Continuously generates realistic telemetry
snapshots and POSTs them to the running FastAPI service's /predict
endpoint, exactly like a real agent would push live infra metrics.

This demonstrates the actual streaming architecture: the API doesn't
care whether data comes from this script or a real cloud agent -- it's
the same endpoint either way. Swapping this script for a real
CloudWatch/Prometheus collector later requires ZERO changes to the API.

Usage:
    python live_simulator.py                  # default: every 2s
    python live_simulator.py --interval 1      # faster stream
    python live_simulator.py --url http://<host>:8000
"""

import argparse
import random
import time
from datetime import datetime

import requests


def generate_normal_snapshot():
    hour = datetime.now().hour
    business_hour = 1.0 if 9 <= hour <= 18 else 0.0

    return {
        "cpu_utilization": round(max(0, min(100, random.gauss(25 + 15 * business_hour, 6))), 2),
        "network_bytes_out": round(max(0, random.gauss(50 + 30 * business_hour, 15)), 2),
        "failed_login_count": max(0, int(random.expovariate(1 / 0.3))) if random.random() < 0.1 else 0,
        "api_calls_per_min": round(max(0, random.gauss(20 + 25 * business_hour, 8)), 2),
        "login_hour": hour,
        "unique_ips_per_user": max(1, int(random.expovariate(1 / 1.1))),
    }


def generate_anomaly_snapshot():
    attack_type = random.choice(["crypto_mining", "data_exfiltration", "brute_force_login", "credential_stuffing"])
    snap = generate_normal_snapshot()

    if attack_type == "crypto_mining":
        snap["cpu_utilization"] = round(random.uniform(90, 100), 2)
    elif attack_type == "data_exfiltration":
        snap["network_bytes_out"] = round(random.uniform(500, 900), 2)
    elif attack_type == "brute_force_login":
        snap["failed_login_count"] = random.randint(15, 40)
        snap["login_hour"] = random.choice([1, 2, 3, 4])
    elif attack_type == "credential_stuffing":
        snap["api_calls_per_min"] = round(random.uniform(150, 300), 2)
        snap["unique_ips_per_user"] = random.randint(8, 20)

    return snap, attack_type


def run(url: str, interval: float, anomaly_rate: float):
    print(f"Live simulator started -> streaming to {url}/predict every {interval}s")
    print(f"Anomaly injection rate: {anomaly_rate:.0%}")
    print("Press Ctrl+C to stop.\n")

    count = 0
    while True:
        count += 1
        is_injected_anomaly = random.random() < anomaly_rate

        if is_injected_anomaly:
            snapshot, attack_type = generate_anomaly_snapshot()
        else:
            snapshot, attack_type = generate_normal_snapshot(), "normal"

        try:
            resp = requests.post(f"{url}/predict", params={"source": "simulator"}, json=snapshot, timeout=5)
            resp.raise_for_status()
            result = resp.json()

            tag = f"{result['severity'].upper()}" if result["is_anomaly"] else "normal"
            marker = "🔴" if result["is_anomaly"] else "🟢"
            print(f"[{count:04d}] {marker} injected={attack_type:<18} -> detected={tag:<8} score={result['anomaly_score']}")

        except requests.exceptions.RequestException as e:
            print(f"[{count:04d}] ERROR reaching API: {e}")

        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live cloud telemetry simulator")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the FastAPI service")
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between telemetry pushes")
    parser.add_argument("--anomaly-rate", type=float, default=0.15, help="Fraction of pushes that are attacks (higher than real 5%% so demos aren't boring)")
    args = parser.parse_args()

    try:
        run(args.url, args.interval, args.anomaly_rate)
    except KeyboardInterrupt:
        print("\nSimulator stopped.")
