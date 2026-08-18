"""
Synthetic Cloud Infrastructure Telemetry Generator
----------------------------------------------------
Generates realistic normal cloud metrics + injects labeled anomaly
patterns (labels kept ONLY for evaluation, never shown to the model
during training — Isolation Forest is unsupervised).

Features generated (per-minute granularity):
    cpu_utilization        - %  (crypto-mining / DoS indicator)
    network_bytes_out      - MB (data exfiltration indicator)
    failed_login_count     - count (brute force indicator)
    api_calls_per_min      - count (bot / credential stuffing indicator)
    login_hour             - 0-23 (off-hours access indicator)
    unique_ips_per_user    - count (account takeover indicator)
"""

import numpy as np
import pandas as pd

np.random.seed(42)  # reproducibility — always mention this in interviews


def generate_normal_traffic(n_samples: int) -> pd.DataFrame:
    """
    Simulate NORMAL cloud infra behavior.
    Distributions are chosen to reflect realistic business-hour patterns,
    not just flat random noise -- this is what makes it 'cloud telemetry'
    rather than generic random numbers.
    """
    timestamps = pd.date_range(start="2026-01-01", periods=n_samples, freq="min")

    # Business-hour effect: CPU/API load higher 9am-6pm
    hours = timestamps.hour
    business_hour_mask = ((hours >= 9) & (hours <= 18)).astype(float)

    cpu_utilization = np.random.normal(
        loc=25 + (15 * business_hour_mask), scale=6, size=n_samples
    ).clip(0, 100)

    network_bytes_out = np.random.normal(
        loc=50 + (30 * business_hour_mask), scale=15, size=n_samples
    ).clip(0, None)

    failed_login_count = np.random.poisson(lam=0.3, size=n_samples)

    api_calls_per_min = np.random.normal(
        loc=20 + (25 * business_hour_mask), scale=8, size=n_samples
    ).clip(0, None)

    login_hour = hours

    unique_ips_per_user = np.random.poisson(lam=1.1, size=n_samples).clip(1, None)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "cpu_utilization": cpu_utilization,
        "network_bytes_out": network_bytes_out,
        "failed_login_count": failed_login_count,
        "api_calls_per_min": api_calls_per_min,
        "login_hour": login_hour,
        "unique_ips_per_user": unique_ips_per_user,
        "is_anomaly": 0,          # ground truth label (evaluation ONLY)
        "anomaly_type": "normal", # for explainability layer testing later
    })
    return df


def inject_anomalies(df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    """
    Replace `contamination` fraction of rows with one of 4 attack patterns.
    Each pattern moves MULTIPLE correlated features together, mimicking
    real attack signatures rather than single-feature noise spikes.

    Attack types:
        1. crypto_mining        -> CPU spike sustained + normal network
        2. data_exfiltration    -> network_bytes_out spike + normal CPU
        3. brute_force_login    -> failed_login_count spike + off-hours
        4. credential_stuffing  -> api_calls_per_min spike + many unique_ips
    """
    df = df.copy()
    n_samples = len(df)
    n_anomalies = int(n_samples * contamination)

    anomaly_indices = np.random.choice(n_samples, size=n_anomalies, replace=False)
    # split anomaly indices roughly evenly across the 4 attack types
    splits = np.array_split(anomaly_indices, 4)

    # --- 1. Crypto-mining: CPU pegged high for sustained period ---
    idx = splits[0]
    df.loc[idx, "cpu_utilization"] = np.random.uniform(90, 100, size=len(idx))
    df.loc[idx, "anomaly_type"] = "crypto_mining"

    # --- 2. Data exfiltration: massive outbound network burst ---
    idx = splits[1]
    df.loc[idx, "network_bytes_out"] = np.random.uniform(500, 900, size=len(idx))
    df.loc[idx, "anomaly_type"] = "data_exfiltration"

    # --- 3. Brute-force login: many failed logins, off-hours ---
    idx = splits[2]
    df.loc[idx, "failed_login_count"] = np.random.randint(15, 40, size=len(idx))
    df["login_hour"] = df["login_hour"].astype("int64")
    df.loc[idx, "login_hour"] = np.random.choice([1, 2, 3, 4], size=len(idx))
    df.loc[idx, "anomaly_type"] = "brute_force_login"

    # --- 4. Credential stuffing: API burst from many different IPs ---
    idx = splits[3]
    df.loc[idx, "api_calls_per_min"] = np.random.uniform(150, 300, size=len(idx))
    df.loc[idx, "unique_ips_per_user"] = np.random.randint(8, 20, size=len(idx))
    df.loc[idx, "anomaly_type"] = "credential_stuffing"

    df.loc[anomaly_indices, "is_anomaly"] = 1
    return df


def generate_dataset(n_samples: int = 10000, contamination: float = 0.05) -> pd.DataFrame:
    """Full pipeline: normal baseline + injected anomalies, shuffled by time order kept."""
    df = generate_normal_traffic(n_samples)
    df = inject_anomalies(df, contamination=contamination)
    return df


if __name__ == "__main__":
    df = generate_dataset(n_samples=10000, contamination=0.05)

    print(df.head(10))
    print("\nAnomaly breakdown:")
    print(df["anomaly_type"].value_counts())
    print(f"\nTotal anomaly rate: {df['is_anomaly'].mean():.2%}")

    df.to_csv("cloud_telemetry_synthetic.csv", index=False)
    print("\nSaved to cloud_telemetry_synthetic.csv")
