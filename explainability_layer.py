"""
Phase 3: Rule-Based Explainability Layer
-------------------------------------------
Takes rows the Isolation Forest flagged as anomalous and applies
human-readable IF-THEN rules to explain WHY, mirroring the attack
signatures used in Phase 1's synthetic data generation.

This is a hybrid ML + rule-based design:
    ML (Isolation Forest)  -> WHAT is anomalous (unsupervised, catches
                               patterns we didn't explicitly define)
    Rules (this file)      -> WHY it's anomalous (explainable, maps to
                               known attack signatures for SOC engineers)

Rows flagged by ML but matching NO rule are labeled "Unclassified
anomaly" -- this is a FEATURE, not a gap: it surfaces genuinely novel
patterns for human review, which is exactly what unsupervised
detection is for.
"""

import pandas as pd


# Thresholds mirror (loosely, not identically) the ranges used in
# generate_data.py's inject_anomalies() -- close enough to catch real
# instances, general enough to not be "cheating" by hardcoding exact values.
THRESHOLDS = {
    "cpu_high": 85,
    "network_high": 400,
    "failed_login_high": 12,
    "off_hours": {0, 1, 2, 3, 4, 5},
    "api_calls_high": 120,
    "unique_ips_high": 7,
}


def classify_anomaly(row: pd.Series) -> tuple[str, str]:
    """
    Apply rule-based logic to a single flagged row.
    Returns (reason, severity).
    Order matters: more specific / higher-severity rules checked first.
    """
    cpu = row["cpu_utilization"]
    net = row["network_bytes_out"]
    failed_logins = row["failed_login_count"]
    login_hour = row["login_hour"]
    api_calls = row["api_calls_per_min"]
    unique_ips = row["unique_ips_per_user"]

    # Rule 1: Crypto-mining -- sustained high CPU, network stays normal
    if cpu > THRESHOLDS["cpu_high"] and net < THRESHOLDS["network_high"]:
        return (
            f"High CPU ({cpu:.1f}%) with normal network activity -> "
            f"possible crypto-mining or resource-hijacking process",
            "High",
        )

    # Rule 2: Data exfiltration -- network spike, CPU stays normal
    if net > THRESHOLDS["network_high"] and cpu < THRESHOLDS["cpu_high"]:
        return (
            f"Abnormal outbound traffic ({net:.0f} MB) with normal CPU -> "
            f"possible data exfiltration",
            "High",
        )

    # Rule 3: Brute-force login -- many failed logins, off-hours
    if failed_logins > THRESHOLDS["failed_login_high"] and login_hour in THRESHOLDS["off_hours"]:
        return (
            f"{int(failed_logins)} failed logins during off-hours ({int(login_hour)}:00) -> "
            f"possible brute-force login attempt",
            "Medium",
        )

    # Rule 4: Credential stuffing -- API burst + many unique IPs per user
    if api_calls > THRESHOLDS["api_calls_high"] and unique_ips > THRESHOLDS["unique_ips_high"]:
        return (
            f"{api_calls:.0f} API calls/min from {int(unique_ips)} unique IPs -> "
            f"possible credential-stuffing / bot activity",
            "High",
        )

    # Rule 5 (partial match): high failed logins alone, even in business hours
    if failed_logins > THRESHOLDS["failed_login_high"]:
        return (
            f"{int(failed_logins)} failed logins (unusual volume) -> "
            f"possible account compromise attempt",
            "Medium",
        )

    # Fallback: ML flagged it, but no rule matched -- genuinely novel pattern
    return (
        "Flagged by ML model but does not match a known attack signature -> "
        "recommend manual review (potential zero-day pattern)",
        "Low",
    )


def apply_explainability_layer(results_path: str = "isolation_forest_results.csv") -> pd.DataFrame:
    df = pd.read_csv(results_path)

    flagged = df[df["predicted_label"] == 1].copy()

    reasons, severities = [], []
    for _, row in flagged.iterrows():
        reason, severity = classify_anomaly(row)
        reasons.append(reason)
        severities.append(severity)

    flagged["explanation"] = reasons
    flagged["severity"] = severities

    return flagged


if __name__ == "__main__":
    explained = apply_explainability_layer()

    print(f"Total ML-flagged anomalies: {len(explained)}\n")

    print("=== Severity breakdown ===")
    print(explained["severity"].value_counts())

    print("\n=== Rule-match rate (vs 'Unclassified') ===")
    unclassified = (explained["severity"] == "Low").sum()
    print(f"Matched a known signature: {len(explained) - unclassified} / {len(explained)}")
    print(f"Unclassified (novel pattern flagged for review): {unclassified}")

    print("\n=== Sample explained alerts ===")
    sample = explained[["anomaly_type", "severity", "explanation"]].head(8)
    for _, row in sample.iterrows():
        print(f"[{row['severity']:>6}] (true type: {row['anomaly_type']:<18}) {row['explanation']}")

    explained.to_csv("explained_alerts.csv", index=False)
    print("\nSaved to explained_alerts.csv")
