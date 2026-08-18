"""
Visualize synthetic cloud telemetry with anomalies highlighted.
Run AFTER generate_data.py has produced cloud_telemetry_synthetic.csv
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("cloud_telemetry_synthetic.csv", parse_dates=["timestamp"])

# Use first 2000 rows for readability (full 10k is too dense to visually inspect)
plot_df = df.iloc[:2000]
normal = plot_df[plot_df["is_anomaly"] == 0]
anomaly = plot_df[plot_df["is_anomaly"] == 1]

fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

features = [
    ("cpu_utilization", "CPU Utilization (%)"),
    ("network_bytes_out", "Network Out (MB)"),
    ("failed_login_count", "Failed Login Count"),
    ("api_calls_per_min", "API Calls / min"),
]

for ax, (col, label) in zip(axes, features):
    ax.plot(normal["timestamp"], normal[col], color="steelblue", lw=0.8, label="Normal")
    ax.scatter(anomaly["timestamp"], anomaly[col], color="red", s=25, zorder=5, label="Anomaly")
    ax.set_ylabel(label)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

axes[0].set_title("Synthetic Cloud Telemetry — Normal vs Injected Anomalies (first 2000 samples)")
axes[-1].set_xlabel("Timestamp")
plt.tight_layout()
plt.savefig("telemetry_visualization.png", dpi=150)
print("Saved telemetry_visualization.png")

# Bonus: anomaly type distribution bar chart
fig2, ax2 = plt.subplots(figsize=(8, 5))
df["anomaly_type"].value_counts().plot(kind="bar", ax=ax2, color=[
    "steelblue" if t == "normal" else "crimson" for t in df["anomaly_type"].value_counts().index
])
ax2.set_title("Sample Distribution by Type")
ax2.set_ylabel("Count")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("anomaly_distribution.png", dpi=150)
print("Saved anomaly_distribution.png")
