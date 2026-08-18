"""
Phase 2: Isolation Forest Anomaly Detection
---------------------------------------------
Trains an UNSUPERVISED Isolation Forest on cloud telemetry features.
Ground-truth labels (is_anomaly, anomaly_type) are stripped before
training and used ONLY afterward for evaluation -- the model never
sees them during fitting. This mirrors real deployment, where you
don't have labeled attack examples in advance.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

# -----------------------------
# 1. Load data
# -----------------------------
df = pd.read_csv("cloud_telemetry_synthetic.csv", parse_dates=["timestamp"])

FEATURE_COLS = [
    "cpu_utilization",
    "network_bytes_out",
    "failed_login_count",
    "api_calls_per_min",
    "login_hour",
    "unique_ips_per_user",
]

X = df[FEATURE_COLS]
y = df["is_anomaly"]  # kept aside ONLY for evaluation, never passed to .fit()

# -----------------------------
# 2. Train/test split
# -----------------------------
# stratify=y ensures both splits keep the same ~5% anomaly ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
print(f"Train anomaly rate: {y_train.mean():.2%} | Test anomaly rate: {y_test.mean():.2%}\n")

# -----------------------------
# 3. Train Isolation Forest
# -----------------------------
# NOTE: y_train is NOT passed here -- .fit(X_train) only.
# This is the crux of "unsupervised" -- worth stating explicitly in any demo.
model = IsolationForest(
    n_estimators=100,
    contamination=0.05,   # expected anomaly proportion, matches our generator
    max_samples="auto",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train)

# -----------------------------
# 4. Predict on test set
# -----------------------------
raw_preds = model.predict(X_test)          # sklearn outputs -1 (anomaly) / 1 (normal)
y_pred = np.where(raw_preds == -1, 1, 0)   # convert to match our label convention (1=anomaly)

# Anomaly score: lower (more negative) = more anomalous.
# Useful later for ranking alerts by severity in the dashboard.
anomaly_scores = model.decision_function(X_test)

# -----------------------------
# 5. Evaluate against ground truth (evaluation ONLY, never used in training)
# -----------------------------
print("=== Confusion Matrix ===")
print("            Predicted Normal | Predicted Anomaly")
cm = confusion_matrix(y_test, y_pred)
print(f"Actual Normal      {cm[0][0]:>10}      |  {cm[0][1]:>10}")
print(f"Actual Anomaly     {cm[1][0]:>10}      |  {cm[1][1]:>10}")

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=["normal", "anomaly"]))

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Precision: {precision:.3f}  (of flagged anomalies, how many were real)")
print(f"Recall:    {recall:.3f}  (of real anomalies, how many were caught)")
print(f"F1-score:  {f1:.3f}")

# -----------------------------
# 6. Save results for Phase 3 (explainability layer will consume this)
# -----------------------------
results_df = X_test.copy()
results_df["true_label"] = y_test.values
results_df["predicted_label"] = y_pred
results_df["anomaly_score"] = anomaly_scores
results_df["anomaly_type"] = df.loc[X_test.index, "anomaly_type"].values

results_df.to_csv("isolation_forest_results.csv", index=False)
print("\nSaved detailed results to isolation_forest_results.csv")

import joblib
joblib.dump(model, "isolation_forest_model.pkl")
print("Saved trained model to isolation_forest_model.pkl")
