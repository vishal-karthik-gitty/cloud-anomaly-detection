"""
Phase 4: FastAPI Service
---------------------------
Wraps the trained Isolation Forest model + rule-based explainability
layer behind a REST API. This is the "plumbing" that lets any external
system (a monitoring agent, a load generator, Prometheus, etc.) submit
live telemetry and get back a prediction -- turning the project from
a script into a deployable service.

Endpoints:
    POST /predict  -> run one telemetry snapshot through model + rules
    GET  /metrics  -> Prometheus-scrapeable operational metrics
    GET  /health   -> liveness check
"""

import time
import joblib
import pandas as pd
from collections import deque
from datetime import datetime
from fastapi import FastAPI, Query
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response, HTMLResponse

from explainability_layer import classify_anomaly, THRESHOLDS  # reuse Phase 3 logic

# -----------------------------
# Load model once at startup (not per-request -- real perf concern)
# -----------------------------
model = joblib.load("isolation_forest_model.pkl")

FEATURE_COLS = [
    "cpu_utilization",
    "network_bytes_out",
    "failed_login_count",
    "api_calls_per_min",
    "login_hour",
    "unique_ips_per_user",
]

app = FastAPI(
    title="Cloud Infra Anomaly Detection API",
    description="Isolation Forest + rule-based explainability for cloud telemetry",
    version="1.0.0",
)

# -----------------------------
# Prometheus metrics
# -----------------------------
REQUEST_COUNT = Counter("anomaly_api_requests_total", "Total prediction requests")
ANOMALY_COUNT = Counter("anomaly_api_anomalies_detected_total", "Total anomalies flagged", ["severity"])
NORMAL_COUNT = Counter("anomaly_api_normal_total", "Total normal predictions")
REQUEST_LATENCY = Histogram("anomaly_api_request_latency_seconds", "Prediction latency")

# -----------------------------
# In-memory rolling event log (last 50 predictions)
# Powers the live feed on /demo -- in production this would be a
# proper time-series store, but for a demo an in-memory ring buffer
# is enough and keeps the project dependency-light.
# -----------------------------
RECENT_EVENTS = deque(maxlen=50)


# -----------------------------
# Request/response schemas
# -----------------------------
class TelemetrySnapshot(BaseModel):
    cpu_utilization: float = Field(..., ge=0, le=100, example=27.5)
    network_bytes_out: float = Field(..., ge=0, example=45.2)
    failed_login_count: int = Field(..., ge=0, example=0)
    api_calls_per_min: float = Field(..., ge=0, example=22.0)
    login_hour: int = Field(..., ge=0, le=23, example=14)
    unique_ips_per_user: int = Field(..., ge=1, example=1)


class PredictionResponse(BaseModel):
    is_anomaly: bool
    anomaly_score: float
    severity: str
    explanation: str


# -----------------------------
# Endpoints
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(snapshot: TelemetrySnapshot, source: str = Query(default="manual", description="Origin tag: 'manual' or 'simulator'")):
    start = time.time()
    REQUEST_COUNT.inc()

    X = pd.DataFrame([snapshot.model_dump()])[FEATURE_COLS]

    raw_pred = model.predict(X)[0]          # -1 = anomaly, 1 = normal
    score = float(model.decision_function(X)[0])
    is_anomaly = raw_pred == -1

    if is_anomaly:
        row = X.iloc[0]
        explanation, severity = classify_anomaly(row)
        ANOMALY_COUNT.labels(severity=severity).inc()
    else:
        explanation, severity = "No anomaly detected -- metrics within normal range", "None"
        NORMAL_COUNT.inc()

    REQUEST_LATENCY.observe(time.time() - start)

    RECENT_EVENTS.appendleft({
        "time": datetime.now().strftime("%H:%M:%S"),
        "source": source,
        "is_anomaly": bool(is_anomaly),
        "severity": severity,
        "anomaly_score": round(score, 5),
        "explanation": explanation,
        "snapshot": snapshot.model_dump(),
    })

    return PredictionResponse(
        is_anomaly=bool(is_anomaly),
        anomaly_score=round(score, 5),
        severity=severity,
        explanation=explanation,
    )


@app.get("/recent")
def recent(limit: int = 30):
    return list(RECENT_EVENTS)[:limit]


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/demo", response_class=HTMLResponse)
def demo():
    return DEMO_HTML


DEMO_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SENTRY // Cloud Anomaly Console</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0B0F14;
    --panel: #12181F;
    --panel-2: #161D26;
    --border: #232B35;
    --text: #E6EDF3;
    --muted: #6B7785;
    --safe: #4CE0B3;
    --medium: #FFB84C;
    --high: #FF5D5D;
    --none-color: #3A4552;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: 'IBM Plex Sans', sans-serif;
    min-height: 100vh;
    background-image:
      radial-gradient(circle at 20% 0%, rgba(76,224,179,0.06), transparent 40%),
      radial-gradient(circle at 80% 100%, rgba(255,93,93,0.05), transparent 40%);
  }
  .topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 32px;
    border-bottom: 1px solid var(--border);
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 10px;
    font-weight: 700;
    font-size: 18px;
    letter-spacing: 0.04em;
  }
  .brand .dot {
    width: 9px; height: 9px; border-radius: 50%;
    background: var(--safe);
    box-shadow: 0 0 8px var(--safe);
    animation: blink 2.4s ease-in-out infinite;
  }
  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.35} }
  .brand small { color: var(--muted); font-weight: 400; font-size: 12px; margin-left: 6px; }
  .status-pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: var(--muted);
    border: 1px solid var(--border);
    padding: 5px 12px;
    border-radius: 20px;
  }
  .wrap {
    max-width: 1080px;
    margin: 0 auto;
    padding: 40px 32px 60px;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 28px;
  }
  @media (max-width: 860px) { .wrap { grid-template-columns: 1fr; } }

  .panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 24px;
  }
  .panel h2 {
    margin: 0 0 4px;
    font-size: 15px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--muted);
    font-weight: 600;
  }
  .panel .sub { color: var(--muted); font-size: 13px; margin-bottom: 20px; }

  .field { margin-bottom: 16px; }
  .field label {
    display: block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: var(--muted);
    margin-bottom: 6px;
  }
  .field input {
    width: 100%;
    background: var(--panel-2);
    border: 1px solid var(--border);
    color: var(--text);
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    padding: 10px 12px;
    border-radius: 6px;
    outline: none;
    transition: border-color 0.15s;
  }
  .field input:focus { border-color: var(--safe); }

  .presets { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px; }
  .preset-btn {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    background: var(--panel-2);
    border: 1px solid var(--border);
    color: var(--muted);
    padding: 6px 10px;
    border-radius: 5px;
    cursor: pointer;
    transition: all 0.15s;
  }
  .preset-btn:hover { border-color: var(--text); color: var(--text); }

  .submit-btn {
    width: 100%;
    background: var(--safe);
    color: #05130E;
    border: none;
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 600;
    font-size: 14px;
    padding: 13px;
    border-radius: 6px;
    cursor: pointer;
    transition: filter 0.15s;
  }
  .submit-btn:hover { filter: brightness(1.08); }
  .submit-btn:active { filter: brightness(0.95); }

  /* Result panel */
  .result-panel {
    display: flex;
    flex-direction: column;
    border-width: 1px;
    transition: border-color 0.4s ease, box-shadow 0.4s ease;
  }
  .result-panel.state-idle { border-color: var(--border); }
  .result-panel.state-safe { border-color: var(--safe); box-shadow: 0 0 24px -8px var(--safe); }
  .result-panel.state-medium { border-color: var(--medium); box-shadow: 0 0 24px -8px var(--medium); }
  .result-panel.state-high { border-color: var(--high); box-shadow: 0 0 24px -8px var(--high); animation: pulse-high 1.4s ease-in-out infinite; }
  @keyframes pulse-high {
    0%,100% { box-shadow: 0 0 24px -8px var(--high); }
    50% { box-shadow: 0 0 40px -6px var(--high); }
  }

  .verdict-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.08em;
    padding: 6px 14px;
    border-radius: 5px;
    display: inline-block;
    width: fit-content;
    margin-bottom: 16px;
  }
  .verdict-badge.idle { background: var(--panel-2); color: var(--muted); }
  .verdict-badge.safe { background: rgba(76,224,179,0.12); color: var(--safe); }
  .verdict-badge.medium { background: rgba(255,184,76,0.12); color: var(--medium); }
  .verdict-badge.high { background: rgba(255,93,93,0.12); color: var(--high); }

  .score-row {
    display: flex;
    justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    color: var(--muted);
    margin-bottom: 20px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border);
  }
  .score-row span.val { color: var(--text); }

  .explanation-label {
    font-size: 12px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 8px;
  }
  .explanation-text {
    font-size: 14px;
    line-height: 1.55;
    color: var(--text);
  }

  .empty-state {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    text-align: center;
    padding: 40px 0;
  }

  .log {
    margin-top: 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--muted);
  }
  .log-row { display: flex; gap: 8px; padding: 3px 0; }
  .log-row .t { color: var(--muted); width: 70px; flex-shrink: 0; }

  .stats-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin: 18px 0;
  }
  .stat {
    background: var(--panel-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px;
    text-align: center;
  }
  .stat-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 22px;
    font-weight: 700;
    color: var(--text);
  }
  .stat-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: var(--muted);
    letter-spacing: 0.06em;
    margin-top: 2px;
  }

  .feed-area {
    max-height: 340px;
    overflow-y: auto;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--panel-2);
  }
  .feed-row {
    display: grid;
    grid-template-columns: 80px 90px 100px 1fr 90px;
    gap: 10px;
    align-items: center;
    padding: 9px 14px;
    border-bottom: 1px solid var(--border);
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
  }
  .feed-row:last-child { border-bottom: none; }
  .feed-row .f-time { color: var(--muted); }
  .feed-row .f-source { color: var(--muted); font-size: 11px; }
  .feed-row .f-explain { color: var(--text); font-size: 11.5px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-family: 'IBM Plex Sans', sans-serif; }
  .feed-row .f-score { color: var(--muted); text-align: right; }
  .f-badge {
    font-size: 10px;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 4px;
    text-align: center;
    letter-spacing: 0.04em;
  }
  .f-badge.high { background: rgba(255,93,93,0.14); color: var(--high); }
  .f-badge.medium { background: rgba(255,184,76,0.14); color: var(--medium); }
  .f-badge.low { background: rgba(255,184,76,0.10); color: var(--medium); }
  .f-badge.safe { background: rgba(76,224,179,0.12); color: var(--safe); }

  footer {
    text-align: center;
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    padding: 20px;
  }
</style>
</head>
<body>

<div class="topbar">
  <div class="brand">SENTRY <small>cloud anomaly console</small><span class="dot"></span></div>
  <div class="status-pill" id="statusPill">MODEL: isolation-forest-v1</div>
</div>

<div class="wrap">

  <div class="panel">
    <h2>Telemetry Input</h2>
    <div class="sub">Submit a live infrastructure snapshot for evaluation.</div>

    <div class="presets">
      <button class="preset-btn" onclick="loadPreset('normal')">Normal traffic</button>
      <button class="preset-btn" onclick="loadPreset('crypto')">Crypto-mining</button>
      <button class="preset-btn" onclick="loadPreset('exfil')">Data exfiltration</button>
      <button class="preset-btn" onclick="loadPreset('brute')">Brute-force login</button>
      <button class="preset-btn" onclick="loadPreset('stuffing')">Credential stuffing</button>
    </div>

    <div class="field">
      <label>CPU_UTILIZATION (%)</label>
      <input type="number" id="cpu_utilization" value="27.5" step="0.1">
    </div>
    <div class="field">
      <label>NETWORK_BYTES_OUT (MB)</label>
      <input type="number" id="network_bytes_out" value="45.2" step="0.1">
    </div>
    <div class="field">
      <label>FAILED_LOGIN_COUNT</label>
      <input type="number" id="failed_login_count" value="0">
    </div>
    <div class="field">
      <label>API_CALLS_PER_MIN</label>
      <input type="number" id="api_calls_per_min" value="22.0" step="0.1">
    </div>
    <div class="field">
      <label>LOGIN_HOUR (0-23)</label>
      <input type="number" id="login_hour" value="14">
    </div>
    <div class="field">
      <label>UNIQUE_IPS_PER_USER</label>
      <input type="number" id="unique_ips_per_user" value="1">
    </div>

    <button class="submit-btn" onclick="runPredict()" id="submitBtn">RUN DETECTION</button>
  </div>

  <div class="panel result-panel state-idle" id="resultPanel">
    <h2>Verdict</h2>
    <div class="sub">Model output + rule-based explanation</div>

    <div id="resultBody">
      <div class="empty-state">— awaiting telemetry submission —</div>
    </div>

    <div class="log" id="logArea"></div>
  </div>

</div>

<div class="wrap" style="grid-template-columns: 1fr; padding-top: 0;">
  <div class="panel">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 4px;">
      <h2 style="margin:0;">Live Cloud Telemetry Feed</h2>
      <div style="display:flex; align-items:center; gap:8px;">
        <span class="dot" style="width:7px;height:7px;"></span>
        <span style="font-family:'JetBrains Mono',monospace; font-size:11px; color:var(--muted);">STREAMING</span>
      </div>
    </div>
    <div class="sub">Auto-refreshing every 2s &middot; real-time telemetry pipeline</div>

    <div class="stats-strip" id="statsStrip">
      <div class="stat"><div class="stat-val" id="statTotal">0</div><div class="stat-label">EVENTS</div></div>
      <div class="stat"><div class="stat-val" style="color:var(--high)" id="statHigh">0</div><div class="stat-label">HIGH</div></div>
      <div class="stat"><div class="stat-val" style="color:var(--medium)" id="statMedium">0</div><div class="stat-label">MEDIUM</div></div>
      <div class="stat"><div class="stat-val" style="color:var(--safe)" id="statNormal">0</div><div class="stat-label">NORMAL</div></div>
    </div>

    <div id="feedArea" class="feed-area">
      <div class="empty-state" style="padding: 30px 0;">— waiting for live_simulator.py to start streaming —</div>
    </div>
  </div>
</div>

<footer>Isolation Forest + rule-based explainability layer &middot; /predict &middot; /recent &middot; /metrics &middot; /docs</footer>

<script>
const presets = {
  normal:   {cpu_utilization:27.5, network_bytes_out:45.2, failed_login_count:0, api_calls_per_min:22.0, login_hour:14, unique_ips_per_user:1},
  crypto:   {cpu_utilization:97.0, network_bytes_out:50.0, failed_login_count:0, api_calls_per_min:20.0, login_hour:3,  unique_ips_per_user:1},
  exfil:    {cpu_utilization:30.0, network_bytes_out:780.0, failed_login_count:0, api_calls_per_min:25.0, login_hour:15, unique_ips_per_user:1},
  brute:    {cpu_utilization:25.0, network_bytes_out:40.0, failed_login_count:30, api_calls_per_min:20.0, login_hour:2,  unique_ips_per_user:1},
  stuffing: {cpu_utilization:28.0, network_bytes_out:60.0, failed_login_count:1,  api_calls_per_min:260.0, login_hour:16, unique_ips_per_user:14},
};

function loadPreset(name) {
  const p = presets[name];
  for (const key in p) document.getElementById(key).value = p[key];
}

let logCount = 0;

async function runPredict() {
  const btn = document.getElementById('submitBtn');
  btn.disabled = true;
  btn.textContent = 'ANALYZING...';

  const payload = {
    cpu_utilization: parseFloat(document.getElementById('cpu_utilization').value),
    network_bytes_out: parseFloat(document.getElementById('network_bytes_out').value),
    failed_login_count: parseInt(document.getElementById('failed_login_count').value),
    api_calls_per_min: parseFloat(document.getElementById('api_calls_per_min').value),
    login_hour: parseInt(document.getElementById('login_hour').value),
    unique_ips_per_user: parseInt(document.getElementById('unique_ips_per_user').value),
  };

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    renderResult(data);
    logRequest(data);
  } catch (e) {
    document.getElementById('resultBody').innerHTML =
      '<div class="empty-state">ERROR: could not reach /predict &mdash; is the server running?</div>';
  }

  btn.disabled = false;
  btn.textContent = 'RUN DETECTION';
}

function severityClass(sev) {
  if (sev === 'High') return 'high';
  if (sev === 'Medium') return 'medium';
  if (sev === 'Low') return 'medium';
  if (sev === 'None') return 'safe';
  return 'idle';
}

function renderResult(data) {
  const panel = document.getElementById('resultPanel');
  const cls = severityClass(data.severity);
  panel.className = 'panel result-panel state-' + cls;

  const badgeText = data.is_anomaly ? ('ANOMALY DETECTED · ' + data.severity.toUpperCase()) : 'NORMAL · NO THREAT';

  document.getElementById('resultBody').innerHTML = `
    <div class="verdict-badge ${cls}">${badgeText}</div>
    <div class="score-row">
      <span>ANOMALY_SCORE</span>
      <span class="val">${data.anomaly_score}</span>
    </div>
    <div class="explanation-label">Explanation</div>
    <div class="explanation-text">${data.explanation}</div>
  `;
}

function logRequest(data) {
  logCount++;
  const logArea = document.getElementById('logArea');
  const now = new Date().toLocaleTimeString();
  const tag = data.is_anomaly ? data.severity.toUpperCase() : 'NORMAL';
  const row = document.createElement('div');
  row.className = 'log-row';
  row.innerHTML = `<span class="t">${now}</span><span>#${logCount} → ${tag} (score ${data.anomaly_score})</span>`;
  logArea.prepend(row);
}

function badgeClass(ev) {
  if (!ev.is_anomaly) return 'safe';
  if (ev.severity === 'High') return 'high';
  if (ev.severity === 'Medium') return 'medium';
  return 'low';
}

async function pollFeed() {
  try {
    const res = await fetch('/recent?limit=30');
    const events = await res.json();

    if (events.length === 0) return;

    let high = 0, medium = 0, normal = 0;
    events.forEach(ev => {
      if (ev.severity === 'High') high++;
      else if (ev.severity === 'Medium' || ev.severity === 'Low') medium++;
      else normal++;
    });

    document.getElementById('statTotal').textContent = events.length;
    document.getElementById('statHigh').textContent = high;
    document.getElementById('statMedium').textContent = medium;
    document.getElementById('statNormal').textContent = normal;

    const feedArea = document.getElementById('feedArea');
    feedArea.innerHTML = events.map(ev => {
      const cls = badgeClass(ev);
      const badgeText = ev.is_anomaly ? ev.severity.toUpperCase() : 'NORMAL';
      return `
        <div class="feed-row">
          <span class="f-time">${ev.time}</span>
          <span class="f-source">${ev.source}</span>
          <span class="f-badge ${cls}">${badgeText}</span>
          <span class="f-explain" title="${ev.explanation}">${ev.explanation}</span>
          <span class="f-score">${ev.anomaly_score}</span>
        </div>`;
    }).join('');
  } catch (e) {
    // server not reachable yet -- silently retry on next interval
  }
}

setInterval(pollFeed, 2000);
pollFeed();
</script>

</body>
</html>
"""
