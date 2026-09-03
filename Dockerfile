# --- Build stage: install dependencies into a clean layer ---
FROM python:3.12-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# --- Runtime stage: slim final image, non-root user ---
FROM python:3.12-slim

# Create a non-root user (same principle as your encryption project setup --
# never run production containers as root)
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy installed packages from the builder stage
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code + trained model artifacts
COPY app.py explainability_layer.py isolation_forest_model.pkl ./

ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

USER appuser

EXPOSE 8000

# Basic container-level healthcheck -- Docker/K8s can use this to know
# if the service is alive without needing external tooling
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
# --- Build stage: install dependencies into a clean layer ---
FROM python:3.12-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# --- Runtime stage: slim final image, non-root user ---
FROM python:3.12-slim

# Create a non-root user (same principle as your encryption project setup --
# never run production containers as root)
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy installed packages from the builder stage
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code + trained model artifacts
COPY app.py explainability_layer.py isolation_forest_model.pkl ./

ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

USER appuser

EXPOSE 8000

# Basic container-level healthcheck -- Docker/K8s can use this to know
# if the service is alive without needing external tooling
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
