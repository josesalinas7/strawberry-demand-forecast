# ── Base ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ── Dependencies ─────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ── Application code ─────────────────────────────────────────────────
COPY . .

# Generate the mock dataset & pre-train the model at build time so the
# container is ready to serve on first request.
RUN python -m src.generate_dataset --dest data/strawberry_demand.csv && \
    python -m src.train --data data/strawberry_demand.csv --model-dir models

# ── Runtime ──────────────────────────────────────────────────────────
# Cloud Run injects $PORT (default 8080).
EXPOSE 8080
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8080"]
