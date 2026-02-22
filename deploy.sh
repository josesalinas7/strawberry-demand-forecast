#!/usr/bin/env bash
# deploy.sh – Build & deploy the forecaster to GCP Cloud Run
#
# Prerequisites:
#   - gcloud CLI authenticated
#   - A GCP project with Artifact Registry + Cloud Run enabled
#
# Usage:
#   bash deploy.sh <GCP_PROJECT_ID> [REGION]
#
set -euo pipefail

PROJECT_ID="${1:?Usage: deploy.sh <GCP_PROJECT_ID> [REGION]}"
REGION="${2:-us-central1}"
SERVICE_NAME="strawberry-forecast"
IMAGE="us-docker.pkg.dev/${PROJECT_ID}/forecast/${SERVICE_NAME}:latest"

echo "──────────────────────────────────────────"
echo "Building & pushing container image …"
echo "──────────────────────────────────────────"
gcloud builds submit \
  --project="${PROJECT_ID}" \
  --tag="${IMAGE}" \
  --timeout=1200

echo "──────────────────────────────────────────"
echo "Deploying to Cloud Run …"
echo "──────────────────────────────────────────"
gcloud run deploy "${SERVICE_NAME}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --image="${IMAGE}" \
  --platform=managed \
  --memory=2Gi \
  --cpu=2 \
  --timeout=300 \
  --min-instances=0 \
  --max-instances=3 \
  --allow-unauthenticated \
  --set-env-vars="MODEL_PATH=models/prophet_strawberry.pkl,DATA_PATH=data/strawberry_demand.csv"

echo "──────────────────────────────────────────"
echo "Creating Cloud Scheduler job (Monday 06:00 UTC) …"
echo "──────────────────────────────────────────"
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --format="value(status.url)")

# Delete existing job if present (idempotent redeploy)
gcloud scheduler jobs delete "${SERVICE_NAME}-weekly" \
  --project="${PROJECT_ID}" \
  --location="${REGION}" \
  --quiet 2>/dev/null || true

gcloud scheduler jobs create http "${SERVICE_NAME}-weekly" \
  --project="${PROJECT_ID}" \
  --location="${REGION}" \
  --schedule="0 6 * * 1" \
  --uri="${SERVICE_URL}/predict" \
  --http-method=POST \
  --headers="Content-Type=application/json" \
  --message-body='{"horizon": 3}' \
  --time-zone="UTC" \
  --attempt-deadline="120s"

echo ""
echo "✔ Deployed: ${SERVICE_URL}"
echo "✔ Scheduler: every Monday 06:00 UTC → POST ${SERVICE_URL}/predict"
