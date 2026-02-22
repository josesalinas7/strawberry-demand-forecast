"""
app.py
======
FastAPI service for the strawberry demand forecaster.
Endpoints:
  GET  /health          → liveness check
  POST /predict         → 3-week-ahead forecast (JSON)
  POST /train           → retrain the model (admin)

Deployed on GCP Cloud Run; scheduled via Cloud Scheduler every Monday 06:00 UTC.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Strawberry Demand Forecast API",
    version="1.0.0",
    description="Weekly Prophet-based strawberry demand forecasting service",
)

logger = logging.getLogger("forecast_api")
logging.basicConfig(level=logging.INFO)

MODEL_PATH = os.getenv("MODEL_PATH", "models/prophet_strawberry.pkl")
DATA_PATH = os.getenv("DATA_PATH", "data/strawberry_demand.csv")
MODEL_DIR = os.getenv("MODEL_DIR", "models")


# ── Schemas ──────────────────────────────────────────────────────────

class ForecastRequest(BaseModel):
    horizon: int = Field(default=3, ge=1, le=12, description="Weeks ahead to forecast")


class ForecastPoint(BaseModel):
    ds: str
    yhat: int
    yhat_lower: int
    yhat_upper: int


class ForecastResponse(BaseModel):
    status: str = "ok"
    horizon_weeks: int
    forecasts: list[ForecastPoint]


class TrainResponse(BaseModel):
    status: str
    metrics: dict


class HealthResponse(BaseModel):
    status: str = "healthy"
    model_loaded: bool


# ── Endpoints ────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    model_exists = Path(MODEL_PATH).is_file()
    return HealthResponse(model_loaded=model_exists)


@app.post("/predict", response_model=ForecastResponse)
def forecast(req: ForecastRequest = ForecastRequest()):
    from src.predict import predict  # lazy import to keep cold-start fast

    if not Path(MODEL_PATH).is_file():
        raise HTTPException(status_code=503, detail="Model not trained yet. POST /train first.")

    try:
        df = predict(model_path=MODEL_PATH, data_path=DATA_PATH, horizon=req.horizon)
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))

    points = [
        ForecastPoint(
            ds=row.ds.strftime("%Y-%m-%d"),
            yhat=row.yhat,
            yhat_lower=row.yhat_lower,
            yhat_upper=row.yhat_upper,
        )
        for row in df.itertuples()
    ]
    return ForecastResponse(horizon_weeks=req.horizon, forecasts=points)


@app.post("/train", response_model=TrainResponse)
def retrain():
    from src.train import train_and_evaluate  # lazy

    try:
        metrics = train_and_evaluate(data_path=DATA_PATH, model_dir=MODEL_DIR)
    except Exception as exc:
        logger.exception("Training failed")
        raise HTTPException(status_code=500, detail=str(exc))

    status = "passed" if metrics.get("quality_gate_passed") else "failed_quality_gate"
    return TrainResponse(status=status, metrics=metrics)
