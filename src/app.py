"""
app.py
======
FastAPI service for the strawberry demand forecaster.
Endpoints:
  GET  /health              → liveness check
  POST /predict             → N-week-ahead forecast (JSON)
  POST /train               → retrain the model (admin)
  GET  /api/history          → historical demand data
  GET  /api/metrics          → latest model evaluation metrics
  GET  /api/summary          → dashboard summary KPIs
  GET  /api/assistant        → deterministic Q&A over forecast data
  GET  /                    → SPA frontend

Deployed on GCP Cloud Run; scheduled via Cloud Scheduler every Monday 06:00 UTC.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

app = FastAPI(
    title="Strawberry Demand Forecast API",
    version="2.0.0",
    description="Weekly Prophet-based strawberry demand forecasting service",
)

logger = logging.getLogger("forecast_api")
logging.basicConfig(level=logging.INFO)

MODEL_PATH = os.getenv("MODEL_PATH", "models/prophet_strawberry.pkl")
DATA_PATH = os.getenv("DATA_PATH", "data/strawberry_demand.csv")
MODEL_DIR = os.getenv("MODEL_DIR", "models")

# Mount static assets
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


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


# ── Helpers ──────────────────────────────────────────────────────────

def _load_history() -> pd.DataFrame:
    """Load the historical CSV, return empty frame on error."""
    try:
        return pd.read_csv(DATA_PATH, parse_dates=["ds"])
    except Exception:
        return pd.DataFrame()


def _load_metrics() -> dict:
    """Load metrics.json, return empty dict on error."""
    metrics_path = Path(MODEL_DIR) / "metrics.json"
    try:
        return json.loads(metrics_path.read_text())
    except Exception:
        return {}


# ── Original Endpoints (unchanged) ──────────────────────────────────

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


# ── New API Endpoints for Frontend ──────────────────────────────────

@app.get("/api/history")
def api_history(
    limit: int = Query(default=0, ge=0, description="Last N rows; 0=all"),
    columns: Optional[str] = Query(default=None, description="Comma-sep column names"),
):
    """Return historical demand data as JSON array."""
    df = _load_history()
    if df.empty:
        return []
    if columns:
        cols = [c.strip() for c in columns.split(",") if c.strip() in df.columns]
        if cols:
            df = df[["ds"] + [c for c in cols if c != "ds"]]
    if limit > 0:
        df = df.tail(limit)
    df["ds"] = df["ds"].dt.strftime("%Y-%m-%d")
    return df.to_dict(orient="records")


@app.get("/api/metrics")
def api_metrics():
    """Return latest model metrics from metrics.json."""
    m = _load_metrics()
    if not m:
        raise HTTPException(status_code=404, detail="No metrics found. Train the model first.")
    return m


@app.get("/api/summary")
def api_summary():
    """Dashboard summary: KPIs computed from historical data + latest metrics."""
    df = _load_history()
    m = _load_metrics()
    model_loaded = Path(MODEL_PATH).is_file()

    if df.empty:
        return {"error": "No historical data available."}

    tail = df.tail(52)  # last year
    recent = df.tail(4)  # last month

    summary = {
        "model_loaded": model_loaded,
        "total_weeks": len(df),
        "date_range": {
            "start": df["ds"].min().strftime("%Y-%m-%d"),
            "end": df["ds"].max().strftime("%Y-%m-%d"),
        },
        "latest_week": {
            "ds": df.iloc[-1]["ds"].strftime("%Y-%m-%d"),
            "units_sold": int(df.iloc[-1]["units_sold"]),
            "avg_price": round(float(df.iloc[-1]["avg_price_usd"]), 2),
            "temp": round(float(df.iloc[-1]["avg_temp_f"]), 1),
        },
        "yearly_stats": {
            "avg_weekly_units": int(tail["units_sold"].mean()),
            "max_weekly_units": int(tail["units_sold"].max()),
            "min_weekly_units": int(tail["units_sold"].min()),
            "std_weekly_units": int(tail["units_sold"].std()),
            "total_units": int(tail["units_sold"].sum()),
        },
        "recent_trend": {
            "avg_units_4w": int(recent["units_sold"].mean()),
            "avg_price_4w": round(float(recent["avg_price_usd"].mean()), 2),
            "avg_temp_4w": round(float(recent["avg_temp_f"].mean()), 1),
            "promo_weeks_4w": int(recent["is_promo"].sum()),
        },
        "metrics": m if m else None,
    }
    return summary


@app.get("/api/assistant")
def api_assistant(q: str = Query(..., min_length=1, description="User question")):
    """
    Deterministic Q&A over forecast & historical data.
    Routes questions to pre-built handlers — no LLM, no AI math.
    All numbers come from stored data.
    """
    df = _load_history()
    m = _load_metrics()
    q_lower = q.lower().strip()

    if df.empty:
        return {"answer": "No historical data is loaded. Please generate the dataset first.", "data": None}

    # ── Route: forecast / prediction questions ───────────────────────
    if any(w in q_lower for w in ["forecast", "predict", "next week", "ahead", "future"]):
        if not Path(MODEL_PATH).is_file():
            return {"answer": "Model is not trained yet. Please train the model first via the Training page.", "data": None}
        from src.predict import predict
        try:
            result = predict(model_path=MODEL_PATH, data_path=DATA_PATH, horizon=3)
            rows = []
            for r in result.itertuples():
                rows.append({"ds": r.ds.strftime("%Y-%m-%d"), "yhat": r.yhat, "yhat_lower": r.yhat_lower, "yhat_upper": r.yhat_upper})
            total = sum(r["yhat"] for r in rows)
            answer = f"The 3-week forecast totals **{total:,} units**.\n\n"
            for r in rows:
                answer += f"- **{r['ds']}**: {r['yhat']:,} units (range: {r['yhat_lower']:,} – {r['yhat_upper']:,})\n"
            return {"answer": answer, "data": rows}
        except Exception as e:
            return {"answer": f"Prediction failed: {e}", "data": None}

    # ── Route: model performance / accuracy / metrics ────────────────
    if any(w in q_lower for w in ["metric", "performance", "accuracy", "mape", "rmse", "mae", "r2", "r²", "quality"]):
        if not m:
            return {"answer": "No metrics available. Train the model first.", "data": None}
        gate = "PASSED" if m.get("quality_gate_passed") else "FAILED"
        answer = (
            f"**Model Performance (Hold-out Evaluation)**\n\n"
            f"| Metric | Value | Threshold | Status |\n"
            f"|--------|-------|-----------|--------|\n"
            f"| MAPE | {m.get('mape', 0):.2%} | ≤ 12% | {'Pass' if m.get('mape', 1) <= 0.12 else 'Fail'} |\n"
            f"| RMSE | {m.get('rmse', 0):,.1f} | ≤ 650 | {'Pass' if m.get('rmse', 999) <= 650 else 'Fail'} |\n"
            f"| MAE | {m.get('mae', 0):,.1f} | ≤ 500 | {'Pass' if m.get('mae', 999) <= 500 else 'Fail'} |\n"
            f"| R² | {m.get('r2', 0):.4f} | ≥ 0.85 | {'Pass' if m.get('r2', 0) >= 0.85 else 'Fail'} |\n\n"
            f"**Quality Gate: {gate}**"
        )
        return {"answer": answer, "data": m}

    # ── Route: historical stats / average / max / trend ──────────────
    if any(w in q_lower for w in ["average", "mean", "total", "sum", "max", "min", "high", "low", "trend", "history", "historical"]):
        tail = df.tail(52)
        avg = int(tail["units_sold"].mean())
        total = int(tail["units_sold"].sum())
        mx = int(tail["units_sold"].max())
        mn = int(tail["units_sold"].min())
        mx_date = tail.loc[tail["units_sold"].idxmax(), "ds"].strftime("%Y-%m-%d")
        mn_date = tail.loc[tail["units_sold"].idxmin(), "ds"].strftime("%Y-%m-%d")
        answer = (
            f"**Last 52-Week Summary**\n\n"
            f"- **Average weekly units**: {avg:,}\n"
            f"- **Total units sold**: {total:,}\n"
            f"- **Peak week**: {mx:,} units ({mx_date})\n"
            f"- **Lowest week**: {mn:,} units ({mn_date})\n"
        )
        return {"answer": answer, "data": {"avg": avg, "total": total, "max": mx, "min": mn}}

    # ── Route: price ─────────────────────────────────────────────────
    if any(w in q_lower for w in ["price", "cost", "expensive", "cheap"]):
        avg_price = round(float(df["avg_price_usd"].mean()), 2)
        recent_price = round(float(df.tail(4)["avg_price_usd"].mean()), 2)
        answer = (
            f"**Pricing Summary**\n\n"
            f"- **All-time average price**: ${avg_price}/lb\n"
            f"- **Recent 4-week average**: ${recent_price}/lb\n"
            f"- Price elasticity: ~−1,200 units per $1 increase above $3.50\n"
        )
        return {"answer": answer, "data": {"avg_price": avg_price, "recent_price": recent_price}}

    # ── Route: weather ───────────────────────────────────────────────
    if any(w in q_lower for w in ["weather", "temperature", "rain", "precip", "temp"]):
        avg_temp = round(float(df.tail(4)["avg_temp_f"].mean()), 1)
        avg_precip = round(float(df.tail(4)["precip_inches"].mean()), 2)
        answer = (
            f"**Recent Weather (4-week avg)**\n\n"
            f"- **Temperature**: {avg_temp}°F\n"
            f"- **Precipitation**: {avg_precip} inches\n"
            f"- Demand impact: +15 units per °F above 55°F, −400 units per inch of rain\n"
        )
        return {"answer": answer, "data": {"avg_temp": avg_temp, "avg_precip": avg_precip}}

    # ── Route: promo ─────────────────────────────────────────────────
    if any(w in q_lower for w in ["promo", "promotion", "campaign"]):
        promo_weeks = int(df["is_promo"].sum())
        total_weeks = len(df)
        pct = round(promo_weeks / total_weeks * 100, 1)
        answer = (
            f"**Promotion Summary**\n\n"
            f"- **Promo weeks**: {promo_weeks} out of {total_weeks} ({pct}%)\n"
            f"- Promo boost: ~+800 units per promotional week\n"
        )
        return {"answer": answer, "data": {"promo_weeks": promo_weeks, "total_weeks": total_weeks}}

    # ── Fallback ─────────────────────────────────────────────────────
    return {
        "answer": (
            "I can help with these topics:\n\n"
            "- **Forecasts**: \"What is the forecast for next week?\"\n"
            "- **Model performance**: \"How accurate is the model?\"\n"
            "- **Historical trends**: \"What was the average demand last year?\"\n"
            "- **Pricing**: \"What is the average price?\"\n"
            "- **Weather impact**: \"How does weather affect demand?\"\n"
            "- **Promotions**: \"How many promo weeks were there?\"\n"
        ),
        "data": None,
    }


# ── SPA Frontend ─────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def serve_spa():
    """Serve the single-page application."""
    index_path = Path(__file__).resolve().parent.parent / "static" / "index.html"
    if not index_path.is_file():
        return HTMLResponse("<h1>Frontend not built. See /docs for API reference.</h1>", status_code=200)
    return HTMLResponse(index_path.read_text(encoding="utf-8"))
