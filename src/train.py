"""
train.py
========
End-to-end training + time-series cross-validation (backtesting) for
the strawberry demand Prophet model.

Optimal split strategy
----------------------
- Training  : 2019-01-07 → 2024-12-30  (~6 years, 313 weeks)
- Hold-out  : 2025-01-06 → 2025-12-29  (~52 weeks, never seen during CV)
- CV / back-test : 5-fold expanding-window with 3-week horizon

Quality gates (production):
  MAPE  < 12 %
  RMSE  < 650 units
  MAE   < 500 units
  R²    > 0.85
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

from src.features import add_regressor_columns, build_future_features, configure_prophet

# ── Quality gate thresholds ──────────────────────────────────────────
QUALITY_GATES = {
    "mape_max": 0.12,
    "rmse_max": 650,
    "mae_max": 500,
    "r2_min": 0.85,
}

TRAIN_END = "2024-12-30"
HOLDOUT_START = "2025-01-06"


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot)


def load_data(path: str = "data/strawberry_demand.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["ds"])
    df = df.rename(columns={"units_sold": "y"})
    df = add_regressor_columns(df)
    return df


def train_and_evaluate(
    data_path: str = "data/strawberry_demand.csv",
    model_dir: str = "models",
) -> dict:
    df = load_data(data_path)

    # ── Split ────────────────────────────────────────────────────────
    train = df[df["ds"] <= TRAIN_END].copy()
    holdout = df[df["ds"] >= HOLDOUT_START].copy()
    print(f"Train  : {train.ds.min().date()} → {train.ds.max().date()}  ({len(train)} wks)")
    print(f"Holdout: {holdout.ds.min().date()} → {holdout.ds.max().date()}  ({len(holdout)} wks)")

    # ── Fit ──────────────────────────────────────────────────────────
    model = configure_prophet()
    model.fit(train)

    # ── Cross-validation (back-test) ─────────────────────────────────
    # initial  = 3 years  (156 weeks)
    # period   = 26 weeks (rolling origin every 6 months)
    # horizon  = 3 weeks  (our forecast horizon)
    cv_results = cross_validation(
        model,
        initial="1092 days",   # ~156 weeks
        period="182 days",     # ~26 weeks
        horizon="21 days",     # 3 weeks
    )
    cv_metrics = performance_metrics(cv_results, rolling_window=1)
    print("\n── Cross-Validation Metrics ──")
    print(cv_metrics[["horizon", "mape", "rmse", "mae", "coverage"]].to_string(index=False))

    # ── Hold-out evaluation ──────────────────────────────────────────
    future_ho = holdout.drop(columns=["y"])
    preds_ho = model.predict(future_ho)
    y_true = holdout["y"].values
    y_pred = preds_ho["yhat"].values

    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    r2 = _r2(y_true, y_pred)

    metrics = {"mape": round(mape, 4), "rmse": round(rmse, 2), "mae": round(mae, 2), "r2": round(r2, 4)}
    print("\n── Hold-out Metrics ──")
    for k, v in metrics.items():
        print(f"  {k:>6s}: {v}")

    # ── Quality gates ────────────────────────────────────────────────
    gate_pass = (
        mape <= QUALITY_GATES["mape_max"]
        and rmse <= QUALITY_GATES["rmse_max"]
        and mae <= QUALITY_GATES["mae_max"]
        and r2 >= QUALITY_GATES["r2_min"]
    )
    metrics["quality_gate_passed"] = gate_pass
    print(f"\n{'✔ QUALITY GATE PASSED' if gate_pass else '✘ QUALITY GATE FAILED'}")

    # ── Persist ──────────────────────────────────────────────────────
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "prophet_strawberry.pkl"
    joblib.dump(model, model_path)

    metrics_path = model_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"\n✔ Model saved  → {model_path}")
    print(f"✔ Metrics saved → {metrics_path}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/strawberry_demand.csv")
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()
    train_and_evaluate(args.data, args.model_dir)
