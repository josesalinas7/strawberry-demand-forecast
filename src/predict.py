"""
predict.py
==========
Load the persisted Prophet model and produce a 3-week-ahead forecast.
Used both by the FastAPI service and by CLI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.features import add_regressor_columns, build_future_features


def predict(
    model_path: str = "models/prophet_strawberry.pkl",
    data_path: str = "data/strawberry_demand.csv",
    horizon: int = 3,
) -> pd.DataFrame:
    """Return a DataFrame with columns [ds, yhat, yhat_lower, yhat_upper]."""
    model = joblib.load(model_path)
    df_hist = pd.read_csv(data_path, parse_dates=["ds"])
    df_hist = add_regressor_columns(df_hist)

    future = build_future_features(df_hist, periods=horizon)
    forecast = model.predict(future)
    result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    result["yhat"] = result["yhat"].round(0).astype(int)
    result["yhat_lower"] = result["yhat_lower"].round(0).astype(int)
    result["yhat_upper"] = result["yhat_upper"].round(0).astype(int)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/prophet_strawberry.pkl")
    parser.add_argument("--data", default="data/strawberry_demand.csv")
    parser.add_argument("--horizon", type=int, default=3)
    args = parser.parse_args()

    result = predict(args.model, args.data, args.horizon)
    print(result.to_json(orient="records", date_format="iso", indent=2))
