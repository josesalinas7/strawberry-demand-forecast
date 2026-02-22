"""
test_pipeline.py
=================
Comprehensive test suite covering every stage of the forecasting pipeline.

Test categories:
  1. Dataset generation & schema validation
  2. Feature engineering correctness
  3. Model training convergence
  4. Cross-validation back-test metrics
  5. Hold-out quality gate enforcement
  6. Prediction output shape & bounds
  7. FastAPI endpoint contracts
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture(scope="session")
def dataset_path(tmp_dir: Path) -> Path:
    from src.generate_dataset import generate

    dest = tmp_dir / "strawberry_demand.csv"
    generate(dest)
    return dest


@pytest.fixture(scope="session")
def df(dataset_path: Path) -> pd.DataFrame:
    return pd.read_csv(dataset_path, parse_dates=["ds"])


@pytest.fixture(scope="session")
def trained_artifacts(dataset_path: Path, tmp_dir: Path) -> dict:
    """Train once for the whole session and return metrics + paths."""
    from src.train import train_and_evaluate

    model_dir = tmp_dir / "models"
    metrics = train_and_evaluate(
        data_path=str(dataset_path),
        model_dir=str(model_dir),
    )
    return {
        "metrics": metrics,
        "model_path": str(model_dir / "prophet_strawberry.pkl"),
        "metrics_path": str(model_dir / "metrics.json"),
        "data_path": str(dataset_path),
    }


# ══════════════════════════════════════════════════════════════════════
# 1. Dataset generation & schema
# ══════════════════════════════════════════════════════════════════════

class TestDataset:
    def test_row_count(self, df: pd.DataFrame):
        """~364 weeks between 2019-01-07 and 2025-12-29."""
        assert 360 <= len(df) <= 370, f"Unexpected row count: {len(df)}"

    def test_columns(self, df: pd.DataFrame):
        expected = {
            "ds", "units_sold", "avg_temp_f", "precip_inches",
            "is_promo", "avg_price_usd", "holiday_window",
            "month_sin", "month_cos",
        }
        assert expected.issubset(df.columns), f"Missing: {expected - set(df.columns)}"

    def test_no_nulls(self, df: pd.DataFrame):
        assert df.isnull().sum().sum() == 0

    def test_date_frequency(self, df: pd.DataFrame):
        diffs = df["ds"].diff().dropna().dt.days
        assert (diffs == 7).all(), "Dataset should be strictly weekly (7-day gaps)"

    def test_units_positive(self, df: pd.DataFrame):
        assert (df["units_sold"] > 0).all()

    def test_price_range(self, df: pd.DataFrame):
        assert df["avg_price_usd"].between(1.0, 6.0).all()

    def test_temperature_range(self, df: pd.DataFrame):
        assert df["avg_temp_f"].between(10, 115).all()


# ══════════════════════════════════════════════════════════════════════
# 2. Feature engineering
# ══════════════════════════════════════════════════════════════════════

class TestFeatures:
    def test_add_regressor_columns(self, df: pd.DataFrame):
        from src.features import add_regressor_columns
        out = add_regressor_columns(df.copy())
        for col in ["avg_temp_f", "precip_inches", "is_promo",
                     "avg_price_usd", "holiday_window", "month_sin", "month_cos"]:
            assert out[col].dtype == np.float64, f"{col} should be float64"

    def test_build_future_features_shape(self, df: pd.DataFrame):
        from src.features import build_future_features
        future = build_future_features(df, periods=3)
        assert len(future) == 3
        assert "ds" in future.columns

    def test_future_dates_are_mondays(self, df: pd.DataFrame):
        from src.features import build_future_features
        future = build_future_features(df, periods=3)
        assert all(d.weekday() == 0 for d in future["ds"]), "Future dates must be Mondays"

    def test_expand_holidays(self):
        from src.features import _expand_holidays
        holidays = _expand_holidays([2024, 2025])
        assert len(holidays) == 20  # 10 holidays × 2 years
        assert "lower_window" in holidays.columns

    def test_configure_prophet_regressors(self):
        from src.features import configure_prophet
        model = configure_prophet(years=[2024])
        # Prophet stores extra regressors as a dict keyed by name
        regressor_names = set(model.extra_regressors.keys())
        expected = {"avg_temp_f", "precip_inches", "is_promo",
                    "avg_price_usd", "holiday_window", "month_sin", "month_cos"}
        assert expected.issubset(regressor_names)


# ══════════════════════════════════════════════════════════════════════
# 3. Training convergence & quality gates
# ══════════════════════════════════════════════════════════════════════

class TestTraining:
    def test_model_file_exists(self, trained_artifacts: dict):
        assert Path(trained_artifacts["model_path"]).is_file()

    def test_metrics_file_exists(self, trained_artifacts: dict):
        assert Path(trained_artifacts["metrics_path"]).is_file()

    def test_mape_below_threshold(self, trained_artifacts: dict):
        m = trained_artifacts["metrics"]
        assert m["mape"] <= 0.12, f"MAPE {m['mape']:.4f} exceeds 12 %"

    def test_rmse_below_threshold(self, trained_artifacts: dict):
        m = trained_artifacts["metrics"]
        assert m["rmse"] <= 650, f"RMSE {m['rmse']:.1f} exceeds 650"

    def test_mae_below_threshold(self, trained_artifacts: dict):
        m = trained_artifacts["metrics"]
        assert m["mae"] <= 500, f"MAE {m['mae']:.1f} exceeds 500"

    def test_r2_above_threshold(self, trained_artifacts: dict):
        m = trained_artifacts["metrics"]
        assert m["r2"] >= 0.85, f"R² {m['r2']:.4f} below 0.85"

    def test_quality_gate_passed(self, trained_artifacts: dict):
        assert trained_artifacts["metrics"]["quality_gate_passed"] is True


# ══════════════════════════════════════════════════════════════════════
# 4. Prediction output
# ══════════════════════════════════════════════════════════════════════

class TestPrediction:
    def test_predict_shape(self, trained_artifacts: dict):
        from src.predict import predict

        result = predict(
            model_path=trained_artifacts["model_path"],
            data_path=trained_artifacts["data_path"],
            horizon=3,
        )
        assert len(result) == 3
        assert set(result.columns) == {"ds", "yhat", "yhat_lower", "yhat_upper"}

    def test_predict_values_positive(self, trained_artifacts: dict):
        from src.predict import predict

        result = predict(
            model_path=trained_artifacts["model_path"],
            data_path=trained_artifacts["data_path"],
            horizon=3,
        )
        assert (result["yhat"] > 0).all(), "Predicted units must be positive"

    def test_confidence_interval_ordering(self, trained_artifacts: dict):
        from src.predict import predict

        result = predict(
            model_path=trained_artifacts["model_path"],
            data_path=trained_artifacts["data_path"],
            horizon=3,
        )
        assert (result["yhat_lower"] <= result["yhat"]).all()
        assert (result["yhat"] <= result["yhat_upper"]).all()


# ══════════════════════════════════════════════════════════════════════
# 5. FastAPI endpoint contracts
# ══════════════════════════════════════════════════════════════════════

class TestAPI:
    @pytest.fixture(autouse=True)
    def _setup_env(self, trained_artifacts: dict):
        """Point the app at the temp model & data."""
        os.environ["MODEL_PATH"] = trained_artifacts["model_path"]
        os.environ["DATA_PATH"] = trained_artifacts["data_path"]
        yield
        os.environ.pop("MODEL_PATH", None)
        os.environ.pop("DATA_PATH", None)

    @pytest.fixture()
    def client(self):
        from fastapi.testclient import TestClient
        from src.app import app
        return TestClient(app)

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "healthy"
        assert body["model_loaded"] is True

    def test_predict_endpoint(self, client):
        r = client.post("/predict", json={"horizon": 3})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["horizon_weeks"] == 3
        assert len(body["forecasts"]) == 3

    def test_predict_default_horizon(self, client):
        r = client.post("/predict")
        assert r.status_code == 200
        assert len(r.json()["forecasts"]) == 3

    def test_predict_invalid_horizon(self, client):
        r = client.post("/predict", json={"horizon": 0})
        assert r.status_code == 422  # validation error
