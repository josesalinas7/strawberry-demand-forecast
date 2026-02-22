<p align="center">
  <h1 align="center">🍓 Strawberry Demand Forecast</h1>
  <p align="center">
    Production-grade weekly demand forecasting with Facebook Prophet, served via FastAPI on GCP Cloud Run
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white" alt="Python 3.11">
  <img src="https://img.shields.io/badge/model-Prophet_1.1-purple?logo=meta" alt="Prophet">
  <img src="https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/deploy-Cloud_Run-4285F4?logo=googlecloud&logoColor=white" alt="Cloud Run">
  <img src="https://img.shields.io/badge/R²-0.9809-brightgreen" alt="R²">
  <img src="https://img.shields.io/badge/MAPE-8.34%25-brightgreen" alt="MAPE">
</p>

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Technical Specifications](#technical-specifications)
  - [Model Type & Framework](#model-type--framework)
  - [Prophet Hyperparameters](#prophet-hyperparameters)
  - [Feature Engineering — Complete Reference](#feature-engineering--complete-reference)
  - [Holiday Modeling](#holiday-modeling)
  - [Seasonality Configuration](#seasonality-configuration)
- [Dataset Specification](#dataset-specification)
  - [Schema](#schema)
  - [Target Variable Construction](#target-variable-construction)
- [Training Pipeline](#training-pipeline)
  - [Train / Hold-out Split](#train--hold-out-split)
  - [Cross-Validation (Backtesting) Strategy](#cross-validation-backtesting-strategy)
  - [Quality Gates](#quality-gates)
  - [Latest Evaluation Metrics](#latest-evaluation-metrics)
- [Prediction Pipeline](#prediction-pipeline)
  - [Future Feature Imputation](#future-feature-imputation)
- [API Reference](#api-reference)
  - [GET /health](#get-health)
  - [POST /predict](#post-predict)
  - [POST /train](#post-train)
- [Test Suite](#test-suite)
- [Docker](#docker)
- [GCP Cloud Run Deployment](#gcp-cloud-run-deployment)
  - [Prerequisites](#prerequisites)
  - [One-Command Deploy](#one-command-deploy)
  - [Cloud Run Service Configuration](#cloud-run-service-configuration)
  - [Cloud Scheduler (Automated Inference)](#cloud-scheduler-automated-inference)
- [Environment Variables](#environment-variables)
- [Dependencies](#dependencies)
- [License](#license)

---

## Overview

This repository contains a **complete, production-ready demand forecasting system** for weekly strawberry unit sales. It predicts `units_sold` **3 weeks ahead** using a Facebook Prophet model enriched with 7 external regressors covering weather, pricing, promotions, holidays, and cyclical time encodings.

| Property | Value |
|---|---|
| **Target variable** | `units_sold` — weekly unit sales of strawberries |
| **Forecast horizon** | 3 weeks ahead (configurable 1–12) |
| **Temporal granularity** | Weekly (ISO Monday, `W-MON` frequency) |
| **Model framework** | Facebook Prophet ≥ 1.1.5 |
| **Serving framework** | FastAPI ≥ 0.110 on Uvicorn |
| **Deployment target** | GCP Cloud Run (fully managed, serverless) |
| **Automated trigger** | Cloud Scheduler — every Monday 06:00 UTC |
| **Prediction interval** | 95% confidence band (`yhat_lower` / `yhat_upper`) |

---

## Architecture

```
                         ┌──────────────────────────────────────┐
                         │          GCP Cloud Run               │
                         │   ┌──────────────────────────────┐   │
  ┌───────────────┐      │   │  FastAPI Application (app.py) │   │
  │ Cloud         │ POST │   │                                │   │
  │ Scheduler     │──────┼──▶│  GET  /health                  │   │
  │ (Mon 06:00)   │      │   │  POST /predict ──▶ predict.py  │   │
  └───────────────┘      │   │  POST /train   ──▶ train.py    │   │
                         │   └──────────┬───────────────────┘   │
  ┌───────────────┐      │              │                        │
  │ HTTP Clients  │──────┼──────────────┘                        │
  │ (curl / apps) │      │   ┌──────────────────────────────┐   │
  └───────────────┘      │   │  prophet_strawberry.pkl       │   │
                         │   │  strawberry_demand.csv        │   │
                         │   │  metrics.json                 │   │
                         │   └──────────────────────────────┘   │
                         └──────────────────────────────────────┘
```

**Data flow:** Historical CSV → Feature Engineering (`features.py`) → Prophet fit (`train.py`) → Serialized `.pkl` → Inference (`predict.py`) → JSON response (`app.py`)

---

## Project Structure

```
strawberry-demand-forecast/
├── src/
│   ├── __init__.py
│   ├── app.py                 # FastAPI service (3 endpoints)
│   ├── features.py            # Feature engineering + Prophet configuration
│   ├── generate_dataset.py    # Synthetic dataset generator (seed=42)
│   ├── predict.py             # Inference pipeline (CLI + importable)
│   └── train.py               # Training, CV, hold-out eval, quality gates
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py       # 257-line comprehensive test suite (20+ tests)
├── data/
│   └── strawberry_demand.csv  # Weekly demand data (~365 rows, 2019–2025)
├── models/
│   ├── prophet_strawberry.pkl # Trained Prophet model (serialized via joblib)
│   └── metrics.json           # Latest hold-out evaluation metrics
├── docs/
│   └── MODEL_AND_DEPLOYMENT_GUIDE.md  # Extended technical documentation
├── Dockerfile                 # Production container (pre-trains at build time)
├── deploy.sh                  # One-command GCP Cloud Run deployment script
├── requirements.txt           # Pinned Python dependencies
├── pyproject.toml             # Pytest configuration
└── README.md                  # ← You are here
```

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/josesalinas7/strawberry-demand-forecast.git
cd strawberry-demand-forecast

# 2. Create virtual environment & install dependencies
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Generate the synthetic dataset (deterministic, seed=42)
python -m src.generate_dataset

# 4. Train the model (includes cross-validation + quality gate check)
python -m src.train

# 5. Run a 3-week-ahead prediction (CLI)
python -m src.predict --horizon 3

# 6. Start the API server locally
uvicorn src.app:app --reload --port 8080

# 7. Test the API
curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" -d '{"horizon": 3}'

# 8. Run the full test suite
pytest -v
```

---

## Technical Specifications

### Model Type & Framework

| Specification | Detail |
|---|---|
| **Algorithm** | Facebook Prophet (Bayesian structural time-series) |
| **Library** | `prophet` ≥ 1.1.5, < 2.0 (Meta open-source) |
| **Backend** | PyStan (MCMC / MAP optimization) |
| **Growth model** | Linear (no logistic saturation cap) |
| **Trend** | Piecewise linear with automatic changepoint detection |
| **Serialization** | `joblib.dump()` → `.pkl` file |
| **Model size** | ~1.5–2 GB serialized on disk |

### Prophet Hyperparameters

Every hyperparameter configured in `src/features.py → configure_prophet()`:

| Parameter | Value | Description |
|---|---|---|
| `growth` | `"linear"` | Linear trend; demand grows ~1.5%/yr with no saturation ceiling |
| `changepoint_prior_scale` | `0.05` | Controls trend flexibility. 0.05 (default) = smooth trend, avoids overfitting to anomalies |
| `holidays_prior_scale` | `10.0` | Regularization for holiday effects. Relaxed (high value) because holidays have strong, known impacts |
| `seasonality_prior_scale` | `10.0` | Regularization for Fourier seasonality terms. Relaxed to let yearly pattern fit freely |
| `interval_width` | `0.95` | Width of the uncertainty interval — 95% prediction bands |
| `yearly_seasonality` | `10` | 10 Fourier order (20 parameters: 10 sine + 10 cosine terms) |
| `weekly_seasonality` | `False` | Disabled — data is weekly-aggregated, sub-weekly patterns don't exist |
| `daily_seasonality` | `False` | Disabled — same reason as weekly |

### Feature Engineering — Complete Reference

All 7 external regressors are registered as **additive** regressors. Every column is cast to `float64` before training/prediction via `add_regressor_columns()`.

#### 1. `avg_temp_f` — Average Weekly Temperature

| Property | Detail |
|---|---|
| Type | Continuous (float) |
| Unit | Degrees Fahrenheit (°F) |
| Valid range | ~10 – 115 °F |
| Prior scale | 10.0 |
| Regressor mode | Additive |
| Effect on demand | **Positive** — +15 units per °F above 55°F baseline |
| Data generation | `55 + 25 × sin(2π(week - 12) / 52) + N(0, 4)` |
| Future imputation | 4-week trailing rolling average |

Warmer temperatures drive strawberry consumption through fresh eating, smoothies, and seasonal desserts.

#### 2. `precip_inches` — Weekly Precipitation

| Property | Detail |
|---|---|
| Type | Continuous (float) |
| Unit | Inches of rainfall |
| Valid range | ≥ 0 (floor-clipped) |
| Prior scale | 10.0 |
| Regressor mode | Additive |
| Effect on demand | **Negative** — −400 units per inch of rainfall |
| Data generation | `max(0, 0.8 + 0.5 × sin(2π(week - 8) / 52) + N(0, 0.3))` |
| Future imputation | 4-week trailing rolling average |

Heavy precipitation suppresses foot traffic and outdoor-associated demand.

#### 3. `avg_price_usd` — Retail Price Per Pound

| Property | Detail |
|---|---|
| Type | Continuous (float) |
| Unit | US Dollars ($) |
| Valid range | ~$1.00 – $6.00 |
| Prior scale | 10.0 |
| Regressor mode | Additive |
| Effect on demand | **Negative (elastic)** — −1,200 units per $1 above $3.50 |
| Data generation | `round(3.50 - 0.80 × sin(2π(week - 14) / 52) + N(0, 0.20), 2)` |
| Future imputation | 4-week trailing rolling average |

This is the **highest-impact single regressor**. Prices are inversely correlated with supply — lower in peak season (spring/summer), higher in winter.

#### 4. `is_promo` — Promotional Campaign Flag

| Property | Detail |
|---|---|
| Type | Binary (0 or 1) |
| Prior scale | 10.0 |
| Regressor mode | Additive |
| Effect on demand | **Positive** — +800 units during promotional weeks |
| Occurrence rate | ~15% of weeks |
| Future imputation | Defaults to `0.0` (no promo); override manually for planned promotions |

Captures BOGO, discount, and ad-feature promotions. In production, this would be driven by an internal marketing/promotion calendar.

#### 5. `holiday_window` — Holiday Demand Window

| Property | Detail |
|---|---|
| Type | Binary (0 or 1) |
| Prior scale | 10.0 |
| Regressor mode | Additive |
| Effect on demand | **Positive** — +600 units during holiday windows |
| Future imputation | Heuristic: months 2, 5, 7, 9, 11, 12 → `1.0`; else `0.0` |

Holiday windows by ISO week-of-year:

| Window | Weeks | Occasion |
|---|---|---|
| 2–6 | Late Jan – early Feb | Super Bowl |
| 5–9 | Spring | Mother's Day / Easter |
| 18–22 | Late May – June | Memorial Day |
| 23–27 | Late June – July | July 4th |
| 35–37 | September | Labor Day |
| 44–48 | November | Thanksgiving + Christmas lead-up |
| 49–52 | December | Christmas / New Year |

#### 6. `month_sin` — Cyclical Month Encoding (Sine)

| Property | Detail |
|---|---|
| Type | Continuous, cyclical |
| Range | [-1.0, 1.0] |
| Prior scale | **5.0** (lower — complements Prophet's built-in seasonality) |
| Regressor mode | Additive |
| Formula | `sin(2π × month / 12)` |

#### 7. `month_cos` — Cyclical Month Encoding (Cosine)

| Property | Detail |
|---|---|
| Type | Continuous, cyclical |
| Range | [-1.0, 1.0] |
| Prior scale | **5.0** |
| Regressor mode | Additive |
| Formula | `cos(2π × month / 12)` |

**Why a sine/cosine pair?** A single value cannot uniquely & continuously encode all 12 months while preserving circular adjacency (December ↔ January). The 2D `(sin, cos)` coordinate gives each month a unique position on the unit circle so the model correctly learns that month 12 and month 1 are neighbors, not 11 units apart.

### Holiday Modeling

Prophet's native holiday mechanism models 10 US holidays with configurable pre/post windows:

| Holiday | Fixed Date | Lower Window (days before) | Upper Window (days after) |
|---|---|---|---|
| New Year | Jan 1 | −3 | +1 |
| Super Bowl | Feb 12 | −7 | 0 |
| Valentine's Day | Feb 14 | −5 | 0 |
| Easter | Apr 10 | −7 | +1 |
| Mother's Day | May 12 | −7 | 0 |
| Memorial Day | May 27 | −3 | +1 |
| July 4th | Jul 4 | −3 | +1 |
| Labor Day | Sep 2 | −3 | +1 |
| Thanksgiving | Nov 28 | −5 | +1 |
| Christmas | Dec 25 | −7 | +1 |

> **Note:** Holidays use fixed date approximations (not dynamically computed for movable feasts). The `lower_window` and `upper_window` fields tell Prophet to model the N days before/after the holiday date as part of the same holiday effect, capturing pre-purchase lead-up demand.

Holiday coverage: Years 2019–2026 (80 total holiday-date entries).

### Seasonality Configuration

| Component | Enabled | Fourier Order | Notes |
|---|---|---|---|
| **Yearly** | ✅ | 10 | Primary demand driver — peak May–Jul for strawberries |
| **Weekly** | ❌ | — | Data is weekly-aggregated; no sub-weekly signal to capture |
| **Daily** | ❌ | — | Data is weekly-aggregated; daily patterns irrelevant |

Yearly seasonality with Fourier order 10 gives Prophet 20 coefficients (10 sine + 10 cosine) to flexibly model the annual demand curve.

---

## Dataset Specification

**Generator:** `python -m src.generate_dataset` (deterministic, `seed=42`)

| Property | Value |
|---|---|
| Date range | 2019-01-07 → 2025-12-29 |
| Frequency | Weekly, every Monday (`W-MON`) |
| Row count | ~365 rows |
| File | `data/strawberry_demand.csv` |
| Reproducibility | `numpy.random.default_rng(42)` — fully deterministic |

### Schema

| Column | Type | Range | Description |
|---|---|---|---|
| `ds` | datetime | 2019-01-07 – 2025-12-29 | ISO week start date (Monday) |
| `units_sold` | int | ≥ 200 | **Target variable** — weekly units sold |
| `avg_temp_f` | float | ~10 – 115 | Average weekly temperature (°F) |
| `precip_inches` | float | ≥ 0 | Weekly precipitation (inches) |
| `is_promo` | int | 0 or 1 | Promotional campaign active |
| `avg_price_usd` | float | ~$1.00 – $6.00 | Retail price per pound |
| `holiday_window` | int | 0 or 1 | Week overlaps a US holiday window |
| `month_sin` | float | [-1, 1] | sin(2π × month / 12) |
| `month_cos` | float | [-1, 1] | cos(2π × month / 12) |

### Target Variable Construction

`units_sold` is constructed as a sum of interpretable components:

```
units_sold = (base_demand × trend)
           + weather_effect
           + promo_effect
           + holiday_effect
           + price_effect
           + noise
```

| Component | Formula | Magnitude |
|---|---|---|
| **Base demand** | `5000 + 3000 × sin(2π(week - 10) / 52)` | 2,000 – 8,000 units |
| **Trend** | `1 + 0.015 × years_elapsed` | +1.5% compounding year-over-year |
| **Weather effect** | `15 × (temp_f - 55) - 400 × precip_in` | ±600 units |
| **Promo effect** | `800 × is_promo` | +800 units |
| **Holiday effect** | `600 × holiday_window` | +600 units |
| **Price effect** | `-1200 × (price_usd - 3.50)` | ±960 units |
| **Noise** | `N(0, 350)` | Gaussian, σ = 350 |

Final values are clipped to a floor of 200 and rounded to integers.

---

## Training Pipeline

**Entry point:** `python -m src.train [--data PATH] [--model-dir DIR]`

```
                ┌────────────────┐
                │   Load CSV     │
                │   rename →  y  │
                └───────┬────────┘
                        │
                ┌───────▼────────┐
                │  Time-based    │
                │  Train/Holdout │
                │  Split         │
                └───────┬────────┘
                        │
            ┌───────────▼───────────┐
            │   Prophet.fit(train)  │
            └───────────┬───────────┘
                        │
         ┌──────────────▼──────────────┐
         │  Cross-Validation (5-fold)  │
         │  expanding window backtest  │
         └──────────────┬──────────────┘
                        │
         ┌──────────────▼──────────────┐
         │  Hold-out evaluation        │
         │  (MAPE, RMSE, MAE, R²)      │
         └──────────────┬──────────────┘
                        │
         ┌──────────────▼──────────────┐
         │  Quality gate check         │
         │  (pass / fail)              │
         └──────────────┬──────────────┘
                        │
         ┌──────────────▼──────────────┐
         │  Persist model (.pkl)       │
         │  Persist metrics (.json)    │
         └─────────────────────────────┘
```

### Train / Hold-out Split

Strict **temporal split** — no data leakage:

| Set | Date Range | Rows | Purpose |
|---|---|---|---|
| **Training** | 2019-01-07 → 2024-12-30 | ~313 weeks | Model fitting + cross-validation |
| **Hold-out** | 2025-01-06 → 2025-12-29 | ~52 weeks | Final unbiased evaluation (never seen during CV) |

### Cross-Validation (Backtesting) Strategy

Uses Prophet's built-in `cross_validation()` with an **expanding-window** protocol:

| Parameter | Value | Meaning |
|---|---|---|
| `initial` | 1,092 days (~156 weeks ≈ 3 years) | Minimum training window before first test fold |
| `period` | 182 days (~26 weeks ≈ 6 months) | Rolling origin shifts every 6 months |
| `horizon` | 21 days (3 weeks) | Forecast horizon, matches production use |
| Folds | ~5 | Expanding window generates approximately 5 cutoff points |
| `rolling_window` | 1 | Metrics are averaged across the full horizon |

Reported CV metrics: `mape`, `rmse`, `mae`, `coverage` (% of actuals within 95% CI).

### Quality Gates

All four metrics must pass against the **hold-out set** before the model is marked production-ready:

| Metric | Threshold | Direction | What It Measures |
|---|---|---|---|
| **MAPE** | ≤ **12%** | Lower is better | Mean Absolute Percentage Error — scale-independent accuracy |
| **RMSE** | ≤ **650 units** | Lower is better | Root Mean Squared Error — penalizes large misses |
| **MAE** | ≤ **500 units** | Lower is better | Mean Absolute Error — average miss in units |
| **R²** | ≥ **0.85** | Higher is better | Coefficient of determination — variance explained |

Implementation in `src/train.py`:
```python
QUALITY_GATES = {
    "mape_max": 0.12,
    "rmse_max": 650,
    "mae_max": 500,
    "r2_min": 0.85,
}
```

If any gate fails, `metrics.json` records `"quality_gate_passed": false` and the `/train` API returns `"status": "failed_quality_gate"`.

### Latest Evaluation Metrics

From `models/metrics.json` (hold-out evaluation on 52-week 2025 data):

| Metric | Target | Achieved | Status |
|---|---|---|---|
| **MAPE** | ≤ 12.00% | **8.34%** | ✅ Pass |
| **RMSE** | ≤ 650.00 | **420.53** | ✅ Pass |
| **MAE** | ≤ 500.00 | **310.07** | ✅ Pass |
| **R²** | ≥ 0.8500 | **0.9809** | ✅ Pass |
| **Quality Gate** | — | **PASSED** | ✅ |

R² of 0.9809 means the model explains **98.1%** of the variance in weekly strawberry demand.

---

## Prediction Pipeline

**Entry point:** `python -m src.predict [--model PATH] [--data PATH] [--horizon N]`

| Argument | Default | Description |
|---|---|---|
| `--model` | `models/prophet_strawberry.pkl` | Path to serialized Prophet model |
| `--data` | `data/strawberry_demand.csv` | Historical data (used for feature imputation) |
| `--horizon` | `3` | Weeks ahead to forecast (1–12) |

### Steps

1. Load trained model from `.pkl` via `joblib.load()`
2. Load historical CSV for deriving future feature values
3. Generate future Monday dates starting 1 week after the last historical date
4. Impute regressor values for future dates (see below)
5. Call `model.predict(future)` → point estimates + 95% prediction intervals
6. Round `yhat`, `yhat_lower`, `yhat_upper` to integers

### Future Feature Imputation

| Feature | Imputation strategy | Production recommendation |
|---|---|---|
| `avg_temp_f` | 4-week trailing rolling average | Weather API forecast |
| `precip_inches` | 4-week trailing rolling average | Weather API forecast |
| `avg_price_usd` | 4-week trailing rolling average | Internal pricing system |
| `is_promo` | Default `0.0` (no promo) | Marketing/promo calendar lookup |
| `holiday_window` | Month heuristic (months 2,5,7,9,11,12 → 1) | Explicit holiday calendar |
| `month_sin` | `sin(2π × month / 12)` | Computed from date |
| `month_cos` | `cos(2π × month / 12)` | Computed from date |

### Output Format (CLI)

```json
[
  { "ds": "2026-01-05T00:00:00.000Z", "yhat": 4523, "yhat_lower": 3812, "yhat_upper": 5234 },
  { "ds": "2026-01-12T00:00:00.000Z", "yhat": 4610, "yhat_lower": 3900, "yhat_upper": 5320 },
  { "ds": "2026-01-19T00:00:00.000Z", "yhat": 4580, "yhat_lower": 3870, "yhat_upper": 5290 }
]
```

---

## API Reference

**Framework:** FastAPI 0.110+ on Uvicorn ASGI server
**Local URL:** `http://localhost:8080`
**Interactive docs:** `http://localhost:8080/docs` (Swagger UI)

### `GET /health`

Liveness probe for load balancers, uptime monitors, and Cloud Run health checks.

**Response `200 OK`:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

`model_loaded` is `false` if the `.pkl` file is missing from disk.

---

### `POST /predict`

Generate an N-week demand forecast with 95% prediction intervals.

**Request body** (optional — defaults to 3-week horizon):
```json
{
  "horizon": 3
}
```

| Field | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `horizon` | int | 3 | 1 ≤ horizon ≤ 12 | Number of weeks to forecast ahead |

**Response `200 OK`:**
```json
{
  "status": "ok",
  "horizon_weeks": 3,
  "forecasts": [
    { "ds": "2026-01-05", "yhat": 4523, "yhat_lower": 3812, "yhat_upper": 5234 },
    { "ds": "2026-01-12", "yhat": 4610, "yhat_lower": 3900, "yhat_upper": 5320 },
    { "ds": "2026-01-19", "yhat": 4580, "yhat_lower": 3870, "yhat_upper": 5290 }
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `ds` | string (YYYY-MM-DD) | Forecast date (Monday) |
| `yhat` | int | Point forecast (units) |
| `yhat_lower` | int | Lower bound of 95% prediction interval |
| `yhat_upper` | int | Upper bound of 95% prediction interval |

**Error responses:**

| Code | Condition |
|---|---|
| `422` | Validation error (e.g. `horizon: 0` or `horizon: 13`) |
| `500` | Internal prediction failure (logged server-side) |
| `503` | Model not trained — no `.pkl` on disk. Call `POST /train` first. |

---

### `POST /train`

Retrain the model end-to-end: fit → cross-validate → hold-out eval → quality gate → persist.

**Request body:** None required.

**Response `200 OK`:**
```json
{
  "status": "passed",
  "metrics": {
    "mape": 0.0834,
    "rmse": 420.53,
    "mae": 310.07,
    "r2": 0.9809,
    "quality_gate_passed": true
  }
}
```

| `status` value | Meaning |
|---|---|
| `"passed"` | All quality gates passed; model saved to disk |
| `"failed_quality_gate"` | One or more metrics below threshold; model still saved but flagged |

**Error responses:**

| Code | Condition |
|---|---|
| `500` | Training crashed (OOM, data corruption, etc.) |

---

## Test Suite

**Location:** `tests/test_pipeline.py` (257 lines, 20+ tests)
**Runner:** pytest ≥ 8.0

```bash
# Run all tests with verbose output
pytest -v

# Run a specific test class
pytest -v -k TestTraining
```

### Test Coverage by Category

| # | Category | Test Class | Tests | What's Verified |
|---|---|---|---|---|
| 1 | Dataset generation | `TestDataset` | 7 | Row count (~365), column schema, no nulls, 7-day frequency, positive units, price range ($1–6), temperature range |
| 2 | Feature engineering | `TestFeatures` | 5 | Regressor dtype (`float64`), future shape, Monday dates, holiday expansion (10 holidays × N years), Prophet regressor registration |
| 3 | Training & quality gates | `TestTraining` | 7 | Model `.pkl` exists, metrics `.json` exists, MAPE ≤ 12%, RMSE ≤ 650, MAE ≤ 500, R² ≥ 0.85, `quality_gate_passed` flag |
| 4 | Prediction output | `TestPrediction` | 3 | Output shape (3 rows, 4 columns), positive predictions, interval ordering (`lower ≤ yhat ≤ upper`) |
| 5 | API contracts | `TestAPI` | 4 | `/health` → 200, `/predict` → 200, default horizon works, invalid horizon → 422 |

All tests use **session-scoped fixtures** — the dataset is generated once and the model is trained once per test session, keeping total runtime manageable despite Prophet's fitting cost.

---

## Docker

### Dockerfile Design

```dockerfile
FROM python:3.11-slim AS base                # Minimal Debian base
ENV PYTHONDONTWRITEBYTECODE=1                # No .pyc files (smaller image)
    PYTHONUNBUFFERED=1                       # Immediate log output for Cloud Run
    PIP_NO_CACHE_DIR=1                       # No pip cache in image
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY . .
# Pre-train at build time → zero cold-start latency on first request
RUN python -m src.generate_dataset && python -m src.train
EXPOSE 8080
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8080"]
```

**Key design choice: pre-training at build time.** The model `.pkl` is baked into the container image so the first request gets an instant response. No startup initialization required.

### Build & Run Locally

```bash
# Build
docker build -t strawberry-forecast .

# Run (maps container port 8080 to host port 8080)
docker run -p 8080:8080 strawberry-forecast

# Verify
curl http://localhost:8080/health
# → {"status":"healthy","model_loaded":true}

curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"horizon": 3}'
```

**Expected image size:** ~1.5–2 GB (Prophet pulls in PyStan and C++ dependencies).

---

## GCP Cloud Run Deployment

### Prerequisites

1. [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) (`gcloud`) installed and authenticated
2. A GCP project with the following **APIs enabled**:
   - Cloud Build API
   - Artifact Registry API
   - Cloud Run API
   - Cloud Scheduler API
3. An **Artifact Registry Docker repository** created:
   ```bash
   gcloud artifacts repositories create forecast \
     --repository-format=docker \
     --location=us \
     --project=<PROJECT_ID>
   ```

### One-Command Deploy

```bash
bash deploy.sh <GCP_PROJECT_ID> [REGION]
# Default region: us-central1
```

This single script performs all three deployment steps:

| Step | What Happens | Command |
|---|---|---|
| **1. Build & Push** | Docker image built remotely on Cloud Build and pushed to Artifact Registry | `gcloud builds submit --tag=... --timeout=1200` |
| **2. Deploy** | Container deployed to Cloud Run with resource limits and env vars | `gcloud run deploy ...` |
| **3. Schedule** | Cloud Scheduler cron job created for automated weekly inference | `gcloud scheduler jobs create http ...` |

### Cloud Run Service Configuration

| Parameter | Value | Rationale |
|---|---|---|
| **Platform** | `managed` | Fully managed — no GKE cluster needed |
| **Memory** | `2 Gi` | Prophet + PyStan require ~1–1.5 GB during training |
| **CPU** | `2` vCPUs | PyStan's sampling benefits from parallelism |
| **Request timeout** | `300s` (5 min) | Training (`POST /train`) can take 2–3 minutes |
| **Min instances** | `0` | Scale to zero when idle (cost optimization) |
| **Max instances** | `3` | Cap concurrency to control costs |
| **Authentication** | `--allow-unauthenticated` | Public access (swap to IAM for production lockdown) |
| **Container port** | `8080` | Cloud Run default `$PORT` |

**Scaling recommendations by use case:**

| Scenario | Memory | CPU | Min Instances | Max Instances |
|---|---|---|---|---|
| Inference only (no retraining) | 1 Gi | 1 | 0 | 3 |
| Inference + retraining via `/train` | 2 Gi | 2 | 0 | 3 |
| Low-latency production (no cold starts) | 2 Gi | 2 | 1 | 5 |
| High-traffic production | 2 Gi | 2 | 2 | 10 |

### Cloud Scheduler (Automated Inference)

| Property | Value |
|---|---|
| Job name | `strawberry-forecast-weekly` |
| Cron schedule | `0 6 * * 1` |
| Meaning | Every **Monday at 06:00 UTC** |
| HTTP method | `POST` |
| Target | `<SERVICE_URL>/predict` |
| Request body | `{"horizon": 3}` |
| Attempt deadline | 120 seconds |
| Time zone | UTC |

**Manage the schedule:**
```bash
# Manually trigger a run
gcloud scheduler jobs run strawberry-forecast-weekly --project=<ID> --location=<REGION>

# Update the schedule
gcloud scheduler jobs update http strawberry-forecast-weekly --schedule="0 8 * * 1" --project=<ID> --location=<REGION>

# Pause the schedule
gcloud scheduler jobs pause strawberry-forecast-weekly --project=<ID> --location=<REGION>
```

**View Cloud Run logs:**
```bash
gcloud logging read \
  "resource.type=cloud_run_revision AND resource.labels.service_name=strawberry-forecast" \
  --project=<PROJECT_ID> --limit=50 --format="table(timestamp, textPayload)"
```

---

## Environment Variables

| Variable | Default | Used By | Description |
|---|---|---|---|
| `MODEL_PATH` | `models/prophet_strawberry.pkl` | `app.py`, `predict.py` | Path to the serialized Prophet model |
| `DATA_PATH` | `data/strawberry_demand.csv` | `app.py`, `predict.py`, `train.py` | Path to the historical demand CSV |
| `MODEL_DIR` | `models` | `app.py`, `train.py` | Directory for saving trained model + metrics |
| `PORT` | `8080` | Uvicorn (injected by Cloud Run) | HTTP server listen port |

---

## Dependencies

All dependencies are pinned in `requirements.txt`:

| Package | Version Constraint | Purpose |
|---|---|---|
| `prophet` | ≥ 1.1.5, < 2.0 | Core time-series forecasting model (Meta/Facebook) |
| `pandas` | ≥ 2.0, < 3.0 | Data manipulation and CSV I/O |
| `numpy` | ≥ 1.24, < 2.0 | Numerical computing (arrays, math) |
| `joblib` | ≥ 1.3, < 2.0 | Model serialization (pickle wrapper with compression) |
| `fastapi` | ≥ 0.110, < 1.0 | REST API framework with automatic OpenAPI docs |
| `uvicorn[standard]` | ≥ 0.27, < 1.0 | ASGI HTTP server (production-grade) |
| `pydantic` | ≥ 2.0, < 3.0 | Request/response schema validation |
| `pytest` | ≥ 8.0, < 9.0 | Test runner and framework |
| `httpx` | ≥ 0.27, < 1.0 | Async HTTP client (used by FastAPI `TestClient`) |

**Python version:** 3.11 (as specified in Dockerfile)

---

## License

This project is provided as-is for educational and demonstration purposes.

Covers: dataset schema, feature correctness, training convergence,
quality-gate enforcement, prediction bounds, and FastAPI contract tests.

## Project Structure

```
strawberry-demand-forecast/
├── data/                        # Generated CSV
├── models/                      # Trained .pkl + metrics.json
├── src/
│   ├── __init__.py
│   ├── generate_dataset.py      # Mock data generator
│   ├── features.py              # Feature engineering + Prophet config
│   ├── train.py                 # Training + CV + quality gates
│   ├── predict.py               # Inference
│   └── app.py                   # FastAPI service
├── tests/
│   └── test_pipeline.py         # Full test suite
├── Dockerfile
├── deploy.sh                    # GCP deployment script
├── requirements.txt
├── pyproject.toml
└── README.md
```

## License

MIT
