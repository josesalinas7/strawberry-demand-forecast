# Strawberry Demand Forecast — Complete Model & Deployment Reference

> **Model type:** Facebook Prophet (additive regression)
> **Target variable:** `units_sold` (weekly strawberry units)
> **Forecast horizon:** 3 weeks ahead
> **Granularity:** Weekly (ISO Monday)
> **Deployment target:** GCP Cloud Run (serverless containers)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Feature Reference (Complete)](#3-feature-reference-complete)
   - 3.1 [avg_temp_f — Average Temperature](#31-avg_temp_f--average-temperature)
   - 3.2 [precip_inches — Precipitation](#32-precip_inches--precipitation)
   - 3.3 [avg_price_usd — Retail Price](#33-avg_price_usd--retail-price)
   - 3.4 [is_promo — Promotional Campaign Flag](#34-is_promo--promotional-campaign-flag)
   - 3.5 [holiday_window — Holiday Demand Window](#35-holiday_window--holiday-demand-window)
   - 3.6 [month_sin / month_cos — Cyclical Month Encoding](#36-month_sin--month_cos--cyclical-month-encoding)
   - 3.7 [US Holidays (Prophet Built-in)](#37-us-holidays-prophet-built-in)
   - 3.8 [Yearly Seasonality (Prophet Built-in)](#38-yearly-seasonality-prophet-built-in)
4. [Prophet Model Configuration](#4-prophet-model-configuration)
5. [Training Pipeline](#5-training-pipeline)
6. [Cross-Validation & Backtesting Strategy](#6-cross-validation--backtesting-strategy)
7. [Quality Gates](#7-quality-gates)
8. [Prediction Pipeline](#8-prediction-pipeline)
9. [API Reference (FastAPI)](#9-api-reference-fastapi)
10. [Test Suite](#10-test-suite)
11. [Dockerization](#11-dockerization)
12. [GCP Cloud Run Deployment](#12-gcp-cloud-run-deployment)
13. [Cloud Scheduler (Automated Weekly Inference)](#13-cloud-scheduler-automated-weekly-inference)
14. [Environment Variables](#14-environment-variables)
15. [Troubleshooting](#15-troubleshooting)

---

## 1. Project Overview

This project provides a **production-grade, end-to-end demand forecasting pipeline** for weekly strawberry sales. It uses Facebook Prophet as the core time-series model, enriched with seven external regressors (weather, price, promotions, holidays, and cyclical time encodings). The model is served via a FastAPI application, containerized with Docker, and deployed on Google Cloud Run with automated weekly inference triggered by Cloud Scheduler.

### Architecture Diagram

```
┌──────────────────┐     Cloud Scheduler      ┌──────────────────────────┐
│  Monday 06:00    │ ──── POST /predict ─────▶ │  Cloud Run               │
│  UTC cron job    │                           │  (FastAPI + Prophet)     │
└──────────────────┘                           └──────────┬───────────────┘
                                                          │
                                  ┌───────────────────────▼──────────────┐
                                  │  prophet_strawberry.pkl (serialized) │
                                  │  + strawberry_demand.csv (history)   │
                                  └──────────────────────────────────────┘
```

### File Structure

| File | Purpose |
|---|---|
| `src/generate_dataset.py` | Creates the synthetic weekly demand dataset (2019–2025) |
| `src/features.py` | Feature engineering, Prophet configuration, holiday definitions |
| `src/train.py` | Training, cross-validation, hold-out evaluation, quality gates |
| `src/predict.py` | Loads the trained model and generates N-week forecasts |
| `src/app.py` | FastAPI web service with `/health`, `/predict`, `/train` endpoints |
| `tests/test_pipeline.py` | 257-line test suite covering every pipeline stage |
| `Dockerfile` | Multi-stage Docker build; pre-trains model at build time |
| `deploy.sh` | One-command deployment to GCP Cloud Run + Cloud Scheduler |
| `data/strawberry_demand.csv` | Weekly demand data (≈365 rows) |
| `models/prophet_strawberry.pkl` | Serialized trained Prophet model |
| `models/metrics.json` | Hold-out evaluation metrics from the last training run |

---

## 2. Dataset

**Source:** `src/generate_dataset.py` generates a realistic synthetic dataset.

| Property | Value |
|---|---|
| Date range | 2019-01-07 → 2025-12-29 |
| Frequency | Weekly (every Monday, `W-MON`) |
| Row count | ~365 rows |
| Target column | `units_sold` (integer, always ≥ 200) |
| Random seed | 42 (reproducible) |

### Data Generation Logic

The target `units_sold` is constructed from multiple additive components:

```
units_sold = base_demand × trend
           + weather_effect
           + promo_effect
           + holiday_effect
           + price_effect
           + noise
```

| Component | Formula | Description |
|---|---|---|
| Base demand | `5000 + 3000 × sin(2π(week - 10) / 52)` | Yearly seasonality peaking May–Jul |
| Trend | `1 + 0.015 × years_elapsed` | +1.5% year-over-year growth |
| Weather | `15 × (temp - 55) - 400 × precip` | Warm → more demand; rain → less |
| Promo | `800 × is_promo` | +800 units during promotions |
| Holiday | `600 × holiday_window` | +600 units during holiday windows |
| Price | `-1200 × (price - 3.50)` | Elastic: cheaper → more demand |
| Noise | `N(0, 350)` | Random Gaussian noise |

### Column Schema

| Column | Type | Range | Description |
|---|---|---|---|
| `ds` | datetime | 2019-01-07 – 2025-12-29 | Monday date (ISO week start) |
| `units_sold` | int | ≥ 200 | Target: weekly units sold |
| `avg_temp_f` | float | ~10 – 115 °F | Average weekly temperature |
| `precip_inches` | float | ≥ 0 | Weekly precipitation |
| `is_promo` | int (0/1) | 0 or 1 | Promotional campaign active |
| `avg_price_usd` | float | ~$1.00 – $6.00 | Retail price per lb |
| `holiday_window` | int (0/1) | 0 or 1 | Week overlaps a holiday window |
| `month_sin` | float | [-1, 1] | sin(2π × month / 12) |
| `month_cos` | float | [-1, 1] | cos(2π × month / 12) |

---

## 3. Feature Reference (Complete)

All features are registered as **additive regressors** in the Prophet model. Each feature is cast to `float64` by `add_regressor_columns()` before training or prediction.

---

### 3.1 `avg_temp_f` — Average Temperature

| Property | Value |
|---|---|
| **Type** | Continuous |
| **Unit** | Degrees Fahrenheit (°F) |
| **Range** | ~10 – 115 °F |
| **Regressor mode** | Additive |
| **Prior scale** | 10.0 |
| **Source** | Weather data (synthetic: seasonal sine + Gaussian noise) |

**How it works:**
Temperature captures the seasonal demand pattern for strawberries. Warmer weather drives higher consumption (smoothies, desserts, fresh eating). The data generation model adds +15 units per degree above 55°F.

**Generation formula:**
```
base = 55 + 25 × sin(2π(week_of_year - 12) / 52)
avg_temp_f = base + N(0, 4)
```

**Future forecasting:** For unseen future dates, the pipeline carries forward a 4-week rolling average from the most recent historical data. In a production environment, this would be replaced by a weather-API forecast.

---

### 3.2 `precip_inches` — Precipitation

| Property | Value |
|---|---|
| **Type** | Continuous |
| **Unit** | Inches of rainfall |
| **Range** | ≥ 0 (clipped at zero) |
| **Regressor mode** | Additive |
| **Prior scale** | 10.0 |
| **Source** | Weather data (synthetic: seasonal sine + Gaussian noise) |

**How it works:**
Precipitation has a **negative** effect on strawberry demand. Heavy rainfall reduces foot traffic to stores and suppresses outdoor-related demand. The data generation model subtracts 400 units per inch of precipitation.

**Generation formula:**
```
base = 0.8 + 0.5 × sin(2π(week_of_year - 8) / 52)
precip_inches = max(0, base + N(0, 0.3))
```

**Future forecasting:** 4-week rolling average carried forward (same as temperature).

---

### 3.3 `avg_price_usd` — Retail Price

| Property | Value |
|---|---|
| **Type** | Continuous |
| **Unit** | US Dollars per pound ($) |
| **Range** | ~$1.00 – $6.00 |
| **Regressor mode** | Additive |
| **Prior scale** | 10.0 |
| **Source** | Retail pricing data (synthetic: seasonal inverse + noise) |

**How it works:**
Price models **demand elasticity**. Strawberry prices are inversely correlated with supply — lower in peak season (spring/summer), higher in winter. The model learns that a $1 increase above $3.50 reduces demand by approximately 1,200 units. This makes price the most impactful single regressor.

**Generation formula:**
```
base = 3.50 - 0.80 × sin(2π(week_of_year - 14) / 52)
avg_price_usd = round(base + N(0, 0.20), 2)
```

**Future forecasting:** 4-week rolling average carried forward.

---

### 3.4 `is_promo` — Promotional Campaign Flag

| Property | Value |
|---|---|
| **Type** | Binary (0 or 1) |
| **Regressor mode** | Additive |
| **Prior scale** | 10.0 |
| **Occurrence rate** | ~15% of weeks |

**How it works:**
Binary flag indicating whether a promotional campaign (e.g. BOGO, discount, ad feature) ran during a given week. Promotions boost demand by approximately +800 units. The flag is set randomly during data generation (~15% probability) but in production would come from an internal promotion calendar.

**Future forecasting:** Defaults to `0` (no promo) for future dates. Override manually when a planned promotion is scheduled by setting `is_promo = 1.0` in the future feature dataframe.

---

### 3.5 `holiday_window` — Holiday Demand Window

| Property | Value |
|---|---|
| **Type** | Binary (0 or 1) |
| **Regressor mode** | Additive |
| **Prior scale** | 10.0 |

**How it works:**
Flags weeks that fall within a US holiday demand window — periods when strawberry demand historically spikes due to holiday cooking, desserts, and entertaining. The data generation model adds +600 units during these windows.

**Holiday windows (by ISO week-of-year):**

| Window | Weeks | Holiday |
|---|---|---|
| Super Bowl | 6 – 9 | Late January / early February |
| Mother's Day / Easter | 18 – 22 | Spring |
| Memorial Day | 23 – 27 | Late May / June |
| July 4th | 23 – 27 | Early July |
| Labor Day | 35 – 37 | September |
| Thanksgiving + Christmas lead-up | 44 – 48 | November |
| Christmas / New Year | 49 – 52 | December |

**Future forecasting heuristic:** For future dates, months 2, 5, 7, 9, 11, 12 trigger `holiday_window = 1.0`, otherwise `0.0`. This is a simplified heuristic; production use should reference an explicit calendar.

---

### 3.6 `month_sin` / `month_cos` — Cyclical Month Encoding

| Property | Value |
|---|---|
| **Type** | Continuous, cyclical |
| **Range** | [-1.0, 1.0] |
| **Regressor mode** | Additive |
| **Prior scale** | 5.0 (lower than other regressors) |

**How it works:**
Standard **Fourier cyclical encoding** of the calendar month, ensuring the model understands that December (month 12) and January (month 1) are adjacent — something a raw integer month feature would not capture.

**Formulas:**
```
month_sin = sin(2π × month / 12)
month_cos = cos(2π × month / 12)
```

| Month | month_sin | month_cos |
|---|---|---|
| January (1) | 0.5000 | 0.8660 |
| April (4) | 0.8660 | -0.5000 |
| July (7) | -0.5000 | -0.8660 |
| October (10) | -0.8660 | 0.5000 |

**Why a pair:** A single sine or cosine cannot uniquely map all 12 months. The pair (`sin`, `cos`) together create a unique 2D coordinate for each month, preserving circular distance.

**Prior scale** is set to 5.0 (vs. 10.0 for other regressors) because Prophet's own yearly seasonality already captures monthly patterns; these features provide a smoother auxiliary signal.

---

### 3.7 US Holidays (Prophet Built-in)

| Property | Value |
|---|---|
| **Type** | Prophet holiday component |
| **Count** | 10 holidays |
| **Years covered** | 2019 – 2026 |
| **holidays_prior_scale** | 10.0 |

Prophet's native holiday mechanism models date-specific demand spikes with configurable **pre-event** and **post-event** windows:

| Holiday | Fixed Date | Lower Window | Upper Window |
|---|---|---|---|
| New Year | Jan 1 | -3 days | +1 day |
| Super Bowl | Feb 12 | -7 days | 0 |
| Valentine's Day | Feb 14 | -5 days | 0 |
| Easter | Apr 10 | -7 days | +1 day |
| Mother's Day | May 12 | -7 days | 0 |
| Memorial Day | May 27 | -3 days | +1 day |
| July 4th | Jul 4 | -3 days | +1 day |
| Labor Day | Sep 2 | -3 days | +1 day |
| Thanksgiving | Nov 28 | -5 days | +1 day |
| Christmas | Dec 25 | -7 days | +1 day |

> **Note:** Holiday dates are fixed approximations (not dynamically computed for movable holidays like Easter or Thanksgiving). The windows act as a buffer to capture lead-up demand.

**Lower window** = days *before* the holiday that are affected (e.g., Valentine's Day has a 5-day pre-purchase window). **Upper window** = days *after* the holiday still affected.

---

### 3.8 Yearly Seasonality (Prophet Built-in)

| Property | Value |
|---|---|
| **Fourier order** | 10 |
| **Weekly seasonality** | Disabled (data is weekly aggregated) |
| **Daily seasonality** | Disabled (data is weekly aggregated) |

Prophet models yearly seasonality using a **Fourier series** with 10 terms. This gives the model 20 parameters (10 sine + 10 cosine) to capture the annual demand curve (peaking May–July for strawberries).

**Why disable weekly/daily:** The dataset is aggregated to weekly granularity, so sub-weekly patterns are irrelevant and would only add noise.

---

## 4. Prophet Model Configuration

Configured in `src/features.py → configure_prophet()`:

```python
Prophet(
    growth="linear",              # Linear trend (no logistic saturation)
    holidays=holidays_df,         # 10 US holidays with windows
    yearly_seasonality=10,        # 10 Fourier terms
    weekly_seasonality=False,     # Disabled (weekly data)
    daily_seasonality=False,      # Disabled (weekly data)
    changepoint_prior_scale=0.05, # Trend flexibility (lower = smoother)
    holidays_prior_scale=10.0,    # Holiday effect regularization
    seasonality_prior_scale=10.0, # Seasonality regularization
    interval_width=0.95,          # 95% prediction intervals
)
```

### Hyperparameter Rationale

| Parameter | Value | Rationale |
|---|---|---|
| `growth` | `"linear"` | Strawberry demand grows steadily (~1.5%/yr); no saturation ceiling |
| `changepoint_prior_scale` | 0.05 | Conservative (default 0.05) — avoids overfitting to COVID-era disruptions |
| `holidays_prior_scale` | 10.0 | Relaxed prior — holidays have a strong, known effect on demand |
| `seasonality_prior_scale` | 10.0 | Relaxed — yearly seasonality is the dominant signal |
| `interval_width` | 0.95 | 95% confidence interval for uncertainty quantification |

### Registered Regressors Summary

| Regressor | Prior Scale | Mode |
|---|---|---|
| `avg_temp_f` | 10.0 | additive |
| `precip_inches` | 10.0 | additive |
| `is_promo` | 10.0 | additive |
| `avg_price_usd` | 10.0 | additive |
| `holiday_window` | 10.0 | additive |
| `month_sin` | 5.0 | additive |
| `month_cos` | 5.0 | additive |

---

## 5. Training Pipeline

**Entry point:** `python -m src.train` (or `POST /train` via API)

### Steps

1. **Load data** — reads `data/strawberry_demand.csv`, renames `units_sold` → `y`, casts all regressor columns to `float64`.
2. **Split** — Time-based split:
   - **Training set:** 2019-01-07 → 2024-12-30 (~313 weeks)
   - **Hold-out set:** 2025-01-06 → 2025-12-29 (~52 weeks)
3. **Fit** — `model.fit(train)` on the training set.
4. **Cross-validate** — 5-fold expanding-window backtesting (see Section 6).
5. **Hold-out evaluation** — predict on the unseen 2025 data; compute MAPE, RMSE, MAE, and R².
6. **Quality gate** — assert all metrics pass thresholds (see Section 7).
7. **Persist** — save model to `models/prophet_strawberry.pkl` (via `joblib`) and metrics to `models/metrics.json`.

### CLI Arguments

```
python -m src.train [--data DATA_PATH] [--model-dir MODEL_DIR]
```

| Argument | Default | Description |
|---|---|---|
| `--data` | `data/strawberry_demand.csv` | Path to training CSV |
| `--model-dir` | `models` | Output directory for model + metrics |

---

## 6. Cross-Validation & Backtesting Strategy

Uses Prophet's built-in `cross_validation()` function with an **expanding window** strategy:

| Parameter | Value | Explanation |
|---|---|---|
| `initial` | 1,092 days (~3 years) | Minimum training window before first test fold |
| `period` | 182 days (~26 weeks) | Rolling origin shifts every 6 months |
| `horizon` | 21 days (3 weeks) | Forecast horizon matches production use |

This produces approximately **5 folds**, each expanding the training set by 26 weeks and evaluating 3-week-ahead forecasts. Metrics are aggregated with `rolling_window=1` (full-horizon average).

**Reported CV metrics:** `mape`, `rmse`, `mae`, `coverage` (% of actuals within 95% CI).

---

## 7. Quality Gates

All four thresholds must pass on the **hold-out set** before the model is considered production-ready:

| Metric | Threshold | Last Measured |
|---|---|---|
| **MAPE** (Mean Absolute Percentage Error) | ≤ 12% | 8.34% ✔ |
| **RMSE** (Root Mean Square Error) | ≤ 650 units | 420.53 ✔ |
| **MAE** (Mean Absolute Error) | ≤ 500 units | 310.07 ✔ |
| **R²** (Coefficient of Determination) | ≥ 0.85 | 0.9809 ✔ |

**Result:** `quality_gate_passed: true`

If any gate fails, `metrics.json` records `quality_gate_passed: false` and the `/train` API endpoint returns `status: "failed_quality_gate"`.

---

## 8. Prediction Pipeline

**Entry point:** `python -m src.predict` (or `POST /predict` via API)

### Steps

1. **Load model** — `joblib.load("models/prophet_strawberry.pkl")`
2. **Load history** — reads the historical CSV to derive future feature values.
3. **Build future features** — `build_future_features(df_hist, periods=N)`:
   - Generates `N` future Monday dates starting one week after the last historical date.
   - Continuous features (`avg_temp_f`, `precip_inches`, `avg_price_usd`) use the **4-week trailing rolling average** from the most recent history.
   - `is_promo` defaults to `0` (no promotion).
   - `holiday_window` uses a month-based heuristic.
   - `month_sin` / `month_cos` are computed from the future date's month.
4. **Predict** — `model.predict(future)` returns point estimates + 95% prediction intervals.
5. **Round** — `yhat`, `yhat_lower`, `yhat_upper` are rounded to integers.

### Output Schema

```json
[
  {
    "ds": "2026-01-05",
    "yhat": 4523,
    "yhat_lower": 3812,
    "yhat_upper": 5234
  }
]
```

### CLI Arguments

```
python -m src.predict [--model MODEL_PATH] [--data DATA_PATH] [--horizon N]
```

| Argument | Default | Description |
|---|---|---|
| `--model` | `models/prophet_strawberry.pkl` | Path to trained model |
| `--data` | `data/strawberry_demand.csv` | Historical data for feature derivation |
| `--horizon` | 3 | Number of weeks ahead to forecast |

---

## 9. API Reference (FastAPI)

**Base URL:** `http://localhost:8080` (local) or Cloud Run service URL.

### `GET /health`

Liveness check for load balancers and uptime monitors.

**Response (200):**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

`model_loaded` is `false` if the `.pkl` file does not exist on disk.

---

### `POST /predict`

Generate a multi-week demand forecast.

**Request body (optional):**
```json
{
  "horizon": 3
}
```

| Field | Type | Default | Constraints |
|---|---|---|---|
| `horizon` | int | 3 | 1 ≤ horizon ≤ 12 |

**Response (200):**
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

**Error responses:**
- `422` — Validation error (e.g. `horizon: 0`)
- `503` — Model not trained yet
- `500` — Internal prediction failure

---

### `POST /train`

Retrain the model on the current dataset. Runs the full pipeline (fit → CV → hold-out → quality gate → persist).

**Response (200):**
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

`status` is `"passed"` if quality gates pass, `"failed_quality_gate"` otherwise.

---

## 10. Test Suite

Located in `tests/test_pipeline.py` (257 lines). Run with:

```bash
pytest -v
```

### Test Categories

| Category | Class | Tests |
|---|---|---|
| Dataset generation & schema | `TestDataset` | Row count, columns, no nulls, weekly frequency, units positive, price range, temperature range |
| Feature engineering | `TestFeatures` | Regressor column dtypes, future feature shape, future dates are Mondays, holiday expansion, Prophet regressor registration |
| Training & quality gates | `TestTraining` | Model file exists, metrics file exists, MAPE/RMSE/MAE/R² thresholds, quality gate flag |
| Prediction output | `TestPrediction` | Output shape (3 rows), positive predictions, confidence interval ordering (`lower ≤ yhat ≤ upper`) |
| FastAPI endpoints | `TestAPI` | `/health` 200, `/predict` 200, default horizon, invalid horizon 422 |

All tests use **session-scoped fixtures** — the dataset is generated once and the model is trained once per test session to minimize runtime.

---

## 11. Dockerization

### Dockerfile Walkthrough

```dockerfile
# ── Base ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \    # No .pyc files
    PYTHONUNBUFFERED=1 \           # Immediate stdout/stderr
    PIP_NO_CACHE_DIR=1             # Smaller image

WORKDIR /app

# ── Dependencies ─────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ── Application code ─────────────────────────────────────────────────
COPY . .

# Pre-train at build time so the container serves on first request
RUN python -m src.generate_dataset --dest data/strawberry_demand.csv && \
    python -m src.train --data data/strawberry_demand.csv --model-dir models

# ── Runtime ──────────────────────────────────────────────────────────
EXPOSE 8080
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8080"]
```

### Key Design Decisions

| Decision | Rationale |
|---|---|
| `python:3.11-slim` base | Minimal Debian image; keeps size down while providing C compiler for Prophet's dependencies (PyStan) |
| Pre-train at build time | Eliminates cold-start model loading latency on first request; the `.pkl` file is baked into the image |
| `PYTHONDONTWRITEBYTECODE=1` | Prevents `.pyc` file generation (reduces image size, avoids permission issues) |
| `PYTHONUNBUFFERED=1` | Ensures all print/log output appears immediately in Cloud Run logs |
| `PIP_NO_CACHE_DIR=1` | Avoids caching pip downloads inside the container image |
| Port 8080 | Cloud Run's default `$PORT` value |

### Build & Run Locally

```bash
# Build the image
docker build -t strawberry-forecast .

# Run locally (maps port 8080)
docker run -p 8080:8080 strawberry-forecast

# Test health endpoint
curl http://localhost:8080/health

# Test prediction
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"horizon": 3}'
```

### Image Size Considerations

Prophet pulls in PyStan and several C++ dependencies. Expect the final image to be **~1.5–2 GB**. To reduce size:
- Use a multi-stage build (compile dependencies in a `builder` stage, copy only `.so` files to a `runtime` stage).
- Pin exact versions in `requirements.txt` to avoid pulling unnecessary extras.

---

## 12. GCP Cloud Run Deployment

### Prerequisites

1. **Google Cloud SDK (`gcloud`)** installed and authenticated.
2. **APIs enabled** on the GCP project:
   - Cloud Build API
   - Artifact Registry API
   - Cloud Run API
   - Cloud Scheduler API
3. **Artifact Registry repository** created:
   ```bash
   gcloud artifacts repositories create forecast \
     --repository-format=docker \
     --location=us \
     --project=<PROJECT_ID>
   ```

### One-Command Deployment

```bash
bash deploy.sh <GCP_PROJECT_ID> [REGION]
```

Default region: `us-central1`

### What `deploy.sh` Does

#### Step 1: Build & Push Container Image

```bash
gcloud builds submit \
  --project="${PROJECT_ID}" \
  --tag="us-docker.pkg.dev/${PROJECT_ID}/forecast/strawberry-forecast:latest" \
  --timeout=1200
```

- Uses **Cloud Build** (serverless) to build the Docker image remotely.
- Pushes the built image to **Artifact Registry** at `us-docker.pkg.dev/<PROJECT>/forecast/strawberry-forecast:latest`.
- Timeout is 1,200 seconds (20 minutes) to accommodate Prophet's large dependency tree.

#### Step 2: Deploy to Cloud Run

```bash
gcloud run deploy strawberry-forecast \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --image="us-docker.pkg.dev/${PROJECT_ID}/forecast/strawberry-forecast:latest" \
  --platform=managed \
  --memory=2Gi \
  --cpu=2 \
  --timeout=300 \
  --min-instances=0 \
  --max-instances=3 \
  --allow-unauthenticated \
  --set-env-vars="MODEL_PATH=models/prophet_strawberry.pkl,DATA_PATH=data/strawberry_demand.csv"
```

| Flag | Value | Explanation |
|---|---|---|
| `--platform=managed` | Fully managed Cloud Run (no GKE needed) |
| `--memory=2Gi` | 2 GB RAM — Prophet + PyStan need ~1–1.5 GB during training |
| `--cpu=2` | 2 vCPUs — Prophet's MCMC sampling benefits from parallelism |
| `--timeout=300` | 5-minute request timeout (training can take 2–3 min) |
| `--min-instances=0` | Scale to zero when idle (cost optimization) |
| `--max-instances=3` | Cap at 3 concurrent instances to control costs |
| `--allow-unauthenticated` | Public access (remove for production; use IAM instead) |
| `--set-env-vars` | Passes model and data paths as environment variables |

#### Step 3: Create Cloud Scheduler Job

```bash
gcloud scheduler jobs create http strawberry-forecast-weekly \
  --schedule="0 6 * * 1" \
  --uri="${SERVICE_URL}/predict" \
  --http-method=POST \
  --headers="Content-Type=application/json" \
  --message-body='{"horizon": 3}' \
  --time-zone="UTC" \
  --attempt-deadline="120s"
```

See Section 13 for details.

### Cloud Run Resource Recommendations

| Scenario | Memory | CPU | Max Instances |
|---|---|---|---|
| Inference only | 1 Gi | 1 | 3 |
| Inference + retraining via `/train` | 2 Gi | 2 | 3 |
| High-traffic production | 2 Gi | 2 | 10 |

---

## 13. Cloud Scheduler (Automated Weekly Inference)

| Property | Value |
|---|---|
| **Job name** | `strawberry-forecast-weekly` |
| **Schedule** | `0 6 * * 1` (every Monday at 06:00 UTC) |
| **Target** | `POST <SERVICE_URL>/predict` |
| **Request body** | `{"horizon": 3}` |
| **Timeout** | 120 seconds |
| **Time zone** | UTC |

The scheduler triggers a 3-week-ahead forecast every Monday morning. In a production system, the `/predict` endpoint would additionally push the forecast results to a downstream data warehouse, dashboard, or alerting system.

To **update the schedule**:
```bash
gcloud scheduler jobs update http strawberry-forecast-weekly \
  --schedule="0 8 * * 1" \
  --project=<PROJECT_ID> \
  --location=<REGION>
```

To **manually trigger** a run:
```bash
gcloud scheduler jobs run strawberry-forecast-weekly \
  --project=<PROJECT_ID> \
  --location=<REGION>
```

---

## 14. Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `models/prophet_strawberry.pkl` | File path to the serialized Prophet model |
| `DATA_PATH` | `data/strawberry_demand.csv` | File path to the historical demand CSV |
| `MODEL_DIR` | `models` | Directory for saving trained models + metrics |
| `PORT` | `8080` | Injected by Cloud Run; used by Uvicorn |

---

## 15. Troubleshooting

### Common Issues

| Issue | Cause | Fix |
|---|---|---|
| `503 Model not trained yet` | No `.pkl` file on disk | `POST /train` or rebuild the Docker image |
| Build timeout on Cloud Build | Prophet dependencies take long to install | Increase `--timeout` in `gcloud builds submit` (default 1200s) |
| OOM (Out of Memory) on Cloud Run | Prophet training exceeds allocated RAM | Increase `--memory` to `4Gi` |
| Cold start latency (~10–30s) | Container scales from 0; model loaded into memory | Set `--min-instances=1` to keep a warm instance |
| `Missing regressor column: X` | Future dataframe missing a required column | Ensure `build_future_features()` returns all 7 regressor columns |
| Incorrect holiday dates | Movable holidays (Easter, Thanksgiving) use fixed approximations | Update `_expand_holidays()` with correct dates per year |

### Viewing Logs

```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=strawberry-forecast" \
  --project=<PROJECT_ID> \
  --limit=50 \
  --format="table(timestamp, textPayload)"
```

### Updating the Model

To retrain with new data:
1. Update `data/strawberry_demand.csv` with fresh data rows.
2. Call `POST /train` to retrain in-place, **or**
3. Rebuild and redeploy the Docker image:
   ```bash
   bash deploy.sh <PROJECT_ID>
   ```

---

## Appendix: Dependencies

From `requirements.txt`:

| Package | Version | Purpose |
|---|---|---|
| `prophet` | ≥1.1.5, <2.0 | Core time-series model (Facebook/Meta) |
| `pandas` | ≥2.0, <3.0 | Data manipulation |
| `numpy` | ≥1.24, <2.0 | Numerical computing |
| `joblib` | ≥1.3, <2.0 | Model serialization (pickle wrapper) |
| `fastapi` | ≥0.110, <1.0 | Web framework for REST API |
| `uvicorn[standard]` | ≥0.27, <1.0 | ASGI server |
| `pydantic` | ≥2.0, <3.0 | Request/response validation |
| `pytest` | ≥8.0, <9.0 | Test runner |
| `httpx` | ≥0.27, <1.0 | Async HTTP client for FastAPI test client |
