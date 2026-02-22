# Strawberry Demand Forecast — Weekly Prophet Model

Production-grade **Facebook Prophet** pipeline that forecasts **weekly strawberry
`units_sold` 3 weeks ahead**, deployed on **GCP Cloud Run** and triggered every
**Monday at 06:00 UTC** via Cloud Scheduler.

---

## Architecture

```
┌──────────────┐   Cloud Scheduler   ┌──────────────────┐
│  Mon 06:00   │ ──── POST /predict ──▶│  Cloud Run       │
│  UTC cron    │                      │  (FastAPI + Prophet) │
└──────────────┘                      └────────┬─────────┘
                                               │
                                       ┌───────▼────────┐
                                       │  prophet_model  │
                                       │  .pkl (2 GB)    │
                                       └────────────────┘
```

## Features Engineered

| Feature | Type | Source |
|---------|------|--------|
| `avg_temp_f` | continuous | Weekly average temperature (°F) |
| `precip_inches` | continuous | Weekly total precipitation |
| `avg_price_usd` | continuous | Retail price per lb |
| `is_promo` | binary | Promotional campaign flag |
| `holiday_window` | binary | Overlaps US holiday demand spike |
| `month_sin` / `month_cos` | cyclical | Fourier-encoded month |
| US Holidays (10) | Prophet holidays | With pre/post windows |
| Yearly seasonality | built-in | 10 Fourier terms |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate mock dataset
python -m src.generate_dataset

# 3. Train model + backtest + quality gate
python -m src.train

# 4. Run predictions (CLI)
python -m src.predict

# 5. Start API server locally
uvicorn src.app:app --reload --port 8080

# 6. Run full test suite
pytest
```

## Train / Test Split & Back-testing Strategy

| Set | Period | Weeks |
|-----|--------|-------|
| **Train** | 2019-01-07 → 2024-12-30 | ~313 |
| **Hold-out** | 2025-01-06 → 2025-12-29 | ~52 |
| **CV (back-test)** | 5-fold expanding window, 3-week horizon | — |

Cross-validation uses `initial=1092 days` (~3 yr), `period=182 days` (~26 wk),
`horizon=21 days` (3 wk) to simulate realistic rolling retrains.

## Quality Gates (must pass before deploy)

| Metric | Threshold |
|--------|-----------|
| MAPE | ≤ 12 % |
| RMSE | ≤ 650 units |
| MAE | ≤ 500 units |
| R² | ≥ 0.85 |

## Deploy to GCP Cloud Run

```bash
bash deploy.sh <GCP_PROJECT_ID> [REGION]
```

This script:
1. Builds & pushes the container via Cloud Build
2. Deploys to Cloud Run (2 vCPU / 2 GB RAM)
3. Creates a Cloud Scheduler job: `0 6 * * 1` (Monday 06:00 UTC)

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness + model status |
| `POST` | `/predict` | 3-week forecast `{"horizon": 3}` |
| `POST` | `/train` | Retrain model on latest data |

## Test Suite

```
pytest -v
```

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
