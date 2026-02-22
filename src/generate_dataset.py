"""
generate_dataset.py
====================
Creates a realistic mock weekly strawberry-demand dataset (2019-01-07 → 2025-12-29).
Each row = one ISO week (Monday date) with:
  - units_sold          : target
  - avg_temp_f          : average weekly temperature (°F)
  - precip_inches       : weekly precipitation (inches)
  - is_promo            : 1 if a promotional campaign ran that week
  - avg_price_usd       : average retail price per lb
  - holiday_window      : 1 if the week overlaps a major US holiday window
  - month_sin / month_cos : cyclical month encoding
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
START = "2019-01-07"
END = "2025-12-29"
FREQ = "W-MON"  # weekly on Monday

# US holiday windows that spike strawberry demand (week-of + 1 week before)
HOLIDAY_WINDOWS: list[tuple[int, int]] = [
    (2, 6),    # Super Bowl (late Jan / early Feb)
    (5, 9),    # Mother's Day / Easter window
    (18, 22),  # Memorial Day / early June
    (23, 27),  # July 4th window
    (35, 37),  # Labor Day
    (44, 48),  # Thanksgiving + Christmas lead-up
    (49, 52),  # Christmas / New Year
]


def _is_holiday_window(week_of_year: int) -> int:
    return int(any(lo <= week_of_year <= hi for lo, hi in HOLIDAY_WINDOWS))


def generate(dest: str | Path = "data/strawberry_demand.csv") -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    dates = pd.date_range(START, END, freq=FREQ)

    n = len(dates)
    week_of_year = dates.isocalendar().week.values.astype(int)
    month = dates.month

    # --- Weather features (seasonal, noisy) ---
    # Temperature peaks in July (~week 27), trough in Jan
    base_temp = 55 + 25 * np.sin(2 * np.pi * (week_of_year - 12) / 52)
    avg_temp_f = base_temp + rng.normal(0, 4, n)

    # Precipitation – higher in spring/fall
    base_precip = 0.8 + 0.5 * np.sin(2 * np.pi * (week_of_year - 8) / 52)
    precip_inches = np.clip(base_precip + rng.normal(0, 0.3, n), 0, None)

    # --- Promotional flag (~15 % of weeks, clustered) ---
    is_promo = (rng.random(n) < 0.15).astype(int)

    # --- Price (inversely correlated with supply peak in spring/summer) ---
    base_price = 3.50 - 0.80 * np.sin(2 * np.pi * (week_of_year - 14) / 52)
    avg_price_usd = np.round(base_price + rng.normal(0, 0.20, n), 2)

    # --- Holiday window ---
    holiday_window = np.array([_is_holiday_window(w) for w in week_of_year])

    # --- Cyclical month encoding ---
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    # ====== TARGET: units_sold ======
    # Base demand with yearly seasonality (peak May-Jul)
    base_demand = 5000 + 3000 * np.sin(2 * np.pi * (week_of_year - 10) / 52)
    # Trend: +1.5 % year-over-year growth
    year_idx = (dates - dates[0]).days / 365.25
    trend = 1 + 0.015 * year_idx
    # Weather effect: warmer → more demand, heavy rain → less
    weather_effect = 15 * (avg_temp_f - 55) - 400 * precip_inches
    # Promo boost ~+800 units
    promo_effect = 800 * is_promo
    # Holiday boost ~+600 units
    holiday_effect = 600 * holiday_window
    # Price elasticity: cheaper → more demand
    price_effect = -1200 * (avg_price_usd - 3.50)
    # Combine
    units_sold = (
        base_demand * trend
        + weather_effect
        + promo_effect
        + holiday_effect
        + price_effect
        + rng.normal(0, 350, n)
    )
    units_sold = np.round(np.clip(units_sold, 200, None)).astype(int)

    df = pd.DataFrame({
        "ds": dates,
        "units_sold": units_sold,
        "avg_temp_f": np.round(avg_temp_f, 1),
        "precip_inches": np.round(precip_inches, 2),
        "is_promo": is_promo,
        "avg_price_usd": avg_price_usd,
        "holiday_window": holiday_window,
        "month_sin": np.round(month_sin, 4),
        "month_cos": np.round(month_cos, 4),
    })

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    print(f"✔ Dataset saved → {dest}  ({len(df)} rows, {df.ds.min().date()} – {df.ds.max().date()})")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default="data/strawberry_demand.csv")
    args = parser.parse_args()
    generate(args.dest)
