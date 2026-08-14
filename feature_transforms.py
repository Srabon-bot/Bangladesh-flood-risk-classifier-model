"""Feature transformations -- MUST exactly match what the model was
trained on (see the research repo's train/build_features.py, which this is
a trimmed, standalone copy of). If you change these, the model's
predictions become meaningless, since it was trained on the ORIGINAL
definitions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stations import STATIC_TERRAIN

LAG_DAYS = [1, 2, 3, 5]
ROLLING_WINDOWS = [7, 14]
SOIL_TREND_DAYS = 30
SWI_HALFLIFE_DAYS = 10


def add_lags_and_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """df must have columns: date, rainfall_local_mm, rainfall_upstream_mm,
    soil_moisture_local, river_discharge_m3s -- sorted or not, this sorts
    by date itself."""
    df = df.sort_values("date").reset_index(drop=True)
    base_cols = ["rainfall_local_mm", "rainfall_upstream_mm", "soil_moisture_local", "river_discharge_m3s"]

    for col in base_cols:
        for lag in LAG_DAYS:
            df[f"{col}_lag{lag}d"] = df[col].shift(lag)

    for col in ["rainfall_local_mm", "rainfall_upstream_mm"]:
        for window in ROLLING_WINDOWS:
            df[f"{col}_sum{window}d"] = df[col].rolling(window, min_periods=1).sum()
        recent7 = df[col].rolling(7, min_periods=1).sum()
        prior7 = df[col].shift(7).rolling(7, min_periods=1).sum()
        df[f"{col}_trend_ratio"] = recent7 / prior7.replace(0, np.nan)

    df["soil_moisture_delta_30d"] = df["soil_moisture_local"] - df["soil_moisture_local"].shift(SOIL_TREND_DAYS)

    # Soil Wetness Index -- exponential recursive filter over raw surface
    # soil moisture (Wagner et al. 1999), a root-zone-like antecedent-
    # wetness proxy. pandas' .ewm(halflife=T) computes this directly.
    df["soil_moisture_swi"] = df["soil_moisture_local"].ewm(halflife=SWI_HALFLIFE_DAYS, adjust=False).mean()

    return df


def add_static_terrain_features(df: pd.DataFrame, station_id: str) -> pd.DataFrame:
    elevation_m, hand_m = STATIC_TERRAIN[station_id]
    df["elevation_m"] = elevation_m
    df["hand_m"] = hand_m
    return df
