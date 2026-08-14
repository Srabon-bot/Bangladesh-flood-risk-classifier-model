"""Loads the 3 trained LightGBM flood-risk models (24h/48h/72h) and their
tuned decision thresholds, and serves predictions from an already-
assembled feature dict (see live_features.py for how to build one).

CASCADE FEATURE: each horizon's model was retrained with one extra feature,
discharge_forecast_<horizon> -- a companion discharge-forecasting model's
OWN prediction for that same horizon, not the classifier's own past-
discharge features (those stay too; this is additional). A/B-tested on
held-out data: positive on PR-AUC at every horizon when each horizon gets
only its own matching forecast, not the other two horizons'. See
models/discharge_cascade/ -- a bundled, self-contained copy of that
discharge model (NOT a live HTTP call to any external service, so this
package has no runtime dependency on anything outside this folder).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from stations import STATIONS

MODELS_DIR = Path(__file__).resolve().parent / "models"
DISCHARGE_CASCADE_DIR = MODELS_DIR / "discharge_cascade"

_STATION_BASIN = {s.station_id: s.basin for s in STATIONS}

BASIN_LABELS = {
    "brahmaputra": "Brahmaputra/Jamuna",
    "meghna": "Surma-Meghna",
    "ganges": "Ganges-Padma",
    "cht": "Chittagong Hill Tracts",
}


@dataclass
class HorizonPrediction:
    horizon: str          # "24h" | "48h" | "72h"
    probability: float    # P(flood within this horizon)
    threshold: float      # this horizon's tuned decision threshold (85% recall)
    risk_level: str       # "low" | "moderate" | "high"


class FloodRiskModel:
    """Usage:
        model = FloodRiskModel.load()
        predictions = model.predict(features, station_id="SW90", prediction_date=pd.Timestamp.now())
    """

    def __init__(self, root: Path):
        self.schema = json.loads((root / "feature_schema.json").read_text())
        # feature_columns_per_horizon (2026-08-14): each horizon has its OWN
        # column list now (one extra discharge_forecast_<horizon> cascade
        # column each) -- no longer one shared list. Falls back to the old
        # flat "feature_columns" key so a pre-cascade model directory (if
        # one is ever loaded) doesn't hard-crash, just runs without the
        # cascade feature.
        if "feature_columns_per_horizon" in self.schema:
            self.feature_columns_per_horizon: dict[str, list[str]] = self.schema["feature_columns_per_horizon"]
        else:
            flat = self.schema["feature_columns"]
            self.feature_columns_per_horizon = {h: flat for h in self.schema["horizons"]}
        self.categorical_values: dict[str, list[str]] = self.schema["categorical_values"]
        self.horizons: list[str] = self.schema["horizons"]

        self._models: dict[str, object] = {}
        self._thresholds: dict[str, float] = {}
        for horizon in self.horizons:
            self._models[horizon] = joblib.load(root / f"model_{horizon}.joblib")
            threshold_info = json.loads((root / f"model_{horizon}_threshold.json").read_text())
            self._thresholds[horizon] = threshold_info["threshold"]

        # Bundled discharge-forecaster copy for the cascade feature -- see
        # module docstring for why this is a bundled copy, not a live call
        # to the other service. Missing directory degrades gracefully (the
        # discharge_forecast_<horizon> column just won't be available,
        # which _build_row surfaces as a clear error rather than a silent
        # wrong prediction) rather than crashing package import.
        self._discharge_models: dict[str, object] = {}
        self._discharge_fit_cols: list[str] | None = None
        if DISCHARGE_CASCADE_DIR.exists():
            discharge_schema = json.loads((DISCHARGE_CASCADE_DIR / "feature_schema.json").read_text())
            self._discharge_fit_cols = discharge_schema["feature_columns"]
            for horizon in self.horizons:
                self._discharge_models[horizon] = joblib.load(DISCHARGE_CASCADE_DIR / f"model_{horizon}.joblib")

    @classmethod
    def load(cls, models_dir: Path | None = None) -> "FloodRiskModel":
        root = models_dir or MODELS_DIR
        if not root.exists():
            raise FileNotFoundError(f"No model artifacts at {root}")
        return cls(root)

    def _compute_discharge_forecasts(self, features: dict, station_id: str, doy_sin: float, doy_cos: float) -> dict[str, float]:
        """Runs the bundled discharge-forecaster models on the SAME base
        feature dict the classifier itself uses -- safe because both
        models' feature schemas are byte-identical (verified directly:
        same feature-engineering pipeline, same 42 base columns, same
        station_id/basin categorical encoding)."""
        if not self._discharge_models:
            raise RuntimeError(
                f"Cascade feature requires the bundled discharge model at {DISCHARGE_CASCADE_DIR}, "
                "which is missing. Re-copy it from the discharge-regression model directory."
            )
        row = dict(features)
        row["station_id"] = station_id
        row["basin"] = _STATION_BASIN[station_id]
        row["doy_sin"] = doy_sin
        row["doy_cos"] = doy_cos
        df = pd.DataFrame([row])[self._discharge_fit_cols]
        df["station_id"] = pd.Categorical(df["station_id"], categories=self.categorical_values["station_id"])
        df["basin"] = pd.Categorical(df["basin"], categories=self.categorical_values["basin"])

        forecasts = {}
        for horizon, dmodel in self._discharge_models.items():
            pred_log = dmodel.predict(df)[0]
            forecasts[horizon] = max(0.0, float(math.expm1(pred_log)))
        return forecasts

    def _build_row(self, features: dict, station_id: str, prediction_date: pd.Timestamp, horizon: str,
                    discharge_forecasts: dict[str, float]) -> pd.DataFrame:
        feature_columns = self.feature_columns_per_horizon[horizon]
        derived = {"station_id", "basin", "doy_sin", "doy_cos"} | {f"discharge_forecast_{h}" for h in self.horizons}
        missing_keys = [c for c in feature_columns if c not in features and c not in derived]
        if missing_keys:
            raise ValueError(f"Missing required feature keys (NaN values are fine, missing KEYS are not): {missing_keys}")
        if station_id not in self.categorical_values["station_id"]:
            raise ValueError(f"Unknown station_id {station_id!r}, expected one of {self.categorical_values['station_id']}")

        row = dict(features)
        row["station_id"] = station_id
        row["basin"] = _STATION_BASIN[station_id]

        doy = prediction_date.dayofyear
        days_in_year = 366 if prediction_date.is_leap_year else 365
        angle = 2 * math.pi * doy / days_in_year
        row["doy_sin"] = math.sin(angle)
        row["doy_cos"] = math.cos(angle)

        # Only this horizon's own matching cascade column is ever consumed
        # (feature_columns already excludes the other two horizons' -- see
        # train_model.py's feature_columns()), but set all 3 on the row dict
        # harmlessly, simpler than conditionally picking one.
        for h, val in discharge_forecasts.items():
            row[f"discharge_forecast_{h}"] = val

        df = pd.DataFrame([row])[feature_columns]
        # Explicit category lists pinned to what training used -- a
        # single-row frame with an implicit astype("category") would
        # otherwise assign category code 0 regardless of the actual
        # station, silently corrupting the prediction.
        df["station_id"] = pd.Categorical(df["station_id"], categories=self.categorical_values["station_id"])
        df["basin"] = pd.Categorical(df["basin"], categories=self.categorical_values["basin"])
        return df

    def predict(self, features: dict, station_id: str, prediction_date: pd.Timestamp) -> list[HorizonPrediction]:
        doy = prediction_date.dayofyear
        days_in_year = 366 if prediction_date.is_leap_year else 365
        angle = 2 * math.pi * doy / days_in_year
        discharge_forecasts = self._compute_discharge_forecasts(
            features, station_id, math.sin(angle), math.cos(angle)
        )
        results = []
        for horizon in self.horizons:
            row = self._build_row(features, station_id, prediction_date, horizon, discharge_forecasts)
            proba = float(self._models[horizon].predict_proba(row)[:, 1][0])
            threshold = self._thresholds[horizon]
            results.append(HorizonPrediction(
                horizon=horizon, probability=proba, threshold=threshold,
                risk_level=self._score_to_level(proba, threshold),
            ))
        return results

    @staticmethod
    def _score_to_level(proba: float, threshold: float) -> str:
        if proba >= threshold:
            return "high"
        if proba >= threshold / 2:
            return "moderate"
        return "low"


def _is_nan(v) -> bool:
    return isinstance(v, float) and math.isnan(v)


def build_reasoning(features: dict, predictions: list[HorizonPrediction], basin: str | None, station_name: str | None) -> list[str]:
    """Plain-language explanation grounded in the same features SHAP found
    most important in training (14-day cumulative local rainfall, season,
    soil moisture)."""
    reasons = []
    basin_label = BASIN_LABELS.get(basin) if basin else None
    if basin_label and station_name:
        reasons.append(
            f"Nearest gauge: {station_name} ({basin_label} basin) -- flooding here is often driven by "
            "upstream monsoon rainfall in India/Nepal as much as local rain."
        )

    sum14d = features.get("rainfall_local_mm_sum14d")
    if sum14d is not None and not _is_nan(sum14d):
        if sum14d > 150:
            reasons.append(f"Heavy local rainfall over the past 14 days ({sum14d:.0f} mm).")
        elif sum14d > 60:
            reasons.append(f"Moderate local rainfall over the past 14 days ({sum14d:.0f} mm).")
        else:
            reasons.append(f"Local rainfall over the past 14 days has been light ({sum14d:.0f} mm).")

    trend = features.get("rainfall_local_mm_trend_ratio")
    if trend is not None and not _is_nan(trend):
        if trend > 1.5:
            reasons.append("Rainfall has been intensifying compared to the prior week.")
        elif trend < 0.5:
            reasons.append("Rainfall has been easing compared to the prior week.")

    delta30 = features.get("soil_moisture_delta_30d")
    if delta30 is not None and not _is_nan(delta30):
        if delta30 > 0.03:
            reasons.append("Soil is notably wetter than it was a month ago, meaning less capacity to absorb more rain.")
        elif delta30 < -0.03:
            reasons.append("Soil is drier than it was a month ago.")

    discharge = features.get("river_discharge_m3s")
    if features.get("river_discharge_m3s_missing"):
        reasons.append("Live river discharge data was unavailable for this forecast; relying on rainfall and soil moisture only.")
    elif discharge is not None and not _is_nan(discharge):
        reasons.append(f"Current river discharge near this gauge is {discharge:,.0f} m3/s.")

    horizon_bits = ", ".join(f"{p.horizon} {p.risk_level}" for p in predictions)
    reasons.append(f"Risk by horizon: {horizon_bits}.")
    return reasons
