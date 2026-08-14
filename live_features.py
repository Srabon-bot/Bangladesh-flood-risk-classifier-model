"""Live feature assembly -- builds the exact feature dict the trained
model needs, using LIVE current data from free Open-Meteo APIs.

History window: WEATHER_PAST_DAYS=90 is a deliberate margin (not the bare
minimum) so soil_moisture_swi's exponential filter has real lookback to
converge from a cold start; this is a close, but not bit-identical,
approximation of what the model saw in training (computed over each
station's full multi-decade history) -- a known, bounded approximation,
not hidden. DISCHARGE_PAST_DAYS=10 only needs to cover the deepest lag
used (5 days, or 3 for the Silchar reference).

Graceful degradation: a rainfall/soil-moisture fetch failure is fatal (no
"today" row to build anything from). A discharge fetch failure is NOT
fatal -- the model handles missing values natively (that's what the
*_missing flags are for), so a discharge-API outage degrades to a
rainfall/soil-moisture-only prediction instead of failing the whole request.
"""

from __future__ import annotations

import pandas as pd

from feature_transforms import add_lags_and_rolling, add_static_terrain_features
from open_meteo import OpenMeteoError, fetch_daily
from stations import (
    STATIONS_BY_ID,
    UP_SILCHAR_LAT,
    UP_SILCHAR_LON,
    UPSTREAM_CHAIN,
    UPSTREAM_REFERENCE_CHAIN,
    query_coords,
    upstream_points,
)

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
FLOOD_URL = "https://flood-api.open-meteo.com/v1/flood"
WEATHER_PAST_DAYS = 90
DISCHARGE_PAST_DAYS = 10

_UPSTREAM_LINK_COLUMNS = [
    "upstream_chain_discharge_lag1d",
    "upstream_chain_discharge_lag2d",
    "upstream_reference_discharge_lag2d",
    "upstream_reference_discharge_lag3d",
]


class LiveFeatureError(RuntimeError):
    pass


async def _fetch_weather(station) -> tuple[pd.DataFrame, pd.Series]:
    grid_points = upstream_points(station.basin)
    lats = [station.lat] + [p[0] for p in grid_points]
    lons = [station.lon] + [p[1] for p in grid_points]
    frames = await fetch_daily(
        WEATHER_URL, lats, lons,
        ["precipitation_sum", "soil_moisture_0_to_7cm_mean"], WEATHER_PAST_DAYS,
    )
    local = frames[0].rename(columns={
        "precipitation_sum": "rainfall_local_mm",
        "soil_moisture_0_to_7cm_mean": "soil_moisture_local",
    })
    upstream_rain = pd.concat([f["precipitation_sum"] for f in frames[1:]], axis=1).mean(axis=1)
    upstream_rain.name = "rainfall_upstream_mm"
    return local, upstream_rain


async def _fetch_discharge_points(points: list[tuple[str, float, float]]) -> dict[str, pd.Series]:
    lats = [p[1] for p in points]
    lons = [p[2] for p in points]
    frames = await fetch_daily(FLOOD_URL, lats, lons, ["river_discharge"], DISCHARGE_PAST_DAYS)
    return {points[i][0]: frames[i]["river_discharge"] for i in range(len(points))}


def _value_n_days_ago(series: pd.Series, n: int) -> float:
    """Returns float('nan'), never None -- a Python None in a single-row
    DataFrame infers `object` dtype instead of `float64`, which LightGBM
    rejects outright. This was a real bug, found by actually calling the
    model with a live-assembled row, not assumed."""
    if len(series) <= n:
        return float("nan")
    v = series.iloc[-1 - n]
    return float(v) if pd.notna(v) else float("nan")


async def build_live_feature_row(station_id: str) -> dict:
    """Builds today's feature row for one station. Raises LiveFeatureError
    -- and only LiveFeatureError -- on any failure."""
    if station_id not in STATIONS_BY_ID:
        raise LiveFeatureError(f"Unknown station_id {station_id!r}")
    station = STATIONS_BY_ID[station_id]

    try:
        try:
            local, upstream_rain = await _fetch_weather(station)
        except OpenMeteoError as exc:
            raise LiveFeatureError(f"weather fetch failed for {station_id}: {exc}") from exc
        if local.empty:
            raise LiveFeatureError(f"weather fetch returned no rows for {station_id}")

        df = pd.concat([local, upstream_rain], axis=1).reset_index()
        df["river_discharge_m3s"] = float("nan")

        link_values: dict[str, float] = dict.fromkeys(_UPSTREAM_LINK_COLUMNS, float("nan"))

        discharge_points: list[tuple[str, float, float]] = []
        own_lat, own_lon = query_coords(station)
        discharge_points.append(("own", own_lat, own_lon))

        chain_pair = UPSTREAM_CHAIN.get(station_id)
        if chain_pair is not None:
            up_id, chain_lag = chain_pair
            up_station = STATIONS_BY_ID.get(up_id)
            if up_station is None:
                chain_pair = None
            else:
                up_lat, up_lon = query_coords(up_station)
                discharge_points.append(("chain", up_lat, up_lon))

        ref_pair = UPSTREAM_REFERENCE_CHAIN.get(station_id)
        if ref_pair is not None:
            _, ref_lag = ref_pair
            discharge_points.append(("reference", UP_SILCHAR_LAT, UP_SILCHAR_LON))

        try:
            discharge_series = await _fetch_discharge_points(discharge_points)
        except OpenMeteoError:
            discharge_series = {}  # degrade, don't fail -- see module docstring

        if "own" in discharge_series:
            own = discharge_series["own"].rename("river_discharge_m3s")
            df = df.drop(columns=["river_discharge_m3s"]).merge(own.reset_index(), on="date", how="left")
        if "chain" in discharge_series and chain_pair is not None:
            link_values[f"upstream_chain_discharge_lag{chain_lag}d"] = _value_n_days_ago(
                discharge_series["chain"], chain_lag
            )
        if "reference" in discharge_series and ref_pair is not None:
            link_values[f"upstream_reference_discharge_lag{ref_lag}d"] = _value_n_days_ago(
                discharge_series["reference"], ref_lag
            )

        for col in ["rainfall_local_mm", "rainfall_upstream_mm", "soil_moisture_local", "river_discharge_m3s"]:
            df[f"{col}_missing"] = df[col].isna()

        df = add_lags_and_rolling(df)
        df = add_static_terrain_features(df, station_id)

        today_row = df.iloc[-1].to_dict()
        today_row.pop("date", None)
        today_row.update(link_values)
        return today_row
    except LiveFeatureError:
        raise
    except Exception as exc:  # noqa: BLE001 -- deliberate catch-all, see module docstring
        raise LiveFeatureError(f"unexpected error building live features for {station_id}: {exc}") from exc
