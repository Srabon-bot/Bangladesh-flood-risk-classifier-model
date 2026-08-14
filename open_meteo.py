"""Low-level client for Open-Meteo's live forecast/flood APIs -- free, no
API key required. Uses the regular forecast/flood endpoints (which return
real values through today), not the historical archive endpoint (which
lags several days behind "today").

Supports multi-point requests (comma-separated lat/lon) -- Open-Meteo
returns a JSON array of per-point objects for these, normalized here into
a uniform list of DataFrames.
"""

from __future__ import annotations

import httpx
import pandas as pd


class OpenMeteoError(RuntimeError):
    pass


async def fetch_daily(
    base_url: str,
    lats: list[float],
    lons: list[float],
    daily_vars: list[str],
    past_days: int,
    forecast_days: int = 1,
    timeout: float = 15.0,
) -> list[pd.DataFrame]:
    """One HTTP request for 1..N points. Returns one DataFrame per point (in
    the same order as lats/lons), indexed by `date`, one column per
    variable in `daily_vars`.

    forecast_days defaults to 1, not 0 -- Open-Meteo's `past_days` window
    alone ends YESTERDAY, not today; forecast_days=1 is what actually
    includes today's (blended observed-so-far/nowcast) value. This was a
    real bug in an earlier version of this pipeline, found by checking the
    actual dates returned, not assumed -- keep forecast_days >= 1 unless
    you have verified otherwise.

    Every failure mode here raises OpenMeteoError and only OpenMeteoError.
    """
    if len(lats) != len(lons):
        raise OpenMeteoError(f"lats/lons length mismatch: {len(lats)} vs {len(lons)}")
    if not lats:
        raise OpenMeteoError("fetch_daily called with zero points")

    params = {
        "latitude": ",".join(str(x) for x in lats),
        "longitude": ",".join(str(x) for x in lons),
        "daily": ",".join(daily_vars),
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": "UTC",
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(base_url, params=params)
            resp.raise_for_status()
            payload = resp.json()

        records = payload if isinstance(payload, list) else [payload]
        if len(records) != len(lats):
            raise OpenMeteoError(
                f"Expected {len(lats)} points in Open-Meteo response, got {len(records)} ({base_url})"
            )

        frames = []
        for rec in records:
            if not isinstance(rec, dict):
                raise OpenMeteoError(f"Unexpected non-object record in Open-Meteo response ({base_url}): {rec!r}")
            daily = rec.get("daily")
            if not isinstance(daily, dict):
                raise OpenMeteoError(f"Response for a point had no 'daily' block ({base_url})")
            dates = daily.get("time", [])
            if not isinstance(dates, list):
                raise OpenMeteoError(f"'daily.time' was not a list ({base_url})")
            df = pd.DataFrame({"date": pd.to_datetime(dates)})
            for var in daily_vars:
                values = daily.get(var)
                if not isinstance(values, list) or len(values) != len(dates):
                    values = [None] * len(dates)
                df[var] = values
            frames.append(df.set_index("date"))
        return frames
    except OpenMeteoError:
        raise
    except Exception as exc:  # noqa: BLE001 -- deliberately broad, see module docstring
        raise OpenMeteoError(
            f"Open-Meteo request/parse failed ({base_url}): {type(exc).__name__}: {exc}"
        ) from exc
