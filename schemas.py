from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

RiskLevel = Literal["low", "moderate", "high"]


class StationInfo(BaseModel):
    station_id: str
    name: str
    river: str
    lat: float
    lon: float
    basin: str


class StationsResponse(BaseModel):
    stations: list[StationInfo]


class DataSources(BaseModel):
    rainfall: str = "Open-Meteo forecast API (local point + 9-point upstream basin grid, past 90 days)"
    soil_moisture: str = "Open-Meteo forecast API (0-7cm depth, local point)"
    discharge: str = "Open-Meteo flood API (GloFAS reanalysis/forecast)"


class HorizonRisk(BaseModel):
    horizon: Literal["24h", "48h", "72h"]
    risk_level: RiskLevel
    probability: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0, description="This horizon's tuned decision threshold (85% recall).")


class RiskResponse(BaseModel):
    station_id: str
    station_name: str
    station_distance_km: float
    basin: str
    horizons: list[HorizonRisk]
    # Mirrors the 24h horizon for callers that just want one headline number.
    risk_level: RiskLevel
    risk_score: float = Field(ge=0.0, le=1.0)
    generated_at: datetime
    reasoning: list[str]
    features_used: dict[str, float | bool | None]
    data_sources: DataSources


class ErrorResponse(BaseModel):
    detail: str
