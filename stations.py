"""The 30 monitoring stations this model was trained on, plus the static/
geographic constants needed to assemble live features. This is a trimmed,
standalone copy of the research repo's train/stations.py -- same data,
same station IDs (must match exactly, since the model was trained on these
specific station_id category codes), just without the research-narrative
comments that only make sense in the context of the full data pipeline.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Station:
    station_id: str
    name: str
    river: str
    lat: float
    lon: float
    basin: str  # "brahmaputra" | "ganges" | "meghna" | "cht"


STATIONS: list[Station] = [
    Station("SW90", "Bahadurabad", "Jamuna", 25.1897, 89.6595, "brahmaputra"),
    Station("SW93", "Sariakandi", "Jamuna", 24.8952, 89.5975, "brahmaputra"),
    Station("SW99", "Sirajganj", "Jamuna", 24.4534, 89.7009, "brahmaputra"),
    Station("SW17", "Chilmari", "Brahmaputra", 25.5333, 89.6833, "brahmaputra"),
    Station("TE01", "Dalia (Teesta Barrage)", "Teesta", 25.9167, 89.2833, "brahmaputra"),
    Station("TE02", "Kaunia", "Teesta", 25.7667, 89.4333, "brahmaputra"),
    Station("OB01", "Mymensingh", "Old Brahmaputra", 24.7471, 90.4203, "brahmaputra"),
    Station("DH01", "Kurigram (Dharla)", "Dharla", 25.8058, 89.6698, "brahmaputra"),
    Station("GA01", "Hardinge Bridge (Kushtia)", "Ganges", 24.0708, 89.0294, "ganges"),
    Station("GA02", "Goalanda (Jamuna-Ganges confluence)", "Padma", 23.7167, 89.7833, "ganges"),
    Station("GA03", "Mawa", "Padma", 23.4342, 90.2650, "ganges"),
    Station("GA04", "Bhagyakul (Munshiganj)", "Padma", 23.5167, 90.2667, "ganges"),
    Station("GO01", "Kamarkhali (Kushtia)", "Gorai", 23.6333, 89.4500, "ganges"),
    Station("GO02", "Gopalganj", "Madhumati", 23.0167, 89.8265, "ganges"),
    Station("ME01", "Chandpur (Padma-Meghna confluence)", "Meghna", 23.2333, 90.6667, "meghna"),
    Station("ME02", "Bhairab Bazar", "Meghna", 24.0500, 90.9833, "meghna"),
    Station("ME03", "Dhaka (Buriganga/Dhaleshwari)", "Dhaleshwari", 23.7000, 90.3667, "meghna"),
    Station("SW267", "Sunamganj", "Surma", 25.0658, 91.3950, "meghna"),
    Station("SW174", "Sylhet", "Surma", 24.8949, 91.8687, "meghna"),
    Station("KU01", "Sherpur (Sylhet)", "Kushiyara", 24.6833, 91.8667, "meghna"),
    Station("KU02", "Amalshid", "Kushiyara", 24.8167, 92.2000, "meghna"),
    Station("NM01", "Durgapur (Netrokona)", "Someshwari", 25.1500, 90.7333, "meghna"),
    Station("CH01", "Rangamati (Karnaphuli)", "Karnaphuli", 22.6333, 92.1833, "cht"),
    Station("CH02", "Bandarban (Sangu)", "Sangu", 22.1953, 92.2183, "cht"),
    Station("CH03", "Chittagong (Halda)", "Halda", 22.5000, 91.8500, "cht"),
    Station("CO01", "Barisal", "Kirtankhola", 22.7010, 90.3535, "ganges"),
    Station("CO02", "Khulna (Rupsha)", "Rupsha", 22.8456, 89.5403, "ganges"),
    Station("CO03", "Bagerhat", "Baleswar", 22.6602, 89.7895, "ganges"),
    Station("CO04", "Patuakhali", "Payra", 22.3596, 90.3296, "ganges"),
    Station("CO05", "Cox's Bazar", "Bakkhali", 21.4272, 92.0058, "cht"),
]

# Static terrain features from MERIT Hydro (elevation_m, hand_m -- Height
# Above Nearest Drainage), sampled once per station. Constant, never fetched
# live.
STATIC_TERRAIN: dict[str, tuple[float, float]] = {
    "SW90": (21.2, 3.10), "SW93": (15.4, 2.70), "SW99": (18.1, 1.00), "SW17": (16.4, 1.50),
    "TE01": (35.4, 0.70), "TE02": (31.1, 1.20), "OB01": (18.4, 13.30), "DH01": (23.2, 0.00),
    "GA01": (2.5, 0.00), "GA02": (11.1, 0.50), "GA03": (0.0, 0.00), "GA04": (6.6, 1.00),
    "GO01": (12.3, 12.10), "GO02": (5.2, 0.70), "ME01": (10.5, 0.60), "ME02": (7.7, 0.00),
    "ME03": (4.2, 0.00), "SW267": (12.6, 4.00), "SW174": (19.3, 2.40), "KU01": (9.8, 1.40),
    "KU02": (12.8, 0.00), "NM01": (16.7, 1.80), "CH01": (35.8, 7.80), "CH02": (22.1, 1.40),
    "CH03": (5.0, 0.80), "CO01": (7.3, 0.40), "CO02": (4.8, 0.00), "CO03": (6.7, 0.00),
    "CO04": (4.8, 0.10), "CO05": (10.8, 6.50),
}

# Upstream catchment sample-grid boxes, one per basin (lat_min, lat_max,
# lon_min, lon_max), used to approximate an area-mean upstream rainfall.
UPSTREAM_BOXES: dict[str, tuple[float, float, float, float]] = {
    "brahmaputra": (26.0, 29.5, 90.0, 96.0),
    "ganges": (24.0, 27.0, 84.0, 89.0),
    "meghna": (24.0, 26.0, 91.5, 93.0),
    "cht": (21.5, 23.5, 92.0, 93.5),
}

# Upstream India-side discharge reference point (Silchar, on the Barak,
# which becomes the Surma/Kushiyara in Bangladesh). NOT one of the 30
# stations -- exists only to supply an upstream discharge feature for the
# stations below.
UP_SILCHAR_LAT, UP_SILCHAR_LON = 24.8333, 92.7789

# station_id -> (reference station lag_days). Which stations get the
# Silchar upstream-discharge feature, and at what lag.
UPSTREAM_REFERENCE_CHAIN: dict[str, tuple[str, int]] = {
    "SW174": ("UP_SILCHAR", 2),
    "SW267": ("UP_SILCHAR", 2),
    "KU01": ("UP_SILCHAR", 2),
    "KU02": ("UP_SILCHAR", 2),
    "ME02": ("UP_SILCHAR", 3),
}

# station_id -> (upstream in-network station_id, travel_time_lag_days).
UPSTREAM_CHAIN: dict[str, tuple[str, int]] = {
    "SW90": ("SW17", 1),
    "SW93": ("SW90", 1),
    "SW99": ("SW93", 1),
    "TE02": ("TE01", 1),
    "GA02": ("GA01", 1),
    "GA03": ("GA02", 1),
    "GA04": ("GA02", 1),
    "ME01": ("GA03", 1),
    "ME02": ("KU01", 2),
}

# GloFAS (the discharge reanalysis grid) is ~0.05 deg resolution and can
# land on the wrong channel near confluences for a handful of stations --
# these coordinates were manually verified against known discharge
# magnitudes and should be used INSTEAD of the station's own lat/lon when
# querying the discharge API specifically (not for anything else).
GLOFAS_COORD_OVERRIDE: dict[str, tuple[float, float]] = {
    "SW93": (24.745, 89.647), "SW174": (24.745, 91.919), "TE01": (25.767, 89.433),
    "GA01": (24.071, 88.979), "GA02": (23.767, 89.733), "GA04": (23.467, 90.117),
    "GO02": (22.867, 89.876), "ME01": (23.233, 90.567), "ME02": (24.100, 91.033),
    "KU02": (24.867, 92.050), "NM01": (25.000, 90.583), "CH01": (22.483, 92.033),
    "CH02": (22.195, 92.068), "CH03": (22.350, 91.750), "CO01": (22.551, 90.504),
    "CO02": (22.746, 89.490), "CO03": (22.760, 89.940), "CO04": (22.210, 90.180),
    "CO05": (21.477, 91.906),
}


def query_coords(station: Station) -> tuple[float, float]:
    """Coordinates to use for DISCHARGE queries specifically (applies the
    GloFAS grid-cell correction above where needed). Use station.lat/lon
    directly for rainfall/soil-moisture queries."""
    return GLOFAS_COORD_OVERRIDE.get(station.station_id, (station.lat, station.lon))


def upstream_points(basin: str, n: int = 3) -> list[tuple[float, float]]:
    """n x n sample grid across a basin's upstream box, used as a coarse
    area-mean approximation for upstream rainfall."""
    lat_min, lat_max, lon_min, lon_max = UPSTREAM_BOXES[basin]
    lats = [lat_min + i * (lat_max - lat_min) / (n - 1) for i in range(n)]
    lons = [lon_min + j * (lon_max - lon_min) / (n - 1) for j in range(n)]
    return [(lat, lon) for lat in lats for lon in lons]


STATIONS_BY_ID = {s.station_id: s for s in STATIONS}
