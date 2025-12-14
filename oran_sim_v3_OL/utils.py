import logging
import sys
from typing import Tuple, List, Dict, Any, Optional, Set
from datetime import datetime
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

EARTH_RADIUS_M = 6371000.0
EARTH_RADIUS_KM = 6371.0

HAP_DTYPE = np.dtype([
    ('hap_id', np.int64),
    ('lat', np.float64),
    ('lon', np.float64),
    ('altitude', np.float64),
    ('max_speed', np.float64),
    ('coverage_radius', np.float64),
    ('target_lat', np.float64),
    ('target_lon', np.float64),
], align=True)

SHIP_DTYPE = np.dtype([
    ('mmsi', np.int64),
    ('lat', np.float64),
    ('lon', np.float64),
    ('passengers', np.int32),
    ('crew', np.int32),
], align=True)


def haps_df_to_array(df: pl.DataFrame) -> np.ndarray:
    n = df.height
    if n == 0:
        return np.empty(0, dtype=HAP_DTYPE)
    arr = np.empty(n, dtype=HAP_DTYPE)
    arr['hap_id'] = df['hap_id'].to_numpy()
    arr['lat'] = df['lat'].to_numpy()
    arr['lon'] = df['lon'].to_numpy()
    arr['altitude'] = df['altitude'].to_numpy()
    arr['max_speed'] = df['max_speed'].to_numpy()
    arr['coverage_radius'] = df['coverage_radius'].to_numpy()
    arr['target_lat'] = df['target_lat'].to_numpy()
    arr['target_lon'] = df['target_lon'].to_numpy()
    return arr


def ships_df_to_array(df: pl.DataFrame) -> np.ndarray:
    n = df.height
    if n == 0:
        return np.empty(0, dtype=SHIP_DTYPE)
    arr = np.empty(n, dtype=SHIP_DTYPE)
    arr['mmsi'] = df['mmsi'].to_numpy()
    arr['lat'] = df['lat'].to_numpy()
    arr['lon'] = df['lon'].to_numpy()
    arr['passengers'] = df['passengers'].fill_null(0).to_numpy()
    arr['crew'] = df['crew'].fill_null(0).to_numpy()
    return arr


def haversine_distance_km_vectorized(
    lat1: np.ndarray, lon1: np.ndarray,
    lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    lat1_r = np.radians(lat1)
    lon1_r = np.radians(lon1)
    lat2_r = np.radians(lat2)
    lon2_r = np.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return EARTH_RADIUS_KM * c


def compute_slant_range_km_vectorized(
    ship_lat: np.ndarray, ship_lon: np.ndarray,
    hap_lat: np.ndarray, hap_lon: np.ndarray, hap_alt_m: np.ndarray
) -> np.ndarray:
    lat1_r = np.radians(ship_lat)
    lon1_r = np.radians(ship_lon)
    lat2_r = np.radians(hap_lat)
    lon2_r = np.radians(hap_lon)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    beta = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    R = EARTH_RADIUS_KM
    h = hap_alt_m / 1000.0
    slant_range = np.sqrt(R ** 2 + (R + h) ** 2 - 2 * R * (R + h) * np.cos(beta))
    return slant_range


def setup_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def get_timestamp() -> str:
    return datetime.now().isoformat()


def pl_haversine_dist(lat1: pl.Expr, lon1: pl.Expr, lat2: pl.Expr, lon2: pl.Expr) -> pl.Expr:
    lat1_rad = lat1.radians()
    lon1_rad = lon1.radians()
    lat2_rad = lat2.radians()
    lon2_rad = lon2.radians()
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (dlat / 2).sin().pow(2) + lat1_rad.cos() * lat2_rad.cos() * (dlon / 2).sin().pow(2)
    c = 2 * a.sqrt().arcsin()
    return c * EARTH_RADIUS_M


def pl_bearing(lat1: pl.Expr, lon1: pl.Expr, lat2: pl.Expr, lon2: pl.Expr) -> pl.Expr:
    lat1_rad = lat1.radians()
    lat2_rad = lat2.radians()
    dlon_rad = (lon2 - lon1).radians()
    y = dlon_rad.sin() * lat2_rad.cos()
    x = lat1_rad.cos() * lat2_rad.sin() - lat1_rad.sin() * lat2_rad.cos() * dlon_rad.cos()
    bearing = np.arctan2(y, x).degrees()
    return (bearing + 360) % 360


def pl_destination_point(lat: pl.Expr, lon: pl.Expr, bearing: pl.Expr, distance: pl.Expr) -> Tuple[pl.Expr, pl.Expr]:
    lat_rad = lat.radians()
    lon_rad = lon.radians()
    bearing_rad = bearing.radians()
    ang_dist = distance / EARTH_RADIUS_M
    lat2_rad = (lat_rad.sin() * ang_dist.cos() +
                lat_rad.cos() * ang_dist.sin() * bearing_rad.cos()).arcsin()
    lon2_rad = lon_rad + np.arctan2(
        bearing_rad.sin() * ang_dist.sin() * lat_rad.cos(),
        ang_dist.cos() - lat_rad.sin() * lat2_rad.sin()
    )
    return lat2_rad.degrees(), lon2_rad.degrees()


def compute_coverage_radius(elevation_deg: float, altitude_km: float) -> float:
    e_rad = np.deg2rad(elevation_deg)
    a = np.rad2deg(np.arcsin((EARTH_RADIUS_KM / (EARTH_RADIUS_KM + altitude_km)) * np.cos(e_rad)))
    theta = np.deg2rad(90 - a - elevation_deg)
    return (EARTH_RADIUS_KM * theta) * 1000.0


from pyproj import Geod
_GEOD_WGS84 = Geod(ellps='WGS84')


def create_geodesic_circle_poly(lon: float, lat: float, radius_m: float, num_points: int = 32) -> Tuple[List[float], List[float]]:
    angles = np.linspace(0, 360, num_points + 1)
    lons, lats, _ = _GEOD_WGS84.fwd(
        lons=np.full(len(angles), lon),
        lats=np.full(len(angles), lat),
        az=angles,
        dist=np.full(len(angles), radius_m)
    )
    return lons.tolist(), lats.tolist()
