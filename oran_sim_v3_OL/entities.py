import polars as pl
import logging
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Packet:
    id: int
    arrival_time: float
    size_bits: float
    destination_ship: int
    priority: int

    remaining_bits: float = 0.0
    service_start_time: Optional[float] = None
    departure_time: Optional[float] = None

    def __post_init__(self):
        self.remaining_bits = self.size_bits


@dataclass
class IPPState:
    active: bool = False


@dataclass
class ShipTrafficState:
    ship_id: int
    ipp_state: IPPState = field(default_factory=IPPState)


class GroundStation:
    def __init__(self, station_id, name, lat, lon, port_url):
        self.station_id = station_id
        self.name = name
        self.lat = lat
        self.lon = lon
        self.port_url = port_url


def load_ais_data_polars(csv_path: str, map_bounds: Dict[str, float] = None) -> pl.DataFrame:
    try:
        df = pl.read_csv(csv_path, ignore_errors=True)

        df = df.select([
            pl.col("IMO").cast(pl.Int64).alias("mmsi"),
            pl.col("BaseDateTime").str.to_datetime(strict=False).alias("datetime"),
            pl.col("LAT").cast(pl.Float64).alias("lat"),
            pl.col("LON").cast(pl.Float64).alias("lon"),
            pl.col("Passengers").cast(pl.Int32).fill_null(0).alias("passengers"),
            pl.col("Crew").cast(pl.Int32).fill_null(0).alias("crew")
        ])

        if map_bounds:
            lat_min = map_bounds.get("lat_min", -90.0)
            lat_max = map_bounds.get("lat_max", 90.0)
            lon_min = map_bounds.get("lon_min", -180.0)
            lon_max = map_bounds.get("lon_max", 180.0)

            df = df.filter(
                (pl.col("lat") >= lat_min) &
                (pl.col("lat") <= lat_max) &
                (pl.col("lon") >= lon_min) &
                (pl.col("lon") <= lon_max)
            )
            logger.info(f"Applied Map Bounds filter. {df.height} rows remaining.")

        df = df.drop_nulls(subset=["datetime"]).with_columns(
            (pl.col("datetime").dt.timestamp("ms").cast(pl.Float64) / 1000.0).alias("timestamp")
        )

        df = df.sort(["mmsi", "timestamp"])

        logger.info(f"Loaded AIS data into Polars: {df.height} rows")
        return df
    except Exception as e:
        logger.error(f"Failed to load AIS data: {e}")
        return pl.DataFrame()
