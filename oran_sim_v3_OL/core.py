from typing import Any, Dict, List, Set, Tuple
import logging
import numpy as np
from pathlib import Path
import polars as pl
import warnings
from sklearn.neighbors import BallTree

from .entities import GroundStation, load_ais_data_polars
from .utils import (
    pl_haversine_dist,
    pl_bearing,
    pl_destination_point,
    compute_coverage_radius,
    EARTH_RADIUS_KM,
    haps_df_to_array,
    ships_df_to_array,
    SHIP_DTYPE,
)
from .communication import CommunicationLayer
from .policies import create_policies, MinimalMobilityPolicy

logger = logging.getLogger(__name__)


class ORANSimulator:
    def __init__(self, config: Dict[str, Any], policy_name: str = "default"):
        self.config = config
        self.timestep = config.get("simulator", {}).get("timestep", 1.0)
        self.mobility_timestep = config.get("simulator", {}).get("mobility_timestep", 600.0)
        self.static_haps = config.get("simulator", {}).get("static_haps", False)
        self.current_time = 0.0
        self.step_count = 0
        self.policy_name = policy_name
        self._last_mobility_update = -float('inf')
        self._cached_connected_haps = set()
        self._cached_hap_candidate_ships = pl.DataFrame()
        self._cached_terrestrial_ships = set()
        self.use_gpu = config.get("performance", {}).get("use_gpu", False)
        gs_config = config.get("ground_stations", {})
        self.terrestrial_range_km = gs_config.get("terrestrial_range_km", 50.0)
        self.mobility_policy, self.association_policy, self.scheduler_policy, self.admission_policy = \
            create_policies(policy_name, config)
        logger.info(f"Using policy set: {policy_name}")
        self._init_haps_df()
        self._init_ships_df()
        self.duration = config.get("simulator", {}).get("duration", 3600.0)
        self.ground_stations: List[GroundStation] = []
        self._init_ground_stations()
        self.comms = CommunicationLayer(
            config,
            association_policy=self.association_policy,
            scheduler_policy=self.scheduler_policy,
            admission_policy=self.admission_policy
        )
        self.comm_stats = pl.DataFrame()
        self.comm_history: List[pl.DataFrame] = []
        self._prev_hap_positions: Dict[int, Tuple[float, float]] = {}
        self._hap_displacements: Dict[int, float] = {}
        self._level2_feedback: Dict[str, Any] = {}
        self._interval_ships_needing_hap: int = 0
        self._interval_ships_served: int = 0
        self._interval_coverage_ratio: float = 0.0
        logger.info(f"ORANSimulator initialized. Duration: {self.duration:.1f}s, Timestep: {self.timestep}s, Policy: {policy_name}")

    def _init_haps_df(self):
        hap_configs = self.config.get("haps", [])
        rows = []
        for hc in hap_configs:
            cov_radius_m = compute_coverage_radius(hc["elevation_angle"], hc["altitude"] / 1000.0)
            rows.append({
                "hap_id": hc["hap_id"],
                "lat": hc["initial_lat"],
                "lon": hc["initial_lon"],
                "altitude": hc["altitude"],
                "max_speed": hc["max_speed"],
                "target_lat": hc["initial_lat"],
                "target_lon": hc["initial_lon"],
                "coverage_radius": cov_radius_m,
                "numerology": hc.get("numerology", 1),
                "total_bandwidth_hz": hc.get("total_bandwidth_hz", 100e6),
            })
        schema = {
            "hap_id": pl.Int64, "lat": pl.Float64, "lon": pl.Float64,
            "altitude": pl.Float64, "max_speed": pl.Float64,
            "target_lat": pl.Float64, "target_lon": pl.Float64,
            "coverage_radius": pl.Float64,
            "numerology": pl.Int64, "total_bandwidth_hz": pl.Float64,
        }
        self.haps_df = pl.DataFrame(rows, schema=schema)
        if not self.haps_df.is_empty():
            self._generate_random_targets_vectorized(self.haps_df.select("hap_id"))

    def _init_ships_df(self):
        ships_config = self.config.get("ships", {})
        ais_csv = ships_config.get("ais_csv_path", "ais_data.csv")
        viz_config = self.config.get("visualization", {})
        bounds = viz_config.get("map_bounds", {})
        path = Path(ais_csv)
        if not path.is_absolute():
            path = Path(__file__).parent / ais_csv
        self.ais_data = load_ais_data_polars(str(path), map_bounds=bounds)
        if not self.ais_data.is_empty():
            self.sim_start_timestamp = self.ais_data["timestamp"].min()
            self.sim_end_timestamp = self.ais_data["timestamp"].max()
            initial_window = self.ais_data.filter(pl.col("timestamp") <= self.sim_start_timestamp + 1.0)
            self.ships_state = initial_window.unique(subset=["mmsi"], keep="last").select([
                "mmsi", "lat", "lon", "passengers", "crew"
            ]).with_columns(pl.lit(True).alias("is_visible"))
        else:
            self.sim_start_timestamp = 0.0
            self.sim_end_timestamp = 3600.0
            self.ships_state = pl.DataFrame(schema={
                "mmsi": pl.Int64, "lat": pl.Float64, "lon": pl.Float64,
                "passengers": pl.Int32, "crew": pl.Int32,
                "is_visible": pl.Boolean
            })

    def _init_ground_stations(self):
        self.port_tree = None
        gs_config = self.config.get("ground_stations", {})
        path_str = gs_config.get("ports_csv_path")
        if not path_str:
            return
        path = Path(path_str)
        if not path.is_absolute():
            path = Path(__file__).parent / path_str
        map_bounds = self.config.get("visualization", {}).get("map_bounds", {})
        lat_min = map_bounds.get("lat_min", -90.0)
        lat_max = map_bounds.get("lat_max", 90.0)
        lon_min = map_bounds.get("lon_min", -180.0)
        lon_max = map_bounds.get("lon_max", 180.0)
        try:
            df = pl.read_csv(str(path), ignore_errors=True)
            total_ports = df.height
            df = df.filter(
                (pl.col("LAT") >= lat_min) &
                (pl.col("LAT") <= lat_max) &
                (pl.col("LON") >= lon_min) &
                (pl.col("LON") <= lon_max)
            )
            for r in df.iter_rows(named=True):
                self.ground_stations.append(GroundStation(
                    r.get("World Port Index Number"), r.get("Main Port Name"),
                    r.get("LAT"), r.get("LON"), r.get("PortUrlString")
                ))
            logger.info(f"Loaded {len(self.ground_stations)} ground stations (filtered from {total_ports} total)")
            if len(self.ground_stations) > 0:
                port_coords_rad = np.radians([[gs.lat, gs.lon] for gs in self.ground_stations])
                self.port_tree = BallTree(port_coords_rad, metric='haversine')
                logger.info(f"Built BallTree with {len(self.ground_stations)} ports")
        except Exception as e:
            logger.error(f"GS Init Error: {e}")

    def _should_update_mobility(self) -> bool:
        return (self.current_time - self._last_mobility_update) >= self.mobility_timestep

    def _generate_random_targets_vectorized(self, hap_ids: pl.DataFrame):
        if hap_ids.is_empty():
            return
        bounds = self.config.get("visualization", {}).get("map_bounds", {})
        count = hap_ids.height
        new_lats = np.random.uniform(bounds.get("lat_min", 20), bounds.get("lat_max", 30), count)
        new_lons = np.random.uniform(bounds.get("lon_min", 50), bounds.get("lon_max", 60), count)
        updates = hap_ids.with_columns([
            pl.Series("target_lat", new_lats),
            pl.Series("target_lon", new_lons)
        ])
        self.haps_df = self.haps_df.join(updates, on="hap_id", how="left", suffix="_new") \
            .with_columns([
                pl.coalesce(["target_lat_new", "target_lat"]).alias("target_lat"),
                pl.coalesce(["target_lon_new", "target_lon"]).alias("target_lon")
            ]).drop(["target_lat_new", "target_lon_new"])

    def _compute_backhaul_connectivity(self) -> Set[int]:
        if not self.ground_stations or self.haps_df.is_empty():
            return set(self.haps_df["hap_id"].to_list())
        connected_haps = set()
        hap_data = self.haps_df.select(["hap_id", "lat", "lon", "coverage_radius"]).to_dicts()
        gs_coords = np.array([[gs.lat, gs.lon] for gs in self.ground_stations])
        if len(gs_coords) == 0:
            return set(self.haps_df["hap_id"].to_list())
        gs_rad = np.radians(gs_coords)
        for hap in hap_data:
            hap_rad = np.radians([hap["lat"], hap["lon"]])
            coverage_radius_m = hap["coverage_radius"]
            dlat = gs_rad[:, 0] - hap_rad[0]
            dlon = gs_rad[:, 1] - hap_rad[1]
            a = np.sin(dlat / 2.0) ** 2 + np.cos(hap_rad[0]) * np.cos(gs_rad[:, 0]) * np.sin(dlon / 2.0) ** 2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            dist_m = EARTH_RADIUS_KM * 1000.0 * c
            if np.min(dist_m) <= coverage_radius_m:
                connected_haps.add(hap["hap_id"])
        return connected_haps

    def _filter_terrestrial_coverage(self, ships: pl.DataFrame) -> pl.DataFrame:
        if ships.is_empty():
            return ships
        if self.port_tree is None:
            return ships
        ship_coords = ships.select(["lat", "lon"]).to_numpy()
        ship_coords_rad = np.radians(ship_coords)
        range_rad = self.terrestrial_range_km / EARTH_RADIUS_KM
        counts = self.port_tree.query_radius(ship_coords_rad, r=range_rad, count_only=True)
        in_terrestrial = counts > 0
        mask = ~in_terrestrial
        return ships.filter(mask)

    def _update_hap_positions_with_policy(self, dt: float):
        haps_arr = haps_df_to_array(self.haps_df)
        if not self.ships_state.is_empty():
            ships_arr = ships_df_to_array(self.ships_state.filter(pl.col("is_visible")))
        else:
            ships_arr = np.empty(0, dtype=SHIP_DTYPE)
        new_positions = self.mobility_policy.decide_positions(
            haps_arr,
            ships_arr,
            self.ground_stations,
            dt,
            self.config,
            level2_feedback=self._level2_feedback
        )
        if new_positions is None:
            self._update_hap_positions_random_waypoint(dt)
            return
        for hap_id, (new_lat, new_lon, _) in new_positions.items():
            if hap_id in self._prev_hap_positions:
                prev_lat, prev_lon = self._prev_hap_positions[hap_id]
                lat1_r, lon1_r = np.radians(prev_lat), np.radians(prev_lon)
                lat2_r, lon2_r = np.radians(new_lat), np.radians(new_lon)
                dlat = lat2_r - lat1_r
                dlon = lon2_r - lon1_r
                a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
                c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
                dist_m = EARTH_RADIUS_KM * 1000.0 * c
                self._hap_displacements[hap_id] = dist_m
            else:
                self._hap_displacements[hap_id] = 0.0
            self._prev_hap_positions[hap_id] = (new_lat, new_lon)
        updates = []
        for hap_id, (lat, lon, alt) in new_positions.items():
            updates.append({
                "hap_id": hap_id,
                "lat": lat,
                "lon": lon,
                "altitude": alt
            })
        if updates:
            updates_df = pl.DataFrame(updates)
            self.haps_df = self.haps_df.join(
                updates_df, on="hap_id", how="left", suffix="_new"
            ).with_columns([
                pl.coalesce(["lat_new", "lat"]).alias("lat"),
                pl.coalesce(["lon_new", "lon"]).alias("lon"),
                pl.coalesce(["altitude_new", "altitude"]).alias("altitude"),
            ]).drop(["lat_new", "lon_new", "altitude_new"])

    def _update_hap_positions_random_waypoint(self, dt: float):
        dist_expr = pl_haversine_dist(
            pl.col("lat"), pl.col("lon"),
            pl.col("target_lat"), pl.col("target_lon")
        )
        bearing_expr = pl_bearing(
            pl.col("lat"), pl.col("lon"),
            pl.col("target_lat"), pl.col("target_lon")
        )
        step_dist_m = pl.col("max_speed") * dt
        updates = self.haps_df.with_columns(
            dist_to_target=dist_expr
        ).with_columns(
            reached=pl.col("dist_to_target") <= step_dist_m
        )
        moving_haps = updates.filter(~pl.col("reached"))
        if not moving_haps.is_empty():
            lat_new, lon_new = pl_destination_point(
                pl.col("lat"), pl.col("lon"), bearing_expr, step_dist_m
            )
            moved = moving_haps.with_columns([
                lat_new.alias("lat"), lon_new.alias("lon")
            ]).drop(["dist_to_target", "reached"])
            reached_haps = updates.filter(pl.col("reached")).with_columns([
                pl.col("target_lat").alias("lat"),
                pl.col("target_lon").alias("lon")
            ]).drop(["dist_to_target", "reached"])
            self.haps_df = pl.concat([moved, reached_haps])
            if not reached_haps.is_empty():
                self._generate_random_targets_vectorized(reached_haps.select("hap_id"))
        else:
            self.haps_df = updates.with_columns([
                pl.col("target_lat").alias("lat"),
                pl.col("target_lon").alias("lon")
            ]).drop(["dist_to_target", "reached"])
            self._generate_random_targets_vectorized(self.haps_df.select("hap_id"))

    def step(self, duration_override: float = None) -> bool:
        if self.current_time >= self.duration:
            return False
        dt = duration_override if duration_override is not None else self.timestep
        prev_epoch = self.sim_start_timestamp + self.current_time
        self.current_time += dt
        self.step_count += 1
        current_epoch = self.sim_start_timestamp + self.current_time
        if not self.ais_data.is_empty():
            q = self.ais_data.lazy().filter(
                (pl.col("timestamp") >= prev_epoch) &
                (pl.col("timestamp") < current_epoch)
            )
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    engine = "gpu" if self.use_gpu else "cpu"
                    interval_data = q.collect(engine=engine)
            except Exception:
                interval_data = q.collect()
            if not interval_data.is_empty():
                updates = interval_data.sort("timestamp") \
                    .unique(subset=["mmsi"], keep="last") \
                    .select(["mmsi", "lat", "lon", "passengers", "crew"]) \
                    .with_columns(pl.lit(True).alias("is_visible"))
                if self.ships_state.is_empty():
                    self.ships_state = updates
                else:
                    self.ships_state = pl.concat([self.ships_state, updates]) \
                        .unique(subset=["mmsi"], keep="last")
        should_recompute = self._should_update_mobility() or self._last_mobility_update < 0
        if not self.haps_df.is_empty() and not self.static_haps:
            if should_recompute:
                use_policy_mobility = not isinstance(self.mobility_policy, MinimalMobilityPolicy)
                if use_policy_mobility:
                    self._update_hap_positions_with_policy(self.mobility_timestep)
                else:
                    self._update_hap_positions_random_waypoint(self.mobility_timestep)
        if should_recompute:
            connected_haps = self._compute_backhaul_connectivity()
            self._cached_connected_haps = connected_haps
            if not self.ships_state.is_empty() and not self.haps_df.is_empty():
                visible_ships = self.ships_state.filter(pl.col("is_visible"))
                hap_candidate_ships = self._filter_terrestrial_coverage(visible_ships)
                all_ship_ids = set(visible_ships["mmsi"].to_list())
                hap_ship_ids = set(hap_candidate_ships["mmsi"].to_list()) if not hap_candidate_ships.is_empty() else set()
                terrestrial_ship_ids = all_ship_ids - hap_ship_ids
                self._cached_hap_candidate_ships = hap_candidate_ships
                self._cached_terrestrial_ships = terrestrial_ship_ids
            else:
                self._cached_hap_candidate_ships = pl.DataFrame()
                self._cached_terrestrial_ships = set()
            self._last_mobility_update = self.current_time
        else:
            connected_haps = self._cached_connected_haps
            hap_candidate_ships = self._cached_hap_candidate_ships
            terrestrial_ship_ids = self._cached_terrestrial_ships
        if not self.ships_state.is_empty() and not self.haps_df.is_empty():
            self.comm_stats = self.comms.update(
                hap_candidate_ships,
                self.haps_df,
                self.ground_stations,
                self.current_time,
                dt,
                connected_haps=connected_haps,
                terrestrial_ships=terrestrial_ship_ids,
                recompute_associations=should_recompute,
                level2_feedback=self._level2_feedback
            )
            if not self.comm_stats.is_empty():
                self.comm_stats = self.comm_stats.with_columns(pl.lit(self.current_time).alias("time"))
                self.comm_history.append(self.comm_stats)

            self._interval_ships_needing_hap = len(hap_candidate_ships) if not hap_candidate_ships.is_empty() else 0
            ships_served = sum(len(node.served_ships) for node in self.comms.haps.values())
            self._interval_ships_served = ships_served
            self._interval_coverage_ratio = (
                ships_served / self._interval_ships_needing_hap
                if self._interval_ships_needing_hap > 0 else 1.0
            )

            qoe_targets = self.config.get('qoe_targets', {})
            self._level2_feedback = self.comms.get_level2_feedback(qoe_targets)

            if self._level2_feedback:
                self._update_scheduler_learner(self._level2_feedback)

            if should_recompute and self._level2_feedback:
                self._update_mobility_learner(self._level2_feedback)
        else:
            self.comm_stats = pl.DataFrame()
        return True

    def get_current_state(self) -> Dict[str, Any]:
        ships_data = []
        if not self.ships_state.is_empty():
            ships_data = self.ships_state.filter(pl.col("is_visible")).to_dicts()
        comms_data = []
        if not self.comm_stats.is_empty():
            comms_data = self.comm_stats.to_dicts()
        return {
            'time': self.current_time,
            'step': self.step_count,
            'haps': self.haps_df.to_dicts(),
            'ships': ships_data,
            'links': comms_data,
            'ground_stations': [
                {'station_id': gs.station_id, 'name': gs.name, 'lat': gs.lat, 'lon': gs.lon, 'port_url': gs.port_url}
                for gs in self.ground_stations
            ],
        }

    def get_interval_metrics(self) -> Dict[str, Any]:
        qoe_targets = self.config.get('qoe_targets', {})
        metrics = self.comms.get_interval_metrics(qoe_targets)
        metrics['ships_needing_hap'] = self._interval_ships_needing_hap
        metrics['ships_served'] = self._interval_ships_served
        metrics['coverage_ratio'] = self._interval_coverage_ratio
        return metrics

    def _update_mobility_learner(self, level2_feedback: Dict[str, Any]) -> None:
        if hasattr(self.mobility_policy, 'update'):
            self.mobility_policy.update(level2_feedback)

    def _update_scheduler_learner(self, level2_feedback: Dict[str, Any]) -> None:
        if hasattr(self.scheduler_policy, 'update_weights'):
            for hap_id, node in self.comms.haps.items():
                ship_losses = self._compute_ship_losses(node, level2_feedback)
                self.scheduler_policy.update_weights(hap_id, ship_losses)

    def _compute_ship_losses(
        self,
        node: 'FastHAPNode',
        level2_feedback: Dict[str, Any]
    ) -> Dict[int, float]:
        losses = {}
        qoe_targets = self.config.get('qoe_targets', {})
        target = qoe_targets.get('per_ship_throughput_target', 1e6)

        for ship_id in node.served_ships:
            ship_idx = node.interval_ship_id_map.get(ship_id)
            if ship_idx is None:
                losses[ship_id] = 0.5
                continue

            served = node.interval_served_per_ship_priority[ship_idx].sum()
            losses[ship_id] = max(0.0, 1.0 - served / target) if target > 0 else 0.0

        return losses
