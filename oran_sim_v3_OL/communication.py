import numpy as np
import polars as pl
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set, Tuple, TYPE_CHECKING

from .utils import (
    EARTH_RADIUS_KM,
    ships_df_to_array,
    haps_df_to_array,
    SHIP_DTYPE,
    HAP_DTYPE,
)

if TYPE_CHECKING:
    from .policies import AssociationPolicy, SchedulerPolicy, AdmissionPolicy

logger = logging.getLogger(__name__)


@dataclass
class AggregateQueueState:
    bits: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    packets: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.int64))
    arrival_time_sum: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    def enqueue_batch(self, priority_idx: int, count: int, size_bits: float, arrival_time: float) -> Tuple[int, float]:
        if count <= 0:
            return 0, 0.0
        total_bits = count * size_bits
        self.bits[priority_idx] += total_bits
        self.packets[priority_idx] += count
        self.arrival_time_sum[priority_idx] += count * arrival_time
        return count, 0.0

    def dequeue_bits(
        self,
        max_bits: float,
        current_time: float,
        rate_bps: float,
        priority_order: Optional[List[int]] = None
    ) -> Tuple[float, Dict[int, List[float]], Dict[int, float]]:
        if priority_order is None:
            priority_order = [0, 1, 2]
        served = 0.0
        delays = {1: [], 2: [], 3: []}
        served_by_priority = {1: 0.0, 2: 0.0, 3: 0.0}
        remaining = max_bits
        for priority_idx in priority_order:
            if remaining <= 0 or self.bits[priority_idx] <= 0:
                continue
            serve_bits = min(remaining, self.bits[priority_idx])
            serve_fraction = serve_bits / self.bits[priority_idx] if self.bits[priority_idx] > 0 else 0
            packets_served = round(self.packets[priority_idx] * serve_fraction)
            packets_served = max(packets_served, 1) if serve_bits > 0 and self.packets[priority_idx] > 0 else 0
            avg_arrival = 0.0
            if self.packets[priority_idx] > 0:
                avg_arrival = self.arrival_time_sum[priority_idx] / self.packets[priority_idx]
            if packets_served > 0 and self.packets[priority_idx] > 0:
                waiting_delay = current_time - avg_arrival
                tx_delay = serve_bits / rate_bps if rate_bps > 0 else 0.0
                total_delay = waiting_delay + tx_delay
                delays[priority_idx + 1] = [total_delay] * packets_served
            self.bits[priority_idx] -= serve_bits
            self.packets[priority_idx] -= packets_served
            self.arrival_time_sum[priority_idx] -= packets_served * avg_arrival
            served += serve_bits
            served_by_priority[priority_idx + 1] = serve_bits
            remaining -= serve_bits
        return served, delays, served_by_priority

    @property
    def total_packets(self) -> int:
        return int(self.packets.sum())

    @property
    def total_bits(self) -> float:
        return float(self.bits.sum())


@dataclass
class FastServerState:
    allocated_rbs: int = 0
    current_rate_bps: float = 0.0
    served_bits: float = 0.0


@dataclass
class IPPState:
    active: bool = False


class FastHAPNode:
    def __init__(self, hap_id: int, config: Dict[str, Any]):
        self.hap_id = hap_id
        self.config = config
        comm_config = config.get("communication", {})
        queue_config = comm_config.get("queue", {})
        self.hap_total_capacity_bits = queue_config.get("hap_total_capacity_bits", np.inf)
        self.hap_current_total_bits = 0.0
        self.numerology = config.get("numerology", 1)
        self.total_bw_hz = config.get("total_bandwidth_hz", 100e6)
        self.scs_khz = 15 * (2 ** self.numerology)
        self.rb_bandwidth = max(12 * self.scs_khz * 1000, 1.0)
        self.total_rbs = int(self.total_bw_hz // self.rb_bandwidth)
        self.served_ships: Set[int] = set()
        self.queues: Dict[int, AggregateQueueState] = {}
        self.servers: Dict[int, FastServerState] = {}
        self.ipp_active: Dict[int, bool] = {}
        self.connected: bool = True
        self.interval_completed_packets: int = 0
        self.interval_dropped_packets: int = 0
        self.interval_dropped_bits: float = 0.0
        self.interval_delay_sum: Dict[int, float] = {1: 0.0, 2: 0.0, 3: 0.0}
        self.interval_delay_count: Dict[int, int] = {1: 0, 2: 0, 3: 0}
        self.interval_served_bits: Dict[int, float] = {1: 0.0, 2: 0.0, 3: 0.0}
        self.drop_reasons: Dict[str, int] = {
            "BUFFER_OVERFLOW": 0, "COVERAGE_LOSS": 0,
            "BACKHAUL_DISCONNECT": 0, "STARVATION": 0,
            "TERRESTRIAL_HANDOFF": 0
        }
        self.interval_arrived_packets: Dict[int, int] = {1: 0, 2: 0, 3: 0}
        self.interval_admitted_packets: Dict[int, int] = {1: 0, 2: 0, 3: 0}
        self.interval_overflow_packets: Dict[int, int] = {1: 0, 2: 0, 3: 0}
        self.qoe_per_slice: Dict[int, float] = {1: 1.0, 2: 1.0, 3: 1.0}
        self.interval_delay_violations: Dict[int, int] = {1: 0, 2: 0, 3: 0}
        self.interval_throughput_violations: Dict[int, int] = {1: 0, 2: 0, 3: 0}
        self.interval_ships_served_count: int = 0
        self._ship_priority_capacity: int = 500
        self.interval_served_per_ship_priority: np.ndarray = np.zeros(
            (self._ship_priority_capacity, 3), dtype=np.float64
        )
        self.interval_ship_id_map: Dict[int, int] = {}
        self._ship_id_list: List[int] = []

    def add_ship(self, ship_id: int):
        if ship_id in self.served_ships:
            return
        self.served_ships.add(ship_id)
        self.queues[ship_id] = AggregateQueueState()
        self.servers[ship_id] = FastServerState()
        self.ipp_active[ship_id] = False
        if ship_id not in self.interval_ship_id_map:
            idx = len(self._ship_id_list)
            if idx >= self._ship_priority_capacity:
                new_capacity = self._ship_priority_capacity * 2
                new_arr = np.zeros((new_capacity, 3), dtype=np.float64)
                new_arr[:self._ship_priority_capacity, :] = self.interval_served_per_ship_priority
                self.interval_served_per_ship_priority = new_arr
                self._ship_priority_capacity = new_capacity
            self.interval_ship_id_map[ship_id] = idx
            self._ship_id_list.append(ship_id)

    def remove_ship(self, ship_id: int, reason: str = "COVERAGE_LOSS"):
        if ship_id not in self.served_ships:
            return
        q = self.queues[ship_id]
        dropped_bits = q.total_bits
        dropped_pkts = q.total_packets
        self.hap_current_total_bits -= dropped_bits
        self.hap_current_total_bits = max(0.0, self.hap_current_total_bits)
        if reason != "TERRESTRIAL_HANDOFF":
            self.interval_dropped_bits += dropped_bits
            self.interval_dropped_packets += dropped_pkts
        self.drop_reasons[reason] = self.drop_reasons.get(reason, 0) + dropped_pkts
        del self.queues[ship_id]
        del self.servers[ship_id]
        self.ipp_active.pop(ship_id, None)
        self.served_ships.remove(ship_id)

    def flush_all(self, reason: str):
        for sid in list(self.served_ships):
            self.remove_ship(sid, reason)

    def compute_utilization(self, dt: float = 1.0) -> float:
        if not self.queues:
            return 0.0
        total_backlog = sum(q.total_bits for q in self.queues.values())
        capacity_per_step = self.total_rbs * self.rb_bandwidth * dt
        if capacity_per_step <= 0:
            return 0.0
        return total_backlog / capacity_per_step

    def get_available_hap_buffer(self) -> float:
        return max(self.hap_total_capacity_bits - self.hap_current_total_bits, 0.0)

    def update_hap_buffer_usage(self, delta_bits: float):
        self.hap_current_total_bits += delta_bits
        self.hap_current_total_bits = max(0.0, self.hap_current_total_bits)

    def clear_interval_metrics(self):
        self.interval_completed_packets = 0
        self.interval_dropped_packets = 0
        self.interval_dropped_bits = 0.0
        self.interval_delay_sum = {1: 0.0, 2: 0.0, 3: 0.0}
        self.interval_delay_count = {1: 0, 2: 0, 3: 0}
        self.interval_served_bits = {1: 0.0, 2: 0.0, 3: 0.0}
        self.drop_reasons = {
            "BUFFER_OVERFLOW": 0, "COVERAGE_LOSS": 0,
            "BACKHAUL_DISCONNECT": 0, "STARVATION": 0,
            "TERRESTRIAL_HANDOFF": 0
        }
        self.interval_arrived_packets = {1: 0, 2: 0, 3: 0}
        self.interval_admitted_packets = {1: 0, 2: 0, 3: 0}
        self.interval_overflow_packets = {1: 0, 2: 0, 3: 0}
        self.interval_delay_violations = {1: 0, 2: 0, 3: 0}
        self.interval_throughput_violations = {1: 0, 2: 0, 3: 0}
        self.interval_ships_served_count = 0
        n_ships = len(self._ship_id_list)
        if n_ships > 0:
            self.interval_served_per_ship_priority[:n_ships, :] = 0.0

    def compute_qoe(self, targets: Dict[str, Any]) -> Dict[int, float]:
        qoe = {}

        p1_delay_target = targets.get('p1_delay_target', 0.01)
        if self.interval_delay_count[1] > 0:
            avg_delay = self.interval_delay_sum[1] / self.interval_delay_count[1]
            qoe[1] = max(0.0, 1.0 - avg_delay / p1_delay_target)
        else:
            qoe[1] = 1.0

        p2_target = targets.get('p2_throughput_target', 10e6)
        if p2_target > 0:
            qoe[2] = min(1.0, self.interval_served_bits[2] / p2_target)
        else:
            qoe[2] = 1.0

        p3_target = targets.get('p3_throughput_target', 50e6)
        if p3_target > 0:
            qoe[3] = min(1.0, self.interval_served_bits[3] / p3_target)
        else:
            qoe[3] = 1.0

        self.qoe_per_slice = qoe
        return qoe

    def compute_violations(self, targets: Dict[str, Any]) -> Dict[str, Any]:
        delay_thresholds = {
            1: targets.get('p1_delay_max_ms', 10.0),
            2: targets.get('p2_delay_max_ms', 100.0),
            3: targets.get('p3_delay_max_ms', 1000.0),
        }
        throughput_thresholds = {
            1: targets.get('p1_throughput_min_bps', 100e3),
            2: targets.get('p2_throughput_min_bps', 10e6),
            3: targets.get('p3_throughput_min_bps', 50e6),
        }
        epsilon_d = {
            1: targets.get('p1_epsilon_d', 0.001),
            2: targets.get('p2_epsilon_d', 0.01),
            3: targets.get('p3_epsilon_d', 0.1),
        }
        epsilon_r = {
            1: targets.get('p1_epsilon_r', 0.01),
            2: targets.get('p2_epsilon_r', 0.05),
            3: targets.get('p3_epsilon_r', 0.1),
        }

        delay_violation_rate = {}
        throughput_violation_rate = {}
        delay_violation_severity = {}
        throughput_violation_severity = {}

        n_ships = len(self.served_ships)
        self.interval_ships_served_count = n_ships

        for p in [1, 2, 3]:
            total_packets = self.interval_delay_count[p]
            if total_packets > 0:
                avg_delay_ms = (self.interval_delay_sum[p] / total_packets) * 1000.0
                j_d = 1.0 if avg_delay_ms > delay_thresholds[p] else 0.0
                delay_violation_rate[p] = j_d
            else:
                delay_violation_rate[p] = 0.0

            if n_ships > 0:
                violations = 0
                for ship_id in self.served_ships:
                    ship_idx = self.interval_ship_id_map.get(ship_id)
                    if ship_idx is not None:
                        ship_throughput = self.interval_served_per_ship_priority[ship_idx, p - 1]
                        if ship_throughput < throughput_thresholds[p]:
                            violations += 1
                throughput_violation_rate[p] = violations / n_ships
            else:
                throughput_violation_rate[p] = 0.0

            delay_violation_severity[p] = max(0.0, delay_violation_rate[p] - epsilon_d[p])
            throughput_violation_severity[p] = max(0.0, throughput_violation_rate[p] - epsilon_r[p])

        return {
            'delay_violation_rate': delay_violation_rate,
            'throughput_violation_rate': throughput_violation_rate,
            'delay_violation_severity': delay_violation_severity,
            'throughput_violation_severity': throughput_violation_severity,
        }


class CommunicationLayer:
    def __init__(
        self,
        config: Dict[str, Any],
        association_policy: Optional["AssociationPolicy"] = None,
        scheduler_policy: Optional["SchedulerPolicy"] = None,
        admission_policy: Optional["AdmissionPolicy"] = None
    ):
        self.config = config
        self.comm_config = config.get("communication", {})
        self.link_params = self.comm_config.get("link_params", {})
        self.traffic_config = self.comm_config.get("traffic", {})
        self.association_policy = association_policy
        self.scheduler_policy = scheduler_policy
        self.admission_policy = admission_policy
        self.haps: Dict[int, FastHAPNode] = {}
        self.k_factor = self.link_params.get("k_factor", 10.0)
        self.tx_power_dbm = self.link_params.get("tx_power_dbm", 43.0)
        self.noise_fig_db = self.link_params.get("noise_figure_db", 5.0)
        self.center_freq = self.link_params.get("center_freq_hz", 2e9)
        self.g_tx_db = self.link_params.get("g_tx_dbi", 0.0)
        self.g_rx_db = self.link_params.get("g_rx_dbi", 0.0)
        c = 3e8
        self.lambda_m = c / self.center_freq
        self.tx_watts = 10 ** ((self.tx_power_dbm - 30) / 10)
        self.g_tx_linear = 10 ** (self.g_tx_db / 10)
        self.g_rx_linear = 10 ** (self.g_rx_db / 10)
        self.los_component = np.sqrt(self.k_factor / (self.k_factor + 1))
        self.nlos_std = np.sqrt(1 / (2 * (self.k_factor + 1)))
        safety_cfg = self.traffic_config.get("safety", {})
        ops_cfg = self.traffic_config.get("operation", {})
        ent_cfg = self.traffic_config.get("entertainment", {})
        self.lambda_1 = safety_cfg.get("arrival_rate_per_sec", 0.05)
        self.lambda_2_embb = ops_cfg.get("embb_arrival_rate_per_sec", 0.1)
        self.lambda_2_mmtc = ops_cfg.get("mmtc_arrival_rate_per_sec", 0.5)
        self.alpha_on = ops_cfg.get("ipp_alpha_on", 0.1)
        self.alpha_off = ops_cfg.get("ipp_alpha_off", 0.05)
        self.lambda_3 = ent_cfg.get("arrival_rate_per_sec", 1.0)
        self.size_1 = safety_cfg.get("packet_size_bytes", 300) * 8
        self.size_2 = ops_cfg.get("packet_size_bytes", 1500) * 8
        self.size_3 = ent_cfg.get("packet_size_bytes", 1500) * 8
        self.use_gpu = config.get("performance", {}).get("use_gpu", False)
        self._cached_assignments = {}
        self._association_valid = False
        self._current_dt = config.get("simulator", {}).get("timestep", 1.0)
        self.interval_handovers: int = 0
        self.interval_handover_ships: Set[int] = set()
        self._max_ships_per_hap = 500
        self._scratch_lats = np.empty(self._max_ships_per_hap, dtype=np.float64)
        self._scratch_lons = np.empty(self._max_ships_per_hap, dtype=np.float64)
        self._scratch_rbs = np.empty(self._max_ships_per_hap, dtype=np.int32)

    def _get_scratch_arrays(self, size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if size > self._max_ships_per_hap:
            self._max_ships_per_hap = size * 2
            self._scratch_lats = np.empty(self._max_ships_per_hap, dtype=np.float64)
            self._scratch_lons = np.empty(self._max_ships_per_hap, dtype=np.float64)
            self._scratch_rbs = np.empty(self._max_ships_per_hap, dtype=np.int32)
        return (
            self._scratch_lats[:size],
            self._scratch_lons[:size],
            self._scratch_rbs[:size]
        )

    def _vectorized_haversine(self, lat1: np.ndarray, lon1: np.ndarray,
                               lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
        lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
        lat2_r, lon2_r = np.radians(lat2), np.radians(lon2)
        dlat = lat2_r - lat1_r
        dlon = lon2_r - lon1_r
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return EARTH_RADIUS_KM * 1000.0 * c

    def _vectorized_slant_range(self, ship_lats: np.ndarray, ship_lons: np.ndarray,
                                 hap_lat: float, hap_lon: float, hap_alt_m: float) -> np.ndarray:
        d_lat_r = np.radians(hap_lat - ship_lats)
        d_lon_r = np.radians(hap_lon - ship_lons)
        a = np.sin(d_lat_r / 2) ** 2 + np.cos(np.radians(ship_lats)) * np.cos(np.radians(hap_lat)) * np.sin(d_lon_r / 2) ** 2
        beta = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        R = EARTH_RADIUS_KM
        h = hap_alt_m / 1000.0
        slant_range = np.sqrt(R ** 2 + (R + h) ** 2 - 2 * R * (R + h) * np.cos(beta))
        return slant_range

    def _vectorized_rate(self, distances_km: np.ndarray, rb_counts: np.ndarray,
                          numerology: int) -> np.ndarray:
        n = len(distances_km)
        rates = np.zeros(n, dtype=np.float64)
        valid = rb_counts > 0
        if not np.any(valid):
            return rates
        dist_m = np.maximum(distances_km[valid] * 1000.0, 1.0)
        fspl_linear = (self.lambda_m / (4 * np.pi * dist_m)) ** 2
        nlos_real = np.random.normal(0, self.nlos_std, len(dist_m))
        nlos_imag = np.random.normal(0, self.nlos_std, len(dist_m))
        h_mag_sq = (self.los_component + nlos_real) ** 2 + nlos_imag ** 2
        bw_hz = rb_counts[valid] * 12 * 15000 * (2 ** numerology)
        noise_power_dbm = -174 + 10 * np.log10(bw_hz) + self.noise_fig_db
        noise_watts = 10 ** ((noise_power_dbm - 30) / 10)
        rx_watts = self.tx_watts * self.g_tx_linear * self.g_rx_linear * fspl_linear * h_mag_sq
        snr_linear = rx_watts / noise_watts
        rates[valid] = bw_hz * np.log2(1 + snr_linear)
        return rates

    def update(self, ships: pl.DataFrame, haps_df: pl.DataFrame,
               ground_stations: List[Any], current_time: float, dt: float,
               connected_haps: set = None,
               terrestrial_ships: set = None,
               recompute_associations: bool = True,
               level2_feedback: Dict[str, Any] = None) -> pl.DataFrame:
        _ = ground_stations
        if terrestrial_ships is None:
            terrestrial_ships = set()
        self._current_dt = dt
        if ships.is_empty() or haps_df.is_empty():
            return pl.DataFrame()
        ships_arr = ships_df_to_array(ships)
        haps_arr = haps_df_to_array(haps_df)
        ship_ids = ships_arr['mmsi']
        ship_lats = ships_arr['lat']
        ship_lons = ships_arr['lon']
        ship_pax = ships_arr['passengers']
        ship_crew = ships_arr['crew']
        hap_ids = haps_arr['hap_id']
        hap_lats = haps_arr['lat']
        hap_lons = haps_arr['lon']
        hap_alts = haps_arr['altitude']
        hap_radii = haps_arr['coverage_radius']
        ship_idx_map = {int(sid): i for i, sid in enumerate(ship_ids)}
        for i in range(len(haps_arr)):
            hid = int(hap_ids[i])
            if hid not in self.haps:
                hap_config = self._get_hap_config(hid)
                self.haps[hid] = FastHAPNode(hid, {**self.config, **hap_config})
        if connected_haps is not None:
            for hid, node in self.haps.items():
                was_connected = node.connected
                node.connected = hid in connected_haps
                if was_connected and not node.connected:
                    node.flush_all("BACKHAUL_DISCONNECT")
                    self._association_valid = False
        for node in self.haps.values():
            for sid in node.served_ships:
                node.servers[sid].served_bits = 0.0
        if recompute_associations or not self._association_valid:
            ship_lats_2d = ship_lats[:, np.newaxis]
            ship_lons_2d = ship_lons[:, np.newaxis]
            hap_lats_2d = hap_lats[np.newaxis, :]
            hap_lons_2d = hap_lons[np.newaxis, :]
            hap_radii_2d = hap_radii[np.newaxis, :]
            dlat = np.radians(hap_lats_2d - ship_lats_2d)
            dlon = np.radians(hap_lons_2d - ship_lons_2d)
            a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(ship_lats_2d)) * np.cos(np.radians(hap_lats_2d)) * np.sin(dlon / 2) ** 2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            distances_m = EARTH_RADIUS_KM * 1000.0 * c
            in_coverage = distances_m <= hap_radii_2d
            connected_mask = np.array([self.haps.get(int(hid), FastHAPNode(0, {})).connected for hid in hap_ids])
            in_coverage = in_coverage & connected_mask[np.newaxis, :]
            hap_loads = {int(hid): len(node.served_ships) for hid, node in self.haps.items()}
            if self.association_policy is not None:
                new_assignments = self.association_policy.decide_associations_batch(
                    ships_arr, haps_arr, in_coverage, hap_loads, self.config,
                    level2_feedback=level2_feedback
                )
            else:
                new_assignments = {}
                distances_m_masked = np.where(in_coverage, distances_m, np.inf)
                best_hap_idx = np.argmin(distances_m_masked, axis=1)
                has_coverage = np.any(in_coverage, axis=1)
                for i in range(len(ships_arr)):
                    sid = int(ship_ids[i])
                    if has_coverage[i]:
                        new_assignments[sid] = int(hap_ids[best_hap_idx[i]])
            self._cached_assignments = new_assignments.copy()
            self._association_valid = True
        else:
            current_ship_ids = set(int(sid) for sid in ship_ids)
            new_assignments = {sid: hap for sid, hap in self._cached_assignments.items()
                               if sid in current_ship_ids}
        current_assignments = {}
        for hid, node in self.haps.items():
            for sid in node.served_ships:
                current_assignments[sid] = hid
        for sid, target_hap in new_assignments.items():
            current_hap = current_assignments.get(sid)
            if current_hap != target_hap:
                if current_hap is not None:
                    self.haps[current_hap].remove_ship(sid, "COVERAGE_LOSS")
                    self.interval_handovers += 1
                    self.interval_handover_ships.add(sid)
                self.haps[target_hap].add_ship(sid)
        for sid, current_hap in current_assignments.items():
            if sid not in new_assignments:
                if sid in terrestrial_ships:
                    self.haps[current_hap].remove_ship(sid, "TERRESTRIAL_HANDOFF")
                else:
                    self.haps[current_hap].remove_ship(sid, "COVERAGE_LOSS")
        for hid, node in self.haps.items():
            if not node.connected or not node.served_ships:
                continue
            served_sids = np.array(list(node.served_ships), dtype=np.int64)
            ship_mask = np.isin(ship_ids, served_sids)
            if not np.any(ship_mask):
                continue
            local_pax = ship_pax[ship_mask]
            local_crew = ship_crew[ship_mask]
            local_sids = ship_ids[ship_mask]
            n_local = len(local_sids)
            n_p1 = np.random.poisson(self.lambda_1 * dt, n_local)
            n_embb = np.random.poisson(self.lambda_2_embb * np.maximum(local_crew, 1) * dt)
            local_sids_int = local_sids.astype(np.int64)
            ipp_active_arr = np.array([node.ipp_active[int(sid)] for sid in local_sids_int], dtype=bool)
            n_mmtc = np.where(
                ipp_active_arr,
                np.random.poisson(self.lambda_2_mmtc * dt, n_local),
                0
            )
            rand_vals = np.random.random(n_local)
            off_mask = ipp_active_arr & (rand_vals < self.alpha_off * dt)
            on_mask = ~ipp_active_arr & (rand_vals < self.alpha_on * dt)
            for i, sid in enumerate(local_sids_int):
                sid_int = int(sid)
                if off_mask[i]:
                    node.ipp_active[sid_int] = False
                elif on_mask[i]:
                    node.ipp_active[sid_int] = True
            n_p2 = n_embb + n_mmtc
            n_p3 = np.random.poisson(self.lambda_3 * np.maximum(local_pax, 1) * dt)
            queue_refs = [node.queues[int(sid)] for sid in local_sids]
            queue_p2_bits = np.array([q.bits[1] for q in queue_refs], dtype=np.float64)
            hap_utilization = node.compute_utilization(self._current_dt)
            available_hap_buffer = node.get_available_hap_buffer()
            if self.admission_policy is not None:
                from .policies import AdmissionResult
                result: AdmissionResult = self.admission_policy.admit_packets(
                    n_p1=n_p1.astype(np.int32),
                    n_p2=n_p2.astype(np.int32),
                    n_p3=n_p3.astype(np.int32),
                    queue_p2_bits=queue_p2_bits,
                    hap_utilization=hap_utilization,
                    available_hap_buffer=available_hap_buffer,
                    packet_sizes=(self.size_1, self.size_2, self.size_3),
                    config=self.config
                )
                admitted_p1 = result.admitted_p1
                admitted_p2 = result.admitted_p2
                admitted_p3 = result.admitted_p3
            else:
                admitted_p1 = n_p1.astype(np.int32).copy()
                admitted_p2 = n_p2.astype(np.int32).copy()
                admitted_p3 = n_p3.astype(np.int32).copy()
            node.interval_arrived_packets[1] += int(n_p1.sum())
            node.interval_arrived_packets[2] += int(n_p2.sum())
            node.interval_arrived_packets[3] += int(n_p3.sum())
            node.interval_admitted_packets[1] += int(admitted_p1.sum())
            node.interval_admitted_packets[2] += int(admitted_p2.sum())
            node.interval_admitted_packets[3] += int(admitted_p3.sum())
            total_enqueued_bits = 0.0
            for i, (sid, q) in enumerate(zip(local_sids, queue_refs)):
                sid = int(sid)
                if admitted_p1[i] > 0:
                    admitted, dropped = q.enqueue_batch(0, int(admitted_p1[i]), self.size_1, current_time)
                    total_enqueued_bits += admitted * self.size_1
                    if dropped > 0:
                        overflow_count = int(admitted_p1[i]) - admitted
                        node.interval_dropped_bits += dropped
                        node.interval_dropped_packets += overflow_count
                        node.drop_reasons["BUFFER_OVERFLOW"] += overflow_count
                        node.interval_overflow_packets[1] += overflow_count
                if admitted_p2[i] > 0:
                    admitted, dropped = q.enqueue_batch(1, int(admitted_p2[i]), self.size_2, current_time)
                    total_enqueued_bits += admitted * self.size_2
                    if dropped > 0:
                        overflow_count = int(admitted_p2[i]) - admitted
                        node.interval_dropped_bits += dropped
                        node.interval_dropped_packets += overflow_count
                        node.drop_reasons["BUFFER_OVERFLOW"] += overflow_count
                        node.interval_overflow_packets[2] += overflow_count
                if admitted_p3[i] > 0:
                    admitted, dropped = q.enqueue_batch(2, int(admitted_p3[i]), self.size_3, current_time)
                    total_enqueued_bits += admitted * self.size_3
                    if dropped > 0:
                        overflow_count = int(admitted_p3[i]) - admitted
                        node.interval_dropped_bits += dropped
                        node.interval_dropped_packets += overflow_count
                        node.drop_reasons["BUFFER_OVERFLOW"] += overflow_count
                        node.interval_overflow_packets[3] += overflow_count
            node.update_hap_buffer_usage(total_enqueued_bits)
        for hid, node in self.haps.items():
            if not node.connected:
                continue
            n_users = len(node.served_ships)
            if n_users == 0:
                continue
            if self.scheduler_policy is not None:
                allocations = self.scheduler_policy.allocate_rbs(
                    hid,
                    node.served_ships,
                    node.total_rbs,
                    node.queues,
                    self.config
                )
                for sid, rbs in allocations.items():
                    if sid in node.servers:
                        node.servers[sid].allocated_rbs = rbs
            else:
                rb_per = node.total_rbs // n_users
                remainder = node.total_rbs % n_users
                for i, sid in enumerate(node.served_ships):
                    extra = 1 if i < remainder else 0
                    node.servers[sid].allocated_rbs = rb_per + extra
        for hid, node in self.haps.items():
            if not node.connected or not node.served_ships:
                continue
            hap_idx = np.where(hap_ids == hid)[0]
            if len(hap_idx) == 0:
                continue
            hap_idx = hap_idx[0]
            hlat, hlon, halt = hap_lats[hap_idx], hap_lons[hap_idx], hap_alts[hap_idx]
            served_list = list(node.served_ships)
            n_served = len(served_list)
            srv_lats, srv_lons, srv_rbs = self._get_scratch_arrays(n_served)
            for i, sid in enumerate(served_list):
                idx = ship_idx_map.get(sid)
                if idx is not None:
                    srv_lats[i] = ship_lats[idx]
                    srv_lons[i] = ship_lons[idx]
                srv_rbs[i] = node.servers[sid].allocated_rbs
            slant_km = self._vectorized_slant_range(srv_lats, srv_lons, hlat, hlon, halt)
            rates_bps = self._vectorized_rate(slant_km, srv_rbs, node.numerology)
            for i, sid in enumerate(served_list):
                server = node.servers[sid]
                server.current_rate_bps = rates_bps[i]
                if rates_bps[i] <= 0 or srv_rbs[i] == 0:
                    continue
                bits_budget = rates_bps[i] * dt
                q = node.queues[sid]
                if self.scheduler_policy is not None:
                    priority_order = self.scheduler_policy.select_priority_for_dequeue(
                        q.bits,
                        q.packets,
                        q.arrival_time_sum,
                        current_time,
                        self.config
                    )
                else:
                    priority_order = [0, 1, 2]
                served, delays, served_by_priority = q.dequeue_bits(
                    bits_budget, current_time, rates_bps[i], priority_order
                )
                server.served_bits = served
                node.update_hap_buffer_usage(-served)
                for priority, delay_list in delays.items():
                    node.interval_delay_sum[priority] += sum(delay_list)
                    node.interval_delay_count[priority] += len(delay_list)
                    node.interval_completed_packets += len(delay_list)
                for priority, bits in served_by_priority.items():
                    node.interval_served_bits[priority] += bits
                ship_idx = node.interval_ship_id_map.get(sid)
                if ship_idx is not None:
                    for priority, bits in served_by_priority.items():
                        if bits > 0:
                            node.interval_served_per_ship_priority[ship_idx, priority - 1] += bits
        return self._build_stats_df(current_time)

    def _get_hap_config(self, hap_id: int) -> Dict[str, Any]:
        for hap_cfg in self.config.get("haps", []):
            if hap_cfg.get("hap_id") == hap_id:
                return {
                    "numerology": hap_cfg.get("numerology", 1),
                    "total_bandwidth_hz": hap_cfg.get("total_bandwidth_hz", 100e6)
                }
        return {}

    def _build_stats_df(self, current_time: float) -> pl.DataFrame:
        stats = []
        for hid, node in self.haps.items():
            for sid in node.served_ships:
                server = node.servers[sid]
                q = node.queues[sid]
                stats.append({
                    "mmsi": sid,
                    "hap_id": hid,
                    "connected": node.connected,
                    "active": q.total_packets > 0,
                    "allocated_rbs": server.allocated_rbs,
                    "q_depth_pkts": q.total_packets,
                    "q_depth_bits": q.total_bits,
                    "current_rate_mbps": server.current_rate_bps / 1e6,
                    "served_bits": server.served_bits,
                    "dropped_bits": 0.0,
                    "completed_pkts": 0,
                    "time": current_time,
                })
        return pl.DataFrame(stats) if stats else pl.DataFrame()

    def get_interval_metrics(self, qoe_targets: Dict[str, Any] = None) -> Dict[str, Any]:
        if qoe_targets is None:
            qoe_targets = {}

        total_completed = 0
        total_dropped = 0
        delay_stats = {1: {"sum": 0.0, "count": 0}, 2: {"sum": 0.0, "count": 0}, 3: {"sum": 0.0, "count": 0}}
        throughput_by_priority = {1: 0.0, 2: 0.0, 3: 0.0}
        drop_reasons = {"BUFFER_OVERFLOW": 0, "COVERAGE_LOSS": 0,
                        "BACKHAUL_DISCONNECT": 0, "STARVATION": 0,
                        "TERRESTRIAL_HANDOFF": 0}

        delay_violation_rate = {1: 0.0, 2: 0.0, 3: 0.0}
        throughput_violation_rate = {1: 0.0, 2: 0.0, 3: 0.0}
        delay_violation_severity = {1: 0.0, 2: 0.0, 3: 0.0}
        throughput_violation_severity = {1: 0.0, 2: 0.0, 3: 0.0}

        utilization_per_hap = {}
        qoe_per_hap = {}
        ships_served_total = 0
        n_haps_with_ships = 0

        for hid, node in self.haps.items():
            total_completed += node.interval_completed_packets
            total_dropped += node.interval_dropped_packets
            for p in [1, 2, 3]:
                delay_stats[p]["sum"] += node.interval_delay_sum[p]
                delay_stats[p]["count"] += node.interval_delay_count[p]
                throughput_by_priority[p] += node.interval_served_bits[p]
            for reason, count in node.drop_reasons.items():
                drop_reasons[reason] = drop_reasons.get(reason, 0) + count

            utilization_per_hap[hid] = node.compute_utilization(self._current_dt)

            qoe_per_hap[hid] = node.compute_qoe(qoe_targets)

            if len(node.served_ships) > 0:
                violations = node.compute_violations(qoe_targets)
                n_ships = len(node.served_ships)
                ships_served_total += n_ships
                n_haps_with_ships += 1

                for p in [1, 2, 3]:
                    delay_violation_rate[p] += violations['delay_violation_rate'][p] * n_ships
                    throughput_violation_rate[p] += violations['throughput_violation_rate'][p] * n_ships
                    delay_violation_severity[p] += violations['delay_violation_severity'][p] * n_ships
                    throughput_violation_severity[p] += violations['throughput_violation_severity'][p] * n_ships

        if ships_served_total > 0:
            for p in [1, 2, 3]:
                delay_violation_rate[p] /= ships_served_total
                throughput_violation_rate[p] /= ships_served_total
                delay_violation_severity[p] /= ships_served_total
                throughput_violation_severity[p] /= ships_served_total

        return {
            "completed_packets": total_completed,
            "dropped_packets": total_dropped,
            "delay_stats_by_priority": delay_stats,
            "throughput_by_priority": throughput_by_priority,
            "drop_reasons": drop_reasons,
            "handover_count": self.interval_handovers,
            "ships_served": ships_served_total,
            "utilization_per_hap": utilization_per_hap,
            "qoe_per_hap": qoe_per_hap,
            "delay_violation_rate": delay_violation_rate,
            "throughput_violation_rate": throughput_violation_rate,
            "delay_violation_severity": delay_violation_severity,
            "throughput_violation_severity": throughput_violation_severity,
        }

    def get_level2_feedback(self, qoe_targets: Dict[str, Any] = None) -> Dict[str, Any]:
        if qoe_targets is None:
            qoe_targets = {}

        qoe_per_slice = {}
        capacity = {}
        load = {}
        backlog = {}

        for hid, node in self.haps.items():
            qoe_per_slice[hid] = node.compute_qoe(qoe_targets)
            capacity[hid] = node.total_rbs * node.rb_bandwidth
            load[hid] = node.compute_utilization(self._current_dt)

            for sid, q in node.queues.items():
                backlog[sid] = q.total_bits

        return {
            'qoe_per_slice': qoe_per_slice,
            'capacity': capacity,
            'load': load,
            'backlog': backlog,
        }

    def clear_interval_metrics(self):
        for node in self.haps.values():
            node.clear_interval_metrics()
        self.interval_handovers = 0
        self.interval_handover_ships.clear()
