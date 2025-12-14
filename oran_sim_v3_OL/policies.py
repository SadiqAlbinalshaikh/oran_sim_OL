from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import logging

from .utils import (
    EARTH_RADIUS_KM,
    HAP_DTYPE,
    SHIP_DTYPE,
    haversine_distance_km_vectorized,
    compute_slant_range_km_vectorized,
)
from .learners import ADER, EG

logger = logging.getLogger(__name__)


class MobilityPolicy(ABC):
    @abstractmethod
    def decide_positions(
        self,
        haps: np.ndarray,
        ships: np.ndarray,
        ports: List[Any],
        dt: float,
        config: Dict[str, Any],
        level2_feedback: Optional[Dict[str, Any]] = None
    ) -> Dict[int, Tuple[float, float, float]]:
        pass

    def update(self, level2_feedback: Dict[str, Any]) -> None:
        pass


class AssociationPolicy(ABC):
    @abstractmethod
    def decide_associations_batch(
        self,
        ships: np.ndarray,
        haps: np.ndarray,
        coverage_matrix: np.ndarray,
        hap_loads: Dict[int, int],
        config: Dict[str, Any],
        level2_feedback: Optional[Dict[str, Any]] = None
    ) -> Dict[int, int]:
        pass


class SchedulerPolicy(ABC):
    @abstractmethod
    def allocate_rbs(
        self,
        hap_id: int,
        served_ships: Set[int],
        total_rbs: int,
        ship_queues: Dict[int, Any],
        config: Dict[str, Any]
    ) -> Dict[int, int]:
        pass

    @abstractmethod
    def select_priority_for_dequeue(
        self,
        queue_bits: np.ndarray,
        queue_packets: np.ndarray,
        arrival_time_sums: np.ndarray,
        current_time: float,
        config: Dict[str, Any]
    ) -> List[int]:
        pass

    def update_weights(self, hap_id: int, losses: Dict[int, float]) -> None:
        pass


@dataclass
class AdmissionResult:
    admitted_p1: np.ndarray
    admitted_p2: np.ndarray
    admitted_p3: np.ndarray
    rejected_p1: np.ndarray
    rejected_p2: np.ndarray
    rejected_p3: np.ndarray


class AdmissionPolicy(ABC):
    @abstractmethod
    def admit_packets(
        self,
        n_p1: np.ndarray,
        n_p2: np.ndarray,
        n_p3: np.ndarray,
        queue_p2_bits: np.ndarray,
        hap_utilization: float,
        available_hap_buffer: float,
        packet_sizes: Tuple[float, float, float],
        config: Dict[str, Any]
    ) -> AdmissionResult:
        pass


class MinimalMobilityPolicy(MobilityPolicy):
    def decide_positions(
        self,
        haps: np.ndarray,
        ships: np.ndarray,
        ports: List[Any],
        dt: float,
        config: Dict[str, Any],
        level2_feedback: Optional[Dict[str, Any]] = None
    ) -> Dict[int, Tuple[float, float, float]]:
        return {
            int(h['hap_id']): (float(h['lat']), float(h['lon']), float(h['altitude']))
            for h in haps
        }


class ADERMobilityPolicy(MobilityPolicy):

    METERS_PER_DEGREE = 111000.0

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ader_per_hap: Dict[int, ADER] = {}

        self.ship_weight = config.get('ship_weight', 1.0)
        self.hap_repulsion_weight = config.get('hap_repulsion_weight', 0.5)
        self.backhaul_penalty_weight = config.get('backhaul_penalty_weight', 10.0)
        self.lipschitz = config.get('lipschitz', 1.0)

        self.max_history_len: int = config.get('max_history_len', 1000)
        self.weight_history: Dict[int, deque] = {}
        self.gradient_history: Dict[int, deque] = {}
        self.loss_history: Dict[int, deque] = {}
        self.action_history: Dict[int, deque] = {}
        self.entropy_history: Dict[int, deque] = {}

        self._cached_haps: Optional[np.ndarray] = None
        self._cached_ships: Optional[np.ndarray] = None
        self._cached_ports: Optional[List[Any]] = None
        self._cached_dt: float = 600.0
        self._cached_config: Optional[Dict[str, Any]] = None

    def decide_positions(
        self,
        haps: np.ndarray,
        ships: np.ndarray,
        ports: List[Any],
        dt: float,
        config: Dict[str, Any],
        level2_feedback: Optional[Dict[str, Any]] = None
    ) -> Dict[int, Tuple[float, float, float]]:
        self._cached_haps = haps.copy() if len(haps) > 0 else haps
        self._cached_ships = ships.copy() if len(ships) > 0 else ships
        self._cached_ports = ports
        self._cached_dt = dt
        self._cached_config = config

        if level2_feedback is None:
            level2_feedback = {}

        results = {}

        for hap in haps:
            hap_id = int(hap['hap_id'])
            max_speed = float(hap['max_speed'])
            coverage_radius = float(hap['coverage_radius'])

            if hap_id not in self.ader_per_hap:
                self.ader_per_hap[hap_id] = self._create_ader(
                    max_speed, dt, config
                )
                self.weight_history[hap_id] = deque(maxlen=self.max_history_len)
                self.gradient_history[hap_id] = deque(maxlen=self.max_history_len)
                self.loss_history[hap_id] = deque(maxlen=self.max_history_len)
                self.action_history[hap_id] = deque(maxlen=self.max_history_len)
                self.entropy_history[hap_id] = deque(maxlen=self.max_history_len)

            ader = self.ader_per_hap[hap_id]

            velocity = ader.get_action()

            new_lat = float(hap['lat']) + velocity[0]
            new_lon = float(hap['lon']) + velocity[1]

            new_lat, new_lon = self._apply_constraints(
                new_lat, new_lon,
                float(hap['lat']), float(hap['lon']),
                max_speed, dt,
                coverage_radius, ports, config
            )

            results[hap_id] = (new_lat, new_lon, float(hap['altitude']))

        return results

    def update(self, level2_feedback: Dict[str, Any]) -> None:
        if self._cached_haps is None or len(self._cached_haps) == 0:
            return

        for hap in self._cached_haps:
            hap_id = int(hap['hap_id'])
            if hap_id not in self.ader_per_hap:
                continue

            ader = self.ader_per_hap[hap_id]

            weights = ader.get_expert_weights().copy()
            self.weight_history[hap_id].append(weights)

            entropy = self._compute_entropy(weights)
            self.entropy_history[hap_id].append(entropy)

            action = ader.get_action().copy()
            self.action_history[hap_id].append(action)

            gradient = self._compute_gradient(
                hap, self._cached_haps, self._cached_ships,
                self._cached_ports, level2_feedback
            )

            self.gradient_history[hap_id].append(gradient.copy())

            loss = float(np.dot(gradient, action))
            self.loss_history[hap_id].append(loss)

            ader.update(gradient)

    def _create_ader(
        self,
        max_speed: float,
        dt: float,
        config: Dict[str, Any]
    ) -> ADER:
        sim_config = config.get('simulator', {})
        duration = sim_config.get('duration', 3600.0)
        mobility_timestep = sim_config.get('mobility_timestep', 600.0)

        T = max(1, int(duration / mobility_timestep))

        max_displacement_deg = (max_speed * dt) / self.METERS_PER_DEGREE
        D = 2 * max_displacement_deg

        L = self.lipschitz

        radius = max_displacement_deg

        def project_ball(v: np.ndarray) -> np.ndarray:
            norm = np.linalg.norm(v)
            if norm > radius:
                return v * (radius / norm)
            return v

        return ADER(
            dim=2,
            T=T,
            D=D,
            L=L,
            constraint_set=project_ball,
            x0=np.zeros(2)
        )

    def _compute_gradient(
        self,
        hap: np.ndarray,
        all_haps: np.ndarray,
        ships: np.ndarray,
        ports: List[Any],
        feedback: Dict[str, Any]
    ) -> np.ndarray:
        grad = np.zeros(2)
        hap_pos = np.array([float(hap['lat']), float(hap['lon'])])
        hap_id = int(hap['hap_id'])

        loads = feedback.get('load', {})
        qoe_per_slice = feedback.get('qoe_per_slice', {})

        if len(ships) > 0:
            for ship in ships:
                ship_pos = np.array([float(ship['lat']), float(ship['lon'])])
                direction = ship_pos - hap_pos
                dist = np.linalg.norm(direction)
                if dist < 1e-6:
                    continue

                weight = self.ship_weight / (dist ** 2 + 1e-6)

                grad -= weight * direction / dist

        for other in all_haps:
            other_id = int(other['hap_id'])
            if other_id == hap_id:
                continue

            other_pos = np.array([float(other['lat']), float(other['lon'])])
            direction = hap_pos - other_pos
            dist = np.linalg.norm(direction)
            if dist < 1e-6:
                continue

            load_weight = loads.get(other_id, 0.5)
            weight = self.hap_repulsion_weight * load_weight / (dist ** 2 + 1e-6)

            grad += weight * direction / dist

        if ports and len(ports) > 0:
            coverage_radius = float(hap['coverage_radius'])
            nearest_port, port_dist = self._find_nearest_port(hap_pos, ports)

            if nearest_port is not None and port_dist > coverage_radius:
                direction = nearest_port - hap_pos
                dist = np.linalg.norm(direction)
                if dist > 1e-6:
                    grad -= self.backhaul_penalty_weight * direction / dist

        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1.0:
            grad = grad / grad_norm

        return grad

    def _find_nearest_port(
        self,
        position: np.ndarray,
        ports: List[Any]
    ) -> Tuple[Optional[np.ndarray], float]:
        if not ports:
            return None, float('inf')

        min_dist = float('inf')
        nearest_pos = None

        for port in ports:
            port_pos = np.array([port.lat, port.lon])
            dist_deg = np.linalg.norm(port_pos - position)
            dist_m = dist_deg * self.METERS_PER_DEGREE

            if dist_m < min_dist:
                min_dist = dist_m
                nearest_pos = port_pos

        return nearest_pos, min_dist

    def _apply_constraints(
        self,
        new_lat: float,
        new_lon: float,
        old_lat: float,
        old_lon: float,
        max_speed: float,
        dt: float,
        coverage_radius: float,
        ports: List[Any],
        config: Dict[str, Any]
    ) -> Tuple[float, float]:
        max_dist_deg = (max_speed * dt) / self.METERS_PER_DEGREE
        dlat = new_lat - old_lat
        dlon = new_lon - old_lon
        dist = np.sqrt(dlat ** 2 + dlon ** 2)

        if dist > max_dist_deg:
            scale = max_dist_deg / dist
            new_lat = old_lat + dlat * scale
            new_lon = old_lon + dlon * scale

        if ports and len(ports) > 0:
            position = np.array([new_lat, new_lon])
            nearest_port, port_dist = self._find_nearest_port(position, ports)

            if nearest_port is not None and port_dist > coverage_radius:
                direction = nearest_port - position
                dir_norm = np.linalg.norm(direction)
                if dir_norm > 1e-6:
                    overshoot_m = port_dist - coverage_radius
                    overshoot_deg = overshoot_m / self.METERS_PER_DEGREE
                    move = (overshoot_deg / dir_norm) * direction
                    new_lat = new_lat + move[0]
                    new_lon = new_lon + move[1]

        return new_lat, new_lon

    @staticmethod
    def _compute_entropy(weights: np.ndarray) -> float:
        eps = 1e-10
        weights = np.clip(weights, eps, 1.0)
        weights = weights / weights.sum()
        return float(-np.sum(weights * np.log(weights)))

    def get_learner_history(self) -> Dict[str, Any]:
        return {
            'weight_history': {hid: list(h) for hid, h in self.weight_history.items()},
            'gradient_history': {hid: list(h) for hid, h in self.gradient_history.items()},
            'loss_history': {hid: list(h) for hid, h in self.loss_history.items()},
            'action_history': {hid: list(h) for hid, h in self.action_history.items()},
            'entropy_history': {hid: list(h) for hid, h in self.entropy_history.items()},
        }

    def compute_cumulative_regret(self, hap_id: int) -> np.ndarray:
        if hap_id not in self.loss_history:
            return np.array([])

        losses = np.array(list(self.loss_history[hap_id]))
        return np.cumsum(losses)


class MinimalAssociationPolicy(AssociationPolicy):
    def decide_associations_batch(
        self,
        ships: np.ndarray,
        haps: np.ndarray,
        coverage_matrix: np.ndarray,
        hap_loads: Dict[int, int],
        config: Dict[str, Any],
        level2_feedback: Optional[Dict[str, Any]] = None
    ) -> Dict[int, int]:
        if len(ships) == 0 or len(haps) == 0:
            return {}
        associations = {}
        ship_lats = ships['lat'][:, np.newaxis]
        ship_lons = ships['lon'][:, np.newaxis]
        hap_lats = haps['lat'][np.newaxis, :]
        hap_lons = haps['lon'][np.newaxis, :]
        dists = haversine_distance_km_vectorized(ship_lats, ship_lons, hap_lats, hap_lons)
        dists[~coverage_matrix] = np.inf
        for i, ship in enumerate(ships):
            if np.any(coverage_matrix[i]):
                best_hap_idx = np.argmin(dists[i])
                associations[int(ship['mmsi'])] = int(haps[best_hap_idx]['hap_id'])
        return associations


class GreedyAssociationPolicy(AssociationPolicy):

    DEFAULT_CAPACITY_BITS = 1e9

    def __init__(self, config: Dict[str, Any]):
        weights = config.get('weights', {})
        self.w_Q = weights.get('queue_transfer', 0.2)
        self.w_L = weights.get('load', 0.4)
        self.w_D = weights.get('distance', 0.25)
        self.w_QoE = weights.get('qoe', 0.15)
        self.max_ships_per_hap = config.get('max_ships_per_hap', 100)

    def decide_associations_batch(
        self,
        ships: np.ndarray,
        haps: np.ndarray,
        coverage_matrix: np.ndarray,
        hap_loads: Dict[int, int],
        config: Dict[str, Any],
        level2_feedback: Optional[Dict[str, Any]] = None
    ) -> Dict[int, int]:
        if len(ships) == 0 or len(haps) == 0:
            return {}

        n_ships = len(ships)
        n_haps = len(haps)

        if level2_feedback is None:
            level2_feedback = {}

        hap_qoe = level2_feedback.get('qoe_per_slice', {})
        hap_capacity = level2_feedback.get('capacity', {})
        ship_backlog = level2_feedback.get('backlog', {})

        ship_lats = ships['lat'][:, np.newaxis]
        ship_lons = ships['lon'][:, np.newaxis]
        hap_lats = haps['lat'][np.newaxis, :]
        hap_lons = haps['lon'][np.newaxis, :]
        distance_matrix_km = haversine_distance_km_vectorized(
            ship_lats, ship_lons, hap_lats, hap_lons
        )

        cost_matrix = np.full((n_ships, n_haps), np.inf, dtype=np.float64)

        for i in range(n_ships):
            ship_id = int(ships[i]['mmsi'])
            B_i = ship_backlog.get(ship_id, 0.0)

            for j in range(n_haps):
                if not coverage_matrix[i, j]:
                    continue

                hap_id = int(haps[j]['hap_id'])

                C_j = hap_capacity.get(hap_id, self.DEFAULT_CAPACITY_BITS)
                C_j = max(C_j, 1.0)
                queue_cost = B_i / C_j

                current_ships = hap_loads.get(hap_id, 0)
                rho_j = current_ships / max(self.max_ships_per_hap, 1)
                load_cost = rho_j

                d_ij = distance_matrix_km[i, j]
                R_j_km = haps[j]['coverage_radius'] / 1000.0
                R_j_km = max(R_j_km, 1.0)
                distance_cost = d_ij / R_j_km

                qoe_j = hap_qoe.get(hap_id, {1: 1.0, 2: 1.0, 3: 1.0})
                min_qoe = min(qoe_j.values()) if qoe_j else 1.0
                qoe_cost = 1.0 - min_qoe

                cost_matrix[i, j] = (
                    self.w_Q * queue_cost +
                    self.w_L * load_cost +
                    self.w_D * distance_cost +
                    self.w_QoE * qoe_cost
                )

        assignments = {}
        current_loads = {int(haps[j]['hap_id']): 0 for j in range(n_haps)}

        min_costs = np.min(cost_matrix, axis=1)
        ship_order = np.argsort(min_costs)

        for i in ship_order:
            if np.isinf(min_costs[i]):
                continue

            ship_id = int(ships[i]['mmsi'])

            valid_mask = coverage_matrix[i] & ~np.isinf(cost_matrix[i])

            for j in range(n_haps):
                if valid_mask[j]:
                    hap_id = int(haps[j]['hap_id'])
                    if current_loads[hap_id] >= self.max_ships_per_hap:
                        valid_mask[j] = False

            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) == 0:
                continue

            sorted_indices = valid_indices[np.argsort(cost_matrix[i, valid_indices])]

            for j in sorted_indices:
                hap_id = int(haps[j]['hap_id'])
                if current_loads[hap_id] < self.max_ships_per_hap:
                    assignments[ship_id] = hap_id
                    current_loads[hap_id] += 1
                    break

        return assignments


class MinimalSchedulerPolicy(SchedulerPolicy):
    def allocate_rbs(
        self,
        hap_id: int,
        served_ships: Set[int],
        total_rbs: int,
        ship_queues: Dict[int, Any],
        config: Dict[str, Any]
    ) -> Dict[int, int]:
        if not served_ships:
            return {}
        rbs_per_ship = max(1, total_rbs // len(served_ships))
        return {sid: rbs_per_ship for sid in served_ships}

    def select_priority_for_dequeue(
        self,
        queue_bits: np.ndarray,
        queue_packets: np.ndarray,
        arrival_time_sums: np.ndarray,
        current_time: float,
        config: Dict[str, Any]
    ) -> List[int]:
        return [0, 1, 2]


class EGRBAllocationPolicy(SchedulerPolicy):

    def __init__(self, config: Dict[str, Any]):
        self.step_size = config.get('step_size', 0.1)
        self.min_weight = config.get('min_weight', 0.01)
        self.dequeue_type = config.get('dequeue_type', 'minimal')

        self.eg_per_hap: Dict[int, EG] = {}
        self.ship_indices: Dict[int, Dict[int, int]] = {}

        self.max_history_len: int = config.get('max_history_len', 1000)
        self.weight_history: Dict[int, deque] = {}
        self.distribution_history: Dict[int, deque] = {}
        self.loss_history: Dict[int, deque] = {}
        self.entropy_history: Dict[int, deque] = {}

    def allocate_rbs(
        self,
        hap_id: int,
        served_ships: Set[int],
        total_rbs: int,
        ship_queues: Dict[int, Any],
        config: Dict[str, Any]
    ) -> Dict[int, int]:
        if not served_ships:
            return {}

        served_list = sorted(served_ships)
        n_ships = len(served_list)

        if hap_id not in self.eg_per_hap:
            self._init_eg(hap_id, served_list)
        else:
            self._sync_eg_arms(hap_id, served_list)

        eg = self.eg_per_hap[hap_id]
        distribution = eg.get_distribution()

        allocations = {}
        remaining_rbs = total_rbs

        for i, ship_id in enumerate(served_list):
            rbs = int(distribution[i] * total_rbs)
            rbs = max(1, min(rbs, remaining_rbs))
            allocations[ship_id] = rbs
            remaining_rbs -= rbs

        if remaining_rbs > 0:
            weight_order = np.argsort(-distribution)
            for i in weight_order:
                if remaining_rbs <= 0:
                    break
                ship_id = served_list[i]
                allocations[ship_id] += 1
                remaining_rbs -= 1

        return allocations

    def select_priority_for_dequeue(
        self,
        queue_bits: np.ndarray,
        queue_packets: np.ndarray,
        arrival_time_sums: np.ndarray,
        current_time: float,
        config: Dict[str, Any]
    ) -> List[int]:
        return [0, 1, 2]

    def update_weights(self, hap_id: int, losses: Dict[int, float]) -> None:
        if hap_id not in self.eg_per_hap:
            return

        eg = self.eg_per_hap[hap_id]
        ship_indices = self.ship_indices.get(hap_id, {})

        weights = eg.get_weights().copy()
        distribution = eg.get_distribution().copy()
        self.weight_history[hap_id].append(weights)
        self.distribution_history[hap_id].append(distribution)

        entropy = self._compute_entropy(distribution)
        self.entropy_history[hap_id].append(entropy)

        self.loss_history[hap_id].append(losses.copy())

        gradient = np.zeros(eg.n_arms)
        for ship_id, loss in losses.items():
            if ship_id in ship_indices:
                gradient[ship_indices[ship_id]] = -loss

        eg.update(gradient)

    def _init_eg(self, hap_id: int, served_list: List[int]) -> None:
        n_ships = len(served_list)
        self.eg_per_hap[hap_id] = EG(
            n_arms=n_ships,
            step_size=self.step_size,
            min_weight=self.min_weight
        )
        self.ship_indices[hap_id] = {sid: i for i, sid in enumerate(served_list)}

        if hap_id not in self.weight_history:
            self.weight_history[hap_id] = deque(maxlen=self.max_history_len)
            self.distribution_history[hap_id] = deque(maxlen=self.max_history_len)
            self.loss_history[hap_id] = deque(maxlen=self.max_history_len)
            self.entropy_history[hap_id] = deque(maxlen=self.max_history_len)

    def _sync_eg_arms(self, hap_id: int, served_list: List[int]) -> None:
        eg = self.eg_per_hap[hap_id]
        current_indices = self.ship_indices[hap_id]
        current_ships = set(current_indices.keys())
        new_ships = set(served_list)

        departed = current_ships - new_ships
        arrived = new_ships - current_ships
        remaining = current_ships - departed

        if len(remaining) == 0:
            self._init_eg(hap_id, served_list)
            return

        if departed:
            indices_to_remove = sorted(
                [current_indices[sid] for sid in departed],
                reverse=True
            )
            for idx in indices_to_remove:
                eg.remove_arm(idx)
            remaining_list = [sid for sid in sorted(current_ships) if sid not in departed]
            current_indices.clear()
            for i, sid in enumerate(remaining_list):
                current_indices[sid] = i

        for ship_id in sorted(arrived):
            eg.add_arm(initial_weight=1.0)
            current_indices[ship_id] = eg.n_arms - 1

    @staticmethod
    def _compute_entropy(distribution: np.ndarray) -> float:
        eps = 1e-10
        distribution = np.clip(distribution, eps, 1.0)
        distribution = distribution / distribution.sum()
        return float(-np.sum(distribution * np.log(distribution)))

    def get_learner_history(self) -> Dict[str, Any]:
        return {
            'weight_history': {hid: list(h) for hid, h in self.weight_history.items()},
            'distribution_history': {hid: list(h) for hid, h in self.distribution_history.items()},
            'loss_history': {hid: list(h) for hid, h in self.loss_history.items()},
            'entropy_history': {hid: list(h) for hid, h in self.entropy_history.items()},
            'ship_indices': self.ship_indices.copy(),
        }

    def compute_cumulative_regret(self, hap_id: int) -> np.ndarray:
        if hap_id not in self.loss_history or len(self.loss_history[hap_id]) == 0:
            return np.array([])

        avg_losses = []
        for loss_dict in self.loss_history[hap_id]:
            if loss_dict:
                avg_loss = sum(loss_dict.values()) / len(loss_dict)
            else:
                avg_loss = 0.0
            avg_losses.append(avg_loss)

        return np.cumsum(np.array(avg_losses))


class MinimalAdmissionPolicy(AdmissionPolicy):
    def admit_packets(
        self,
        n_p1: np.ndarray,
        n_p2: np.ndarray,
        n_p3: np.ndarray,
        queue_p2_bits: np.ndarray,
        hap_utilization: float,
        available_hap_buffer: float,
        packet_sizes: Tuple[float, float, float],
        config: Dict[str, Any]
    ) -> AdmissionResult:
        return AdmissionResult(
            admitted_p1=n_p1.copy(),
            admitted_p2=n_p2.copy(),
            admitted_p3=n_p3.copy(),
            rejected_p1=np.zeros_like(n_p1),
            rejected_p2=np.zeros_like(n_p2),
            rejected_p3=np.zeros_like(n_p3),
        )


class ThresholdAdmissionPolicy(AdmissionPolicy):

    def __init__(self, config: Dict[str, Any]):
        self.theta_high = config.get('theta_high', 0.9)
        self.theta_mid = config.get('theta_mid', 0.7)
        self.p2_threshold = config.get('p2_backlog_threshold', 1e6)

    def admit_packets(
        self,
        n_p1: np.ndarray,
        n_p2: np.ndarray,
        n_p3: np.ndarray,
        queue_p2_bits: np.ndarray,
        hap_utilization: float,
        available_hap_buffer: float,
        packet_sizes: Tuple[float, float, float],
        config: Dict[str, Any]
    ) -> AdmissionResult:
        size_p1, size_p2, size_p3 = packet_sizes
        n_ships = len(n_p1)

        admitted_p1 = np.zeros(n_ships, dtype=np.int32)
        admitted_p2 = np.zeros(n_ships, dtype=np.int32)
        admitted_p3 = np.zeros(n_ships, dtype=np.int32)

        for i in range(n_ships):
            bits_needed = n_p1[i] * size_p1
            if bits_needed <= available_hap_buffer:
                admitted_p1[i] = n_p1[i]
                available_hap_buffer -= bits_needed
            else:
                admitted_p1[i] = int(available_hap_buffer // size_p1) if size_p1 > 0 else 0
                available_hap_buffer -= admitted_p1[i] * size_p1
                available_hap_buffer = max(0.0, available_hap_buffer)

            if queue_p2_bits[i] < self.p2_threshold:
                admit_count = n_p2[i]
                if hap_utilization > self.theta_high:
                    admit_count = admit_count // 2

                bits_needed = admit_count * size_p2
                if bits_needed <= available_hap_buffer:
                    admitted_p2[i] = admit_count
                    available_hap_buffer -= bits_needed
                else:
                    admitted_p2[i] = int(available_hap_buffer // size_p2) if size_p2 > 0 else 0
                    available_hap_buffer -= admitted_p2[i] * size_p2
                    available_hap_buffer = max(0.0, available_hap_buffer)

            if hap_utilization < 1.0:
                admit_count = n_p3[i]
                if hap_utilization > self.theta_high:
                    admit_count = 0
                elif hap_utilization > self.theta_mid:
                    admit_count = admit_count // 2

                bits_needed = admit_count * size_p3
                if bits_needed <= available_hap_buffer:
                    admitted_p3[i] = admit_count
                    available_hap_buffer -= bits_needed
                else:
                    admitted_p3[i] = int(available_hap_buffer // size_p3) if size_p3 > 0 else 0
                    available_hap_buffer -= admitted_p3[i] * size_p3
                    available_hap_buffer = max(0.0, available_hap_buffer)

        return AdmissionResult(
            admitted_p1=admitted_p1,
            admitted_p2=admitted_p2,
            admitted_p3=admitted_p3,
            rejected_p1=n_p1 - admitted_p1,
            rejected_p2=n_p2 - admitted_p2,
            rejected_p3=n_p3 - admitted_p3,
        )


def create_policies(
    policy_name: str,
    config: Dict[str, Any] = None
) -> Tuple[MobilityPolicy, AssociationPolicy, SchedulerPolicy, AdmissionPolicy]:
    if config is None:
        config = {}

    policy_config = config.get('policies', {})

    mobility_cfg = policy_config.get('mobility', {})
    mobility_type = mobility_cfg.get('type', 'minimal')
    if mobility_type == 'ader':
        mobility_policy = ADERMobilityPolicy(mobility_cfg)
    elif mobility_type == 'minimal':
        mobility_policy = MinimalMobilityPolicy()
    else:
        logger.warning(f"Unknown mobility type '{mobility_type}', using minimal")
        mobility_policy = MinimalMobilityPolicy()

    assoc_cfg = policy_config.get('association', {})
    assoc_type = assoc_cfg.get('type', 'minimal')
    if assoc_type == 'greedy':
        association_policy = GreedyAssociationPolicy(assoc_cfg)
    elif assoc_type == 'minimal':
        association_policy = MinimalAssociationPolicy()
    else:
        logger.warning(f"Unknown association type '{assoc_type}', using minimal")
        association_policy = MinimalAssociationPolicy()

    sched_cfg = policy_config.get('scheduler', {})
    rb_alloc_type = sched_cfg.get('rb_allocation_type', sched_cfg.get('type', 'minimal'))
    dequeue_type = sched_cfg.get('dequeue_type', 'minimal')

    if rb_alloc_type == 'eg':
        scheduler_policy = EGRBAllocationPolicy({**sched_cfg, 'dequeue_type': dequeue_type})
    elif rb_alloc_type == 'minimal':
        scheduler_policy = MinimalSchedulerPolicy()
    else:
        logger.warning(f"Unknown rb_allocation_type '{rb_alloc_type}', using minimal")
        scheduler_policy = MinimalSchedulerPolicy()

    admission_cfg = policy_config.get('admission', {})
    admission_type = admission_cfg.get('type', 'minimal')
    if admission_type == 'threshold':
        admission_policy = ThresholdAdmissionPolicy(admission_cfg)
    elif admission_type == 'minimal':
        admission_policy = MinimalAdmissionPolicy()
    else:
        logger.warning(f"Unknown admission type '{admission_type}', using minimal")
        admission_policy = MinimalAdmissionPolicy()

    return (mobility_policy, association_policy, scheduler_policy, admission_policy)
