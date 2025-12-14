from typing import Any, Dict, List, Optional, TYPE_CHECKING
import numpy as np
import polars as pl

if TYPE_CHECKING:
    from .core import ORANSimulator

DEFAULT_QOE_TARGETS = {
    'p1_delay_max_ms': 10.0,
    'p1_throughput_min_bps': 100e3,
    'p1_epsilon_d': 0.001,
    'p1_epsilon_r': 0.01,
    'p1_delay_target': 0.01,
    'p2_delay_max_ms': 100.0,
    'p2_throughput_min_bps': 10e6,
    'p2_epsilon_d': 0.01,
    'p2_epsilon_r': 0.05,
    'p2_throughput_target': 10e6,
    'p3_delay_max_ms': 1000.0,
    'p3_throughput_min_bps': 50e6,
    'p3_epsilon_d': 0.1,
    'p3_epsilon_r': 0.1,
    'p3_throughput_target': 50e6,
    'per_ship_throughput_target': 1e6,
}

QOE_REFERENCE = {
    1: 100e3,
    2: 10e6,
    3: 50e6,
}

QOE_BETA = {
    1: 10.0,
    2: 1.0,
    3: 0.1,
}


class MetricsCollector:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.qoe_targets = {**DEFAULT_QOE_TARGETS, **config.get('qoe_targets', {})}

        self.time_series: Dict[str, List] = {
            'time': [],
            'step': [],
            'ships_needing_hap': [],
            'ships_served': [],
            'coverage_ratio': [],
            'delay_p1_ms': [],
            'delay_p2_ms': [],
            'delay_p3_ms': [],
            'throughput_p1_bps': [],
            'throughput_p2_bps': [],
            'throughput_p3_bps': [],
            'throughput_total_bps': [],
            'delay_violation_rate_p1': [],
            'delay_violation_rate_p2': [],
            'delay_violation_rate_p3': [],
            'throughput_violation_rate_p1': [],
            'throughput_violation_rate_p2': [],
            'throughput_violation_rate_p3': [],
            'qoe_p1': [],
            'qoe_p2': [],
            'qoe_p3': [],
            'completed_packets': [],
            'dropped_packets': [],
            'handover_count': [],
            'drops_overflow': [],
            'drops_coverage': [],
            'drops_backhaul': [],
            'drops_starvation': [],
            'drops_terrestrial': [],
        }

        self.utilization_per_hap: Dict[int, List[float]] = {}
        self.qoe_per_hap: Dict[int, Dict[int, List[float]]] = {}

        self.ader_history: Optional[Dict[str, Any]] = None
        self.eg_history: Optional[Dict[str, Any]] = None

    def collect(self, simulator: 'ORANSimulator', state: Dict[str, Any]) -> None:
        metrics = state.get('metrics', {})

        self.time_series['time'].append(state.get('time', 0.0))
        self.time_series['step'].append(state.get('step', 0))

        self.time_series['ships_needing_hap'].append(metrics.get('ships_needing_hap', 0))
        self.time_series['ships_served'].append(metrics.get('ships_served', 0))
        self.time_series['coverage_ratio'].append(metrics.get('coverage_ratio', 1.0))

        delay_stats = metrics.get('delay_stats_by_priority', {})
        for p, key in [(1, 'delay_p1_ms'), (2, 'delay_p2_ms'), (3, 'delay_p3_ms')]:
            stats = delay_stats.get(p, {})
            count = stats.get('count', 0)
            if count > 0:
                avg_delay_s = stats.get('sum', 0.0) / count
                self.time_series[key].append(avg_delay_s * 1000.0)
            else:
                self.time_series[key].append(0.0)

        tput = metrics.get('throughput_by_priority', {})
        self.time_series['throughput_p1_bps'].append(tput.get(1, 0.0))
        self.time_series['throughput_p2_bps'].append(tput.get(2, 0.0))
        self.time_series['throughput_p3_bps'].append(tput.get(3, 0.0))
        self.time_series['throughput_total_bps'].append(sum(tput.values()))

        delay_viol = metrics.get('delay_violation_rate', {})
        tput_viol = metrics.get('throughput_violation_rate', {})
        self.time_series['delay_violation_rate_p1'].append(delay_viol.get(1, 0.0))
        self.time_series['delay_violation_rate_p2'].append(delay_viol.get(2, 0.0))
        self.time_series['delay_violation_rate_p3'].append(delay_viol.get(3, 0.0))
        self.time_series['throughput_violation_rate_p1'].append(tput_viol.get(1, 0.0))
        self.time_series['throughput_violation_rate_p2'].append(tput_viol.get(2, 0.0))
        self.time_series['throughput_violation_rate_p3'].append(tput_viol.get(3, 0.0))

        qoe_per_hap = metrics.get('qoe_per_hap', {})
        if qoe_per_hap:
            qoe_agg = {1: [], 2: [], 3: []}
            for hap_id, qoe in qoe_per_hap.items():
                for p in [1, 2, 3]:
                    qoe_agg[p].append(qoe.get(p, 1.0))
            for p in [1, 2, 3]:
                if qoe_agg[p]:
                    raw_qoe = np.mean(qoe_agg[p])
                    normalized = self._normalize_qoe(p, raw_qoe)
                    self.time_series[f'qoe_p{p}'].append(normalized)
                else:
                    self.time_series[f'qoe_p{p}'].append(1.0)
        else:
            for p in [1, 2, 3]:
                self.time_series[f'qoe_p{p}'].append(1.0)

        self.time_series['completed_packets'].append(metrics.get('completed_packets', 0))
        self.time_series['dropped_packets'].append(metrics.get('dropped_packets', 0))

        self.time_series['handover_count'].append(metrics.get('handover_count', 0))

        drops = metrics.get('drop_reasons', {})
        self.time_series['drops_overflow'].append(drops.get('BUFFER_OVERFLOW', 0))
        self.time_series['drops_coverage'].append(drops.get('COVERAGE_LOSS', 0))
        self.time_series['drops_backhaul'].append(drops.get('BACKHAUL_DISCONNECT', 0))
        self.time_series['drops_starvation'].append(drops.get('STARVATION', 0))
        self.time_series['drops_terrestrial'].append(drops.get('TERRESTRIAL_HANDOFF', 0))

        util_per_hap = metrics.get('utilization_per_hap', {})
        for hap_id, util in util_per_hap.items():
            if hap_id not in self.utilization_per_hap:
                self.utilization_per_hap[hap_id] = []
            self.utilization_per_hap[hap_id].append(util)

    def _normalize_qoe(self, priority: int, raw_qoe: float) -> float:
        return float(np.clip(raw_qoe, 0.0, 1.0))

    def collect_learner_history(self, simulator: 'ORANSimulator') -> None:
        if hasattr(simulator.mobility_policy, 'get_learner_history'):
            self.ader_history = simulator.mobility_policy.get_learner_history()

        if hasattr(simulator.scheduler_policy, 'get_learner_history'):
            self.eg_history = simulator.scheduler_policy.get_learner_history()

    def get_metrics_dict(self) -> Dict[str, Any]:
        return {
            'time_series': self.time_series,
            'utilization_per_hap': self.utilization_per_hap,
            'ader_history': self.ader_history,
            'eg_history': self.eg_history,
            'qoe_targets': self.qoe_targets,
        }

    def get_metrics_df(self) -> pl.DataFrame:
        return pl.DataFrame(self.time_series)

    def export_csv(self, path: str) -> None:
        df = self.get_metrics_df()
        df.write_csv(path)

    def get_summary_stats(self) -> Dict[str, Any]:
        n = len(self.time_series['time'])
        if n == 0:
            return {}

        half_n = n // 2

        def safe_mean(lst, start=0):
            if not lst or len(lst) <= start:
                return 0.0
            return float(np.mean(lst[start:]))

        return {
            'mean_coverage_ratio': safe_mean(self.time_series['coverage_ratio'], half_n),
            'mean_qoe_p1': safe_mean(self.time_series['qoe_p1'], half_n),
            'mean_qoe_p2': safe_mean(self.time_series['qoe_p2'], half_n),
            'mean_qoe_p3': safe_mean(self.time_series['qoe_p3'], half_n),
            'mean_delay_p1_ms': safe_mean(self.time_series['delay_p1_ms'], half_n),
            'mean_delay_p2_ms': safe_mean(self.time_series['delay_p2_ms'], half_n),
            'mean_delay_p3_ms': safe_mean(self.time_series['delay_p3_ms'], half_n),
            'mean_delay_violation_p1': safe_mean(self.time_series['delay_violation_rate_p1'], half_n),
            'mean_delay_violation_p2': safe_mean(self.time_series['delay_violation_rate_p2'], half_n),
            'mean_delay_violation_p3': safe_mean(self.time_series['delay_violation_rate_p3'], half_n),
            'mean_throughput_violation_p1': safe_mean(self.time_series['throughput_violation_rate_p1'], half_n),
            'mean_throughput_violation_p2': safe_mean(self.time_series['throughput_violation_rate_p2'], half_n),
            'mean_throughput_violation_p3': safe_mean(self.time_series['throughput_violation_rate_p3'], half_n),
            'total_completed_packets': sum(self.time_series['completed_packets']),
            'total_dropped_packets': sum(self.time_series['dropped_packets']),
            'total_handovers': sum(self.time_series['handover_count']),
        }
