import argparse
import logging
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from typing import Dict, Any, List, Tuple
from pathlib import Path

from .core import ORANSimulator
from .utils import create_geodesic_circle_poly

logger = logging.getLogger(__name__)

SLICE_CONFIG = {
    1: {"name": "URLLC (Safety)", "color": "#e74c3c", "short": "URLLC"},
    2: {"name": "eMBB+mMTC (Ops)", "color": "#3498db", "short": "eMBB+mMTC"},
    3: {"name": "eMBB (Entertainment)", "color": "#2ecc71", "short": "eMBB"},
}


def apply_ewma(data: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    if len(data) == 0 or alpha >= 1.0:
        return data
    if alpha <= 0:
        alpha = 0.001
    result = np.empty_like(data, dtype=np.float64)
    result[0] = data[0]
    complement = 1.0 - alpha
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + complement * result[i - 1]
    return result


AVAILABLE_PLOTS = {
    "map": "Geographic map with HAPs, ships, and coverage areas",
    "delay_urllc": "Delay (ms) for URLLC slice",
    "delay_embb_mtc": "Delay (ms) for eMBB+mMTC slice",
    "delay_embb": "Delay (ms) for eMBB slice",
    "throughput": "Total throughput (Mbps)",
    "queue": "Total queue depth (packets)",
    "packets": "Completed and dropped packets per interval",
    "drops": "Cumulative drops by reason",
    "ships": "Number of ships over time",
    "coverage": "Coverage ratio (ships served / ships needing HAP)",
    "throughput_per_slice": "Throughput per slice with R_min thresholds",
    "violations": "Delay and throughput violation rates",
    "utilization": "Per-HAP utilization over time",
    "qoe": "Normalized QoE per slice",
    "ader_weights": "ADER expert weight evolution (stacked area)",
    "ader_entropy": "ADER weight entropy over time",
    "eg_weights": "EG weight distribution heatmap",
    "regret": "Cumulative regret curves",
    "gradient_mag": "ADER gradient magnitude over time",
}

QOS_THRESHOLDS = {
    1: {"delay_max_ms": 10, "throughput_min_mbps": 0.1, "epsilon_d": 0.001, "epsilon_r": 0.01},
    2: {"delay_max_ms": 100, "throughput_min_mbps": 10, "epsilon_d": 0.01, "epsilon_r": 0.05},
    3: {"delay_max_ms": 1000, "throughput_min_mbps": 50, "epsilon_d": 0.1, "epsilon_r": 0.1},
}


def run_simulation(config: Dict[str, Any], policy_name: str = "default", progress_interval: int = 100) -> Tuple[ORANSimulator, List[Dict], Dict[str, Any]]:
    sim = ORANSimulator(config, policy_name=policy_name)
    sim_duration = config.get("simulator", {}).get("duration", 1000)
    states = []
    step = 0
    start_time = time.time()
    logger.info(f"Running simulation for {sim_duration}s...")
    while sim.step():
        state = sim.get_current_state()
        metrics = sim.get_interval_metrics()
        state["metrics"] = metrics
        states.append(state)
        step += 1
        if step % progress_interval == 0:
            pct = (sim.current_time / sim_duration) * 100
            elapsed = time.time() - start_time
            rate = step / elapsed if elapsed > 0 else 0
            logger.info(f"Step {step} | Time: {sim.current_time:.0f}s ({pct:.1f}%) | Rate: {rate:.1f} steps/s")
    total_time = time.time() - start_time
    logger.info(f"Simulation complete: {step} steps in {total_time:.1f}s ({step/total_time:.1f} steps/s)")

    learner_history = {}
    if hasattr(sim.mobility_policy, 'get_learner_history'):
        learner_history['ader'] = sim.mobility_policy.get_learner_history()
    if hasattr(sim.scheduler_policy, 'get_learner_history'):
        learner_history['eg'] = sim.scheduler_policy.get_learner_history()

    return sim, states, learner_history


def extract_metrics(states: List[Dict], smoothing_alpha: float = 0.0, config: Dict[str, Any] = None) -> Dict[str, Any]:
    n = len(states)
    if n == 0:
        return {k: [] for k in ["time", "delay_urllc_ms", "delay_embb_mtc_ms",
                                "delay_embb_ms", "throughput_mbps", "queue_depth",
                                "completed_packets", "dropped_packets", "cum_overflow",
                                "cum_coverage", "cum_backhaul", "cum_starvation",
                                "cum_terrestrial", "n_ships", "coverage_ratio",
                                "throughput_p1_mbps", "throughput_p2_mbps", "throughput_p3_mbps",
                                "delay_violation_p1", "delay_violation_p2", "delay_violation_p3",
                                "throughput_violation_p1", "throughput_violation_p2", "throughput_violation_p3",
                                "qoe_p1", "qoe_p2", "qoe_p3"]}

    qoe_targets = {}
    if config:
        qoe_targets = config.get('qoe_targets', {})

    time_arr = np.zeros(n, dtype=np.float64)
    delay_urllc = np.zeros(n, dtype=np.float64)
    delay_embb_mtc = np.zeros(n, dtype=np.float64)
    delay_embb = np.zeros(n, dtype=np.float64)
    throughput_arr = np.zeros(n, dtype=np.float64)
    queue_depth_arr = np.zeros(n, dtype=np.int64)
    completed_arr = np.zeros(n, dtype=np.int64)
    dropped_arr = np.zeros(n, dtype=np.int64)
    n_ships_arr = np.zeros(n, dtype=np.int64)

    coverage_ratio_arr = np.zeros(n, dtype=np.float64)
    throughput_p1_arr = np.zeros(n, dtype=np.float64)
    throughput_p2_arr = np.zeros(n, dtype=np.float64)
    throughput_p3_arr = np.zeros(n, dtype=np.float64)
    delay_viol_p1_arr = np.zeros(n, dtype=np.float64)
    delay_viol_p2_arr = np.zeros(n, dtype=np.float64)
    delay_viol_p3_arr = np.zeros(n, dtype=np.float64)
    tput_viol_p1_arr = np.zeros(n, dtype=np.float64)
    tput_viol_p2_arr = np.zeros(n, dtype=np.float64)
    tput_viol_p3_arr = np.zeros(n, dtype=np.float64)
    qoe_p1_arr = np.zeros(n, dtype=np.float64)
    qoe_p2_arr = np.zeros(n, dtype=np.float64)
    qoe_p3_arr = np.zeros(n, dtype=np.float64)

    overflow_raw = np.zeros(n, dtype=np.int64)
    coverage_raw = np.zeros(n, dtype=np.int64)
    backhaul_raw = np.zeros(n, dtype=np.int64)
    starvation_raw = np.zeros(n, dtype=np.int64)
    terrestrial_raw = np.zeros(n, dtype=np.int64)

    utilization_per_hap: Dict[int, List[float]] = {}

    delay_arrays = {1: delay_urllc, 2: delay_embb_mtc, 3: delay_embb}

    for i, state in enumerate(states):
        links = state.get("links", [])
        interval_metrics = state.get("metrics", {})

        time_arr[i] = state.get("time", 0)

        delay_stats = interval_metrics.get("delay_stats_by_priority", {})
        for p, arr in delay_arrays.items():
            stats = delay_stats.get(p, {})
            count = stats.get("count", 0)
            if count > 0:
                arr[i] = (stats["sum"] / count) * 1000

        tput = interval_metrics.get("throughput_by_priority", {})
        throughput_arr[i] = sum(tput.values()) / 1e6
        throughput_p1_arr[i] = tput.get(1, 0) / 1e6
        throughput_p2_arr[i] = tput.get(2, 0) / 1e6
        throughput_p3_arr[i] = tput.get(3, 0) / 1e6

        if links:
            queue_depth_arr[i] = sum(link.get("q_depth_pkts", 0) for link in links)

        completed_arr[i] = interval_metrics.get("completed_packets", 0)
        dropped_arr[i] = interval_metrics.get("dropped_packets", 0)

        coverage_ratio_arr[i] = interval_metrics.get("coverage_ratio", 1.0)

        delay_viol = interval_metrics.get("delay_violation_rate", {})
        tput_viol = interval_metrics.get("throughput_violation_rate", {})
        delay_viol_p1_arr[i] = delay_viol.get(1, 0.0)
        delay_viol_p2_arr[i] = delay_viol.get(2, 0.0)
        delay_viol_p3_arr[i] = delay_viol.get(3, 0.0)
        tput_viol_p1_arr[i] = tput_viol.get(1, 0.0)
        tput_viol_p2_arr[i] = tput_viol.get(2, 0.0)
        tput_viol_p3_arr[i] = tput_viol.get(3, 0.0)

        qoe_per_hap = interval_metrics.get("qoe_per_hap", {})
        if qoe_per_hap:
            qoe_vals = {1: [], 2: [], 3: []}
            for hap_id, qoe in qoe_per_hap.items():
                for p in [1, 2, 3]:
                    qoe_vals[p].append(qoe.get(p, 1.0))
            qoe_p1_arr[i] = np.mean(qoe_vals[1]) if qoe_vals[1] else 1.0
            qoe_p2_arr[i] = np.mean(qoe_vals[2]) if qoe_vals[2] else 1.0
            qoe_p3_arr[i] = np.mean(qoe_vals[3]) if qoe_vals[3] else 1.0
        else:
            qoe_p1_arr[i] = 1.0
            qoe_p2_arr[i] = 1.0
            qoe_p3_arr[i] = 1.0

        util_per_hap = interval_metrics.get("utilization_per_hap", {})
        for hap_id, util in util_per_hap.items():
            if hap_id not in utilization_per_hap:
                utilization_per_hap[hap_id] = [0.0] * i
            utilization_per_hap[hap_id].append(util)
        for hap_id in utilization_per_hap:
            if len(utilization_per_hap[hap_id]) <= i:
                utilization_per_hap[hap_id].append(0.0)

        drop_reasons = interval_metrics.get("drop_reasons", {})
        overflow_raw[i] = drop_reasons.get("BUFFER_OVERFLOW", 0)
        coverage_raw[i] = drop_reasons.get("COVERAGE_LOSS", 0)
        backhaul_raw[i] = drop_reasons.get("BACKHAUL_DISCONNECT", 0)
        starvation_raw[i] = drop_reasons.get("STARVATION", 0)
        terrestrial_raw[i] = drop_reasons.get("TERRESTRIAL_HANDOFF", 0)

        n_ships_arr[i] = len(state.get("ships", []))

    cum_overflow = np.cumsum(overflow_raw)
    cum_coverage = np.cumsum(coverage_raw)
    cum_backhaul = np.cumsum(backhaul_raw)
    cum_starvation = np.cumsum(starvation_raw)
    cum_terrestrial = np.cumsum(terrestrial_raw)

    if smoothing_alpha > 0:
        delay_urllc = apply_ewma(delay_urllc, smoothing_alpha)
        delay_embb_mtc = apply_ewma(delay_embb_mtc, smoothing_alpha)
        delay_embb = apply_ewma(delay_embb, smoothing_alpha)
        throughput_arr = apply_ewma(throughput_arr, smoothing_alpha)
        queue_depth_arr = apply_ewma(queue_depth_arr.astype(np.float64), smoothing_alpha)
        coverage_ratio_arr = apply_ewma(coverage_ratio_arr, smoothing_alpha)
        throughput_p1_arr = apply_ewma(throughput_p1_arr, smoothing_alpha)
        throughput_p2_arr = apply_ewma(throughput_p2_arr, smoothing_alpha)
        throughput_p3_arr = apply_ewma(throughput_p3_arr, smoothing_alpha)
        qoe_p1_arr = apply_ewma(qoe_p1_arr, smoothing_alpha)
        qoe_p2_arr = apply_ewma(qoe_p2_arr, smoothing_alpha)
        qoe_p3_arr = apply_ewma(qoe_p3_arr, smoothing_alpha)
        util_smoothing_alpha = 0.05
        for hap_id in utilization_per_hap:
            util_arr = np.array(utilization_per_hap[hap_id], dtype=np.float64)
            utilization_per_hap[hap_id] = apply_ewma(util_arr, util_smoothing_alpha).tolist()

    return {
        "time": time_arr.tolist(),
        "delay_urllc_ms": delay_urllc.tolist(),
        "delay_embb_mtc_ms": delay_embb_mtc.tolist(),
        "delay_embb_ms": delay_embb.tolist(),
        "throughput_mbps": throughput_arr.tolist(),
        "queue_depth": queue_depth_arr.tolist(),
        "completed_packets": completed_arr.tolist(),
        "dropped_packets": dropped_arr.tolist(),
        "cum_overflow": cum_overflow.tolist(),
        "cum_coverage": cum_coverage.tolist(),
        "cum_backhaul": cum_backhaul.tolist(),
        "cum_starvation": cum_starvation.tolist(),
        "cum_terrestrial": cum_terrestrial.tolist(),
        "n_ships": n_ships_arr.tolist(),
        "coverage_ratio": coverage_ratio_arr.tolist(),
        "throughput_p1_mbps": throughput_p1_arr.tolist(),
        "throughput_p2_mbps": throughput_p2_arr.tolist(),
        "throughput_p3_mbps": throughput_p3_arr.tolist(),
        "delay_violation_p1": delay_viol_p1_arr.tolist(),
        "delay_violation_p2": delay_viol_p2_arr.tolist(),
        "delay_violation_p3": delay_viol_p3_arr.tolist(),
        "throughput_violation_p1": tput_viol_p1_arr.tolist(),
        "throughput_violation_p2": tput_viol_p2_arr.tolist(),
        "throughput_violation_p3": tput_viol_p3_arr.tolist(),
        "qoe_p1": qoe_p1_arr.tolist(),
        "qoe_p2": qoe_p2_arr.tolist(),
        "qoe_p3": qoe_p3_arr.tolist(),
        "utilization_per_hap": utilization_per_hap,
    }


def render_video(
    config: Dict[str, Any],
    output_path: str = "simulation.mp4",
    policy_name: str = "default",
    frame_skip: int = 1,
    smoothing_alpha: float = 0.1,
    plots: List[str] = None,
    dpi: int = 100,
    states: List[Dict] = None,
    learner_history: Dict[str, Any] = None
):
    if plots is None:
        plots = list(AVAILABLE_PLOTS.keys())
    logger.info(f"Starting video render with plots: {plots}")
    if states is None:
        sim, states, learner_history = run_simulation(config, policy_name=policy_name)
    elif learner_history is None:
        learner_history = {}
    if not states:
        logger.warning("No simulation states to render")
        return
    t0 = time.time()
    metrics = extract_metrics(states, smoothing_alpha)
    logger.info(f"extract_metrics completed in {time.time() - t0:.2f}s")
    t0 = time.time()
    states_to_render = states[::frame_skip] if frame_skip > 1 else states
    n_frames = len(states_to_render)
    logger.info(f"Rendering {n_frames} frames (frame_skip={frame_skip}) [slice took {time.time() - t0:.4f}s]")
    t0 = time.time()
    n_plots = len(plots)
    if n_plots <= 1:
        n_rows, n_cols = 1, 1
    elif n_plots <= 2:
        n_rows, n_cols = 1, 2
    elif n_plots <= 4:
        n_rows, n_cols = 2, 2
    elif n_plots <= 6:
        n_rows, n_cols = 2, 3
    else:
        n_rows, n_cols = 3, 3
    figsize = config.get("visualization", {}).get("figsize", (14, 10))
    fig = plt.figure(figsize=figsize, dpi=dpi)
    axes = {}
    for i, plot_name in enumerate(plots):
        if plot_name == "map":
            ax = fig.add_subplot(n_rows, n_cols, i + 1, projection=ccrs.PlateCarree())
        else:
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
        axes[plot_name] = ax
    plt.tight_layout(pad=2.0)
    logger.info(f"Figure setup completed in {time.time() - t0:.2f}s")
    map_bounds = config.get("visualization", {}).get("map_bounds", {})
    coverage_cache = {}
    hap_coverage_patches = []
    if "map" in axes:
        ax_map = axes["map"]
        max_haps = max(len(s.get("haps", [])) for s in states_to_render)
        for _ in range(max_haps):
            patch = Polygon([(0, 0)], closed=True, alpha=0.2,
                           facecolor='blue', edgecolor='blue',
                           linewidth=1.5, transform=ccrs.PlateCarree())
            patch.set_visible(False)
            ax_map.add_patch(patch)
            hap_coverage_patches.append(patch)

    def init():
        if "map" in axes:
            ax = axes["map"]
            ax.clear()
            ax.set_extent([map_bounds.get("lon_min", 50), map_bounds.get("lon_max", 60),
                          map_bounds.get("lat_min", 20), map_bounds.get("lat_max", 30)],
                         crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND, facecolor='lightgray')
            ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
        return []

    def update(frame_idx):
        state = states_to_render[frame_idx]
        original_idx = frame_idx * frame_skip
        current_time = state.get("time", 0)
        if "map" in axes:
            ax = axes["map"]
            ax.clear()
            ax.set_extent([map_bounds.get("lon_min", 50), map_bounds.get("lon_max", 60),
                          map_bounds.get("lat_min", 20), map_bounds.get("lat_max", 30)],
                         crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND, facecolor='lightgray')
            ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
            for gs in state.get("ground_stations", []):
                ax.plot(gs["lon"], gs["lat"], 's', color='blue', markersize=6,
                       alpha=0.6, transform=ccrs.PlateCarree())
            for hap in state.get("haps", []):
                hid = hap["hap_id"]
                lat, lon = hap["lat"], hap["lon"]
                radius = hap.get("coverage_radius", 100000)
                cache_key = (hid, round(lat, 2), round(lon, 2))
                if cache_key not in coverage_cache:
                    try:
                        lons, lats = create_geodesic_circle_poly(lon, lat, radius, 32)
                        coverage_cache[cache_key] = (lons, lats)
                    except:
                        coverage_cache[cache_key] = None
                if coverage_cache[cache_key]:
                    clons, clats = coverage_cache[cache_key]
                    ax.fill(clons, clats, alpha=0.15, color='blue', transform=ccrs.PlateCarree())
                ax.plot(lon, lat, '^', color='red', markersize=10, transform=ccrs.PlateCarree())
                ax.text(lon, lat + 0.3, f"HAP{hid}", fontsize=8, ha='center',
                       transform=ccrs.PlateCarree())
            max_people = max((s.get("passengers", 0) or 0) + (s.get("crew", 0) or 0) for s in state.get("ships", [])) or 1
            for ship in state.get("ships", []):
                passengers = ship.get("passengers", 0) or 0
                crew = ship.get("crew", 0) or 0
                total_people = passengers + crew 
                ship_size = 2 + (5 * (total_people / max_people))
                ax.plot(ship["lon"], ship["lat"], 'o', color='yellow', markersize=ship_size,
                       markeredgecolor='black', markeredgewidth=0.3, transform=ccrs.PlateCarree())
            legend_elements = [
                Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=10, label='HAP'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markeredgecolor='black',
                       markeredgewidth=0.3, markersize=8, label='Ship'),
                Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', alpha=0.6, markersize=8, label='Port'),
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.9)
            ax.set_title(f"Time: {current_time:.0f}s | Ships: {len(state.get('ships', []))}")
        if "queue" in axes:
            ax = axes["queue"]
            ax.clear()
            t = metrics["time"][:original_idx+1]
            ax.plot(t, metrics["queue_depth"][:original_idx+1], 'b-', linewidth=1)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Queue Depth (packets)")
            ax.set_title("Total Queue Depth")
            ax.grid(True, alpha=0.3)
        if "packets" in axes:
            ax = axes["packets"]
            ax.clear()
            t = metrics["time"][:original_idx+1]
            ax.plot(t, metrics["completed_packets"][:original_idx+1], 'g-', label="Completed", linewidth=1)
            ax.plot(t, metrics["dropped_packets"][:original_idx+1], 'r-', label="Dropped", linewidth=1)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Packets")
            ax.set_title("Packets per Interval")
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
        if "drops" in axes:
            ax = axes["drops"]
            ax.clear()
            t = metrics["time"][:original_idx+1]
            ax.stackplot(t,
                        metrics["cum_overflow"][:original_idx+1],
                        metrics["cum_coverage"][:original_idx+1],
                        metrics["cum_backhaul"][:original_idx+1],
                        metrics["cum_starvation"][:original_idx+1],
                        metrics["cum_terrestrial"][:original_idx+1],
                        labels=["Overflow", "Coverage", "Backhaul", "Starvation", "Terrestrial"],
                        colors=['red', 'orange', 'purple', 'brown', 'gray'])
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Cumulative Drops")
            ax.set_title("Drops by Reason")
            ax.legend(loc='upper left', fontsize=7)
            ax.grid(True, alpha=0.3)
        if "ships" in axes:
            ax = axes["ships"]
            ax.clear()
            t = metrics["time"][:original_idx+1]
            ax.plot(t, metrics["n_ships"][:original_idx+1], 'b-', linewidth=1)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Ships")
            ax.set_title("Number of Ships")
            ax.grid(True, alpha=0.3)
        if "throughput" in axes:
            ax = axes["throughput"]
            ax.clear()
            t = metrics["time"][:original_idx+1]
            ax.plot(t, metrics["throughput_mbps"][:original_idx+1], 'b-', linewidth=1)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Throughput (Mbps)")
            ax.set_title("Total Throughput")
            ax.grid(True, alpha=0.3)
        for p, key, name in [(1, "delay_urllc", "URLLC"), (2, "delay_embb_mtc", "eMBB+mMTC"), (3, "delay_embb", "eMBB")]:
            if key in axes:
                ax = axes[key]
                ax.clear()
                t = metrics["time"][:original_idx+1]
                ax.plot(t, metrics[f"{key}_ms"][:original_idx+1], '-',
                       color=SLICE_CONFIG[p]["color"], linewidth=1)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Delay (ms)")
                ax.set_title(f"Delay - {name}")
                ax.grid(True, alpha=0.3)
        return []

    fps = config.get("visualization", {}).get("fps", 10)
    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=n_frames,
        interval=1000/fps, blit=False
    )
    logger.info(f"Saving video to {output_path}...")
    writer = animation.FFMpegWriter(fps=fps, metadata={'title': 'ORAN Simulation'})
    anim.save(output_path, writer=writer)
    plt.close(fig)
    logger.info(f"Video saved: {output_path}")


def render_stats_png(
    config: Dict[str, Any],
    output_dir: str = "stats",
    policy_name: str = "default",
    smoothing_alpha: float = 0.1,
    plots: List[str] = None,
    dpi: int = 150,
    states: List[Dict] = None,
    learner_history: Dict[str, Any] = None
):
    if plots is None:
        plots = list(AVAILABLE_PLOTS.keys())
        plots.remove("map")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if states is None:
        logger.info(f"Running simulation for stats generation...")
        sim, states, learner_history = run_simulation(config, policy_name=policy_name)
    elif learner_history is None:
        learner_history = {}
    if not states:
        logger.warning("No simulation states")
        return
    metrics = extract_metrics(states, smoothing_alpha, config)
    t = metrics["time"]

    qoe_targets = config.get('qoe_targets', {})
    thresholds = {
        1: {
            "delay_max_ms": qoe_targets.get('p1_delay_max_ms', QOS_THRESHOLDS[1]["delay_max_ms"]),
            "throughput_min_mbps": qoe_targets.get('p1_throughput_min_bps', QOS_THRESHOLDS[1]["throughput_min_mbps"] * 1e6) / 1e6,
            "epsilon_d": qoe_targets.get('p1_epsilon_d', QOS_THRESHOLDS[1]["epsilon_d"]),
            "epsilon_r": qoe_targets.get('p1_epsilon_r', QOS_THRESHOLDS[1]["epsilon_r"]),
        },
        2: {
            "delay_max_ms": qoe_targets.get('p2_delay_max_ms', QOS_THRESHOLDS[2]["delay_max_ms"]),
            "throughput_min_mbps": qoe_targets.get('p2_throughput_min_bps', QOS_THRESHOLDS[2]["throughput_min_mbps"] * 1e6) / 1e6,
            "epsilon_d": qoe_targets.get('p2_epsilon_d', QOS_THRESHOLDS[2]["epsilon_d"]),
            "epsilon_r": qoe_targets.get('p2_epsilon_r', QOS_THRESHOLDS[2]["epsilon_r"]),
        },
        3: {
            "delay_max_ms": qoe_targets.get('p3_delay_max_ms', QOS_THRESHOLDS[3]["delay_max_ms"]),
            "throughput_min_mbps": qoe_targets.get('p3_throughput_min_bps', QOS_THRESHOLDS[3]["throughput_min_mbps"] * 1e6) / 1e6,
            "epsilon_d": qoe_targets.get('p3_epsilon_d', QOS_THRESHOLDS[3]["epsilon_d"]),
            "epsilon_r": qoe_targets.get('p3_epsilon_r', QOS_THRESHOLDS[3]["epsilon_r"]),
        },
    }

    for plot_name in plots:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
        rendered = True

        if plot_name == "queue":
            ax.plot(t, metrics["queue_depth"], 'b-', linewidth=1)
            ax.set_ylabel("Queue Depth (packets)")
            ax.set_title("Total Queue Depth")
        elif plot_name == "packets":
            ax.plot(t, metrics["completed_packets"], 'g-', label="Completed", linewidth=1)
            ax.plot(t, metrics["dropped_packets"], 'r-', label="Dropped", linewidth=1)
            ax.set_ylabel("Packets")
            ax.set_title("Packets per Interval")
            ax.legend()
        elif plot_name == "drops":
            ax.stackplot(t,
                        metrics["cum_overflow"],
                        metrics["cum_coverage"],
                        metrics["cum_backhaul"],
                        metrics["cum_starvation"],
                        metrics["cum_terrestrial"],
                        labels=["Overflow", "Coverage", "Backhaul", "Starvation", "Terrestrial"],
                        colors=['red', 'orange', 'purple', 'brown', 'gray'])
            ax.set_ylabel("Cumulative Drops")
            ax.set_title("Drops by Reason")
            ax.legend(loc='upper left')
        elif plot_name == "ships":
            ax.plot(t, metrics["n_ships"], 'b-', linewidth=1)
            ax.set_ylabel("Ships")
            ax.set_title("Number of Ships")
        elif plot_name == "throughput":
            ax.plot(t, metrics["throughput_mbps"], 'b-', linewidth=1)
            ax.set_ylabel("Throughput (Mbps)")
            ax.set_title("Total Throughput")
        elif plot_name == "delay_urllc":
            ax.plot(t, metrics["delay_urllc_ms"], '-', color=SLICE_CONFIG[1]["color"], linewidth=1)
            ax.axhline(y=thresholds[1]["delay_max_ms"], color='red', linestyle='--',
                      alpha=0.7, label=f'D_max = {thresholds[1]["delay_max_ms"]} ms')
            ax.set_ylabel("Delay (ms)")
            ax.set_title("Delay - URLLC (P1)")
            ax.legend()
        elif plot_name == "delay_embb_mtc":
            ax.plot(t, metrics["delay_embb_mtc_ms"], '-', color=SLICE_CONFIG[2]["color"], linewidth=1)
            ax.axhline(y=thresholds[2]["delay_max_ms"], color='red', linestyle='--',
                      alpha=0.7, label=f'D_max = {thresholds[2]["delay_max_ms"]} ms')
            ax.set_ylabel("Delay (ms)")
            ax.set_title("Delay - eMBB+mMTC (P2)")
            ax.legend()
        elif plot_name == "delay_embb":
            ax.plot(t, metrics["delay_embb_ms"], '-', color=SLICE_CONFIG[3]["color"], linewidth=1)
            ax.axhline(y=thresholds[3]["delay_max_ms"], color='red', linestyle='--',
                      alpha=0.7, label=f'D_max = {thresholds[3]["delay_max_ms"]} ms')
            ax.set_ylabel("Delay (ms)")
            ax.set_title("Delay - eMBB (P3)")
            ax.legend()

        elif plot_name == "coverage":
            ax.plot(t, metrics["coverage_ratio"], 'b-', linewidth=1)
            ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Target (100%)')
            ax.set_ylabel("Coverage Ratio")
            ax.set_ylim(0, 1.1)
            ax.set_title("Coverage Ratio (Ships Served / Ships Needing HAP)")
            ax.legend()

        elif plot_name == "throughput_per_slice":
            plt.close(fig)
            fig, axes_arr = plt.subplots(3, 1, figsize=(10, 10), dpi=dpi, sharex=True)
            for i, (p, label) in enumerate([(1, "P1 (URLLC)"), (2, "P2 (eMBB+mMTC)"), (3, "P3 (eMBB)")]):
                ax_sub = axes_arr[i]
                ax_sub.plot(t, metrics[f"throughput_p{p}_mbps"], '-',
                           color=SLICE_CONFIG[p]["color"], linewidth=1)
                ax_sub.axhline(y=thresholds[p]["throughput_min_mbps"], color='red',
                              linestyle='--', alpha=0.7, label=f'R_min = {thresholds[p]["throughput_min_mbps"]} Mbps')
                ax_sub.set_ylabel("Throughput (Mbps)")
                ax_sub.set_title(f"Throughput - {label}")
                ax_sub.legend(loc='upper right')
                ax_sub.grid(True, alpha=0.3)
            axes_arr[-1].set_xlabel("Time (s)")
            fig.tight_layout()
            fig_path = output_path / f"{plot_name}.png"
            fig.savefig(fig_path)
            plt.close(fig)
            logger.info(f"Saved: {fig_path}")
            continue

        elif plot_name == "violations":
            plt.close(fig)
            fig, axes_arr = plt.subplots(2, 1, figsize=(10, 8), dpi=dpi, sharex=True)

            ax_delay = axes_arr[0]
            for p in [1, 2, 3]:
                ax_delay.plot(t, metrics[f"delay_violation_p{p}"], '-',
                             color=SLICE_CONFIG[p]["color"], linewidth=1,
                             label=f'{SLICE_CONFIG[p]["short"]}')
                ax_delay.axhline(y=thresholds[p]["epsilon_d"], color=SLICE_CONFIG[p]["color"],
                                linestyle=':', alpha=0.5)
            ax_delay.set_ylabel("Delay Violation Rate")
            ax_delay.set_title("Delay Violation Rates (J^D_k) with ε thresholds")
            ax_delay.legend(loc='upper right')
            ax_delay.grid(True, alpha=0.3)

            ax_tput = axes_arr[1]
            for p in [1, 2, 3]:
                ax_tput.plot(t, metrics[f"throughput_violation_p{p}"], '-',
                            color=SLICE_CONFIG[p]["color"], linewidth=1,
                            label=f'{SLICE_CONFIG[p]["short"]}')
                ax_tput.axhline(y=thresholds[p]["epsilon_r"], color=SLICE_CONFIG[p]["color"],
                               linestyle=':', alpha=0.5)
            ax_tput.set_ylabel("Throughput Violation Rate")
            ax_tput.set_xlabel("Time (s)")
            ax_tput.set_title("Throughput Violation Rates (J^R_k) with ε thresholds")
            ax_tput.legend(loc='upper right')
            ax_tput.grid(True, alpha=0.3)

            fig.tight_layout()
            fig_path = output_path / f"{plot_name}.png"
            fig.savefig(fig_path)
            plt.close(fig)
            logger.info(f"Saved: {fig_path}")
            continue

        elif plot_name == "utilization":
            util_per_hap = metrics.get("utilization_per_hap", {})
            if not util_per_hap:
                logger.warning(f"No utilization data for {plot_name}")
                plt.close(fig)
                continue
            colors = plt.cm.tab10(np.linspace(0, 1, len(util_per_hap)))
            for i, (hap_id, util_list) in enumerate(sorted(util_per_hap.items())):
                t_util = t[:len(util_list)]
                ax.plot(t_util, util_list, '-', color=colors[i], linewidth=1,
                       label=f'HAP {hap_id}')
            ax.set_ylabel("Utilization")
            ax.set_ylim(0, 1.1)
            ax.set_title("Per-HAP Utilization (ρ_j)")
            ax.legend(loc='upper right')

        elif plot_name == "qoe":
            for p in [1, 2, 3]:
                ax.plot(t, metrics[f"qoe_p{p}"], '-', color=SLICE_CONFIG[p]["color"],
                       linewidth=1, label=f'{SLICE_CONFIG[p]["short"]}')
            ax.set_ylabel("Normalized QoE")
            ax.set_ylim(0, 1.1)
            ax.set_title("Normalized QoE per Slice (Q̂_k ∈ [0,1])")
            ax.legend(loc='lower right')

        elif plot_name == "ader_weights":
            ader = learner_history.get('ader', {})
            weight_hist = ader.get('weight_history', {})
            if not weight_hist:
                logger.warning(f"No ADER weight history for {plot_name}")
                plt.close(fig)
                continue
            first_hap_id = next(iter(weight_hist.keys()))
            weights = list(weight_hist[first_hap_id])
            if not weights:
                plt.close(fig)
                continue
            weights_arr = np.array(weights)
            n_experts = weights_arr.shape[1]
            t_ader = np.arange(len(weights_arr))
            colors = plt.cm.viridis(np.linspace(0, 1, n_experts))
            ax.stackplot(t_ader, weights_arr.T, labels=[f'Expert {i}' for i in range(n_experts)],
                        colors=colors, alpha=0.8)
            ax.set_ylabel("Expert Weight")
            ax.set_ylim(0, 1)
            ax.set_title(f"ADER Expert Weights (HAP {first_hap_id})")
            ax.legend(loc='upper right', fontsize=8)

        elif plot_name == "ader_entropy":
            ader = learner_history.get('ader', {})
            entropy_hist = ader.get('entropy_history', {})
            if not entropy_hist:
                logger.warning(f"No ADER entropy history for {plot_name}")
                plt.close(fig)
                continue
            colors = plt.cm.tab10(np.linspace(0, 1, len(entropy_hist)))
            for i, (hap_id, entropy_list) in enumerate(sorted(entropy_hist.items())):
                t_ent = np.arange(len(entropy_list))
                ax.plot(t_ent, list(entropy_list), '-', color=colors[i], linewidth=1,
                       label=f'HAP {hap_id}')
            ax.set_ylabel("Entropy H(t)")
            ax.set_title("ADER Weight Entropy (H = -Σ p_i log(p_i))")
            ax.legend(loc='upper right')

        elif plot_name == "eg_weights":
            eg = learner_history.get('eg', {})
            weight_hist = eg.get('weight_history', {})
            ship_indices_hist = eg.get('ship_indices', {})
            if not weight_hist:
                logger.warning(f"No EG weight history for {plot_name}")
                plt.close(fig)
                continue
            first_hap_id = next(iter(weight_hist.keys()))
            weights = list(weight_hist[first_hap_id])
            ship_indices = list(ship_indices_hist.get(first_hap_id, []))
            if not weights or len(weights) < 2:
                plt.close(fig)
                continue
            all_ships = set()
            for si in ship_indices:
                if hasattr(si, '__iter__') and not isinstance(si, (str, int)):
                    all_ships.update(si)
                elif si is not None:
                    all_ships.add(si)
            if not all_ships:
                max_size = max(len(w) for w in weights)
                all_ships = set(range(max_size))
            ship_ids = sorted(all_ships)
            ship_to_idx = {s: i for i, s in enumerate(ship_ids)}
            n_ships = len(ship_ids)
            n_steps = len(weights)
            weight_matrix = np.zeros((n_steps, n_ships))
            for ti, (w, si) in enumerate(zip(weights, ship_indices)):
                if hasattr(si, '__iter__') and not isinstance(si, (str, int)):
                    si_list = list(si)
                elif si is not None:
                    si_list = [si]
                else:
                    si_list = list(range(len(w)))
                for wi, ship_id in enumerate(si_list):
                    if wi < len(w) and ship_id in ship_to_idx:
                        weight_matrix[ti, ship_to_idx[ship_id]] = w[wi]
            im = ax.imshow(weight_matrix.T, aspect='auto', cmap='viridis',
                          origin='lower', vmin=0, vmax=1)
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Ship Index")
            ax.set_title(f"EG Weight Distribution (HAP {first_hap_id})")
            fig.colorbar(im, ax=ax, label='Weight')

        elif plot_name == "regret":
            ader = learner_history.get('ader', {})
            eg = learner_history.get('eg', {})

            mobility_timestep = config.get("simulator", {}).get("mobility_timestep", 600.0)
            timestep = config.get("simulator", {}).get("timestep", 1.0)

            has_regret = False
            if ader.get('loss_history'):
                first_hap_id = next(iter(ader['loss_history'].keys()))
                losses = list(ader['loss_history'][first_hap_id])
                if losses:
                    cum_loss = np.cumsum(losses)
                    best_loss_per_step = np.min(losses)
                    cum_best = np.arange(1, len(losses) + 1) * best_loss_per_step
                    regret_ader = cum_loss - cum_best
                    t_regret = np.arange(len(regret_ader)) * mobility_timestep
                    ax.plot(t_regret, regret_ader, '-', color='blue', linewidth=1, label='ADER Regret')
                    has_regret = True

            if eg.get('loss_history'):
                first_hap_id = next(iter(eg['loss_history'].keys()))
                loss_dicts = list(eg['loss_history'][first_hap_id])
                if loss_dicts:
                    total_losses = [sum(ld.values()) for ld in loss_dicts]
                    cum_loss = np.cumsum(total_losses)
                    best_loss_per_step = np.min([min(ld.values()) if ld else 0 for ld in loss_dicts])
                    cum_best = np.arange(1, len(loss_dicts) + 1) * best_loss_per_step
                    regret_eg = cum_loss - cum_best
                    t_regret = np.arange(len(regret_eg)) * timestep
                    ax.plot(t_regret, regret_eg, '-', color='green', linewidth=1, label='EG Regret')
                    has_regret = True

            if not has_regret:
                logger.warning(f"No regret data for {plot_name}")
                plt.close(fig)
                continue

            T = len(t)
            n_experts = 8
            t_bound = np.arange(T) * timestep
            bound = np.sqrt(np.arange(1, T + 1) * np.log(n_experts))
            ax.plot(t_bound, bound, '--', color='gray', alpha=0.7, label='O(√(T log n))')
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Cumulative Regret")
            ax.set_title("Cumulative Regret with Theoretical Bound")
            ax.legend()

        elif plot_name == "gradient_mag":
            ader = learner_history.get('ader', {})
            gradient_hist = ader.get('gradient_history', {})
            if not gradient_hist:
                logger.warning(f"No ADER gradient history for {plot_name}")
                plt.close(fig)
                continue
            colors = plt.cm.tab10(np.linspace(0, 1, len(gradient_hist)))
            for i, (hap_id, grad_list) in enumerate(sorted(gradient_hist.items())):
                grad_mags = [np.linalg.norm(g) for g in grad_list]
                t_grad = np.arange(len(grad_mags))
                ax.plot(t_grad, grad_mags, '-', color=colors[i], linewidth=1,
                       label=f'HAP {hap_id}')
            ax.set_ylabel("||g||₂")
            ax.set_title("ADER Gradient Magnitude over Time")
            ax.legend(loc='upper right')

        else:
            rendered = False

        if not rendered:
            plt.close(fig)
            continue

        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig_path = output_path / f"{plot_name}.png"
        fig.savefig(fig_path)
        plt.close(fig)
        logger.info(f"Saved: {fig_path}")

    logger.info(f"Stats saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="ORAN HAP Simulator Video Renderer")
    parser.add_argument("--config", "-c", type=str, default="config.json", help="Config file path")
    parser.add_argument("--output", "-o", type=str, default="simulation.mp4", help="Output video path")
    parser.add_argument("--policy", "-p", type=str, default="default", help="Policy name")
    parser.add_argument("--fps", type=int, default=10, help="Video FPS")
    parser.add_argument("--frame-skip", type=int, default=1, help="Skip frames")
    parser.add_argument("--smoothing", type=float, default=0.1, help="EWMA smoothing")
    parser.add_argument("--plots", type=str, nargs="+", default=list(AVAILABLE_PLOTS.keys()), help="Plots to include")
    parser.add_argument("--stats-only", action="store_true", help="Generate PNG stats only")
    parser.add_argument("--stats-dir", type=str, default="stats", help="Stats output directory")
    parser.add_argument("--dpi", type=int, default=100, help="Output DPI")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    from .config import load_config
    config = load_config(args.config)
    config["visualization"]["fps"] = args.fps
    if args.stats_only:
        render_stats_png(config, output_dir=args.stats_dir, policy_name=args.policy,
                        smoothing_alpha=args.smoothing, plots=args.plots, dpi=args.dpi)
    else:
        render_video(config, output_path=args.output, policy_name=args.policy,
                    frame_skip=args.frame_skip, smoothing_alpha=args.smoothing,
                    plots=args.plots, dpi=args.dpi)


if __name__ == "__main__":
    main()
