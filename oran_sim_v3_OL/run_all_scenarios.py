import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .config import load_config
from .video_renderer import (
    run_simulation,
    extract_metrics,
    render_video,
    render_stats_png,
    SLICE_CONFIG,
    QOS_THRESHOLDS,
)
from .metrics_collector import MetricsCollector

logger = logging.getLogger(__name__)

SCENARIOS = {
    "static_with_no_rules": {
        "config_path": "config_static.json",
        "description": "Baseline: static HAPs, FIFO dequeing, no admission control, equal RB allocation, associate with closest ship if overlapped",
        "has_ader": False,
        "has_eg": False,
        "color": "#272727",
    },
    "static_with_rules": {
        "config_path": "config_static_rules.json",
        "description": "Static HAPs with rule-based policies, no mobility, and equal RB allocation.",
        "has_ader": False,
        "has_eg": False,
        "color": "#3498db",
    },
    "learned": {
        "config_path": "config_ader.json",
        "description": "Full learned: ADER + EG",
        "has_ader": True,
        "has_eg": True,
        "color": "#FF0000",
    },
}

TIER1_PLOTS = [
    "coverage",
    "delay_urllc",
    "delay_embb_mtc",
    "delay_embb",
    "throughput_per_slice",
    "violations",
    "utilization",
    "qoe",
]

TIER2_PLOTS = [
    "ader_weights",
    "ader_entropy",
    "eg_weights",
    "regret",
    "gradient_mag",
]


def run_scenario(
    scenario_name: str,
    config_path: str,
    output_dir: Path,
    duration: int = None,
    generate_video: bool = True,
    dpi: int = 150,
) -> Dict[str, Any]:
    logger.info(f"Running scenario: {scenario_name}")
    logger.info(f"  Config: {config_path}")

    config = load_config(config_path)
    if duration is not None:
        config["simulator"]["duration"] = duration

    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    sim, states, learner_history = run_simulation(config, policy_name="default")
    elapsed = time.time() - start_time
    logger.info(f"  Simulation completed in {elapsed:.1f}s")

    if not states:
        logger.warning(f"  No states collected for {scenario_name}")
        return {}

    metrics = extract_metrics(states, smoothing_alpha=0.1, config=config)

    collector = MetricsCollector(config)
    for state in states:
        collector.collect(sim, state)
    collector.collect_learner_history(sim)

    csv_path = output_dir / "metrics.csv"
    collector.export_csv(str(csv_path))
    logger.info(f"  Saved: {csv_path}")

    summary = collector.get_summary_stats()

    scenario_info = SCENARIOS.get(scenario_name, {})
    plots_to_generate = TIER1_PLOTS.copy()
    if scenario_info.get("has_ader"):
        plots_to_generate.extend(["ader_weights", "ader_entropy", "gradient_mag"])
    if scenario_info.get("has_eg"):
        plots_to_generate.append("eg_weights")
    if scenario_info.get("has_ader") or scenario_info.get("has_eg"):
        plots_to_generate.append("regret")

    render_stats_png(
        config,
        output_dir=str(output_dir),
        policy_name="default",
        smoothing_alpha=0.1,
        plots=plots_to_generate,
        dpi=dpi,
        states=states,
        learner_history=learner_history,
    )

    if generate_video:
        video_path = output_dir / "simulation.mp4"
        render_video(
            config,
            output_path=str(video_path),
            policy_name="default",
            frame_skip=int(config.get("simulator", {}).get("mobility_timestep", 600)),
            smoothing_alpha=0.1,
            plots=["map"],
            dpi=dpi,
            states=states,
            learner_history=learner_history,
        )

    return {
        "metrics": metrics,
        "learner_history": learner_history,
        "summary": summary,
        "config": config,
    }


def run_all_scenarios(
    output_base: Path,
    duration: int = None,
    generate_videos: bool = True,
    dpi: int = 150,
    scenarios: Dict[str, Dict] = None,
) -> Dict[str, Dict]:
    if scenarios is None:
        scenarios = SCENARIOS

    results = {}
    output_base.mkdir(parents=True, exist_ok=True)

    for scenario_name, scenario_info in scenarios.items():
        scenario_dir = output_base / f"scenario_{scenario_name}"
        config_path = Path(__file__).parent / scenario_info["config_path"]

        results[scenario_name] = run_scenario(
            scenario_name=scenario_name,
            config_path=str(config_path),
            output_dir=scenario_dir,
            duration=duration,
            generate_video=generate_videos,
            dpi=dpi,
        )

    return results


def generate_comparison_plots(
    results: Dict[str, Dict],
    output_dir: Path,
    dpi: int = 150,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating comparison plots...")

    valid_results = {k: v for k, v in results.items() if v.get("metrics")}
    if not valid_results:
        logger.warning("No valid results for comparison plots")
        return

    scenario_names = list(valid_results.keys())
    scenario_colors = [SCENARIOS[name]["color"] for name in scenario_names]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), dpi=dpi, sharex=True)
    for p, (ax, label) in enumerate(zip(axes, ["P1 (URLLC)", "P2 (eMBB+mMTC)", "P3 (eMBB)"]), start=1):
        for scenario_name, color in zip(scenario_names, scenario_colors):
            metrics = valid_results[scenario_name]["metrics"]
            t = metrics["time"]
            qoe = metrics[f"qoe_p{p}"]
            ax.plot(t, qoe, '-', color=color, linewidth=1, label=scenario_name)
        ax.set_ylabel("Normalized QoE")
        ax.set_title(f"QoE - {label}")
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("QoE Comparison Across Scenarios", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "qoe_comparison.png")
    plt.close(fig)
    logger.info(f"  Saved: {output_dir / 'qoe_comparison.png'}")

    fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)
    for scenario_name, color in zip(scenario_names, scenario_colors):
        metrics = valid_results[scenario_name]["metrics"]
        t = metrics["time"]
        coverage = metrics["coverage_ratio"]
        ax.plot(t, coverage, '-', color=color, linewidth=1, label=scenario_name)
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Target (100%)')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Coverage Ratio")
    ax.set_title("Coverage Ratio Comparison Across Scenarios")
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    fig.tight_layout()
    fig.savefig(output_dir / "coverage_comparison.png")
    plt.close(fig)
    logger.info(f"  Saved: {output_dir / 'coverage_comparison.png'}")

    baseline_name = "baseline"
    if baseline_name not in valid_results:
        baseline_name = scenario_names[0]

    baseline_summary = valid_results[baseline_name].get("summary", {})
    metrics_to_compare = [
        ("mean_coverage_ratio", "Coverage", True),
        ("mean_qoe_p1", "QoE P1", True),
        ("mean_qoe_p2", "QoE P2", True),
        ("mean_qoe_p3", "QoE P3", True),
        ("mean_delay_violation_p1", "Delay Viol P1", False),
        ("mean_delay_violation_p2", "Delay Viol P2", False),
        ("mean_delay_violation_p3", "Delay Viol P3", False),
    ]

    fig, ax = plt.subplots(figsize=(14, 6), dpi=dpi)
    x = np.arange(len(metrics_to_compare))
    width = 0.8 / len(scenario_names)

    for i, (scenario_name, color) in enumerate(zip(scenario_names, scenario_colors)):
        summary = valid_results[scenario_name].get("summary", {})
        improvements = []
        for metric_key, _, higher_better in metrics_to_compare:
            baseline_val = baseline_summary.get(metric_key, 0)
            scenario_val = summary.get(metric_key, 0)
            if baseline_val == 0:
                improvement = 0
            elif higher_better:
                improvement = ((scenario_val - baseline_val) / max(baseline_val, 1e-9)) * 100
            else:
                improvement = ((baseline_val - scenario_val) / max(baseline_val, 1e-9)) * 100
            improvements.append(improvement)
        ax.bar(x + i * width, improvements, width, label=scenario_name, color=color, alpha=0.8)

    ax.set_ylabel("% Improvement over Baseline")
    ax.set_title("Performance Improvement Relative to Baseline")
    ax.set_xticks(x + width * (len(scenario_names) - 1) / 2)
    ax.set_xticklabels([label for _, label, _ in metrics_to_compare], rotation=45, ha='right')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(output_dir / "improvement_bar.png")
    plt.close(fig)
    logger.info(f"  Saved: {output_dir / 'improvement_bar.png'}")

    fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)
    has_regret = False

    for scenario_name, color in zip(scenario_names, scenario_colors):
        learner_history = valid_results[scenario_name].get("learner_history", {})
        scenario_config = valid_results[scenario_name].get("config", {})
        mobility_timestep = scenario_config.get("simulator", {}).get("mobility_timestep", 600.0)
        timestep = scenario_config.get("simulator", {}).get("timestep", 1.0)

        ader = learner_history.get("ader", {})
        if ader.get("loss_history"):
            first_hap_id = next(iter(ader["loss_history"].keys()))
            losses = list(ader["loss_history"][first_hap_id])
            if losses:
                cum_loss = np.cumsum(losses)
                best_loss = np.min(losses)
                cum_best = np.arange(1, len(losses) + 1) * best_loss
                regret = cum_loss - cum_best
                t_regret = np.arange(len(regret)) * mobility_timestep
                ax.plot(t_regret, regret, '-', color=color, linewidth=1,
                       label=f"{scenario_name} (ADER)")
                has_regret = True

        eg = learner_history.get("eg", {})
        if eg.get("loss_history"):
            first_hap_id = next(iter(eg["loss_history"].keys()))
            loss_dicts = list(eg["loss_history"][first_hap_id])
            if loss_dicts:
                total_losses = [sum(ld.values()) for ld in loss_dicts]
                cum_loss = np.cumsum(total_losses)
                best_loss = np.min([min(ld.values()) if ld else 0 for ld in loss_dicts])
                cum_best = np.arange(1, len(loss_dicts) + 1) * best_loss
                regret = cum_loss - cum_best
                t_regret = np.arange(len(regret)) * timestep
                ax.plot(t_regret, regret, '--', color=color, linewidth=1,
                       label=f"{scenario_name} (EG)")
                has_regret = True

    if has_regret:
        first_config = next(iter(valid_results.values())).get("config", {})
        timestep = first_config.get("simulator", {}).get("timestep", 1.0)
        max_t = max(len(r["metrics"]["time"]) for r in valid_results.values())
        n_experts = 8
        t_bound = np.arange(max_t) * timestep
        bound = np.sqrt(np.arange(1, max_t + 1) * np.log(n_experts))
        ax.plot(t_bound, bound, ':', color='gray', alpha=0.7, label='O(sqrt(T log n))')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Cumulative Regret")
        ax.set_title("Regret Comparison Across Scenarios")
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "regret_comparison.png")
        logger.info(f"  Saved: {output_dir / 'regret_comparison.png'}")
    else:
        logger.warning("  No regret data available for comparison")
    plt.close(fig)

    summary_lines = ["Scenario Summary Statistics", "=" * 60, ""]
    for scenario_name in scenario_names:
        summary = valid_results[scenario_name].get("summary", {})
        summary_lines.append(f"\n{scenario_name.upper()}")
        summary_lines.append("-" * 40)
        for key, value in sorted(summary.items()):
            if isinstance(value, float):
                summary_lines.append(f"  {key}: {value:.4f}")
            else:
                summary_lines.append(f"  {key}: {value}")
    summary_path = output_dir / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    logger.info(f"  Saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run all ORAN HAP Simulator scenarios and generate comparison plots"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output",
        help="Base output directory"
    )
    parser.add_argument(
        "--duration", "-d",
        type=int,
        default=None,
        help="Override simulation duration (for testing)"
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip video generation"
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        choices=list(SCENARIOS.keys()),
        default=None,
        help="Specific scenarios to run (default: all)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output DPI for plots"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    output_base = Path(args.output)

    scenarios_to_run = SCENARIOS
    if args.scenarios:
        scenarios_to_run = {k: v for k, v in SCENARIOS.items() if k in args.scenarios}

    logger.info(f"Running scenarios: {list(scenarios_to_run.keys())}")
    results = run_all_scenarios(
        output_base=output_base,
        duration=args.duration,
        generate_videos=not args.no_video,
        dpi=args.dpi,
        scenarios=scenarios_to_run,
    )

    comparison_dir = output_base / "comparison"
    generate_comparison_plots(results, comparison_dir, dpi=args.dpi)

    logger.info(f"All outputs saved to: {output_base}")


if __name__ == "__main__":
    main()
