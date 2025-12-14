import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "simulator": {
        "name": "ORAN HAP Simulator",
        "version": "0.2.1",
        "log_level": "INFO",
        "timestep": 0.5,
        "mobility_timestep": 600.0,
        "duration": 172800,
        "static_haps": True
    },
    "haps": [
        {
            "hap_id": 1,
            "initial_lat": 26.449601539517168,
            "initial_lon": -79.75173077372065,
            "altitude": 25000.0,
            "max_speed": 110.0,
            "elevation_angle": 10.0,
            "numerology": 1,
            "total_bandwidth_hz": 20e6
        },
        {
            "hap_id": 2,
            "initial_lat": 18.72423980917694,
            "initial_lon": -65.68145872369568,
            "altitude": 25000.0,
            "max_speed": 110.0,
            "elevation_angle": 10.0,
            "numerology": 1,
            "total_bandwidth_hz": 20e6
        },
        {
            "hap_id": 3,
            "initial_lat": 28.19668870928491,
            "initial_lon": -93.49899362691876,
            "altitude": 25000.0,
            "max_speed": 110.0,
            "elevation_angle": 10.0,
            "numerology": 1,
            "total_bandwidth_hz": 20e6
        },
        {
            "hap_id": 4,
            "initial_lat": 29.34839797872341,
            "initial_lon": -89.11856729417205,
            "altitude": 25000.0,
            "max_speed": 110.0,
            "elevation_angle": 10.0,
            "numerology": 1,
            "total_bandwidth_hz": 20e6
        },
        {
            "hap_id": 5,
            "initial_lat": 23.932000660843084,
            "initial_lon": -82.04402545603979,
            "altitude": 25000.0,
            "max_speed": 110.0,
            "elevation_angle": 10.0,
            "numerology": 1,
            "total_bandwidth_hz": 20e6
        },
        {
            "hap_id": 6,
            "initial_lat": 19.635136484293223,
            "initial_lon": -69.18881004827348,
            "altitude": 25000.0,
            "max_speed": 110.0,
            "elevation_angle": 10.0,
            "numerology": 1,
            "total_bandwidth_hz": 20e6
        }
    ],
    "ground_stations": {
        "ports_csv_path": "data/wpi_2025.csv",
        "terrestrial_range_km": 20.0
    },
    "ships": {
        "ais_csv_path": "data/interpolated_5_speeds (3 months).csv",
        "mmsi_csv_path": "data/cruises_mmsi_imos.csv"
    },
    "communication": {
        "link_params": {
            "center_freq_hz": 2e9,
            "tx_power_dbm": 43.0,
            "noise_figure_db": 5.0,
            "k_factor": 10.0,
            "g_tx_dbi": 0.0,
            "g_rx_dbi": 0.0
        },
        "queue": {
            "hap_total_capacity_bits": 100e6
        },
        "traffic": {
            "safety": {
                "arrival_rate_per_sec": 1,
                "packet_size_bytes": 300
            },
            "operation": {
                "embb_arrival_rate_per_sec": 2,
                "mmtc_arrival_rate_per_sec": 3,
                "ipp_alpha_on": 0.1,
                "ipp_alpha_off": 0.05,
                "packet_size_bytes": 1000
            },
            "entertainment": {
                "arrival_rate_per_sec": 5.0,
                "packet_size_bytes": 1500
            }
        },
        "scheduling": {
            "priority_weights": {
                "1": 1.0,
                "2": 2.0,
                "3": 3.0
            }
        }
    },
    "visualization": {
        "enabled": True,
        "fps": 2,
        "map_bounds": {
            "lat_min": 11.678602603234797,
            "lat_max": 30.83177123893535,
            "lon_min": -98.15860405725323,
            "lon_max": -58.42952973793629
        },
        "figsize": [14, 10]
    },
    "performance": {
        "use_gpu": False
    },
    "policies": {
        "mobility": {
            "type": "minimal",
            "lipschitz": 1.0,
            "ship_weight": 1.0,
            "hap_repulsion_weight": 0.5,
            "backhaul_penalty_weight": 10.0
        },
        "scheduler": {
            "rb_allocation_type": "minimal",
            "dequeue_type": "strict_priority",
            "step_size": 0.1,
            "min_weight": 0.01
      },
        "association": {
          "type": "minimal",
          "weights": {
              "queue_transfer": 0.2,
              "load": 0.4,
              "distance": 0.25,
              "qoe": 0.15
          },
          "max_ships_per_hap": 100
      },
        "admission": {
          "type": "minimal",
          "theta_high": 0.9,
          "theta_mid": 0.7,
          "p2_backlog_threshold": 1000000
      }
    }
}


def deep_merge(base: Dict, override: Dict) -> Dict:
    result = base.copy()
    for key, value in override.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    if config_path is None:
        config_path = "config.json"
    config_file = Path(config_path)
    if not config_file.exists():
        logger.info(f"Config file {config_path} not found, using defaults")
        return DEFAULT_CONFIG.copy()
    try:
        with open(config_file, 'r') as f:
            user_config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return deep_merge(DEFAULT_CONFIG, user_config)
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        logger.info("Using default configuration")
        return DEFAULT_CONFIG.copy()
