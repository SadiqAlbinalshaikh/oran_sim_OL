# HAPs in Maritime Network Simulator

A time-stepped simulator for High Altitude Platform (HAP) communications serving maritime vessels.

## Data

Download the required data files from:
**[Google Drive](https://drive.google.com/drive/folders/1osyrMEaBlxS-vfICi93ngTHJqSlHXkO0?usp=sharing)**

Place the downloaded files in the `data/` directory.

## Usage

### Run All Scenarios

Run all predefined scenarios and generate comparison plots (Needs good CPU and high memory for the current predefined duration, reduce it for testing using --duration 1200 for example):

```bash
python -m oran_sim_v3_OL.run_all_scenarios --output output/
```

**Options:**
| Flag | Description |
|------|-------------|
| `--output`, `-o` | Base output directory (default: `output`) |
| `--duration`, `-d` | Override simulation duration in seconds |
| `--no-video` | Skip video generation |
| `--scenarios` | Specific scenarios to run: `static_with_no_rules`, `static_with_rules`, `learned` |
| `--dpi` | Output DPI for plots (default: 150) |
| `--log-level` | Logging level (default: INFO) |

**Example:**
```bash
python -m oran_sim_v3_OL.run_all_scenarios -o results/ --scenarios learned static_with_rules --duration 3600
```

### Video Renderer

Run a single simulation and render video or static plots:

```bash
# Render simulation video
python -m oran_sim_v3_OL.video_renderer --config config.json --output simulation.mp4

# Generate static stats plots only
python -m oran_sim_v3_OL.video_renderer --stats-only --stats-dir stats/
```

**Options:**
| Flag | Description |
|------|-------------|
| `--config`, `-c` | Config file path (default: `config.json`) |
| `--output`, `-o` | Output video path (default: `simulation.mp4`) |
| `--policy`, `-p` | Policy name (default: `default`) |
| `--fps` | Video frames per second (default: 10) |
| `--frame-skip` | Skip frames to speed up rendering (default: 1) |
| `--smoothing` | EWMA smoothing alpha (default: 0.1) |
| `--plots` | Plots to include (see available plots below) |
| `--stats-only` | Generate PNG stats instead of video |
| `--stats-dir` | Stats output directory (default: `stats`) |
| `--dpi` | Output DPI (default: 100) |
| `--log-level` | Logging level (default: INFO) |

**Available plots:** `map`, `delay_urllc`, `delay_embb_mtc`, `delay_embb`, `throughput`, `queue`, `packets`, `drops`, `ships`, `coverage`, `throughput_per_slice`, `violations`, `utilization`, `qoe`

## Parameters and Configurations

Each scenario uses a JSON config file containing all essential simulation parameters:

| Scenario | Config File | Description |
|----------|-------------|-------------|
| `static_with_no_rules` | `config_static.json` | **Baseline**: Static HAPs with fixed positions, FIFO dequeuing, no admission control, equal RB allocation, ships associate with closest HAP |
| `static_with_rules` | `config_static_rules.json` | **Rule-based**: Static HAPs with handcrafted rule-based policies for scheduling and admission, no mobility |
| `learned` | `config_ader.json` | **Learned**: ADER for adaptive HAP mobility + EG for dynamic resource scheduling. Other policies (association, admission, dequeue) are rule-based and are used similary to `static_with_rules` for comparison |

## Learners

The `learners/` module contains online learning algorithms that we directly import and use in `policies.py`:

- **ADER** (`ader.py`) - Adaptive learning for HAP mobility decisions
- **EG** (`eg.py`) - Exponentiated Gradient for resource scheduling
- **OMD** (`omd.py`) - Online Mirror Descent (base algorithm)

## Tests

Run learner tests with visualization outputs:

```bash
# Run specific learner test with plots
python tests/test_ader.py --plots
python tests/test_eg.py --plots
python tests/test_omd.py --plots
```

Test outputs (convergence plots, regret curves) are saved as PNGs in `tests/`.
