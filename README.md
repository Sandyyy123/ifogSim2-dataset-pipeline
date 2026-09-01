> **⚠️ Proprietary — All Rights Reserved.** © 2026 Sandeep Grover. This repository is licensed to Sandeep Grover and may **not** be used, run, copied, modified, distributed, or used to train models without prior written permission. Public visibility does not grant a license. See [LICENSE](LICENSE).

---

# iFogSim2 Dataset Generation Pipeline

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Publication Ready](https://img.shields.io/badge/Data-Publication--Ready-purple)](datasets/)

Research-grade dataset generation pipeline for **energy-aware scheduling and rescheduling** in heterogeneous fog/cloud/edge environments. Produces structured, reproducible datasets for training ML and reinforcement learning models.

---

## What This Generates

### Scheduler Dataset
Captures **task-routing decisions**: given a task's resource requirements and the current state of fog nodes, which node should receive the task?

| Column | Type | Description |
|--------|------|-------------|
| episode_id | int | Simulation episode index |
| task_cpu_mips | float | Task CPU requirement |
| task_data_size_kb | float | Task data payload |
| deadline_ms | float | QoS deadline constraint |
| fog_load_fraction | float | Target node load (0-1) |
| edge_to_fog_latency_ms | float | Current network latency |
| energy_consumed_j | float | Energy consumed for this routing |
| qos_met | int | 1 if deadline met, 0 otherwise |
| **routing_decision** | **int** | **Label: fog node index selected** |

*Full schema: 21 columns. See `src/schema.py`.*

### Rescheduler Dataset
Captures **module migration decisions**: should a currently-placed application module migrate to a different fog node?

| Column | Type | Description |
|--------|------|-------------|
| current_node_load | float | Current node utilization |
| candidate_node_load | float | Migration target utilization |
| migration_energy_j | float | Energy cost of migration |
| latency_improvement_ms | float | QoS gain if migration occurs |
| energy_improvement_j | float | Energy saving if migration occurs |
| load_balance_improvement | float | Load balance delta |
| **migration_decision** | **int** | **Label: 1=migrate, 0=stay** |

*Full schema: 19 columns. See `src/schema.py`.*

---

## Quick Start

```bash
pip install -r requirements.txt

# Fast test - 100 episodes (~30 seconds)
python generate_dataset.py --quick

# Full dataset - 1000 episodes (publication scale)
python generate_dataset.py --config configs/scenario_default.yaml

# Custom scenario
python generate_dataset.py --config configs/scenario_default.yaml --episodes 2000 --seed 123
```

Output: `datasets/scheduler_dataset.csv` and `datasets/rescheduler_dataset.csv`

---

## Architecture

```
configs/scenario_default.yaml
        |
        v
FogTopology (topology_generator.py)
  - n fog nodes (heterogeneous CPU, memory, energy)
  - n user devices with mobility
  - pairwise latency/bandwidth matrix
        |
        v
iFogSim2Simulator (simulator.py)
  - iFogSim2 energy model: E = E_idle*t + (E_act - E_idle)*util*t
  - QoS latency model: tx_time + exec_time + propagation
  - Migration cost model: E_mig = data_size * energy_coeff
  - 4 scheduling policies: first_fit, round_robin, energy_aware, random
        |
        v
DatasetBuilder (dataset_builder.py)
  - n_episodes x timesteps_per_episode simulation runs
  - Rotating scheduling policies for dataset diversity
        |
        +--> datasets/scheduler_dataset.csv
        +--> datasets/rescheduler_dataset.csv
        |
        v
DataValidator (data_validator.py)
  - Class balance check
  - Missing value detection
  - Physical constraint verification (no negative energy)
  - High-correlation pair flagging
```

---

## Configuration

All parameters in `configs/scenario_default.yaml`:

```yaml
fog_topology:
  n_fog_nodes: 5           # heterogeneous nodes
  n_user_devices: 50
  fog_node_cpu_mips: [500, 1000, 2000, 4000, 8000]

simulation:
  n_episodes: 1000
  timesteps_per_episode: 100
  seed: 42                 # fixed for reproducibility
  scheduling_policies: [first_fit, round_robin, energy_aware, random]

migration:
  load_threshold_trigger: 0.80
  qos_degradation_trigger_ms: 50
```

---

## Publication Quality Standards

- **Reproducible**: Fixed seed in config; all randomness seeded from config value
- **Documented schema**: Full column descriptions in `src/schema.py`
- **Validated**: Automatic class balance + physical constraint checks on generation
- **Configurable**: All scenario parameters externalized to YAML
- **Split-ready**: Config includes train/val/test split ratios

---

## Using with ML/RL Frameworks

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Scheduler classification
df = pd.read_csv("datasets/scheduler_dataset.csv")
features = df.drop(columns=["routing_decision", "episode_id", "timestep"])
labels = df["routing_decision"]
clf = RandomForestClassifier(n_estimators=100).fit(features, labels)

# Rescheduler binary classification
df2 = pd.read_csv("datasets/rescheduler_dataset.csv")
# Use migration_decision as binary label
```

For RL: use `iFogSim2Simulator` directly as a gym-compatible environment wrapper.

---

## Java iFogSim2 Integration

The Python simulator (`src/simulator.py`) replicates iFogSim2's energy and latency equations. To use the actual Java iFogSim2:

1. Install iFogSim2: https://github.com/Cloudslab/iFogSim
2. Replace the `_compute_energy` and `schedule` methods in `simulator.py` with subprocess calls to the Java JAR
3. The `DatasetBuilder` interface remains identical

---

## Author

**Dr. Sandeep Grover**
PhD Data Science | 12 Years Academic Research | 60+ Peer-Reviewed Publications
