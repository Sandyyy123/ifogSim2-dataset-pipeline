"""
iFogSim2 Dataset Pipeline - Entry Point

Generates all 6 CSV datasets for scheduler and migration decisions.

Usage:
    python generate_dataset.py --demo           # quick 10-episode test
    python generate_dataset.py --episodes 500   # full run
    python generate_dataset.py --config config/custom.yaml --seed 99
    python generate_dataset.py --quick          # 20 episodes, fast validation

Outputs (in results/):
    scheduler_decisions.csv    - one row per task arrival
    scheduler_candidates.csv   - one row per candidate node evaluated
    scheduler_outcomes.csv     - one row per executed task (actual results)
    migration_decisions.csv    - one row per migration evaluation
    migration_candidates.csv   - one row per candidate (NO_MIGRATION always rank 0)
    migration_outcomes.csv     - one row per migration event (actual results)
"""

import argparse
import yaml
import json
import pandas as pd
from pathlib import Path

from src.dataset_builder import DatasetBuilder
from src.data_validator import DataValidator

DEFAULT_CFG = {
    "fog_topology": {
        "n_fog_nodes": 3,
        "n_edge_nodes": 4,
        "n_user_devices": 20,
        "n_applications": 5,
        "cloud_latency_base_ms": 150.0,
        "fog_cpu_mips_range": [4000, 16000],
        "fog_memory_mb_range": [8192, 65536],
        "fog_energy_idle_w_range": [100, 300],
        "fog_energy_active_w_range": [300, 800],
    },
    "network": {
        "bandwidth_mbps_range": [10, 1000],
        "base_latency_ms_range": [5, 60],
    },
    "simulation": {
        "n_episodes": 100,
        "timesteps_per_episode": 50,
        "scheduling_policies": ["teacher", "first_fit", "round_robin", "energy_aware"],
        "migration_eval_freq": 10,
    },
    "teacher_policy": {
        "w_lat": 0.40,
        "w_energy": 0.35,
        "w_load": 0.25,
        "teacher_fraction": 0.87,
    },
    "migration": {
        "sla_latency_ms": 200.0,
        "load_threshold_trigger": 0.85,
        "qos_degradation_trigger_ms": 20.0,
        "energy_per_byte_j": 5e-9,
        "fixed_overhead_j": 0.1,
    },
}


def main():
    parser = argparse.ArgumentParser(description="iFogSim2 6-CSV Dataset Pipeline")
    parser.add_argument("--demo", action="store_true", help="Quick 10-episode demo")
    parser.add_argument("--quick", action="store_true", help="20 episodes with validation")
    parser.add_argument("--episodes", type=int, help="Override n_episodes from config")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--no-validate", action="store_true", help="Skip validation")
    args = parser.parse_args()

    cfg = DEFAULT_CFG.copy()
    if args.config:
        with open(args.config) as f:
            cfg.update(yaml.safe_load(f))

    if args.demo:
        cfg["simulation"]["n_episodes"] = 10
        cfg["simulation"]["timesteps_per_episode"] = 20
        print("Demo mode: 10 episodes x 20 timesteps")
    elif args.quick:
        cfg["simulation"]["n_episodes"] = 20
        cfg["simulation"]["timesteps_per_episode"] = 30
        print("Quick mode: 20 episodes x 30 timesteps")
    elif args.episodes:
        cfg["simulation"]["n_episodes"] = args.episodes

    n_ep = cfg["simulation"]["n_episodes"]
    n_ts = cfg["simulation"]["timesteps_per_episode"]
    print(f"Seed: {args.seed} | Episodes: {n_ep} | Timesteps/ep: {n_ts}")
    print(f"Teacher fraction: {cfg['teacher_policy']['teacher_fraction']:.0%} of rows\n")

    builder = DatasetBuilder(cfg=cfg, seed=args.seed)
    datasets = builder.build_dataset()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in datasets.items():
        path = out_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"  {name:<35} {len(df):>8,} rows  -> {path}")

    summary = {k: {"rows": len(v), "cols": len(v.columns)} for k, v in datasets.items()}
    with open(out_dir / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if not args.no_validate:
        print("\nRunning validation suite...")
        validator = DataValidator(datasets)
        report = validator.print_report()

        def _serialize(obj):
            if isinstance(obj, float) and obj != obj:
                return None
            return str(obj)

        with open(out_dir / "validation_report.json", "w") as f:
            json.dump(report, f, indent=2, default=_serialize)
        print(f"Validation report -> {out_dir}/validation_report.json")

    print(f"\nAll outputs -> {out_dir}/")


if __name__ == "__main__":
    main()
