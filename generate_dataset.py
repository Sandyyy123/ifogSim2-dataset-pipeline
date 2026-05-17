"""
Generate iFogSim2-compatible scheduler and rescheduler datasets.

Usage:
    python generate_dataset.py --config configs/scenario_default.yaml
    python generate_dataset.py --config configs/scenario_default.yaml --episodes 500 --seed 99
    python generate_dataset.py --quick   # 100 episodes for fast testing
"""

import argparse
import yaml
from pathlib import Path
import pandas as pd

from src.dataset_builder import DatasetBuilder
from src.data_validator import DataValidator


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main(args):
    cfg = load_config(args.config)

    if args.episodes:
        cfg["simulation"]["n_episodes"] = args.episodes
    if args.seed is not None:
        cfg["simulation"]["seed"] = args.seed
    if args.quick:
        cfg["simulation"]["n_episodes"] = 100
        cfg["simulation"]["timesteps_per_episode"] = 50
        print("Quick mode: 100 episodes x 50 timesteps")

    output_dir = Path(args.output_dir or cfg["dataset"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nBuilding dataset: {cfg['simulation']['n_episodes']} episodes x "
          f"{cfg['simulation']['timesteps_per_episode']} timesteps")
    print(f"Fog nodes: {cfg['fog_topology']['n_fog_nodes']} | "
          f"User devices: {cfg['fog_topology']['n_user_devices']} | "
          f"Seed: {cfg['simulation']['seed']}\n")

    builder = DatasetBuilder(cfg, seed=cfg["simulation"]["seed"])
    scheduler_df, rescheduler_df = builder.build_dataset()

    sched_path = output_dir / cfg["dataset"]["scheduler_file"]
    resched_path = output_dir / cfg["dataset"]["rescheduler_file"]
    scheduler_df.to_csv(sched_path, index=False)
    rescheduler_df.to_csv(resched_path, index=False)

    print(f"\nSaved:")
    print(f"  Scheduler   -> {sched_path} ({len(scheduler_df):,} rows)")
    print(f"  Rescheduler -> {resched_path} ({len(rescheduler_df):,} rows)")

    if cfg["dataset"].get("validate", True):
        validator = DataValidator(scheduler_df, rescheduler_df)
        validator.print_report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="iFogSim2 Dataset Generator")
    parser.add_argument("--config", default="configs/scenario_default.yaml")
    parser.add_argument("--episodes", type=int, help="Override n_episodes from config")
    parser.add_argument("--seed", type=int, help="Override random seed")
    parser.add_argument("--output-dir", help="Override output directory")
    parser.add_argument("--quick", action="store_true", help="100 episodes for fast testing")
    args = parser.parse_args()
    main(args)
