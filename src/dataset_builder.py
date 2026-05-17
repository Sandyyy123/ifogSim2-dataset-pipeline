"""
Dataset builder - orchestrates simulation episodes to produce scheduler and rescheduler datasets.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple, List

from .schema import SchedulerRecord, ReschedulerRecord, records_to_dataframe
from .topology_generator import FogTopology
from .simulator import iFogSim2Simulator


class DatasetBuilder:
    """
    Runs simulation episodes using multiple scheduling policies to generate
    diverse, publication-quality training datasets for ML/RL models.

    Args:
        cfg: Full config dict (from YAML)
        seed: Random seed for reproducibility
    """

    def __init__(self, cfg: dict, seed: int = 42):
        self.cfg = cfg
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.topology = FogTopology(cfg, seed=seed)
        self.simulator = iFogSim2Simulator(self.topology, cfg, rng=self.rng)

    def run_scheduler_episode(
        self, episode_id: int, n_timesteps: int
    ) -> List[SchedulerRecord]:
        """Run one simulation episode and collect scheduler decision records."""
        records = []
        self.simulator.reset_loads()
        policies = self.cfg["simulation"]["scheduling_policies"]

        task_id = 0
        for t in range(n_timesteps):
            # Mobility update every 10 steps
            if t % 10 == 0:
                self.topology.update_device_locations()

            # Sample application and device
            n_apps = self.cfg["fog_topology"]["n_applications"]
            app_id = int(self.rng.integers(0, n_apps))
            dev_id = int(self.rng.integers(0, self.cfg["fog_topology"]["n_user_devices"]))

            task = self.topology.generate_task(app_id, task_id, dev_id)
            task["episode_id"] = episode_id
            task["timestep"] = t

            # Use one scheduling policy per timestep (rotate for diversity)
            policy = policies[t % len(policies)]
            fog_node = self.simulator.apply_scheduling_policy(task, policy)

            record = self.simulator.schedule(task, fog_node, policy)
            records.append(record)
            task_id += 1

        return records

    def run_rescheduler_episode(
        self, episode_id: int, n_timesteps: int
    ) -> List[ReschedulerRecord]:
        """Run one simulation episode and collect rescheduler decision records."""
        records = []
        self.simulator.reset_loads()
        n_fog = len(self.topology.fog_nodes)
        n_apps = self.cfg["fog_topology"]["n_applications"]

        module_id = 0
        for t in range(n_timesteps):
            # Evaluate migration for each fog node pair combination
            n_pairs = min(n_fog * (n_fog - 1), 5)  # max 5 pairs per timestep
            evaluated = set()
            for _ in range(n_pairs):
                cur = int(self.rng.integers(0, n_fog))
                cand = int(self.rng.integers(0, n_fog))
                if cur == cand or (cur, cand) in evaluated:
                    continue
                evaluated.add((cur, cand))
                app_id = int(self.rng.integers(0, n_apps))
                record = self.simulator.evaluate_migration(
                    module_id, app_id, cur, cand, episode_id, t
                )
                records.append(record)
                module_id += 1

        return records

    def build_dataset(self, n_episodes: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run all episodes and return (scheduler_df, rescheduler_df).

        Args:
            n_episodes: Override config value if provided.
        """
        n = n_episodes or self.cfg["simulation"]["n_episodes"]
        timesteps = self.cfg["simulation"]["timesteps_per_episode"]

        scheduler_records, rescheduler_records = [], []

        for ep in tqdm(range(n), desc="Simulating episodes"):
            scheduler_records.extend(self.run_scheduler_episode(ep, timesteps))
            rescheduler_records.extend(self.run_rescheduler_episode(ep, timesteps))

        sched_df = records_to_dataframe(scheduler_records)
        resched_df = records_to_dataframe(rescheduler_records)

        return sched_df, resched_df
