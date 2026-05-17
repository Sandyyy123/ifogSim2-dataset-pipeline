"""
Dataset builder - orchestrates episodes to produce all 6 CSV streams.

Outputs:
  scheduler_decisions    - one row per task arrival
  scheduler_candidates   - one row per node evaluated per decision
  scheduler_outcomes     - one row per executed task
  migration_decisions    - one row per migration evaluation
  migration_candidates   - one row per candidate (NO_MIGRATION always rank 0)
  migration_outcomes     - one row per migration event
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict

from .schema import records_to_dataframe, DATASET_NAMES
from .topology_generator import FogTopology
from .simulator import iFogSim2Simulator

MIGRATION_TRIGGERS = ["mobility", "overload", "sla_breach", "energy"]


class DatasetBuilder:
    """
    Runs simulation episodes using teacher policy + policy rotation
    to produce 6 linked CSVs for ML/RL training.

    Args:
        cfg: Full config dict (from YAML)
        seed: Random seed for full reproducibility
    """

    def __init__(self, cfg: dict, seed: int = 42):
        self.cfg = cfg
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.topology = FogTopology(cfg, seed=seed)
        self.simulator = iFogSim2Simulator(self.topology, cfg, rng=self.rng)

    def build_dataset(self, n_episodes: int = None) -> Dict[str, pd.DataFrame]:
        """
        Run all episodes. Returns dict keyed by DATASET_NAMES.

        Args:
            n_episodes: Override config value if provided.
        """
        n = n_episodes or self.cfg["simulation"]["n_episodes"]
        timesteps = self.cfg["simulation"]["timesteps_per_episode"]
        mig_freq = self.cfg.get("simulation", {}).get("migration_eval_freq", 10)

        sched_decisions, sched_candidates, sched_outcomes = [], [], []
        mig_decisions, mig_candidates, mig_outcomes = [], [], []

        module_counter = 0

        for ep in tqdm(range(n), desc="Simulating episodes"):
            self.simulator.reset_loads()
            task_counter = 0

            # Mobility + task scheduling
            for step in range(timesteps):
                if step % 10 == 0:
                    self.topology.update_device_locations()

                n_apps = self.cfg["fog_topology"]["n_applications"]
                n_devs = self.cfg["fog_topology"]["n_user_devices"]
                app_id = int(self.rng.integers(0, n_apps))
                dev_id = f"dev-{int(self.rng.integers(0, n_devs))}"

                task = self.topology.generate_task(app_id, task_counter, dev_id)
                task_id = f"ep{ep}-t{step}-{task_counter}"
                task["task_id"] = task_id
                task["device_id"] = dev_id

                dec, cands, out = self.simulator.schedule(task, ep, step)
                sched_decisions.append(dec)
                sched_candidates.extend(cands)
                sched_outcomes.append(out)
                task_counter += 1

                # Evaluate migration periodically
                if step % mig_freq == 0 and step > 0:
                    n_fog = len(self.topology.fog_nodes)
                    cur_node = int(self.rng.integers(0, n_fog))
                    trigger = MIGRATION_TRIGGERS[step % len(MIGRATION_TRIGGERS)]
                    module_id = f"mod-ep{ep}-s{step}-{module_counter}"
                    m_dec, m_cands, m_out = self.simulator.evaluate_migration(
                        module_id=module_id,
                        app_id=app_id,
                        device_id=dev_id,
                        current_node_idx=cur_node,
                        trigger_type=trigger,
                        episode_id=ep,
                        step_id=step,
                    )
                    mig_decisions.append(m_dec)
                    mig_candidates.extend(m_cands)
                    mig_outcomes.append(m_out)
                    module_counter += 1

        datasets = {
            "scheduler_decisions": records_to_dataframe(sched_decisions),
            "scheduler_candidates": records_to_dataframe(sched_candidates),
            "scheduler_outcomes": records_to_dataframe(sched_outcomes),
            "migration_decisions": records_to_dataframe(mig_decisions),
            "migration_candidates": records_to_dataframe(mig_candidates),
            "migration_outcomes": records_to_dataframe(mig_outcomes),
        }

        return datasets
