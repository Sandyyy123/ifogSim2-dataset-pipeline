"""
iFogSim2 simulation core - produces all 6 CSV streams.

Energy model (iFogSim2 PowerModel):
    E_total = E_idle * t + (E_active - E_idle) * CPU_utilization * t

Per-task energy is a proportional estimate (device-level delta / total tasks),
NOT a per-task measurement. All energy columns are labeled *_estimated or *_j
with a disclaimer in schema docs. This is standard in fog simulation research.

When Java iFogSim2 is integrated (M4), replace _compute_* methods with
subprocess calls to the JVM; the 6-dataset interface remains identical.
"""

import math
import numpy as np
from typing import List, Tuple

from .schema import (
    SchedulerDecisionRecord, SchedulerCandidateRecord, SchedulerOutcomeRecord,
    MigrationDecisionRecord, MigrationCandidateRecord, MigrationOutcomeRecord,
)
from .topology_generator import FogTopology, FogNode
from .teacher_policy import TeacherPolicy, PolicyRotation, NodeScore

NO_MIGRATION = "NO_MIGRATION"


class iFogSim2Simulator:
    """
    Simulates iFogSim2 fog environment and emits all 6 CSV record streams.

    Produces:
      - scheduler_decisions + scheduler_candidates + scheduler_outcomes
      - migration_decisions + migration_candidates + migration_outcomes
    """

    def __init__(self, topology: FogTopology, cfg: dict, rng: np.random.Generator):
        self.topology = topology
        self.cfg = cfg
        self.rng = rng
        n = len(topology.fog_nodes)
        self.node_loads = np.zeros(n)
        self.teacher = TeacherPolicy(
            w_lat=cfg.get("teacher_policy", {}).get("w_lat", 0.40),
            w_energy=cfg.get("teacher_policy", {}).get("w_energy", 0.35),
            w_load=cfg.get("teacher_policy", {}).get("w_load", 0.25),
        )
        self.policy_rotation = PolicyRotation(
            teacher_fraction=cfg.get("teacher_policy", {}).get("teacher_fraction", 0.87)
        )

    # ------------------------------------------------------------------
    # SCHEDULER: decision + candidates + outcome
    # ------------------------------------------------------------------

    def schedule(
        self, task: dict, episode_id: int, step_id: int
    ) -> Tuple[SchedulerDecisionRecord, List[SchedulerCandidateRecord], SchedulerOutcomeRecord]:
        """
        Process one scheduling event. Returns (decision, [candidates], outcome).

        All nodes in topology are evaluated as candidates.
        Teacher policy selects argmax; policy_rotation may override for diversity.
        """
        task_id = task["task_id"]
        dev_id = task["device_id"]

        # Build candidate scores for all nodes
        candidates = self._score_scheduler_candidates(task)
        policy = self.policy_rotation.select_policy(self.rng, step_id)
        selected = self.policy_rotation.apply(policy, candidates, self.rng, step_id)
        teacher_choice = self.teacher.select(candidates)

        # Mark teacher choice on all candidates
        for c in candidates:
            c.is_teacher_choice = 1 if c.node_id == teacher_choice.node_id else 0

        # Rank all candidates by teacher score
        ranked = self.teacher.rank_candidates(candidates)

        # Build decision record
        dev = self.topology.user_devices[int(dev_id.split("-")[1])]
        nearest_fog_idx = self._nearest_fog(dev_id)
        nearest_fog = self.topology.fog_nodes[nearest_fog_idx]

        decision = SchedulerDecisionRecord(
            episode_id=episode_id,
            step_id=step_id,
            task_id=task_id,
            app_id=task["app_id"],
            task_cpu_mi=task["cpu_mips"],
            task_deadline_ms=task["deadline_ms"],
            task_data_size_kb=task["data_size_kb"],
            task_output_size_kb=task.get("output_size_kb", task["data_size_kb"] * 0.1),
            device_id=dev_id,
            device_location_x=float(dev.x),
            device_location_y=float(dev.y),
            device_battery_pct=float(self.rng.uniform(20, 100)),  # instrumented hook in Java
            nearest_fog_id=f"fog-{nearest_fog_idx}",
            nearest_fog_latency_ms=float(self.topology.device_to_fog_latency(dev_id, nearest_fog_idx)),
            selected_node_id=selected.node_id,
            selected_node_tier=selected.node_tier,
            teacher_label=teacher_choice.node_id,
            teacher_score=float(teacher_choice.teacher_score),
            policy_used=policy,
            node_cpu_util_pct=float(selected.cpu_util_pct),
            node_ram_used_mb=float(selected.ram_used_mb),
            node_queue_depth=int(selected.queue_depth),
            estimated_latency_ms=float(selected.latency_estimate_ms),
            estimated_energy_j=float(selected.energy_estimate_j),
        )

        # Build candidate records
        candidate_records = []
        for c in ranked:
            candidate_records.append(SchedulerCandidateRecord(
                episode_id=episode_id,
                step_id=step_id,
                task_id=task_id,
                candidate_node_id=c.node_id,
                candidate_node_tier=c.node_tier,
                candidate_rank=c.rank,
                cpu_util_pct=float(c.cpu_util_pct),
                ram_used_mb=float(c.ram_used_mb),
                queue_depth=int(c.queue_depth),
                latency_estimate_ms=float(c.latency_estimate_ms),
                energy_estimate_j=float(c.energy_estimate_j),
                rtt_ms=float(c.rtt_ms),
                score_latency_component=float(c.score_latency_component),
                score_energy_component=float(c.score_energy_component),
                score_load_component=float(c.score_load_component),
                teacher_score=float(c.teacher_score),
                is_teacher_choice=int(c.is_teacher_choice),
            ))

        # Simulate execution and build outcome
        fn_idx = self._node_id_to_idx(selected.node_id)
        noise = self.rng.uniform(0.90, 1.12)
        actual_lat = selected.latency_estimate_ms * noise
        actual_queue = self.rng.uniform(0, 5)
        actual_exec = actual_lat - selected.rtt_ms - actual_queue
        sla_met = int(actual_lat <= task["deadline_ms"])
        actual_energy = selected.energy_estimate_j * self.rng.uniform(0.85, 1.15)

        # Update node load state
        if 0 <= fn_idx < len(self.node_loads):
            load_delta = task["cpu_mips"] / self.topology.fog_nodes[fn_idx].cpu_mips
            self.node_loads[fn_idx] = float(np.clip(
                self.node_loads[fn_idx] + load_delta * 0.1, 0.0, 1.0
            ))

        outcome = SchedulerOutcomeRecord(
            episode_id=episode_id,
            step_id=step_id,
            task_id=task_id,
            selected_node_id=selected.node_id,
            actual_latency_ms=round(float(actual_lat), 3),
            actual_energy_j=round(float(actual_energy), 6),
            actual_execution_time_ms=round(float(max(0, actual_exec)), 3),
            actual_queue_wait_ms=round(float(actual_queue), 3),
            sla_met=sla_met,
            latency_vs_estimate_delta_ms=round(float(actual_lat - selected.latency_estimate_ms), 3),
            task_completed=1,
        )

        return decision, candidate_records, outcome

    def _score_scheduler_candidates(self, task: dict) -> List[NodeScore]:
        """Score all topology nodes as scheduler candidates."""
        candidates = []
        dev_id = task["device_id"]

        for i, fn in enumerate(self.topology.fog_nodes):
            load = float(self.node_loads[i])
            rtt = float(self.topology.device_to_fog_latency(dev_id, i))
            bw = self.rng.uniform(*self.cfg["network"]["bandwidth_mbps_range"])
            tx_ms = (task["data_size_kb"] / 1024.0) / bw * 1000.0
            eff_cpu = fn.cpu_mips * max(0.1, 1.0 - load)
            exec_ms = (task["cpu_mips"] / eff_cpu) * 1000.0
            lat_ms = rtt + tx_ms + exec_ms + self.rng.uniform(0, 3)
            t_sec = lat_ms / 1000.0
            cpu_util = min(1.0, task["cpu_mips"] / fn.cpu_mips)
            energy_j = self._compute_energy(fn, t_sec, cpu_util)
            s_lat, s_energy, s_load, total = self.teacher.score_candidate(lat_ms, energy_j, load)
            queue_depth = int(self.rng.integers(0, 8))
            ram_used = fn.memory_mb * load

            candidates.append(NodeScore(
                node_id=f"fog-{i}",
                node_tier="fog",
                latency_estimate_ms=round(lat_ms, 3),
                energy_estimate_j=round(energy_j, 6),
                load_pct=load,
                rtt_ms=round(rtt, 3),
                queue_depth=queue_depth,
                ram_used_mb=round(ram_used, 1),
                cpu_util_pct=round(cpu_util, 4),
                score_latency_component=round(s_lat, 8),
                score_energy_component=round(s_energy, 8),
                score_load_component=round(s_load, 6),
                teacher_score=round(total, 8),
            ))

        # Add cloud as a candidate (always available, high latency)
        cloud_lat = float(self.cfg["fog_topology"]["cloud_latency_base_ms"]) + self.rng.uniform(-20, 20)
        cloud_energy = 0.05  # cloud energy not attributable locally
        s_lat, s_energy, s_load, total = self.teacher.score_candidate(cloud_lat, cloud_energy + 1e-6, 0.1)
        candidates.append(NodeScore(
            node_id="cloud-dc",
            node_tier="cloud",
            latency_estimate_ms=round(cloud_lat, 3),
            energy_estimate_j=round(cloud_energy, 6),
            load_pct=0.1,
            rtt_ms=round(cloud_lat * 0.7, 3),
            queue_depth=0,
            ram_used_mb=0.0,
            cpu_util_pct=0.05,
            score_latency_component=round(s_lat, 8),
            score_energy_component=round(s_energy, 8),
            score_load_component=round(s_load, 6),
            teacher_score=round(total, 8),
        ))

        return candidates

    # ------------------------------------------------------------------
    # MIGRATION: decision + candidates (with NO_MIGRATION) + outcome
    # ------------------------------------------------------------------

    def evaluate_migration(
        self, module_id: str, app_id: int, device_id: str,
        current_node_idx: int, trigger_type: str,
        episode_id: int, step_id: int
    ) -> Tuple[MigrationDecisionRecord, List[MigrationCandidateRecord], MigrationOutcomeRecord]:
        """
        Evaluate migration for a module. NO_MIGRATION is always candidate rank 0.

        Returns (decision, [candidates], outcome).
        """
        fn_cur = self.topology.fog_nodes[current_node_idx]
        load_cur = float(self.node_loads[current_node_idx])
        lat_cur = self._node_latency(fn_cur, load_cur)
        energy_rate_cur = fn_cur.energy_idle_w + (fn_cur.energy_active_w - fn_cur.energy_idle_w) * load_cur

        # Device position
        dev_idx = int(device_id.split("-")[1])
        dev = self.topology.user_devices[dev_idx]

        # Build candidate list - NO_MIGRATION ALWAYS at rank 0
        migration_candidates = self._build_migration_candidates(
            module_id, episode_id, step_id,
            current_node_idx, fn_cur, lat_cur, energy_rate_cur
        )

        # Teacher picks best candidate (may be NO_MIGRATION)
        teacher_choice = max(migration_candidates, key=lambda c: c.teacher_score)
        for c in migration_candidates:
            c.is_teacher_choice = 1 if c.candidate_node_id == teacher_choice.candidate_node_id else 0

        migration_chosen = teacher_choice.candidate_node_id

        decision = MigrationDecisionRecord(
            episode_id=episode_id,
            step_id=step_id,
            module_id=module_id,
            app_id=app_id,
            trigger_type=trigger_type,
            current_node_id=f"fog-{current_node_idx}",
            current_node_tier="fog",
            current_cpu_util_pct=round(load_cur, 4),
            current_latency_ms=round(lat_cur, 3),
            current_energy_rate_w=round(energy_rate_cur, 4),
            device_id=device_id,
            device_location_x=float(dev.x),
            device_location_y=float(dev.y),
            migration_chosen=migration_chosen,
            teacher_label=migration_chosen,
            n_candidates_evaluated=len(migration_candidates),
        )

        # Build outcome
        if migration_chosen == NO_MIGRATION:
            outcome = MigrationOutcomeRecord(
                episode_id=episode_id,
                step_id=step_id,
                module_id=module_id,
                migration_chosen=NO_MIGRATION,
                actual_migration_cost_j=0.0,
                actual_downtime_ms=0.0,
                actual_bytes_transferred=0,
                latency_delta_ms=0.0,
                energy_delta_j=0.0,
                sla_met_post=int(lat_cur <= self.cfg.get("migration", {}).get("sla_latency_ms", 200)),
                migration_beneficial=0,
                migration_roi_actual=float("nan"),
            )
        else:
            chosen_cand = next(c for c in migration_candidates if c.candidate_node_id == migration_chosen)
            noise = self.rng.uniform(0.9, 1.1)
            actual_cost = chosen_cand.migration_cost_j * noise
            actual_lat_post = chosen_cand.estimated_latency_ms * self.rng.uniform(0.95, 1.05)
            lat_delta = actual_lat_post - lat_cur
            energy_delta = (chosen_cand.estimated_energy_rate_w - energy_rate_cur) * 60  # 1-min window
            beneficial = int(actual_cost < abs(energy_delta) * 0.5 and lat_delta < 0)
            roi = (abs(energy_delta) - actual_cost) / (actual_cost + 1e-9)

            outcome = MigrationOutcomeRecord(
                episode_id=episode_id,
                step_id=step_id,
                module_id=module_id,
                migration_chosen=migration_chosen,
                actual_migration_cost_j=round(float(actual_cost), 4),
                actual_downtime_ms=round(float(chosen_cand.downtime_ms * noise), 2),
                actual_bytes_transferred=int(chosen_cand.state_transfer_bytes),
                latency_delta_ms=round(float(lat_delta), 3),
                energy_delta_j=round(float(energy_delta), 4),
                sla_met_post=int(actual_lat_post <= self.cfg.get("migration", {}).get("sla_latency_ms", 200)),
                migration_beneficial=beneficial,
                migration_roi_actual=round(float(roi), 4),
            )

        return decision, migration_candidates, outcome

    def _build_migration_candidates(
        self, module_id: str, episode_id: int, step_id: int,
        current_node_idx: int, fn_cur: FogNode, lat_cur: float, energy_rate_cur: float
    ) -> List[MigrationCandidateRecord]:
        """
        Build all candidate rows. NO_MIGRATION is ALWAYS rank 0.
        """
        mc_cfg = self.cfg.get("migration", {})
        sla_lat = mc_cfg.get("sla_latency_ms", 200.0)

        # Rank 0: NO_MIGRATION (mandatory baseline)
        no_mig_score_lat, no_mig_score_e, no_mig_score_l, no_mig_total = self.teacher.score_candidate(
            lat_cur, energy_rate_cur, float(self.node_loads[current_node_idx])
        )
        candidates = [MigrationCandidateRecord(
            episode_id=episode_id,
            step_id=step_id,
            module_id=module_id,
            candidate_node_id=NO_MIGRATION,
            candidate_rank=0,
            candidate_tier="none",
            migration_cost_j=0.0,
            state_transfer_bytes=0,
            downtime_ms=0.0,
            estimated_latency_ms=round(lat_cur, 3),
            estimated_energy_rate_w=round(energy_rate_cur, 4),
            estimated_latency_improvement_ms=0.0,
            migration_roi=float("nan"),
            teacher_score=round(no_mig_total, 8),
            is_teacher_choice=0,
        )]

        # Rank 1+: actual migration targets (all other fog nodes)
        for i, fn in enumerate(self.topology.fog_nodes):
            if i == current_node_idx:
                continue
            load_cand = float(self.node_loads[i])
            lat_cand = self._node_latency(fn, load_cand)
            energy_rate_cand = fn.energy_idle_w + (fn.energy_active_w - fn.energy_idle_w) * load_cand

            # Migration cost model: state_size * energy_per_byte + fixed overhead
            state_bytes = int(self.rng.integers(5_000_000, 200_000_000))  # 5-200 MB state
            energy_per_byte = mc_cfg.get("energy_per_byte_j", 5e-9)
            fixed_overhead_j = mc_cfg.get("fixed_overhead_j", 0.1)
            mig_cost = state_bytes * energy_per_byte + fixed_overhead_j
            downtime_ms = (state_bytes / (100 * 1e6)) * 1000  # 100 Mbps link

            lat_improvement = lat_cur - lat_cand
            energy_improvement = (energy_rate_cur - energy_rate_cand) * 60
            roi = (energy_improvement - mig_cost) / (mig_cost + 1e-9)

            s_lat, s_energy, s_load, total = self.teacher.score_candidate(
                lat_cand, energy_rate_cand + 1e-9, load_cand
            )
            # Penalize score for migration cost
            total = total - mig_cost * 1e-4

            rank = len(candidates)
            candidates.append(MigrationCandidateRecord(
                episode_id=episode_id,
                step_id=step_id,
                module_id=module_id,
                candidate_node_id=f"fog-{i}",
                candidate_rank=rank,
                candidate_tier="fog",
                migration_cost_j=round(float(mig_cost), 4),
                state_transfer_bytes=state_bytes,
                downtime_ms=round(float(downtime_ms), 2),
                estimated_latency_ms=round(lat_cand, 3),
                estimated_energy_rate_w=round(energy_rate_cand, 4),
                estimated_latency_improvement_ms=round(lat_improvement, 3),
                migration_roi=round(float(roi), 4),
                teacher_score=round(total, 8),
                is_teacher_choice=0,
            ))

        return candidates

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_energy(self, fn: FogNode, t_sec: float, cpu_util: float) -> float:
        """iFogSim2 PowerModel: E = E_idle*t + (E_active - E_idle) * util * t"""
        return fn.energy_idle_w * t_sec + (fn.energy_active_w - fn.energy_idle_w) * cpu_util * t_sec

    def _node_latency(self, fn: FogNode, load: float) -> float:
        eff_cpu = fn.cpu_mips * max(0.1, 1.0 - load)
        exec_ms = (500.0 / eff_cpu) * 1000.0
        return float(exec_ms + self.rng.uniform(2, 15))

    def _nearest_fog(self, dev_id: str) -> int:
        dev_idx = int(dev_id.split("-")[1])
        min_lat = math.inf
        nearest = 0
        for i in range(len(self.topology.fog_nodes)):
            lat = self.topology.device_to_fog_latency(dev_id, i)
            if lat < min_lat:
                min_lat = lat
                nearest = i
        return nearest

    def _node_id_to_idx(self, node_id: str) -> int:
        if node_id.startswith("fog-"):
            return int(node_id.split("-")[1])
        return -1

    def reset_loads(self):
        self.node_loads = self.rng.uniform(0.0, 0.5, size=len(self.topology.fog_nodes))
