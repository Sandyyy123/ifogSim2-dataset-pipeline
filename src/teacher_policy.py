"""
Teacher scheduling policy: energy-latency-load weighted heuristic.

Generates 85-90% of dataset rows. Remaining rows use policy rotation
(first_fit, round_robin, energy_aware) for diversity.

Score formula per candidate node n given task t:
    score(n, t) = w_lat * (1/latency(n,t)) + w_energy * (1/energy(n,t)) + w_load * (1 - load(n))

Default weights: w_lat=0.40, w_energy=0.35, w_load=0.25 (configurable).
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class NodeScore:
    node_id: str
    node_tier: str
    latency_estimate_ms: float
    energy_estimate_j: float
    load_pct: float
    rtt_ms: float
    queue_depth: int
    ram_used_mb: float
    cpu_util_pct: float
    score_latency_component: float
    score_energy_component: float
    score_load_component: float
    teacher_score: float
    rank: int = 0
    is_teacher_choice: int = 0


class TeacherPolicy:
    """
    Weighted energy-latency-load heuristic for scheduling and migration.

    Weights are configurable per scenario. Default calibrated for
    latency-sensitive edge workloads.
    """

    def __init__(self, w_lat: float = 0.40, w_energy: float = 0.35, w_load: float = 0.25):
        assert abs(w_lat + w_energy + w_load - 1.0) < 1e-6, "Weights must sum to 1.0"
        self.w_lat = w_lat
        self.w_energy = w_energy
        self.w_load = w_load

    def score_candidate(
        self,
        latency_ms: float,
        energy_j: float,
        load_pct: float,
    ) -> tuple:
        """Return (score_lat, score_energy, score_load, total_score)."""
        eps = 1e-9
        s_lat = self.w_lat * (1.0 / (latency_ms + eps))
        s_energy = self.w_energy * (1.0 / (energy_j + eps))
        s_load = self.w_load * (1.0 - load_pct)
        return s_lat, s_energy, s_load, s_lat + s_energy + s_load

    def rank_candidates(self, candidates: List[NodeScore]) -> List[NodeScore]:
        """Sort candidates by teacher_score descending, assign ranks, mark winner."""
        sorted_cands = sorted(candidates, key=lambda c: c.teacher_score, reverse=True)
        for i, c in enumerate(sorted_cands):
            c.rank = i
            c.is_teacher_choice = 1 if i == 0 else 0
        return sorted_cands

    def select(self, candidates: List[NodeScore]) -> NodeScore:
        """Return the highest-scoring candidate (teacher choice)."""
        return max(candidates, key=lambda c: c.teacher_score)


class PolicyRotation:
    """Rotate through non-teacher policies for diversity in 10-15% of rows."""

    POLICIES = ["first_fit", "round_robin", "energy_aware"]

    def __init__(self, teacher_fraction: float = 0.87):
        self.teacher_fraction = teacher_fraction

    def select_policy(self, rng: np.random.Generator, step_id: int) -> str:
        """Return policy name for this step."""
        if rng.random() < self.teacher_fraction:
            return "teacher"
        return self.POLICIES[step_id % len(self.POLICIES)]

    def apply(self, policy: str, candidates: List[NodeScore], rng: np.random.Generator, step_id: int) -> NodeScore:
        """Apply named policy to select from candidates."""
        if policy == "teacher":
            return max(candidates, key=lambda c: c.teacher_score)
        elif policy == "first_fit":
            for c in candidates:
                if c.load_pct < 0.8:
                    return c
            return candidates[0]
        elif policy == "round_robin":
            return candidates[step_id % len(candidates)]
        elif policy == "energy_aware":
            return min(candidates, key=lambda c: c.energy_estimate_j)
        return candidates[0]
