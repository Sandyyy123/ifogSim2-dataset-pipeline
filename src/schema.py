"""
Dataset schemas for iFogSim2 scheduler and rescheduler datasets.

These schemas define the exact column sets for the two CSV outputs.
SchedulerRecord: captures task-routing decisions (where to send a task).
ReschedulerRecord: captures module migration decisions (move or stay).
"""

from dataclasses import dataclass, fields, asdict
from typing import List
import pandas as pd


@dataclass
class SchedulerRecord:
    """One row of the scheduler dataset - one scheduling decision."""
    episode_id: int
    timestep: int
    task_id: int
    app_id: int
    # Task resource requirements
    task_cpu_mips: float
    task_memory_mb: float
    task_data_size_kb: float
    deadline_ms: float
    # Current fog node state at routing time
    fog_node_id: int
    fog_cpu_available_mips: float
    fog_memory_available_mb: float
    fog_load_fraction: float          # 0.0 - 1.0
    fog_energy_idle_w: float
    fog_energy_active_w: float
    # Network conditions
    edge_to_fog_latency_ms: float
    cloud_latency_ms: float
    network_bandwidth_mbps: float
    # Outcome metrics
    energy_consumed_j: float
    execution_latency_ms: float
    qos_met: int                      # 1 if deadline met, 0 otherwise
    # Label: which node was selected (fog node index or -1 for cloud)
    routing_decision: int

    @classmethod
    def columns(cls) -> List[str]:
        return [f.name for f in fields(cls)]

    def to_dict(self):
        return asdict(self)


@dataclass
class ReschedulerRecord:
    """One row of the rescheduler dataset - one migration decision."""
    episode_id: int
    timestep: int
    module_id: int
    app_id: int
    # Current placement
    current_node_id: int
    current_node_load: float
    current_node_energy_w: float
    current_qos_latency_ms: float
    # Migration candidate
    candidate_node_id: int
    candidate_node_load: float
    candidate_node_energy_w: float
    candidate_qos_latency_ms: float
    # Migration cost
    migration_data_size_mb: float
    migration_latency_ms: float
    migration_energy_j: float
    # Net improvement if migration occurs
    latency_improvement_ms: float     # positive = candidate is better
    energy_improvement_j: float       # positive = candidate uses less
    load_balance_improvement: float   # positive = load more balanced
    # Label: 1 = migrate to candidate, 0 = stay
    migration_decision: int

    @classmethod
    def columns(cls) -> List[str]:
        return [f.name for f in fields(cls)]

    def to_dict(self):
        return asdict(self)


def records_to_dataframe(records: list) -> pd.DataFrame:
    """Convert list of SchedulerRecord or ReschedulerRecord to DataFrame."""
    if not records:
        return pd.DataFrame()
    return pd.DataFrame([r.to_dict() for r in records])
