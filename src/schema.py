"""
Dataset schemas for the 6-CSV iFogSim2 pipeline.

Three CSVs for scheduling: decisions, candidates, outcomes.
Three CSVs for migration: decisions, candidates, outcomes.

Join key: episode_id + step_id + task_id (scheduler) or module_id (migration).
"""

from dataclasses import dataclass, fields, asdict
from typing import List, Optional
import pandas as pd


# ---------------------------------------------------------------------------
# SCHEDULER DATASET TRIPLE
# ---------------------------------------------------------------------------

@dataclass
class SchedulerDecisionRecord:
    """One scheduling decision event. One row per task arrival."""
    episode_id: int
    step_id: int
    task_id: str
    app_id: int
    # Task properties
    task_cpu_mi: float          # million instructions (Cloudlet.getLength())
    task_deadline_ms: float     # SLA deadline
    task_data_size_kb: float    # input payload
    task_output_size_kb: float  # output payload
    # Source device state (instrumented)
    device_id: str
    device_location_x: float   # normalized 0-1
    device_location_y: float   # normalized 0-1
    device_battery_pct: float  # remaining battery (instrumented hook)
    # Nearest fog node (derived)
    nearest_fog_id: str
    nearest_fog_latency_ms: float   # RTT to nearest fog
    # Selected node (teacher or policy output)
    selected_node_id: str
    selected_node_tier: str         # fog / edge / cloud
    # Teacher policy fields (custom)
    teacher_label: str              # recommended node ID
    teacher_score: float            # composite score for chosen node
    policy_used: str                # first_fit / energy_aware / round_robin / teacher
    # Context at selected node (instrumented)
    node_cpu_util_pct: float        # CPU utilization 0-1
    node_ram_used_mb: float
    node_queue_depth: int
    # Pre-execution estimates (derived)
    estimated_latency_ms: float
    estimated_energy_j: float       # labeled _estimated - approximation

    @classmethod
    def columns(cls) -> List[str]:
        return [f.name for f in fields(cls)]

    def to_dict(self):
        return asdict(self)


@dataclass
class SchedulerCandidateRecord:
    """One candidate node evaluated per scheduling decision."""
    episode_id: int
    step_id: int
    task_id: str
    candidate_node_id: str
    candidate_node_tier: str        # fog / edge / cloud
    candidate_rank: int             # 0 = best, ascending
    # Node state at evaluation time
    cpu_util_pct: float
    ram_used_mb: float
    queue_depth: int
    # Per-candidate estimates
    latency_estimate_ms: float
    energy_estimate_j: float        # approximation
    rtt_ms: float                   # round-trip to this node
    # Teacher scoring components
    score_latency_component: float
    score_energy_component: float
    score_load_component: float
    teacher_score: float            # composite score
    is_teacher_choice: int          # 1 if this is the argmax node

    @classmethod
    def columns(cls) -> List[str]:
        return [f.name for f in fields(cls)]

    def to_dict(self):
        return asdict(self)


@dataclass
class SchedulerOutcomeRecord:
    """Post-execution actual results for a scheduling decision."""
    episode_id: int
    step_id: int
    task_id: str
    selected_node_id: str
    # Actual measurements
    actual_latency_ms: float
    actual_energy_j: float          # labeled actual but per-task approx (see docs)
    actual_execution_time_ms: float
    actual_queue_wait_ms: float
    # SLA outcome
    sla_met: int                    # 1 = deadline met, 0 = violated
    latency_vs_estimate_delta_ms: float  # actual - estimated
    # Throughput
    task_completed: int             # 1 = completed, 0 = failed/dropped

    @classmethod
    def columns(cls) -> List[str]:
        return [f.name for f in fields(cls)]

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# MIGRATION DATASET TRIPLE
# ---------------------------------------------------------------------------

@dataclass
class MigrationDecisionRecord:
    """One migration evaluation event. First candidate is always NO_MIGRATION."""
    episode_id: int
    step_id: int
    module_id: str
    app_id: int
    # Migration trigger
    trigger_type: str               # mobility / overload / sla_breach / energy
    # Current placement state
    current_node_id: str
    current_node_tier: str
    current_cpu_util_pct: float
    current_latency_ms: float
    current_energy_rate_w: float    # power draw at current node
    # Device position (for mobility-triggered)
    device_id: str
    device_location_x: float
    device_location_y: float
    # Teacher decision
    migration_chosen: str           # NO_MIGRATION or target node ID
    teacher_label: str              # teacher recommendation
    n_candidates_evaluated: int     # includes NO_MIGRATION

    @classmethod
    def columns(cls) -> List[str]:
        return [f.name for f in fields(cls)]

    def to_dict(self):
        return asdict(self)


@dataclass
class MigrationCandidateRecord:
    """One candidate per migration evaluation. NO_MIGRATION always at rank 0."""
    episode_id: int
    step_id: int
    module_id: str
    candidate_node_id: str          # "NO_MIGRATION" for baseline row
    candidate_rank: int             # 0 = NO_MIGRATION, 1+ = actual nodes
    candidate_tier: str             # none / fog / edge / cloud
    # Migration cost (0 for NO_MIGRATION)
    migration_cost_j: float
    state_transfer_bytes: int       # 0 for NO_MIGRATION
    downtime_ms: float              # 0 for NO_MIGRATION
    # Post-migration estimates for this candidate
    estimated_latency_ms: float
    estimated_energy_rate_w: float
    estimated_latency_improvement_ms: float  # vs current node
    migration_roi: float            # (benefit_j - cost_j) / cost_j; NaN for NO_MIGRATION
    teacher_score: float
    is_teacher_choice: int          # 1 if teacher selected this candidate

    @classmethod
    def columns(cls) -> List[str]:
        return [f.name for f in fields(cls)]

    def to_dict(self):
        return asdict(self)


@dataclass
class MigrationOutcomeRecord:
    """Actual post-migration results."""
    episode_id: int
    step_id: int
    module_id: str
    migration_chosen: str           # node ID or NO_MIGRATION
    # Actual costs incurred (0 if NO_MIGRATION)
    actual_migration_cost_j: float
    actual_downtime_ms: float
    actual_bytes_transferred: int
    # Performance delta
    latency_delta_ms: float         # post - pre; negative = improvement
    energy_delta_j: float           # post - pre; negative = improvement
    sla_met_post: int               # 1 = SLA met after migration
    migration_beneficial: int       # 1 = benefit > cost
    migration_roi_actual: float     # realized ROI

    @classmethod
    def columns(cls) -> List[str]:
        return [f.name for f in fields(cls)]

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def records_to_dataframe(records: list) -> pd.DataFrame:
    """Convert list of any record dataclass to a DataFrame."""
    if not records:
        return pd.DataFrame()
    return pd.DataFrame([r.to_dict() for r in records])


DATASET_NAMES = [
    "scheduler_decisions",
    "scheduler_candidates",
    "scheduler_outcomes",
    "migration_decisions",
    "migration_candidates",
    "migration_outcomes",
]
