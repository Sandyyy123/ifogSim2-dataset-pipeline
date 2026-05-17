"""
High-fidelity Python simulation of iFogSim2's energy and latency models.

Implements iFogSim2's core equations:
  Energy model:  E_total = E_idle * t + (E_active - E_idle) * CPU_utilization * t
  Migration cost: E_mig = data_size_kb * energy_coeff + latency_penalty
  QoS model:    latency = transmission_time + execution_time + propagation_latency

When the actual Java iFogSim2 is available, replace the _compute_* methods
with subprocess calls to the Java process. The DatasetBuilder interface remains identical.
"""

import numpy as np
from .schema import SchedulerRecord, ReschedulerRecord
from .topology_generator import FogTopology


class iFogSim2Simulator:
    """
    Simulates iFogSim2 fog computing environment for dataset generation.

    Args:
        topology: FogTopology instance
        cfg: Full config dict
        rng: numpy random generator (for reproducibility)
    """

    def __init__(self, topology: FogTopology, cfg: dict, rng: np.random.Generator):
        self.topology = topology
        self.cfg = cfg
        self.rng = rng
        self.node_loads = np.zeros(len(topology.fog_nodes))  # current load per node

    def schedule(self, task: dict, fog_node_id: int, policy: str) -> SchedulerRecord:
        """
        Simulate scheduling task to fog_node_id and return a SchedulerRecord.

        Args:
            task: Task dict from FogTopology.generate_task()
            fog_node_id: Selected fog node (-1 for cloud)
            policy: Scheduling policy label (for provenance)
        """
        fn = self.topology.fog_nodes[fog_node_id]
        dev_id = task["device_id"]

        # Network conditions
        edge_to_fog_lat = self.topology.device_to_fog_latency(dev_id, fog_node_id)
        cloud_lat = float(self.cfg["fog_topology"]["cloud_latency_base_ms"])
        bw = self.rng.uniform(*self.cfg["network"]["bandwidth_mbps_range"])

        # Transmission time (ms)
        tx_time_ms = (task["data_size_kb"] / 1024.0) / bw * 1000.0

        # Execution time (ms)
        load = self.node_loads[fog_node_id]
        effective_cpu = fn.cpu_mips * max(0.1, 1.0 - load)
        exec_time_ms = (task["cpu_mips"] / effective_cpu) * 1000.0

        total_latency_ms = edge_to_fog_lat + tx_time_ms + exec_time_ms
        qos_met = int(total_latency_ms <= task["deadline_ms"])

        # Energy (iFogSim2 model)
        t_sec = total_latency_ms / 1000.0
        cpu_util = min(1.0, task["cpu_mips"] / fn.cpu_mips)
        energy_j = self._compute_energy(fn, t_sec, cpu_util)

        # Update load
        load_delta = task["cpu_mips"] / fn.cpu_mips
        self.node_loads[fog_node_id] = float(np.clip(load + load_delta * 0.1, 0.0, 1.0))

        node_state = self.topology.fog_node_state(fog_node_id, load)

        return SchedulerRecord(
            episode_id=task.get("episode_id", 0),
            timestep=task.get("timestep", 0),
            task_id=task["task_id"],
            app_id=task["app_id"],
            task_cpu_mips=task["cpu_mips"],
            task_memory_mb=task["memory_mb"],
            task_data_size_kb=task["data_size_kb"],
            deadline_ms=task["deadline_ms"],
            fog_node_id=fog_node_id,
            fog_cpu_available_mips=node_state["fog_cpu_available_mips"],
            fog_memory_available_mb=node_state["fog_memory_available_mb"],
            fog_load_fraction=node_state["fog_load_fraction"],
            fog_energy_idle_w=fn.energy_idle_w,
            fog_energy_active_w=fn.energy_active_w,
            edge_to_fog_latency_ms=edge_to_fog_lat,
            cloud_latency_ms=cloud_lat,
            network_bandwidth_mbps=float(bw),
            energy_consumed_j=energy_j,
            execution_latency_ms=total_latency_ms,
            qos_met=qos_met,
            routing_decision=fog_node_id,
        )

    def evaluate_migration(
        self, module_id: int, app_id: int, current_node: int, candidate_node: int,
        episode_id: int, timestep: int
    ) -> ReschedulerRecord:
        """
        Evaluate whether migrating module from current_node to candidate_node is beneficial.
        Returns a ReschedulerRecord with the ground-truth migration_decision label.
        """
        fn_cur = self.topology.fog_nodes[current_node]
        fn_cand = self.topology.fog_nodes[candidate_node]
        mc = self.cfg["migration"]

        load_cur = self.node_loads[current_node]
        load_cand = self.node_loads[candidate_node]

        # Latency at each node
        lat_cur = self._qos_latency(fn_cur, load_cur)
        lat_cand = self._qos_latency(fn_cand, load_cand)

        # Migration cost
        data_size_mb = self.rng.uniform(10, 200)
        mig_lat_ms = data_size_mb * 1024 * mc["migration_latency_coeff"]
        mig_energy_j = data_size_mb * 1024 * mc["migration_bandwidth_energy_coeff"]

        # Net improvements
        lat_improvement = lat_cur - lat_cand  # positive = candidate better
        energy_cur = self._compute_energy(fn_cur, 1.0, load_cur)
        energy_cand = self._compute_energy(fn_cand, 1.0, load_cand)
        energy_improvement = energy_cur - energy_cand

        load_balance_improvement = abs(load_cur - load_cand) - abs(
            (load_cur - 0.1) - (load_cand + 0.1)
        )

        # Ground truth: migrate if improvement outweighs migration cost and QoS threshold met
        qos_thresh = mc["qos_degradation_trigger_ms"]
        migrate = int(
            lat_improvement > qos_thresh
            and energy_improvement > mig_energy_j * 0.5
            and load_cand < mc["load_threshold_trigger"]
        )

        # Occasionally flip label for diversity
        if self.rng.random() < 0.05:
            migrate = 1 - migrate

        return ReschedulerRecord(
            episode_id=episode_id,
            timestep=timestep,
            module_id=module_id,
            app_id=app_id,
            current_node_id=current_node,
            current_node_load=float(load_cur),
            current_node_energy_w=fn_cur.energy_active_w,
            current_qos_latency_ms=lat_cur,
            candidate_node_id=candidate_node,
            candidate_node_load=float(load_cand),
            candidate_node_energy_w=fn_cand.energy_active_w,
            candidate_qos_latency_ms=lat_cand,
            migration_data_size_mb=float(data_size_mb),
            migration_latency_ms=mig_lat_ms,
            migration_energy_j=mig_energy_j,
            latency_improvement_ms=lat_improvement,
            energy_improvement_j=energy_improvement,
            load_balance_improvement=float(load_balance_improvement),
            migration_decision=migrate,
        )

    def _compute_energy(self, fn, t_sec: float, cpu_util: float) -> float:
        """iFogSim2 energy model: E = E_idle*t + (E_active - E_idle) * util * t"""
        return fn.energy_idle_w * t_sec + (fn.energy_active_w - fn.energy_idle_w) * cpu_util * t_sec

    def _qos_latency(self, fn, load: float) -> float:
        """Estimate end-to-end latency for a module hosted on fn at given load."""
        effective_cpu = fn.cpu_mips * max(0.1, 1.0 - load)
        exec_ms = (500.0 / effective_cpu) * 1000.0  # normalized 500 MIPS task
        return float(exec_ms + self.rng.uniform(2, 10))

    def apply_scheduling_policy(self, task: dict, policy: str) -> int:
        """Select fog node index using named policy."""
        n = len(self.topology.fog_nodes)
        if policy == "first_fit":
            for i in range(n):
                fn = self.topology.fog_nodes[i]
                if self.node_loads[i] < 0.8 and fn.cpu_mips >= task["cpu_mips"]:
                    return i
            return 0
        elif policy == "round_robin":
            return int(task["task_id"] % n)
        elif policy == "energy_aware":
            scores = [
                fn.energy_idle_w + fn.energy_active_w * self.node_loads[i]
                for i, fn in enumerate(self.topology.fog_nodes)
            ]
            return int(np.argmin(scores))
        else:  # random
            return int(self.rng.integers(0, n))

    def reset_loads(self):
        self.node_loads = self.rng.uniform(0.0, 0.5, size=len(self.topology.fog_nodes))
