"""
Fog/cloud/edge topology generator.

Produces a randomized but configurable topology of fog nodes, user devices,
and network links. Matches iFogSim2's physical layer model.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


@dataclass
class FogNode:
    node_id: int
    cpu_mips: float
    memory_mb: float
    energy_idle_w: float
    energy_active_w: float
    location: Tuple[float, float]  # (x, y) coordinates


@dataclass
class UserDevice:
    device_id: int
    cpu_mips: float
    memory_mb: float
    location: Tuple[float, float]
    assigned_fog_node: int  # nearest fog node at init


@dataclass
class NetworkLink:
    src: int
    dst: int
    bandwidth_mbps: float
    latency_ms: float


class FogTopology:
    """
    Generates and manages the fog/cloud/edge topology for simulation.

    The topology is a two-tier structure:
      Tier 1 (Edge): User devices with constrained resources
      Tier 2 (Fog): Fog nodes with moderate resources
      Tier 3 (Cloud): Single cloud endpoint with unlimited resources + high latency
    """

    def __init__(self, cfg: dict, seed: int = 42):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.fog_nodes: List[FogNode] = []
        self.user_devices: List[UserDevice] = []
        self.links: List[NetworkLink] = []
        self.latency_matrix: np.ndarray = None
        self.bandwidth_matrix: np.ndarray = None
        self._build()

    def _build(self):
        tc = self.cfg["fog_topology"]
        nc = self.cfg["network"]

        # Generate fog nodes
        cpu_list = tc["fog_node_cpu_mips"]
        mem_list = tc["fog_node_memory_mb"]
        idle_list = tc["fog_node_energy_idle_w"]
        act_list = tc["fog_node_energy_active_w"]
        n = tc["n_fog_nodes"]

        for i in range(n):
            self.fog_nodes.append(FogNode(
                node_id=i,
                cpu_mips=float(cpu_list[i % len(cpu_list)]),
                memory_mb=float(mem_list[i % len(mem_list)]),
                energy_idle_w=float(idle_list[i % len(idle_list)]),
                energy_active_w=float(act_list[i % len(act_list)]),
                location=(self.rng.uniform(0, 100), self.rng.uniform(0, 100)),
            ))

        # Generate user devices
        dc = self.cfg["user_devices"]
        for i in range(tc["n_user_devices"]):
            loc = (self.rng.uniform(0, 100), self.rng.uniform(0, 100))
            nearest = self._nearest_fog_node(loc)
            self.user_devices.append(UserDevice(
                device_id=i,
                cpu_mips=self.rng.uniform(*dc["cpu_mips_range"]),
                memory_mb=self.rng.uniform(*dc["memory_mb_range"]),
                location=loc,
                assigned_fog_node=nearest,
            ))

        # Build latency and bandwidth matrices (fog-to-fog + device-to-fog)
        n_fog = len(self.fog_nodes)
        self.latency_matrix = np.zeros((n_fog, n_fog))
        self.bandwidth_matrix = np.zeros((n_fog, n_fog))
        lat_lo, lat_hi = nc["latency_base_ms_range"]
        bw_lo, bw_hi = nc["bandwidth_mbps_range"]

        for i in range(n_fog):
            for j in range(n_fog):
                if i != j:
                    dist = np.linalg.norm(
                        np.array(self.fog_nodes[i].location) - np.array(self.fog_nodes[j].location)
                    )
                    base_lat = lat_lo + (lat_hi - lat_lo) * dist / 141.4
                    jitter = self.rng.uniform(0, nc["latency_jitter_ms"])
                    self.latency_matrix[i, j] = base_lat + jitter
                    self.bandwidth_matrix[i, j] = self.rng.uniform(bw_lo, bw_hi)

    def _nearest_fog_node(self, loc: Tuple[float, float]) -> int:
        dists = [
            np.linalg.norm(np.array(loc) - np.array(fn.location))
            for fn in self.fog_nodes
        ]
        return int(np.argmin(dists))

    def device_to_fog_latency(self, device_id: int, fog_node_id: int) -> float:
        """Latency from user device to fog node in ms."""
        dev = self.user_devices[device_id]
        fn = self.fog_nodes[fog_node_id]
        dist = np.linalg.norm(np.array(dev.location) - np.array(fn.location))
        nc = self.cfg["network"]
        lat_lo, lat_hi = nc["latency_base_ms_range"]
        base = lat_lo + (lat_hi - lat_lo) * dist / 141.4
        return float(base + self.rng.uniform(0, nc["latency_jitter_ms"]))

    def generate_task(self, app_id: int, task_id: int, device_id: int) -> dict:
        """Generate a random task for a given application."""
        apps = self.cfg["applications"]
        app = next(a for a in apps if a["id"] == app_id)
        return {
            "task_id": task_id,
            "app_id": app_id,
            "device_id": device_id,
            "cpu_mips": float(self.rng.uniform(*app["task_cpu_mips_range"])),
            "memory_mb": float(self.rng.uniform(*app["task_memory_mb_range"])),
            "data_size_kb": float(self.rng.uniform(*app["task_data_size_kb_range"])),
            "deadline_ms": float(self.rng.uniform(*app["deadline_ms_range"])),
        }

    def update_device_locations(self):
        """Simulate user device mobility - update positions and reassign fog nodes."""
        dc = self.cfg["user_devices"]
        if not dc.get("mobility", False):
            return
        for dev in self.user_devices:
            dx, dy = self.rng.uniform(-5, 5, size=2)
            new_x = np.clip(dev.location[0] + dx, 0, 100)
            new_y = np.clip(dev.location[1] + dy, 0, 100)
            dev.location = (float(new_x), float(new_y))
            dev.assigned_fog_node = self._nearest_fog_node(dev.location)

    def fog_node_state(self, node_id: int, current_load: float) -> dict:
        """Return current state snapshot for a fog node."""
        fn = self.fog_nodes[node_id]
        return {
            "fog_node_id": node_id,
            "fog_cpu_available_mips": fn.cpu_mips * (1.0 - current_load),
            "fog_memory_available_mb": fn.memory_mb * (1.0 - current_load * 0.8),
            "fog_load_fraction": current_load,
            "fog_energy_idle_w": fn.energy_idle_w,
            "fog_energy_active_w": fn.energy_active_w,
        }
