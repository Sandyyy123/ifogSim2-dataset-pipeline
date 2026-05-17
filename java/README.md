# Java Instrumentation: iFogSim2 Integration

This folder contains Java instrumentation hooks for integrating with a real iFogSim2 simulation.

## Files

| File | Purpose |
|------|---------|
| `EnergyMonitor.java` | Per-step device energy capture via PowerModel hooks |
| `SchedulerCallback.java` | FogBroker scheduling decision intercept |
| `MigrationCallback.java` | AppModule migration pre/post event capture |

## Integration Steps (Milestone 4)

### 1. Add to your iFogSim2 project

Place `instrumentation/` under `src/main/java/` in your iFogSim2 Maven/Gradle project.

```
src/main/java/
  instrumentation/
    EnergyMonitor.java
    SchedulerCallback.java
    MigrationCallback.java
```

### 2. Initialize monitors in your simulation class

```java
EnergyMonitor energyMon = new EnergyMonitor("results/energy_log.csv");
SchedulerCallback schedCb = new SchedulerCallback("results/scheduler_log.csv");
MigrationCallback migCb = new MigrationCallback(
    "results/migration_pre_log.csv",
    "results/migration_post_log.csv"
);
```

### 3. Hook into FogBroker

Subclass `FogBroker` and override `processEvent()`:

```java
@Override
protected void processEvent(SimEvent ev) {
    super.processEvent(ev);
    if (ev.getTag() == FogEvents.TUPLE_ARRIVAL) {
        Tuple tuple = (Tuple) ev.getData();
        FogDevice selected = getSelectedDevice(tuple);  // your routing logic
        schedCb.record(tuple, selected, stepId, episodeId, currentPolicy);
    }
}
```

### 4. Record energy per step

In your simulation loop (or time-advance handler):

```java
for (FogDevice device : fogDevices) {
    energyMon.record(device, stepId);
}
```

### 5. Hook into module migration

Subclass `FogDevice` and override `migrateAppModule()`:

```java
@Override
protected void migrateAppModule(String moduleId, String targetDeviceName) {
    FogDevice target = getFogDeviceByName(targetDeviceName);
    migCb.recordPre(module, this, target, stepId, episodeId, triggerType);
    super.migrateAppModule(moduleId, targetDeviceName);
    // In the completion event handler:
    migCb.recordPost(module, target, stepId, episodeId, cost, downtime, bytes, latPost, slaMs);
}
```

### 6. Feed Java logs to Python pipeline

The Python pipeline expects Java logs in `results/`:

```bash
python generate_dataset.py --java-logs results/ --episodes 500
```

The `--java-logs` mode replaces the Python simulation with log parsing from real iFogSim2 runs.
Python adds: teacher labels, NO_MIGRATION candidates, feature derivation, all 6 CSV assembly.

## Field Feasibility Notes

| Field | Java Source | Instrumentation Required |
|-------|-------------|--------------------------|
| `task_cpu_mi` | `Cloudlet.getLength()` | None - direct API |
| `task_deadline_ms` | `AppEdge.deadline` | None - direct API |
| `node_cpu_util_pct` | `Host.getAvailableMips()` delta | Requires step-level hook |
| `device_battery_pct` | Not in base iFogSim2 | Custom EnergyMonitor |
| `teacher_label` | Python TeacherPolicy | Python post-processing |
| `estimated_energy_j` | Python model | Approximation (see docs) |

## Energy Accounting Warning

iFogSim2 `PowerModel` computes energy at the **device level**, not per-task.
`energy_delta_j` in `EnergyMonitor` is the device-level delta per step.
Per-task energy attribution in Python is `(task_mi / total_mi_this_step) * device_energy_delta`.
This is an approximation standard in fog simulation research - label these columns accordingly.
