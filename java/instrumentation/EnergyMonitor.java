package instrumentation;

import org.cloudbus.cloudsim.core.SimEvent;
import org.fog.entities.FogDevice;
import org.fog.utils.FogEvents;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.Map;

/**
 * EnergyMonitor - instrumentation hook for iFogSim2 device-level energy tracking.
 *
 * Attach to each FogDevice to capture per-step energy measurements.
 * Output: CSV rows appended to energy_log.csv for Python ingestion.
 *
 * Integration:
 *   EnergyMonitor monitor = new EnergyMonitor("results/energy_log.csv");
 *   monitor.attach(fogDevice);           // call for each FogDevice
 *   monitor.record(fogDevice, stepId);   // call in simulation loop
 *   monitor.close();                     // flush and close at end
 *
 * Note: iFogSim2 PowerModel accounts for energy at device level, not per-task.
 * Per-task attribution in the Python pipeline is a proportional estimate.
 */
public class EnergyMonitor {

    private final PrintWriter writer;
    private final Map<String, Double> lastEnergySnapshot = new HashMap<>();

    public EnergyMonitor(String outputPath) throws IOException {
        this.writer = new PrintWriter(new FileWriter(outputPath, true));
        writer.println("step_id,device_id,device_type,energy_total_j,energy_delta_j," +
                       "cpu_util_pct,mips_total,mips_used,ram_total_mb,ram_used_mb," +
                       "energy_idle_w,energy_active_w,sim_time_ms");
    }

    /**
     * Record device energy state at the given simulation step.
     * Call this after each CloudSim clock tick advance.
     */
    public void record(FogDevice device, int stepId) {
        double energyTotal = device.getEnergyConsumption();
        String deviceId = device.getName();
        double lastEnergy = lastEnergySnapshot.getOrDefault(deviceId, 0.0);
        double energyDelta = energyTotal - lastEnergy;
        lastEnergySnapshot.put(deviceId, energyTotal);

        double mipsTotal = device.getHost().getTotalMips();
        double mipsUsed = mipsTotal - device.getHost().getAvailableMips();
        double cpuUtil = mipsTotal > 0 ? mipsUsed / mipsTotal : 0.0;

        double ramTotal = device.getHost().getRam();
        double ramUsed = ramTotal - device.getHost().getRamProvisioner().getAvailableRam();

        // PowerModel parameters (iFogSim2 LinearPowerModel)
        double energyIdleW = device.getHost().getPowerModel().getPower(0.0);
        double energyActiveW = device.getHost().getPowerModel().getPower(1.0);

        writer.printf("%d,%s,%s,%.6f,%.6f,%.4f,%.1f,%.1f,%.1f,%.1f,%.2f,%.2f,%.3f%n",
                stepId,
                deviceId,
                device.getClass().getSimpleName(),
                energyTotal,
                energyDelta,
                cpuUtil,
                mipsTotal,
                mipsUsed,
                ramTotal,
                ramUsed,
                energyIdleW,
                energyActiveW,
                org.cloudbus.cloudsim.core.CloudSim.clock()
        );
    }

    public void flush() {
        writer.flush();
    }

    public void close() {
        writer.flush();
        writer.close();
    }
}
