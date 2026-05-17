package instrumentation;

import org.fog.application.AppModule;
import org.fog.entities.FogDevice;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

/**
 * MigrationCallback - intercepts AppModule migration events in iFogSim2.
 *
 * Hook into FogDevice.migrateAppModule() to capture raw migration decisions.
 * The Python pipeline adds NO_MIGRATION candidates and teacher labels in post-processing.
 *
 * Integration:
 *   Override or subclass FogDevice and call this in migrateAppModule():
 *
 *     MigrationCallback migCb = new MigrationCallback("results/migration_log.csv");
 *     // Before migration executes:
 *     migCb.recordPre(module, sourceDevice, targetDevice, stepId, episodeId, triggerType);
 *     // After migration completes (in event handler):
 *     migCb.recordPost(module, targetDevice, stepId, episodeId, actualCostJ, actualDowntimeMs);
 *
 * IMPORTANT: The Python pipeline enforces the NO_MIGRATION rule by adding it as rank-0
 * candidate for every evaluation. This Java callback only captures decisions that
 * actually triggered a migration evaluation in iFogSim2.
 */
public class MigrationCallback {

    private final PrintWriter preWriter;
    private final PrintWriter postWriter;

    public MigrationCallback(String prePath, String postPath) throws IOException {
        this.preWriter = new PrintWriter(new FileWriter(prePath, true));
        this.postWriter = new PrintWriter(new FileWriter(postPath, true));

        preWriter.println("episode_id,step_id,module_id,app_id,trigger_type," +
                          "source_node_id,source_cpu_util,source_latency_ms,source_energy_w," +
                          "target_node_id,target_cpu_util,target_latency_ms,target_energy_w," +
                          "module_state_size_mb,device_id,sim_time_ms");

        postWriter.println("episode_id,step_id,module_id,migration_chosen," +
                           "actual_cost_j,actual_downtime_ms,bytes_transferred," +
                           "latency_post_ms,energy_rate_post_w,sla_met,sim_time_ms");
    }

    /**
     * Record pre-migration state. Call when migration is being evaluated.
     */
    public void recordPre(AppModule module, FogDevice source, FogDevice target,
                          int stepId, int episodeId, String triggerType) {

        double srcMips = source.getHost().getTotalMips();
        double srcUsed = srcMips - source.getHost().getAvailableMips();
        double tgtMips = target.getHost().getTotalMips();
        double tgtUsed = tgtMips - target.getHost().getAvailableMips();

        preWriter.printf("%d,%d,%s,%s,%s,%s,%.4f,%.3f,%.2f,%s,%.4f,%.3f,%.2f,%.2f,%s,%.3f%n",
                episodeId,
                stepId,
                module.getName(),
                module.getAppId(),
                triggerType,
                source.getName(),
                srcMips > 0 ? srcUsed / srcMips : 0.0,
                0.0,  // source latency - derived in Python from latency matrix
                source.getHost().getPowerModel().getPower(srcMips > 0 ? srcUsed / srcMips : 0.0),
                target.getName(),
                tgtMips > 0 ? tgtUsed / tgtMips : 0.0,
                0.0,  // target latency - derived in Python
                target.getHost().getPowerModel().getPower(tgtMips > 0 ? tgtUsed / tgtMips : 0.0),
                module.getSize() / (1024.0 * 1024.0),  // bytes to MB
                "dev-unknown",  // fill from MobilityModel if available
                org.cloudbus.cloudsim.core.CloudSim.clock()
        );
    }

    /**
     * Record post-migration outcome. Call after migration completes.
     */
    public void recordPost(AppModule module, FogDevice target, int stepId, int episodeId,
                           double actualCostJ, double actualDowntimeMs, long bytesTransferred,
                           double latencyPostMs, double slaThresholdMs) {

        double tgtMips = target.getHost().getTotalMips();
        double tgtUsed = tgtMips - target.getHost().getAvailableMips();
        double energyRatePost = target.getHost().getPowerModel().getPower(tgtMips > 0 ? tgtUsed / tgtMips : 0.0);
        int slaMet = latencyPostMs <= slaThresholdMs ? 1 : 0;

        postWriter.printf("%d,%d,%s,%s,%.6f,%.3f,%d,%.3f,%.4f,%d,%.3f%n",
                episodeId,
                stepId,
                module.getName(),
                target.getName(),
                actualCostJ,
                actualDowntimeMs,
                bytesTransferred,
                latencyPostMs,
                energyRatePost,
                slaMet,
                org.cloudbus.cloudsim.core.CloudSim.clock()
        );
    }

    public void close() {
        preWriter.flush();
        preWriter.close();
        postWriter.flush();
        postWriter.close();
    }
}
