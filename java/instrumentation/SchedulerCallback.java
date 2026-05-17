package instrumentation;

import org.fog.application.AppModule;
import org.fog.entities.FogDevice;
import org.fog.entities.Tuple;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

/**
 * SchedulerCallback - intercepts FogBroker scheduling decisions in iFogSim2.
 *
 * Hook into FogBroker.submitTupleToActuators() or the module placement event
 * to capture raw scheduling data for the Python pipeline.
 *
 * Integration:
 *   Override FogBroker.processEvent() for TUPLE_ARRIVAL events:
 *
 *     SchedulerCallback callback = new SchedulerCallback("results/scheduler_log.csv");
 *     // In processEvent() after routing decision is made:
 *     callback.record(tuple, selectedDevice, stepId, episodeId, policyName);
 *
 * Fields written match scheduler_decisions.csv schema (raw + instrumented columns).
 * Derived and custom columns (teacher_label, estimated_latency_ms) are added
 * by the Python pipeline post-processing step.
 */
public class SchedulerCallback {

    private final PrintWriter writer;

    public SchedulerCallback(String outputPath) throws IOException {
        this.writer = new PrintWriter(new FileWriter(outputPath, true));
        writer.println("episode_id,step_id,task_id,app_id," +
                       "task_cpu_mi,task_deadline_ms,task_data_size_kb," +
                       "device_id,device_location_x,device_location_y," +
                       "selected_node_id,selected_node_mips,selected_node_ram_mb," +
                       "selected_node_cpu_util,selected_node_queue_depth," +
                       "nearest_fog_latency_ms,sim_time_ms,policy_used");
    }

    /**
     * Record one scheduling decision. Call immediately after FogBroker selects
     * a destination node for a tuple.
     *
     * @param tuple         The tuple being scheduled (Cloudlet wrapper)
     * @param selectedNode  The FogDevice selected as destination
     * @param stepId        Current simulation step (from clock-tick counter)
     * @param episodeId     Current episode number
     * @param policyName    Policy used (first_fit / round_robin / energy_aware)
     */
    public void record(Tuple tuple, FogDevice selectedNode,
                       int stepId, int episodeId, String policyName) {

        double mipsTotal = selectedNode.getHost().getTotalMips();
        double mipsUsed = mipsTotal - selectedNode.getHost().getAvailableMips();
        double cpuUtil = mipsTotal > 0 ? mipsUsed / mipsTotal : 0.0;
        double ramTotal = selectedNode.getHost().getRam();
        double ramUsed = ramTotal - selectedNode.getHost().getRamProvisioner().getAvailableRam();

        writer.printf("%d,%d,%s,%s,%.2f,%.2f,%.2f,%s,%.4f,%.4f,%s,%.1f,%.1f,%.4f,%d,%.3f,%.3f,%s%n",
                episodeId,
                stepId,
                tuple.getCloudletId(),
                tuple.getAppId(),
                (double) tuple.getCloudletLength(),      // MI (getLength returns long)
                tuple.getExpiryTime() > 0 ? tuple.getExpiryTime() : -1.0,
                tuple.getCloudletFileSize() / 1024.0,   // bytes to KB
                "dev-" + tuple.getSourceDeviceId(),
                0.0,  // device_location_x - filled by MobilityModel callback
                0.0,  // device_location_y - filled by MobilityModel callback
                selectedNode.getName(),
                mipsTotal,
                ramTotal,
                cpuUtil,
                0,    // queue_depth - approximated in Python from logs
                0.0,  // nearest_fog_latency_ms - derived in Python
                org.cloudbus.cloudsim.core.CloudSim.clock(),
                policyName
        );
    }

    public void close() {
        writer.flush();
        writer.close();
    }
}
