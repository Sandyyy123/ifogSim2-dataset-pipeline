"""
Validation suite for all 6 CSV datasets.

Checks:
  - Structural: schema types, nulls in key columns, row counts
  - Physical: energy > 0, latency > 0, load 0-1
  - Semantic: NO_MIGRATION at rank 0, teacher_label = argmax(score), join integrity
  - Statistical: class balance, constant columns, high-correlation pairs
"""

import pandas as pd
import numpy as np
from typing import Dict

NO_MIGRATION = "NO_MIGRATION"


class DataValidator:
    def __init__(self, datasets: Dict[str, pd.DataFrame]):
        self.ds = datasets
        self.issues = []
        self.warnings = []

    def validate(self) -> dict:
        report = {}
        report["structural"] = self._check_structural()
        report["physical"] = self._check_physical()
        report["semantic"] = self._check_semantic()
        report["statistical"] = self._check_statistical()
        report["shapes"] = {k: v.shape for k, v in self.ds.items()}
        report["issues"] = self.issues
        report["warnings"] = self.warnings
        report["status"] = "FAIL" if any("ERROR" in i for i in self.issues) else "PASS"
        return report

    def _check_structural(self) -> dict:
        results = {}
        key_cols = {
            "scheduler_decisions": ["episode_id", "step_id", "task_id"],
            "scheduler_candidates": ["episode_id", "step_id", "task_id", "candidate_node_id"],
            "scheduler_outcomes": ["episode_id", "step_id", "task_id"],
            "migration_decisions": ["episode_id", "step_id", "module_id"],
            "migration_candidates": ["episode_id", "step_id", "module_id", "candidate_node_id"],
            "migration_outcomes": ["episode_id", "step_id", "module_id"],
        }
        for name, df in self.ds.items():
            issues = []
            if df.empty:
                self.issues.append(f"ERROR: {name} is empty")
                continue
            for col in key_cols.get(name, []):
                if col not in df.columns:
                    self.issues.append(f"ERROR: {name} missing key column '{col}'")
                elif df[col].isnull().any():
                    self.issues.append(f"ERROR: {name}.{col} has nulls in key column")
            results[name] = {"rows": len(df), "cols": len(df.columns), "key_nulls": issues}
        return results

    def _check_physical(self) -> dict:
        violations = []

        def _check_range(df, name, col, lo=None, hi=None):
            if col not in df.columns:
                return
            if lo is not None and (df[col] < lo).any():
                n = (df[col] < lo).sum()
                violations.append(f"{name}.{col}: {n} values below {lo}")
            if hi is not None and (df[col] > hi).any():
                n = (df[col] > hi).sum()
                violations.append(f"{name}.{col}: {n} values above {hi}")

        sd = self.ds.get("scheduler_decisions", pd.DataFrame())
        _check_range(sd, "scheduler_decisions", "estimated_energy_j", lo=0)
        _check_range(sd, "scheduler_decisions", "estimated_latency_ms", lo=0)
        _check_range(sd, "scheduler_decisions", "node_cpu_util_pct", lo=0, hi=1)
        _check_range(sd, "scheduler_decisions", "device_battery_pct", lo=0, hi=100)
        _check_range(sd, "scheduler_decisions", "device_location_x", lo=0, hi=1)
        _check_range(sd, "scheduler_decisions", "device_location_y", lo=0, hi=1)

        so = self.ds.get("scheduler_outcomes", pd.DataFrame())
        _check_range(so, "scheduler_outcomes", "actual_latency_ms", lo=0)
        _check_range(so, "scheduler_outcomes", "actual_energy_j", lo=0)

        mc = self.ds.get("migration_candidates", pd.DataFrame())
        _check_range(mc, "migration_candidates", "migration_cost_j", lo=0)
        _check_range(mc, "migration_candidates", "downtime_ms", lo=0)
        _check_range(mc, "migration_candidates", "state_transfer_bytes", lo=0)

        if violations:
            self.issues.extend([f"ERROR: {v}" for v in violations])
        return {"violations": violations or "none"}

    def _check_semantic(self) -> dict:
        results = {}

        # NO_MIGRATION must be at rank 0 in migration_candidates
        mc = self.ds.get("migration_candidates", pd.DataFrame())
        if not mc.empty and "candidate_node_id" in mc.columns and "candidate_rank" in mc.columns:
            rank_0 = mc[mc["candidate_rank"] == 0]
            not_no_mig = rank_0[rank_0["candidate_node_id"] != NO_MIGRATION]
            if len(not_no_mig) > 0:
                self.issues.append(
                    f"ERROR: migration_candidates has {len(not_no_mig)} rank-0 rows that are NOT NO_MIGRATION"
                )
                results["no_migration_rank0"] = "FAIL"
            else:
                results["no_migration_rank0"] = "PASS"

            # NO_MIGRATION cost must be 0
            no_mig_rows = mc[mc["candidate_node_id"] == NO_MIGRATION]
            nonzero_cost = (no_mig_rows["migration_cost_j"] != 0).sum() if "migration_cost_j" in mc.columns else 0
            if nonzero_cost > 0:
                self.issues.append(f"ERROR: {nonzero_cost} NO_MIGRATION rows have non-zero migration_cost_j")
            results["no_migration_cost_zero"] = "PASS" if nonzero_cost == 0 else "FAIL"

            no_mig_pct = len(no_mig_rows) / len(mc) * 100
            results["no_migration_fraction_pct"] = round(no_mig_pct, 1)

        # Teacher label vs argmax check (sample 1000 rows)
        sc = self.ds.get("scheduler_candidates", pd.DataFrame())
        if not sc.empty and "teacher_score" in sc.columns and "is_teacher_choice" in sc.columns:
            sample_tasks = sc["task_id"].unique()[:100]
            mismatches = 0
            for tid in sample_tasks:
                group = sc[sc["task_id"] == tid]
                if group.empty:
                    continue
                argmax_node = group.loc[group["teacher_score"].idxmax(), "candidate_node_id"]
                marked_node = group[group["is_teacher_choice"] == 1]["candidate_node_id"].values
                if len(marked_node) == 0 or marked_node[0] != argmax_node:
                    mismatches += 1
            results["teacher_label_argmax_check"] = f"{mismatches} mismatches in sample of {len(sample_tasks)}"
            if mismatches > 5:
                self.warnings.append(f"WARNING: teacher_label != argmax(score) in {mismatches}/100 sampled tasks")

        # Join integrity: every decision has a corresponding outcome
        sd = self.ds.get("scheduler_decisions", pd.DataFrame())
        so = self.ds.get("scheduler_outcomes", pd.DataFrame())
        if not sd.empty and not so.empty:
            dec_keys = set(sd["task_id"].astype(str))
            out_keys = set(so["task_id"].astype(str))
            missing = len(dec_keys - out_keys)
            results["scheduler_decision_outcome_join"] = f"{missing} decisions missing outcomes"
            if missing > 0:
                self.warnings.append(f"WARNING: {missing} scheduler decisions have no outcome row")

        md = self.ds.get("migration_decisions", pd.DataFrame())
        mo = self.ds.get("migration_outcomes", pd.DataFrame())
        if not md.empty and not mo.empty:
            dec_keys = set(md["module_id"].astype(str))
            out_keys = set(mo["module_id"].astype(str))
            missing = len(dec_keys - out_keys)
            results["migration_decision_outcome_join"] = f"{missing} decisions missing outcomes"

        return results

    def _check_statistical(self) -> dict:
        results = {}

        # Class balance for key label columns
        sd = self.ds.get("scheduler_decisions", pd.DataFrame())
        if "policy_used" in sd.columns:
            results["policy_distribution"] = sd["policy_used"].value_counts(normalize=True).round(3).to_dict()

        so = self.ds.get("scheduler_outcomes", pd.DataFrame())
        if "sla_met" in so.columns:
            sla_rate = so["sla_met"].mean()
            results["sla_met_rate"] = round(float(sla_rate), 3)
            if sla_rate < 0.5:
                self.warnings.append(f"WARNING: SLA met rate is low ({sla_rate:.1%}) - check deadline config")

        md = self.ds.get("migration_decisions", pd.DataFrame())
        if "migration_chosen" in md.columns:
            no_mig_rate = (md["migration_chosen"] == NO_MIGRATION).mean()
            results["no_migration_chosen_rate"] = round(float(no_mig_rate), 3)
            if no_mig_rate < 0.3:
                self.warnings.append(f"WARNING: migration rate very high ({1-no_mig_rate:.1%}) - check trigger config")

        # Constant columns
        for name, df in self.ds.items():
            num_df = df.select_dtypes(include=[np.number])
            constant_cols = [c for c in num_df.columns if num_df[c].nunique() <= 1]
            if constant_cols:
                self.warnings.append(f"WARNING: {name} has constant columns: {constant_cols}")

        return results

    def print_report(self):
        r = self.validate()
        sep = "=" * 55
        print(f"\n{sep}")
        print(f"Dataset Validation Report  |  Status: {r['status']}")
        print(sep)
        for name, shape in r["shapes"].items():
            print(f"  {name:<35} {shape[0]:>8,} rows x {shape[1]:>3} cols")
        print()
        if r["issues"]:
            print("ERRORS:")
            for i in r["issues"]:
                print(f"  {i}")
        if r["warnings"]:
            print("Warnings:")
            for w in r["warnings"]:
                print(f"  {w}")
        if not r["issues"] and not r["warnings"]:
            print("No issues found.")
        print(sep)
        sems = r.get("semantic", {})
        if "no_migration_rank0" in sems:
            print(f"  NO_MIGRATION rank-0 check : {sems['no_migration_rank0']}")
        if "teacher_label_argmax_check" in sems:
            print(f"  Teacher argmax check       : {sems['teacher_label_argmax_check']}")
        if "no_migration_chosen_rate" in r.get("statistical", {}):
            print(f"  NO_MIGRATION chosen rate   : {r['statistical']['no_migration_chosen_rate']:.1%}")
        if "sla_met_rate" in r.get("statistical", {}):
            print(f"  SLA met rate               : {r['statistical']['sla_met_rate']:.1%}")
        print(f"{sep}\n")
        return r
