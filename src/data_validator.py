"""Dataset quality validation - checks class balance, correlations, physical constraints."""

import pandas as pd
import numpy as np
from pathlib import Path


class DataValidator:
    def __init__(self, scheduler_df: pd.DataFrame, rescheduler_df: pd.DataFrame):
        self.sched = scheduler_df
        self.resched = rescheduler_df
        self.issues = []

    def check_class_balance(self) -> dict:
        results = {}
        for name, df, col in [
            ("scheduler", self.sched, "qos_met"),
            ("rescheduler", self.resched, "migration_decision"),
        ]:
            if col not in df.columns:
                continue
            counts = df[col].value_counts(normalize=True)
            minority_frac = counts.min()
            results[name] = {"class_balance": counts.to_dict(), "minority_fraction": float(minority_frac)}
            if minority_frac < 0.10:
                self.issues.append(f"WARNING: {name} dataset severely imbalanced ({minority_frac:.1%} minority class)")
        return results

    def check_missing_values(self) -> dict:
        results = {}
        for name, df in [("scheduler", self.sched), ("rescheduler", self.resched)]:
            nulls = df.isnull().sum()
            nulls = nulls[nulls > 0]
            results[name] = nulls.to_dict() if len(nulls) > 0 else "clean"
            if len(nulls) > 0:
                self.issues.append(f"ERROR: {name} has {len(nulls)} columns with missing values")
        return results

    def check_physical_constraints(self) -> dict:
        issues = []
        if "energy_consumed_j" in self.sched.columns:
            neg_energy = (self.sched["energy_consumed_j"] < 0).sum()
            if neg_energy > 0:
                issues.append(f"energy_consumed_j: {neg_energy} negative values (physical violation)")
        if "fog_load_fraction" in self.sched.columns:
            invalid_load = ((self.sched["fog_load_fraction"] < 0) | (self.sched["fog_load_fraction"] > 1)).sum()
            if invalid_load > 0:
                issues.append(f"fog_load_fraction: {invalid_load} values outside [0,1]")
        if issues:
            self.issues.extend(issues)
        return {"physical_constraint_violations": issues or "none"}

    def check_correlation_matrix(self, threshold: float = 0.95) -> dict:
        high_corr_pairs = []
        for df in [self.sched, self.resched]:
            num_df = df.select_dtypes(include=[np.number])
            corr = num_df.corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            pairs = upper.stack()
            high = pairs[pairs > threshold]
            for (c1, c2), val in high.items():
                high_corr_pairs.append({"col1": c1, "col2": c2, "correlation": round(float(val), 3)})
        return {"high_correlation_pairs": high_corr_pairs}

    def validate(self) -> dict:
        report = {
            "class_balance": self.check_class_balance(),
            "missing_values": self.check_missing_values(),
            "physical_constraints": self.check_physical_constraints(),
            "high_correlations": self.check_correlation_matrix(),
            "scheduler_shape": self.sched.shape,
            "rescheduler_shape": self.resched.shape,
            "issues": self.issues,
            "status": "PASS" if not any("ERROR" in i for i in self.issues) else "FAIL",
        }
        return report

    def print_report(self):
        r = self.validate()
        print(f"\n{'='*50}")
        print(f"Dataset Validation Report | Status: {r['status']}")
        print(f"{'='*50}")
        print(f"Scheduler  : {r['scheduler_shape'][0]:,} rows x {r['scheduler_shape'][1]} cols")
        print(f"Rescheduler: {r['rescheduler_shape'][0]:,} rows x {r['rescheduler_shape'][1]} cols")
        cb = r["class_balance"]
        if "scheduler" in cb:
            print(f"\nClass balance (qos_met): {cb['scheduler']['class_balance']}")
        if "rescheduler" in cb:
            print(f"Class balance (migration): {cb['rescheduler']['class_balance']}")
        if r["issues"]:
            print("\nIssues found:")
            for issue in r["issues"]:
                print(f"  {issue}")
        else:
            print("\nNo issues found.")
        print(f"{'='*50}\n")
        return r
