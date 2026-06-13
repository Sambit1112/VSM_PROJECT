"""
Optimization Engine — VSM Lean Manufacturing Optimizer
Implements bottleneck detection, workload balancing, and lean recommendations.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


@dataclass
class ProcessNode:
    name: str
    cycle_time: float
    setup_time: float
    waiting_time: float
    utilization: float
    defect_rate: float
    inventory: float
    output: float

    @property
    def total_time(self) -> float:
        return self.cycle_time + self.setup_time + self.waiting_time

    @property
    def value_add_ratio(self) -> float:
        """Ratio of value-adding time (cycle) to total time."""
        return self.cycle_time / (self.total_time + 1e-5)

    @property
    def process_efficiency(self) -> float:
        return self.utilization * (1 - self.defect_rate) * self.value_add_ratio


@dataclass
class VSMOptimizer:
    processes: List[ProcessNode] = field(default_factory=list)

    def load_from_dataframe(self, df: pd.DataFrame):
        agg = df.groupby("Process_Name").agg(
            cycle_time  = ("Cycle_Time_min", "mean"),
            setup_time  = ("Setup_Time_min", "mean"),
            waiting_time= ("Waiting_Time_min", "mean"),
            utilization = ("Machine_Utilization", "mean"),
            defect_rate = ("Defect_Rate", "mean"),
            inventory   = ("Inventory_Level", "mean"),
            output      = ("Production_Output_units_hr", "mean"),
        ).reset_index()

        self.processes = [
            ProcessNode(
                name=row["Process_Name"],
                cycle_time=row["cycle_time"],
                setup_time=row["setup_time"],
                waiting_time=row["waiting_time"],
                utilization=row["utilization"],
                defect_rate=row["defect_rate"],
                inventory=row["inventory"],
                output=row["output"],
            )
            for _, row in agg.iterrows()
        ]

    def detect_bottlenecks(self) -> List[Dict]:
        """
        Identify bottlenecks using a composite score.
        Theory of Constraints: bottleneck = process with lowest throughput
        relative to demand.
        """
        max_wait  = max(p.waiting_time for p in self.processes) + 1e-5
        max_util  = max(p.utilization  for p in self.processes) + 1e-5
        max_defect= max(p.defect_rate  for p in self.processes) + 1e-5
        max_inv   = max(p.inventory    for p in self.processes) + 1e-5

        bottlenecks = []
        for p in self.processes:
            score = (
                (p.waiting_time / max_wait) * 0.35 +
                (p.utilization  / max_util) * 0.30 +
                (p.defect_rate  / max_defect) * 0.20 +
                (p.inventory    / max_inv) * 0.15
            )
            is_bn = score > 0.55 or p.utilization > 0.88 or p.waiting_time > 15
            bottlenecks.append({
                "process": p.name,
                "bottleneck_score": round(score, 4),
                "is_bottleneck": bool(is_bn),
                "utilization": round(p.utilization, 3),
                "waiting_time": round(p.waiting_time, 2),
                "defect_rate": round(p.defect_rate, 4),
            })

        return sorted(bottlenecks, key=lambda x: -x["bottleneck_score"])

    def calculate_takt_time(self, daily_demand: int, available_time_min: int = 480) -> float:
        """Takt Time = Available Production Time / Customer Demand."""
        return available_time_min / daily_demand if daily_demand > 0 else 0

    def identify_waste_categories(self) -> Dict[str, List[str]]:
        """Map to 8 Lean Wastes (TIMWOODS)."""
        wastes = {
            "Transport": [], "Inventory": [], "Motion": [],
            "Waiting": [], "Overproduction": [], "Overprocessing": [],
            "Defects": [], "Skills":[],
        }
        for p in self.processes:
            if p.inventory > 50:
                wastes["Inventory"].append(f"{p.name}: {p.inventory:.0f} units")
            if p.waiting_time > 10:
                wastes["Waiting"].append(f"{p.name}: {p.waiting_time:.1f} min")
            if p.defect_rate > 0.07:
                wastes["Defects"].append(f"{p.name}: {p.defect_rate*100:.1f}%")
            if p.utilization < 0.40:
                wastes["Skills"].append(f"{p.name}: {p.utilization*100:.0f}% util — potential skill underuse")
        return {k: v for k, v in wastes.items() if v}

    def workload_balancing_plan(self, takt_time: float) -> List[Dict]:
        """
        Generate workload rebalancing plan.
        If cycle_time > takt_time → overloaded → split/accelerate.
        If cycle_time << takt_time → underloaded → merge/reduce capacity.
        """
        plans = []
        for p in self.processes:
            gap = p.cycle_time - takt_time
            if gap > 2:
                action = "SPLIT or ADD RESOURCE"
                detail = (f"Cycle time ({p.cycle_time:.1f} min) exceeds takt "
                          f"({takt_time:.1f} min) by {gap:.1f} min. "
                          f"Add parallel station or redistribute tasks.")
                urgency = "HIGH"
            elif gap < -5:
                action = "CONSOLIDATE or REDUCE CAPACITY"
                detail = (f"Cycle time ({p.cycle_time:.1f} min) significantly "
                          f"below takt ({takt_time:.1f} min). "
                          f"Merge with adjacent process or reduce machine count.")
                urgency = "MEDIUM"
            else:
                action = "BALANCED"
                detail = f"Cycle time ({p.cycle_time:.1f} min) within takt tolerance."
                urgency = "LOW"

            plans.append({
                "process": p.name,
                "action": action,
                "detail": detail,
                "urgency": urgency,
                "cycle_time_vs_takt": round(gap, 2),
            })

        return sorted(plans, key=lambda x: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}[x["urgency"]])

    def estimate_improvement_potential(self) -> Dict:
        """Estimate % improvement if all identified wastes are eliminated."""
        current_lead_time = sum(p.total_time for p in self.processes)
        current_value_add = sum(p.cycle_time  for p in self.processes)

        # Theoretical minimum (eliminate all waiting and reduce defects by 50%)
        target_waiting = sum(min(p.waiting_time, 5) for p in self.processes)
        target_cycle   = sum(p.cycle_time * (1 - p.defect_rate * 0.5) for p in self.processes)
        target_setup   = sum(p.setup_time * 0.7 for p in self.processes)  # SMED assumption
        target_lead    = target_cycle + target_setup + target_waiting

        lead_time_reduction = (current_lead_time - target_lead) / current_lead_time * 100
        value_add_ratio_now = current_value_add / current_lead_time * 100

        return {
            "current_total_lead_time_min":   round(current_lead_time, 2),
            "target_total_lead_time_min":    round(target_lead, 2),
            "lead_time_reduction_pct":       round(lead_time_reduction, 2),
            "current_value_add_ratio_pct":   round(value_add_ratio_now, 2),
            "current_total_value_add_min":   round(current_value_add, 2),
            "estimated_efficiency_gain_pct": round(lead_time_reduction * 0.8, 2),
        }

    def generate_full_report(self, daily_demand: int = 100) -> Dict:
        takt = self.calculate_takt_time(daily_demand)
        return {
            "takt_time_min":         round(takt, 3),
            "bottlenecks":           self.detect_bottlenecks(),
            "waste_categories":      self.identify_waste_categories(),
            "workload_balance":      self.workload_balancing_plan(takt),
            "improvement_potential": self.estimate_improvement_potential(),
        }


# ── Demo ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = pd.read_csv("data/processed_data.csv")
    optimizer = VSMOptimizer()
    optimizer.load_from_dataframe(df)
    report = optimizer.generate_full_report(daily_demand=120)

    print("\n=== VSM Optimization Report ===")
    print(f"\nTakt Time: {report['takt_time_min']} min")

    print("\n--- Top Bottlenecks ---")
    for b in report["bottlenecks"][:5]:
        flag = "⚠️ " if b["is_bottleneck"] else "  "
        print(f"  {flag}{b['process']:30s}  Score={b['bottleneck_score']}")

    print("\n--- Lean Waste Identified ---")
    for waste_type, items in report["waste_categories"].items():
        print(f"  {waste_type}: {', '.join(items)}")

    print("\n--- Improvement Potential ---")
    imp = report["improvement_potential"]
    print(f"  Current Lead Time : {imp['current_total_lead_time_min']} min")
    print(f"  Target Lead Time  : {imp['target_total_lead_time_min']} min")
    print(f"  Lead Time Reduction: {imp['lead_time_reduction_pct']}%")
    print(f"  Value-Add Ratio   : {imp['current_value_add_ratio_pct']}%")
