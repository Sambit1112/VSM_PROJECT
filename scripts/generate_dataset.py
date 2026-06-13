"""
VSM Manufacturing Dataset Generator
Generates realistic synthetic manufacturing process data for ML training.
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)
N = 2000  # number of records

MACHINES = [f"M{i:02d}" for i in range(1, 11)]
PROCESSES = [
    "Raw Material Intake", "Cutting", "Forming", "Welding",
    "Surface Treatment", "Assembly", "Quality Inspection",
    "Packaging", "Storage", "Dispatch"
]

def generate_dataset(n=N):
    records = []
    for i in range(1, n + 1):
        process_idx = np.random.randint(0, len(PROCESSES))
        process_name = PROCESSES[process_idx]
        machine_id = np.random.choice(MACHINES)

        # Base cycle time varies by process (minutes)
        base_cycle = [5, 12, 15, 20, 18, 25, 10, 8, 3, 5][process_idx]
        cycle_time = max(1.0, np.random.normal(base_cycle, base_cycle * 0.2))

        # Setup time (minutes)
        setup_time = max(0.5, np.random.exponential(5))

        # Waiting time — higher for bottleneck processes (Welding, Assembly)
        bottleneck = process_name in ["Welding", "Assembly", "Quality Inspection"]
        waiting_time = max(0, np.random.exponential(20 if bottleneck else 5))

        # Defect rate (0–1)
        base_defect = 0.08 if bottleneck else 0.03
        defect_rate = min(1.0, max(0.0, np.random.beta(2, 25) + base_defect))

        # Inventory level (units)
        inventory = int(np.random.exponential(50 if bottleneck else 20))

        # Machine utilization (0–1)
        utilization = min(1.0, max(0.1, np.random.beta(7 if bottleneck else 4, 3)))

        # Production output (units/hour)
        theoretical_max = 60 / cycle_time
        production_output = theoretical_max * utilization * (1 - defect_rate)
        production_output = max(0.1, production_output + np.random.normal(0, 0.5))

        # Lead time = cycle + setup + waiting
        lead_time = cycle_time + setup_time + waiting_time + np.random.normal(0, 1)
        lead_time = max(cycle_time, lead_time)

        # Derived features
        efficiency_score = (production_output / theoretical_max) * 100 if theoretical_max > 0 else 0
        is_bottleneck = int(waiting_time > 15 or utilization > 0.88 or defect_rate > 0.10)

        # Shift
        shift = np.random.choice(["Morning", "Afternoon", "Night"], p=[0.4, 0.35, 0.25])

        # Day of week (1=Monday)
        day_of_week = np.random.randint(1, 6)

        records.append({
            "Record_ID": i,
            "Process_ID": f"P{process_idx + 1:02d}",
            "Process_Name": process_name,
            "Machine_ID": machine_id,
            "Cycle_Time_min": round(cycle_time, 2),
            "Setup_Time_min": round(setup_time, 2),
            "Waiting_Time_min": round(waiting_time, 2),
            "Defect_Rate": round(defect_rate, 4),
            "Inventory_Level": inventory,
            "Machine_Utilization": round(utilization, 4),
            "Production_Output_units_hr": round(production_output, 2),
            "Lead_Time_min": round(lead_time, 2),
            "Efficiency_Score": round(efficiency_score, 2),
            "Is_Bottleneck": is_bottleneck,
            "Shift": shift,
            "Day_Of_Week": day_of_week,
        })

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    df = generate_dataset()
    out_path = os.path.join(os.path.dirname(__file__), "../data/manufacturing_data.csv")
    df.to_csv(out_path, index=False)
    print(f"Dataset saved: {out_path}")
    print(df.describe())
    print("\nBottleneck distribution:")
    print(df["Is_Bottleneck"].value_counts())
