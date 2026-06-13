"""
Flask API — VSM Optimization System
Endpoints for predictions, VSM data, and optimization recommendations.
"""

from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os, json

app = Flask(__name__)

MODEL_DIR = "models"
DATA_PATH = "data/processed_data.csv"

# ── Load Models ──────────────────────────────────────────────────────────────
def load_model(name):
    path = os.path.join(MODEL_DIR, name)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

bottleneck_bundle = load_model("bottleneck_model.pkl")
cycletime_bundle  = load_model("cycle_time_model.pkl")
delay_bundle      = load_model("delay_model.pkl")
defect_bundle     = load_model("defect_model.pkl")
cluster_bundle    = load_model("clustering_model.pkl")
encoders          = load_model("encoders.pkl")

df_global = pd.read_csv(DATA_PATH) if os.path.exists(DATA_PATH) else None

# ── Helpers ──────────────────────────────────────────────────────────────────
def safe_predict(bundle, features, input_dict):
    """Generic predict helper."""
    try:
        feat_vals = [input_dict.get(f, 0) for f in features]
        X = np.array(feat_vals).reshape(1, -1)
        X_s = bundle["scaler"].transform(X)
        return bundle["model"].predict(X_s)[0]
    except Exception as e:
        return {"error": str(e)}


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "models_loaded": {
        "bottleneck": bottleneck_bundle is not None,
        "cycle_time": cycletime_bundle is not None,
        "delay": delay_bundle is not None,
        "defect": defect_bundle is not None,
        "clustering": cluster_bundle is not None,
    }})


@app.route("/api/vsm/summary", methods=["GET"])
def vsm_summary():
    """Return aggregated VSM metrics per process."""
    if df_global is None:
        return jsonify({"error": "Dataset not loaded"}), 500

    summary = df_global.groupby("Process_Name").agg(
        avg_cycle_time    = ("Cycle_Time_min", "mean"),
        avg_waiting_time  = ("Waiting_Time_min", "mean"),
        avg_lead_time     = ("Lead_Time_min", "mean"),
        avg_utilization   = ("Machine_Utilization", "mean"),
        avg_defect_rate   = ("Defect_Rate", "mean"),
        avg_inventory     = ("Inventory_Level", "mean"),
        avg_output        = ("Production_Output_units_hr", "mean"),
        bottleneck_pct    = ("Is_Bottleneck", "mean"),
    ).reset_index().round(3)

    return jsonify(summary.to_dict(orient="records"))


@app.route("/api/vsm/bottlenecks", methods=["GET"])
def get_bottlenecks():
    """Return top bottleneck processes sorted by composite score."""
    if df_global is None:
        return jsonify({"error": "Dataset not loaded"}), 500

    agg = df_global.groupby("Process_Name").agg(
        avg_waiting   = ("Waiting_Time_min", "mean"),
        avg_util      = ("Machine_Utilization", "mean"),
        avg_defect    = ("Defect_Rate", "mean"),
        bottleneck_rt = ("Is_Bottleneck", "mean"),
    ).reset_index()

    # Composite bottleneck score (normalized)
    agg["score"] = (
        (agg["avg_waiting"] / agg["avg_waiting"].max()) * 0.4 +
        (agg["avg_util"]    / agg["avg_util"].max())    * 0.3 +
        (agg["avg_defect"]  / agg["avg_defect"].max())  * 0.15 +
        agg["bottleneck_rt"] * 0.15
    )
    agg = agg.sort_values("score", ascending=False).round(4)
    return jsonify(agg.to_dict(orient="records"))


@app.route("/api/predict/bottleneck", methods=["POST"])
def predict_bottleneck():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body"}), 400
    prediction = safe_predict(bottleneck_bundle, bottleneck_bundle["features"], data)
    label = "BOTTLENECK" if prediction == 1 else "NORMAL"
    return jsonify({"prediction": int(prediction), "label": label})


@app.route("/api/predict/cycle_time", methods=["POST"])
def predict_cycle_time():
    data = request.get_json()
    prediction = safe_predict(cycletime_bundle, cycletime_bundle["features"], data)
    return jsonify({"predicted_cycle_time_min": round(float(prediction), 2)})


@app.route("/api/predict/delay", methods=["POST"])
def predict_delay():
    data = request.get_json()
    prediction = safe_predict(delay_bundle, delay_bundle["features"], data)
    label = "HIGH_DELAY" if prediction == 1 else "NORMAL"
    return jsonify({"prediction": int(prediction), "label": label,
                    "delay_threshold_min": round(delay_bundle["threshold"], 2)})


@app.route("/api/predict/defect", methods=["POST"])
def predict_defect():
    data = request.get_json()
    prediction = safe_predict(defect_bundle, defect_bundle["features"], data)
    return jsonify({"predicted_defect_rate": round(float(prediction), 4)})


@app.route("/api/optimize/recommendations", methods=["GET"])
def optimization_recommendations():
    """Rule-based + ML-driven optimization recommendations."""
    if df_global is None:
        return jsonify({"error": "Dataset not loaded"}), 500

    agg = df_global.groupby("Process_Name").agg(
        avg_waiting  = ("Waiting_Time_min", "mean"),
        avg_util     = ("Machine_Utilization", "mean"),
        avg_defect   = ("Defect_Rate", "mean"),
        avg_inventory= ("Inventory_Level", "mean"),
        avg_cycle    = ("Cycle_Time_min", "mean"),
    ).reset_index()

    recommendations = []
    for _, row in agg.iterrows():
        recs = []
        priority = "LOW"

        if row["avg_waiting"] > 15:
            recs.append(f"Reduce waiting time ({row['avg_waiting']:.1f} min) — consider parallel workstations or WIP limits.")
            priority = "HIGH"
        if row["avg_util"] > 0.88:
            recs.append(f"Machine over-utilized ({row['avg_util']*100:.1f}%) — evaluate capacity expansion or load balancing.")
            priority = "HIGH"
        if row["avg_util"] < 0.45:
            recs.append(f"Machine under-utilized ({row['avg_util']*100:.1f}%) — consider consolidating with another process.")
        if row["avg_defect"] > 0.08:
            recs.append(f"High defect rate ({row['avg_defect']*100:.1f}%) — implement error-proofing (Poka-Yoke) and SPC.")
            priority = "HIGH" if priority != "HIGH" else priority
        if row["avg_inventory"] > 60:
            recs.append(f"Excessive inventory ({row['avg_inventory']:.0f} units) — introduce pull-based Kanban scheduling.")

        if not recs:
            recs.append("Process within acceptable parameters. Monitor for drift.")

        recommendations.append({
            "process": row["Process_Name"],
            "priority": priority,
            "recommendations": recs,
            "metrics": {
                "avg_waiting_min": round(row["avg_waiting"], 2),
                "avg_utilization": round(row["avg_util"], 3),
                "avg_defect_rate": round(row["avg_defect"], 4),
                "avg_inventory":   round(row["avg_inventory"], 1),
                "avg_cycle_min":   round(row["avg_cycle"], 2),
            }
        })

    # Sort HIGH priority first
    recommendations.sort(key=lambda x: 0 if x["priority"] == "HIGH" else 1)
    return jsonify(recommendations)


@app.route("/api/metrics/efficiency", methods=["GET"])
def efficiency_metrics():
    """Overall production efficiency KPIs."""
    if df_global is None:
        return jsonify({"error": "Dataset not loaded"}), 500

    kpis = {
        "overall_avg_efficiency_pct":   round(df_global["Efficiency_Score"].mean(), 2),
        "avg_lead_time_min":            round(df_global["Lead_Time_min"].mean(), 2),
        "avg_cycle_time_min":           round(df_global["Cycle_Time_min"].mean(), 2),
        "avg_waiting_time_min":         round(df_global["Waiting_Time_min"].mean(), 2),
        "avg_machine_utilization":      round(df_global["Machine_Utilization"].mean(), 4),
        "avg_defect_rate":              round(df_global["Defect_Rate"].mean(), 4),
        "bottleneck_process_count":     int(df_global.groupby("Process_Name")["Is_Bottleneck"].mean().gt(0.5).sum()),
        "total_records_analyzed":       len(df_global),
    }
    return jsonify(kpis)


if __name__ == "__main__":
    print("Starting VSM API on http://0.0.0.0:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
