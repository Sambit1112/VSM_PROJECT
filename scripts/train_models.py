"""
ML Training Pipeline — VSM Optimization System
Trains: Bottleneck Detector, Cycle Time Predictor, Delay Predictor,
        Defect Predictor, Process Clustering
"""

import pandas as pd
import numpy as np
import pickle, os, json
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, mean_squared_error, mean_absolute_error, r2_score)
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "data/manufacturing_data.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────
# 1. LOAD & PREPROCESS
# ─────────────────────────────────────────
def load_and_preprocess():
    df = pd.read_csv(DATA_PATH)

    # Encode categorical columns
    le_process = LabelEncoder()
    le_machine = LabelEncoder()
    le_shift   = LabelEncoder()
    df["Process_Enc"] = le_process.fit_transform(df["Process_Name"])
    df["Machine_Enc"] = le_machine.fit_transform(df["Machine_ID"])
    df["Shift_Enc"]   = le_shift.fit_transform(df["Shift"])

    encoders = {"process": le_process, "machine": le_machine, "shift": le_shift}
    with open(f"{MODEL_DIR}/encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)

    # Drop missing values (none expected but good practice)
    df.dropna(inplace=True)
    return df

# ─────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────
def engineer_features(df):
    # Interaction features
    df["Util_x_CycleTime"] = df["Machine_Utilization"] * df["Cycle_Time_min"]
    df["Wait_to_Cycle_Ratio"] = df["Waiting_Time_min"] / (df["Cycle_Time_min"] + 1e-5)
    df["Total_Process_Time"] = df["Cycle_Time_min"] + df["Setup_Time_min"] + df["Waiting_Time_min"]
    df["Defect_x_Output"] = df["Defect_Rate"] * df["Production_Output_units_hr"]

    # Binary flag: high inventory
    df["High_Inventory"] = (df["Inventory_Level"] > df["Inventory_Level"].quantile(0.75)).astype(int)

    # Log-transform skewed features
    df["Log_Lead_Time"] = np.log1p(df["Lead_Time_min"])
    df["Log_Waiting"] = np.log1p(df["Waiting_Time_min"])
    return df

BASE_FEATURES = [
    "Cycle_Time_min", "Setup_Time_min", "Waiting_Time_min",
    "Defect_Rate", "Inventory_Level", "Machine_Utilization",
    "Production_Output_units_hr", "Process_Enc", "Machine_Enc",
    "Shift_Enc", "Day_Of_Week",
    "Util_x_CycleTime", "Wait_to_Cycle_Ratio",
    "Total_Process_Time", "Defect_x_Output", "High_Inventory"
]

# ─────────────────────────────────────────
# 3. EVALUATION HELPER
# ─────────────────────────────────────────
def eval_classifier(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y_test, y_pred, zero_division=0), 4),
    }

def eval_regressor(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return {
        "Model": name,
        "RMSE": round(rmse, 4),
        "MAE": round(mean_absolute_error(y_test, y_pred), 4),
        "R2": round(r2_score(y_test, y_pred), 4),
    }

# ─────────────────────────────────────────
# 4. TASK A — BOTTLENECK DETECTION (Classification)
# ─────────────────────────────────────────
def train_bottleneck_detector(df):
    print("\n── Bottleneck Detection ─────────────────")
    X = df[BASE_FEATURES]
    y = df["Is_Bottleneck"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42),
    }

    results = []
    best_score, best_model, best_name = 0, None, ""
    for name, m in models.items():
        m.fit(X_train_s, y_train)
        r = eval_classifier(name, m, X_test_s, y_test)
        results.append(r)
        print(f"  {name:25s}  Acc={r['Accuracy']}  F1={r['F1']}")
        if r["F1"] > best_score:
            best_score, best_model, best_name = r["F1"], m, name

    print(f"  → Best: {best_name} (F1={best_score})")
    bundle = {"model": best_model, "scaler": scaler, "features": BASE_FEATURES}
    with open(f"{MODEL_DIR}/bottleneck_model.pkl", "wb") as f:
        pickle.dump(bundle, f)
    return pd.DataFrame(results)

# ─────────────────────────────────────────
# 5. TASK B — CYCLE TIME PREDICTION (Regression)
# ─────────────────────────────────────────
def train_cycle_time_predictor(df):
    print("\n── Cycle Time Prediction ────────────────")
    feat = [f for f in BASE_FEATURES if f != "Cycle_Time_min"]
    X = df[feat]
    y = df["Cycle_Time_min"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression":  Ridge(alpha=1.0),
        "Decision Tree":     DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random Forest":     RandomForestRegressor(n_estimators=150, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier.__mro__[0].__subclasses__()[0]
                             if False else __import__('sklearn.ensemble', fromlist=['GradientBoostingRegressor']).GradientBoostingRegressor(n_estimators=150, random_state=42),
    }
    # Simpler model dict
    from sklearn.ensemble import GradientBoostingRegressor
    models["Gradient Boosting"] = GradientBoostingRegressor(n_estimators=150, random_state=42)

    results = []
    best_rmse, best_model, best_name = 1e9, None, ""
    for name, m in models.items():
        m.fit(X_train_s, y_train)
        r = eval_regressor(name, m, X_test_s, y_test)
        results.append(r)
        print(f"  {name:25s}  RMSE={r['RMSE']}  MAE={r['MAE']}  R2={r['R2']}")
        if r["RMSE"] < best_rmse:
            best_rmse, best_model, best_name = r["RMSE"], m, name

    print(f"  → Best: {best_name} (RMSE={best_rmse})")
    bundle = {"model": best_model, "scaler": scaler, "features": feat}
    with open(f"{MODEL_DIR}/cycle_time_model.pkl", "wb") as f:
        pickle.dump(bundle, f)
    return pd.DataFrame(results)

# ─────────────────────────────────────────
# 6. TASK C — DELAY PREDICTION (Classification: High Waiting Time)
# ─────────────────────────────────────────
def train_delay_predictor(df):
    print("\n── Production Delay Prediction ──────────")
    threshold = df["Waiting_Time_min"].quantile(0.75)
    df["High_Delay"] = (df["Waiting_Time_min"] > threshold).astype(int)

    feat = [f for f in BASE_FEATURES if "Wait" not in f]
    X = df[feat]
    y = df["High_Delay"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    clf.fit(X_train_s, y_train)
    r = eval_classifier("Random Forest", clf, X_test_s, y_test)
    print(f"  Random Forest  Acc={r['Accuracy']}  F1={r['F1']}")

    bundle = {"model": clf, "scaler": scaler, "features": feat, "threshold": threshold}
    with open(f"{MODEL_DIR}/delay_model.pkl", "wb") as f:
        pickle.dump(bundle, f)
    return r

# ─────────────────────────────────────────
# 7. TASK D — DEFECT PREDICTION (Regression)
# ─────────────────────────────────────────
def train_defect_predictor(df):
    print("\n── Defect Rate Prediction ───────────────")
    feat = [f for f in BASE_FEATURES if "Defect" not in f]
    X = df[feat]
    y = df["Defect_Rate"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    from sklearn.ensemble import GradientBoostingRegressor
    gbr = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
    gbr.fit(X_train_s, y_train)
    r = eval_regressor("Gradient Boosting", gbr, X_test_s, y_test)
    print(f"  Gradient Boosting  RMSE={r['RMSE']}  R2={r['R2']}")

    bundle = {"model": gbr, "scaler": scaler, "features": feat}
    with open(f"{MODEL_DIR}/defect_model.pkl", "wb") as f:
        pickle.dump(bundle, f)
    return r

# ─────────────────────────────────────────
# 8. TASK E — PROCESS CLUSTERING (K-Means)
# ─────────────────────────────────────────
def train_clustering(df):
    print("\n── Process Clustering (K-Means) ─────────")
    cluster_feat = ["Cycle_Time_min", "Waiting_Time_min", "Machine_Utilization",
                    "Defect_Rate", "Inventory_Level", "Production_Output_units_hr"]
    X = df[cluster_feat]
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    # Elbow check k=2..8
    inertias = {}
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_s)
        inertias[k] = round(km.inertia_, 2)

    # Choose k=4 (elbow heuristic for this domain)
    km_best = KMeans(n_clusters=4, random_state=42, n_init=10)
    km_best.fit(X_s)
    df["Cluster"] = km_best.labels_

    cluster_summary = df.groupby("Cluster")[cluster_feat].mean().round(3)
    print(cluster_summary.to_string())

    bundle = {"model": km_best, "scaler": scaler, "features": cluster_feat, "inertias": inertias}
    with open(f"{MODEL_DIR}/clustering_model.pkl", "wb") as f:
        pickle.dump(bundle, f)

    with open(f"{MODEL_DIR}/inertias.json", "w") as f:
        json.dump(inertias, f)
    return cluster_summary

# ─────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    df = load_and_preprocess()
    df = engineer_features(df)
    df.to_csv("data/processed_data.csv", index=False)
    print(f"Processed dataset: {df.shape}")

    r_bottleneck = train_bottleneck_detector(df)
    r_cycletime  = train_cycle_time_predictor(df)
    r_delay      = train_delay_predictor(df)
    r_defect     = train_defect_predictor(df)
    r_cluster    = train_clustering(df)

    print("\n═══════════════════════════════════════")
    print("  TRAINING COMPLETE — All models saved to /models/")
    print("═══════════════════════════════════════")
