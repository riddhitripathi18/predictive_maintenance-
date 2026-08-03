import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import ttf_engine
from ttf_engine import run_ttf_forecast, DEFAULT_DEGRADATION
from machine_registry import MachineRegistry, ML_COMPATIBLE_TYPES
from heat_exchanger_analyzer import HXUnit, analyze_hx_fleet

# ─────────────────────────────────────────────
# INITIALIZATION & SETUP
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
REGISTRY = MachineRegistry(BASE_DIR)

app = FastAPI(title=" SYS-CONTROL Predictive Maintenance API")

# Enable CORS for frontend connection (typically Vite dev server on 5173 or custom)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
try:
    with open(os.path.join(MODEL_DIR, "xgb_model.pkl"), "rb") as f:
        xgb_model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "rf_model.pkl"), "rb") as f:
        rf_model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
        
    with open(os.path.join(MODEL_DIR, "xgb_threshold.pkl"), "rb") as f:
        xgb_thresh_f1 = float(pickle.load(f))
    with open(os.path.join(MODEL_DIR, "xgb_safety_threshold.pkl"), "rb") as f:
        xgb_thresh_safety = float(pickle.load(f))
    with open(os.path.join(MODEL_DIR, "rf_threshold.pkl"), "rb") as f:
        rf_thresh_f1 = float(pickle.load(f))
    with open(os.path.join(MODEL_DIR, "rf_safety_threshold.pkl"), "rb") as f:
        rf_thresh_safety = float(pickle.load(f))
        
    SUB_TARGETS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    sub_models = {}
    for sub in SUB_TARGETS:
        with open(os.path.join(MODEL_DIR, f"xgb_{sub.lower()}_model.pkl"), "rb") as f:
            sub_models[sub] = pickle.load(f)
            
    with open(os.path.join(MODEL_DIR, "feature_cols.pkl"), "rb") as f:
        FEATURE_COLS = pickle.load(f)
        
    with open(os.path.join(MODEL_DIR, "fleet_medians.pkl"), "rb") as f:
        fleet_medians = pickle.load(f)
    models_loaded = True
except Exception as e:
    print(f"Error loading models: {e}")
    models_loaded = False
    xgb_thresh_f1, xgb_thresh_safety, rf_thresh_f1, rf_thresh_safety = 0.3, 0.1, 0.3, 0.1
    FEATURE_COLS = []
    fleet_medians = {}

# Helper to engineer features
def engineer_features(air_temp: float, proc_temp: float, rpm: float, torque: float, tool_wear: float, type_enc: int):
    temp_delta = proc_temp - air_temp
    power_W = torque * rpm * (2 * np.pi / 60)
    torque_x_wear = torque * tool_wear
    wear_pct = tool_wear / 253.0
    high_torque = 1 if torque > 55 else 0
    high_wear = 1 if tool_wear > 200 else 0
    return [type_enc, air_temp, proc_temp, rpm, torque, tool_wear,
            temp_delta, power_W, torque_x_wear, wear_pct, high_torque, high_wear]

# Helper to extract anomalies
def get_anomalies(air_temp: float, proc_temp: float, rpm: float, torque: float, tool_wear: float):
    temp_delta = proc_temp - air_temp
    power_W = torque * rpm * (2 * np.pi / 60)
    
    current_vals = {
        "Air temperature [K]": air_temp,
        "Process temperature [K]": proc_temp,
        "Rotational speed [rpm]": rpm,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
        "temp_delta": temp_delta,
        "power_W": power_W
    }
    
    feat_display_names = {
        "Air temperature [K]": "Ambient Temp",
        "Process temperature [K]": "Process Temp",
        "Rotational speed [rpm]": "Spindle RPM",
        "Torque [Nm]": "Torque Load",
        "Tool wear [min]": "Cumulative Wear",
        "temp_delta": "Thermal Delta",
        "power_W": "Rotary Power"
    }
    
    deviations = []
    for f_name, curr_val in current_vals.items():
        median_val = fleet_medians.get(f_name, 1.0)
        pct_dev = ((curr_val - median_val) / median_val) * 100
        deviations.append({
            "parameter": feat_display_names[f_name],
            "current": round(curr_val, 2),
            "median": round(median_val, 2),
            "deviation": round(pct_dev, 2)
        })
    return deviations

# Helper to build AI Operator Guidelines
def get_guidelines(air_temp: float, proc_temp: float, rpm: float, torque: float, tool_wear: float, prob: float):
    reasons = []
    if torque > 60:
        reasons.append("🔴 **CRITICAL TORQUE OVERLOAD:** Motor is operating at **" + str(round(torque, 1)) + " Nm**, exceeding physical safety margins (>60 Nm). High risk of mechanical shear.")
    if tool_wear > 200:
        reasons.append("⚠️ **DEGRADED COMPONENT LIFE:** Cumulative tool wear has reached **" + str(round(tool_wear, 1)) + " min**. Micro-fractures and thermal degradation are expected.")
    if rpm < 1300:
        reasons.append("⚠️ **ROTATION SLOWDOWN:** Speed dropped to **" + str(round(rpm, 1)) + " RPM**. Combined with high load, this indicates potential motor stalling or severe friction.")
    temp_delta = proc_temp - air_temp
    if temp_delta < 8.6:
        reasons.append("🔴 **INSUFFICIENT HEAT DISSIPATION:** Temperature delta is **" + str(round(temp_delta, 1)) + " K** (Limit: >8.6 K). Cooling loop blockage suspected.")
    power = torque * rpm * (2 * np.pi / 60)
    if power > 9000 or power < 3500:
        reasons.append("⚠️ **ELECTRICAL OUT-OF-BOUNDS:** Power draws **" + str(round(power, 0)) + " W** which is outside nominal rating limits (3500W-9000W). Risk of electrical circuit breaker failure.")
        
    if not reasons:
        if prob > 0.5:
            reasons.append("🧬 **MULTIVARIATE AI FLAG:** No single sensor exceeds standard limits, but the cross-parameter pattern indicates early signs of impending failure.")
        else:
            reasons.append("💚 **NOMINAL STATE:** All diagnostic streams are within safe engineering operational medians.")
    return reasons

# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/api/config")
def get_config():
    return {
        "models_loaded": models_loaded,
        "thresholds": {
            "xgb_f1": xgb_thresh_f1,
            "xgb_safety": xgb_thresh_safety,
            "rf_f1": rf_thresh_f1,
            "rf_safety": rf_thresh_safety
        }
    }

class SimulationInput(BaseModel):
    air_temp: float
    proc_temp: float
    rpm: float
    torque: float
    tool_wear: float
    type_str: str  # "L (Low)", "M (Medium)", "H (High)"
    active_thresh: float

@app.post("/api/simulate")
def simulate_override(data: SimulationInput):
    if not models_loaded:
        raise HTTPException(status_code=500, detail="ML models not loaded")
        
    type_enc = {"L (Low)": 0, "M (Medium)": 1, "H (High)": 2}.get(data.type_str, 1)
    features = engineer_features(data.air_temp, data.proc_temp, data.rpm, data.torque, data.tool_wear, type_enc)
    X_scaled = scaler.transform(np.array([features]))
    
    prob = float(xgb_model.predict_proba(X_scaled)[0, 1])
    
    # Sub-models failures
    sub_probs = {}
    for sub in SUB_TARGETS:
        if sub in sub_models:
            sub_probs[sub] = float(sub_models[sub].predict_proba(X_scaled)[0, 1])
            
    # Deviations
    deviations = get_anomalies(data.air_temp, data.proc_temp, data.rpm, data.torque, data.tool_wear)
    # Action guidelines
    guidelines = get_guidelines(data.air_temp, data.proc_temp, data.rpm, data.torque, data.tool_wear, prob)
    
    return {
        "probability": prob,
        "sub_probs": sub_probs,
        "deviations": deviations,
        "guidelines": guidelines,
        "temp_delta": data.proc_temp - data.air_temp,
        "power_W": data.torque * data.rpm * (2 * np.pi / 60),
        "torque_x_wear": data.torque * data.tool_wear,
        "wear_pct": data.tool_wear / 253.0
    }

class ForecastInput(BaseModel):
    air_temp: float
    proc_temp: float
    rpm: float
    torque: float
    tool_wear: float
    type_str: str
    active_thresh: float
    horizon_months: int
    wear_rate: float
    rpm_drift: float
    torque_drift: float
    sc_rpm: float = 0.0
    sc_torque: float = 0.0
    sc_wear: float = 0.0

@app.post("/api/forecast")
def run_forecast(data: ForecastInput):
    if not models_loaded:
        raise HTTPException(status_code=500, detail="ML models not loaded")
        
    type_enc = {"L (Low)": 0, "M (Medium)": 1, "H (High)": 2}.get(data.type_str, 1)
    
    deg = {
        "wear_rate_per_month": data.wear_rate,
        "rpm_drift_per_month": data.rpm_drift,
        "torque_drift_per_month": data.torque_drift,
        "air_temp_drift_per_month": 0.05,
        "proc_temp_drift_per_month": 0.1,
    }
    
    result_base = run_ttf_forecast(
        xgb_model=xgb_model,
        scaler=scaler,
        air_temp=data.air_temp,
        proc_temp=data.proc_temp,
        rpm=data.rpm,
        torque=data.torque,
        tool_wear=data.tool_wear,
        type_enc=type_enc,
        active_threshold=data.active_thresh,
        horizon_months=data.horizon_months,
        degradation=deg,
        scenario_overrides={"rpm": 0.0, "torque": 0.0, "tool_wear": 0.0}
    )
    
    result_scen = None
    if data.sc_rpm != 0.0 or data.sc_torque != 0.0 or data.sc_wear != 0.0:
        result_scen = run_ttf_forecast(
            xgb_model=xgb_model,
            scaler=scaler,
            air_temp=data.air_temp,
            proc_temp=data.proc_temp,
            rpm=data.rpm,
            torque=data.torque,
            tool_wear=data.tool_wear,
            type_enc=type_enc,
            active_threshold=data.active_thresh,
            horizon_months=data.horizon_months,
            degradation=deg,
            scenario_overrides={"rpm": data.sc_rpm, "torque": data.sc_torque, "tool_wear": data.sc_wear}
        )
        
    return {
        "base": {
            "months": result_base.months,
            "probabilities": result_base.probabilities,
            "predicted_fail_month": result_base.predicted_fail_month,
            "fail_message": result_base.fail_message,
            "projected_params": result_base.projected_params,
            "interventions": result_base.interventions
        },
        "scenario": {
            "months": result_scen.months,
            "probabilities": result_scen.probabilities,
            "predicted_fail_month": result_scen.predicted_fail_month,
            "fail_message": result_scen.fail_message,
            "projected_params": result_scen.projected_params,
            "interventions": result_scen.interventions
        } if result_scen else None
    }

# ─────────────────────────────────────────────
# BATCH ANALYSIS
# ─────────────────────────────────────────────
@app.post("/api/batch")
async def batch_process(file: UploadFile = File(...), active_thresh: float = Form(0.35)):
    if not models_loaded:
        raise HTTPException(status_code=500, detail="ML models not loaded")
        
    try:
        df_raw = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {e}")
        
    # Check headers
    needed = ["Type", "Air temperature [K]", "Process temperature [K]",
              "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
    missing = [c for c in needed if c not in df_raw.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")
        
    type_enc = {"L": 0, "M": 1, "H": 2}
    df = df_raw.copy()
    df["Type_encoded"]  = df["Type"].map(type_enc).fillna(1).astype(int)
    df["temp_delta"]    = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["power_W"]       = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * (2 * np.pi / 60)
    df["torque_x_wear"] = df["Torque [Nm]"] * df["Tool wear [min]"]
    df["wear_pct"]      = df["Tool wear [min]"] / 253.0
    df["high_torque"]   = (df["Torque [Nm]"] > 55).astype(int)
    df["high_wear"]     = (df["Tool wear [min]"] > 200).astype(int)
    
    X_scaled = scaler.transform(df[FEATURE_COLS])
    proba = xgb_model.predict_proba(X_scaled)[:, 1]
    
    df["Failure Probability (%)"] = (proba * 100).round(2)
    df["Predicted Failure"]       = (proba >= active_thresh).astype(int)
    df["Risk Level"] = df["Failure Probability (%)"].apply(
        lambda p: "🔴 CRITICAL" if p > 70 else ("🟡 WARNING" if p >= 35 else "🟢 SAFE")
    )
    
    total = len(df)
    n_critical = int((df["Risk Level"] == "🔴 CRITICAL").sum())
    n_warning  = int((df["Risk Level"] == "🟡 WARNING").sum())
    n_safe     = int((df["Risk Level"] == "🟢 SAFE").sum())
    n_predicted = int(df["Predicted Failure"].sum())
    
    # Check if target is present
    eval_metrics = None
    if "Machine failure" in df.columns:
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, average_precision_score, confusion_matrix
        )
        y_true = df["Machine failure"].astype(int)
        y_pred = df["Predicted Failure"].astype(int)
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        pr_auc = average_precision_score(y_true, proba)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        eval_metrics = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "pr_auc": float(pr_auc),
            "false_positives": int(fp),
            "false_negatives": int(fn)
        }
        
    return {
        "records": df.to_dict(orient="records"),
        "stats": {
            "total": total,
            "critical": n_critical,
            "warning": n_warning,
            "safe": n_safe,
            "alarms": n_predicted
        },
        "evaluation": eval_metrics
    }

# ─────────────────────────────────────────────
# ASSETS REGISTRY ENDPOINTS
# ─────────────────────────────────────────────
@app.get("/api/assets")
def get_assets():
    return REGISTRY.get_all()

@app.post("/api/assets")
def create_asset(asset: Dict):
    asset_id = REGISTRY.add_asset(asset)
    return {"status": "success", "id": asset_id}

@app.put("/api/assets/{asset_id}")
def update_asset(asset_id: str, updates: Dict):
    success = REGISTRY.update_asset(asset_id, updates)
    if not success:
        raise HTTPException(status_code=404, detail="Asset not found")
    return {"status": "success"}

@app.delete("/api/assets/{asset_id}")
def delete_asset(asset_id: str):
    success = REGISTRY.delete_asset(asset_id)
    if not success:
        raise HTTPException(status_code=404, detail="Asset not found")
    return {"status": "success"}

# ─────────────────────────────────────────────
# HEAT EXCHANGERS ENDPOINTS
# ─────────────────────────────────────────────
@app.get("/api/exchangers")
def get_exchangers():
    # Load registered heat exchangers
    hxs = REGISTRY.get_hx_units()
    
    # If registry is empty of HXs, register a couple of template units for the UI
    if not hxs:
        hx1 = REGISTRY.build_default_asset("Heat Exchanger")
        hx1["id"] = "HX-101"
        hx1["description"] = "Distillation Reflux Cooler"
        hx1["shell_temp_in"] = 120.0
        hx1["shell_temp_out"] = 92.5
        hx1["shell_flow_kg_s"] = 12.5
        
        hx2 = REGISTRY.build_default_asset("Heat Exchanger")
        hx2["id"] = "HX-102"
        hx2["description"] = "Reactor Bottoms Preheater"
        hx2["shell_temp_in"] = 145.0
        hx2["shell_temp_out"] = 120.0
        hx2["shell_flow_kg_s"] = 8.0
        hx2["days_since_last_clean"] = 180
        
        REGISTRY.add_asset(hx1)
        REGISTRY.add_asset(hx2)
        hxs = [hx1, hx2]
        
    # Analyze
    hx_units = []
    for h in hxs:
        u = HXUnit(
            unit_id=h["id"],
            description=h.get("description", ""),
            fluid_type=h.get("fluid_type", "process_liquid"),
            shell_temp_in=float(h.get("shell_temp_in", 120.0)),
            shell_temp_out=float(h.get("shell_temp_out", 95.0)),
            shell_flow_kg_s=float(h.get("shell_flow_kg_s", 10.0)),
            shell_cp=float(h.get("shell_cp", 4.18)),
            tube_temp_in=float(h.get("tube_temp_in", 60.0)),
            tube_temp_out=float(h.get("tube_temp_out", 80.0)),
            tube_flow_kg_s=float(h.get("tube_flow_kg_s", 12.0)),
            tube_cp=float(h.get("tube_cp", 4.18)),
            design_U=float(h.get("design_U", 1000.0)),
            heat_area_m2=float(h.get("heat_area_m2", 50.0)),
            days_since_last_clean=int(h.get("days_since_last_clean", 0))
        )
        u.analyze()
        hx_units.append(u)
        
    # Sort by criticality score (highest first)
    hx_units_sorted = sorted(hx_units, key=lambda x: x.criticality_score, reverse=True)
    
    return [
        {
            "id": u.unit_id,
            "description": u.description,
            "fluid_type": u.fluid_type,
            "effectiveness": u.effectiveness,
            "design_effectiveness": u.design_effectiveness,
            "fouling_factor": u.fouling_factor,
            "fouling_limit": u.fouling_limit,
            "foul_pct": (u.fouling_factor / u.fouling_limit * 100) if u.fouling_limit else 0,
            "heat_duty_kw": u.heat_duty_kw,
            "current_U": u.current_U,
            "criticality_score": u.criticality_score,
            "days_to_clean": u.days_to_clean,
            "status_color": "#ef4444" if u.criticality_score >= 80 else ("#f59e0b" if u.criticality_score >= 50 else "#10b981"),
            "status_label": "CRITICAL" if u.criticality_score >= 80 else ("WARNING" if u.criticality_score >= 50 else "SAFE"),
            "lmtd": u.lmtd,
            "temperature_pinch": u.temperature_pinch,
            "diagnostics": u.diagnostics,
            "health_score": u.health_score
        } for u in hx_units_sorted
    ]

# ─────────────────────────────────────────────
# HISTORICAL STATS ENDPOINTS
# ─────────────────────────────────────────────
@app.get("/api/historical-stats")
def get_historical_stats():
    dp = os.path.join(BASE_DIR, "ai4i2020.csv")
    if not os.path.exists(dp):
        raise HTTPException(status_code=404, detail="ai4i2020.csv missing")
        
    df = pd.read_csv(dp, encoding="utf-8-sig")
    total = len(df)
    fails = int(df["Machine failure"].sum())
    
    type_vc = df["Type"].value_counts().to_dict()
    
    fbt = df.groupby("Type")["Machine failure"].agg(["sum","count"])
    fbt_dict = {}
    for t, r in fbt.iterrows():
        fbt_dict[t] = {
            "sum": int(r["sum"]),
            "count": int(r["count"]),
            "fail_pct": float(r["sum"] / r["count"] * 100)
        }
        
    return {
        "size": total,
        "faults": fails,
        "fail_rate_pct": float(fails/total*100),
        "class_ratio": f"{int((total-fails)/fails)}:1",
        "grades": type_vc,
        "grade_fail_rates": fbt_dict
    }

@app.get("/api/historical-distribution")
def get_historical_distribution(column: str):
    dp = os.path.join(BASE_DIR, "ai4i2020.csv")
    if not os.path.exists(dp):
        raise HTTPException(status_code=404, detail="ai4i2020.csv missing")
        
    df = pd.read_csv(dp, encoding="utf-8-sig")
    if column not in df.columns:
        raise HTTPException(status_code=400, detail="Invalid column")
        
    # sample down to 500 rows for lightweight chart transmission
    df_sample = df[[column, "Machine failure"]].dropna()
    return df_sample.to_dict(orient="records")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

# Force reload - retrained 12-feature model 2