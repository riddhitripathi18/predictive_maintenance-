"""
dashboard.py
Predictive Maintenance POC -- Premium SCADA Control Dashboard (v3.0)
Navigation via st.tabs (always visible at top of page).
Aesthetics: Premium dark UI, glassmorphic cards, custom glowing SVG ring dials, pulsing indicators.
Run: python -m streamlit run dashboard.py
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings("ignore")

# New modules
import importlib
import ttf_engine
import machine_registry
import heat_exchanger_analyzer

importlib.reload(ttf_engine)
importlib.reload(machine_registry)
importlib.reload(heat_exchanger_analyzer)

from ttf_engine import run_ttf_forecast, conditional_safe_forecast, DEFAULT_DEGRADATION
from machine_registry import (
    MachineRegistry, MACHINE_TYPES, FAILURE_MODES_BY_TYPE,
    SENSOR_FIELDS_BY_TYPE, ML_COMPATIBLE_TYPES
)
from heat_exchanger_analyzer import (
    HXUnit, analyze_hx_fleet, FOULING_LIMITS
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SYS-CONTROL | Predictive Maintenance",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# PREMIUM UI GLASSMORPHIC STYLING
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 10% 20%, #0d121c 0%, #060910 90%);
    color: #e2e8f0;
}

/* Glassmorphism Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(17, 25, 40, 0.75);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 14px;
    padding: 8px;
    gap: 8px;
    margin-bottom: 24px;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
    backdrop-filter: blur(12px);
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #94a3b8;
    border-radius: 10px;
    font-weight: 500;
    font-size: 0.95rem;
    padding: 12px 28px;
    border: none;
    transition: all 0.3s ease;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important;
    color: #ffffff !important;
    box-shadow: 0 0 15px rgba(29, 78, 216, 0.4);
}

/* Neon alerts */
.alert-critical {
    background: linear-gradient(135deg, rgba(61, 15, 15, 0.8) 0%, rgba(92, 26, 26, 0.8) 100%);
    border: 2px solid #ef4444;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 0 20px rgba(239, 68, 68, 0.25);
    backdrop-filter: blur(12px);
}
.alert-warning {
    background: linear-gradient(135deg, rgba(61, 43, 0, 0.8) 0%, rgba(92, 66, 0, 0.8) 100%);
    border: 2px solid #f59e0b;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 0 20px rgba(245, 158, 11, 0.25);
    backdrop-filter: blur(12px);
}
.alert-safe {
    background: linear-gradient(135deg, rgba(10, 45, 31, 0.8) 0%, rgba(15, 64, 48, 0.8) 100%);
    border: 2px solid #10b981;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 0 20px rgba(16, 185, 129, 0.25);
    backdrop-filter: blur(12px);
}

.alert-critical h2 { color: #f87171; font-weight: 700; margin: 0; }
.alert-warning h2 { color: #fbbf24; font-weight: 700; margin: 0; }
.alert-safe h2 { color: #34d399; font-weight: 700; margin: 0; }

.section-header {
    background: linear-gradient(90deg, rgba(29, 78, 216, 0.15), rgba(13, 18, 28, 0));
    border-left: 4px solid #3b82f6;
    padding: 14px 20px;
    border-radius: 0 10px 10px 0;
    margin: 24px 0 14px 0;
    color: #f1f5f9;
    font-weight: 600;
    font-size: 1.1rem;
    letter-spacing: 0.5px;
}

.insight-box {
    background: rgba(20, 31, 48, 0.5);
    border: 1px solid rgba(59, 130, 246, 0.15);
    border-radius: 12px;
    padding: 16px 20px;
    margin: 8px 0;
    color: #cbd5e1;
    font-size: 0.92rem;
    line-height: 1.6;
    backdrop-filter: blur(12px);
}

/* Sidebar Custom Tech-Theme */
.stSidebar { background: #080d16 !important; border-right: 1px solid rgba(255, 255, 255, 0.05); }

/* Blinking Indicator Animation */
@keyframes pulse-green {
    0% { transform: scale(0.9); opacity: 0.4; }
    50% { transform: scale(1.2); opacity: 1; filter: drop-shadow(0 0 8px #10b981); }
    100% { transform: scale(0.9); opacity: 0.4; }
}
.pulse-indicator {
    width: 10px;
    height: 10px;
    background-color: #10b981;
    border-radius: 50%;
    display: inline-block;
    animation: pulse-green 1.8s infinite ease-in-out;
}

/* Custom styling for metrics */
div[data-testid="stMetric"] {
    background: rgba(17, 24, 39, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 12px !important;
    padding: 14px !important;
}

/* Scrollbars */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: #060910; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #334155; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODELS & DATA
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")

SUB_TARGETS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

@st.cache_resource
def load_models():
    try:
        with open(os.path.join(MODEL_DIR, "xgb_model.pkl"), "rb") as f:
            xgb_model = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "rf_model.pkl"), "rb") as f:
            rf_model = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
            
        with open(os.path.join(MODEL_DIR, "xgb_threshold.pkl"), "rb") as f:
            xgb_thresh_f1 = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "xgb_safety_threshold.pkl"), "rb") as f:
            xgb_thresh_safety = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "rf_threshold.pkl"), "rb") as f:
            rf_thresh_f1 = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "rf_safety_threshold.pkl"), "rb") as f:
            rf_thresh_safety = pickle.load(f)
            
        sub_models = {}
        for sub in SUB_TARGETS:
            with open(os.path.join(MODEL_DIR, f"xgb_{sub.lower()}_model.pkl"), "rb") as f:
                sub_models[sub] = pickle.load(f)
                
        with open(os.path.join(MODEL_DIR, "feature_cols.pkl"), "rb") as f:
            feat_cols = pickle.load(f)
            
        with open(os.path.join(MODEL_DIR, "fleet_medians.pkl"), "rb") as f:
            fleet_medians = pickle.load(f)
            
        return (xgb_model, rf_model, scaler, 
                float(xgb_thresh_f1), float(xgb_thresh_safety),
                float(rf_thresh_f1), float(rf_thresh_safety),
                sub_models, feat_cols, fleet_medians, True)
    except FileNotFoundError as e:
        return None, None, None, 0.3, 0.1, 0.3, 0.1, {}, [], {}, False

(xgb_model, rf_model, scaler, 
 xgb_thresh_f1, xgb_thresh_safety,
 rf_thresh_f1, rf_thresh_safety,
 sub_models, FEATURE_COLS, fleet_medians, models_loaded) = load_models()


# ─────────────────────────────────────────────
# DYNAMIC PREMIUM COMPONENTS
# ─────────────────────────────────────────────
TYPE_MAP = {"L (Low)": 0, "M (Medium)": 1, "H (High)": 2}

def engineer_features(air_temp, proc_temp, rpm, torque, tool_wear, type_enc):
    temp_delta    = proc_temp - air_temp
    power_W       = torque * rpm * (2 * np.pi / 60)
    torque_x_wear = torque * tool_wear
    wear_pct      = tool_wear / 253.0
    high_torque   = 1 if torque > 55 else 0
    high_wear     = 1 if tool_wear > 200 else 0
    return [type_enc, air_temp, proc_temp, rpm, torque, tool_wear,
            temp_delta, power_W, torque_x_wear, wear_pct, high_torque, high_wear]

def get_risk(prob):
    if prob >= 0.65:   return "#ef4444", "CRITICAL", "🔴"
    elif prob >= 0.35: return "#f59e0b", "WARNING",  "🟡"
    else:              return "#10b981", "SAFE",      "🟢"

def make_premium_gauge(prob, active_thresh):
    # Generates a premium vector radial dial with animation and dropshadows instead of generic plotly meters
    color, label, icon = get_risk(prob)
    r = 80
    circ = 2 * np.pi * r  # ~502.6
    offset = circ - (prob * circ)
    
    # Calculate threshold marker angle on standard dial gauge scale
    t_angle = (active_thresh * 360) - 90  # convert ratio to degrees
    
    pulse_class = "pulse-glow" if prob >= active_thresh else ""
    
    svg_html = f"""
    <style>
    @keyframes pulse-ring {{
        0% {{ opacity: 0.8; filter: drop-shadow(0px 0px 4px {color}); }}
        50% {{ opacity: 1; filter: drop-shadow(0px 0px 16px {color}); }}
        100% {{ opacity: 0.8; filter: drop-shadow(0px 0px 4px {color}); }}
    }}
    .glowing-ring {{
        animation: pulse-ring 2s infinite ease-in-out;
    }}
    </style>
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; position: relative; height: 260px; margin-top: 10px;">
        <svg width="220" height="220" viewBox="0 0 200 200" style="transform: rotate(-90deg);">
            <!-- Background base ring -->
            <circle cx="100" cy="100" r="{r}" stroke="rgba(255,255,255,0.03)" stroke-width="12" fill="transparent" />
            <!-- Segment boundaries -->
            <circle cx="100" cy="100" r="{r}" stroke="#1e293b" stroke-width="14" fill="transparent" stroke-dasharray="{circ}" stroke-dashoffset="0" />
            <!-- Active Fill Ring -->
            <circle class="glowing-ring" cx="100" cy="100" r="{r}" stroke="{color}" stroke-width="12" fill="transparent"
                    stroke-dasharray="{circ}" stroke-dashoffset="{offset}" stroke-linecap="round"
                    style="transition: stroke-dashoffset 0.8s ease-in-out;" />
        </svg>
        <div style="position: absolute; text-align: center; top: 50%; left: 50%; transform: translate(-50%, -50%);">
            <h1 style="margin: 0; font-size: 2.8rem; font-weight: 700; color: #ffffff; text-shadow: 0 0 10px {color}aa; font-family: 'Outfit', sans-serif;">{prob*100:.1f}%</h1>
            <p style="margin: 2px 0 0 0; font-size: 0.85rem; font-weight: 600; color: {color}; text-transform: uppercase; letter-spacing: 2px;">{label}</p>
        </div>
    </div>
    """
    return svg_html

def custom_metric_card(title, value, unit, status_color="#3b82f6", bg_color="rgba(17, 24, 39, 0.45)"):
    # Futuristic custom layout telemetry card
    card_html = f"""
    <div style="background: {bg_color};
                border: 1px solid rgba(255, 255, 255, 0.05);
                border-top: 3px solid {status_color};
                border-radius: 12px;
                padding: 18px;
                margin: 6px 0;
                box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.25);
                backdrop-filter: blur(12px);
                transition: transform 0.2s ease-in-out;">
        <p style="margin: 0; color: #94a3b8; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600;">{title}</p>
        <div style="display: flex; align-items: baseline; gap: 6px; margin-top: 10px;">
            <h2 style="margin: 0; color: #ffffff; font-size: 1.85rem; font-weight: 700; font-family: 'Outfit', sans-serif;">{value}</h2>
            <span style="color: #64748b; font-size: 0.8rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px;">{unit}</span>
        </div>
    </div>
    """
    return card_html

def explain_prediction(air_temp, proc_temp, rpm, torque, tool_wear, prob, mode_label):
    reasons = []
    if torque > 60:
        reasons.append(f"🔴 **CRITICAL TORQUE OVERLOAD:** Motor is operating at **{torque} Nm**, exceeding physical safety margins (>60 Nm). High risk of mechanical shear.")
    if tool_wear > 200:
        reasons.append(f"⚠️ **DEGRADED COMPONENT LIFE:** Cumulative tool wear has reached **{tool_wear} min**. Micro-fractures and thermal degradation are expected.")
    if rpm < 1300:
        reasons.append(f"⚠️ **ROTATION SLOWDOWN:** Speed dropped to **{rpm} RPM**. Combined with high load, this indicates potential motor stalling or severe friction.")
    temp_delta = proc_temp - air_temp
    if temp_delta < 8.6:
        reasons.append(f"🔴 **INSUFFICIENT HEAT DISSIPATION:** Temperature delta is **{temp_delta:.1f} K** (Limit: >8.6 K). Cooling loop blockage suspected.")
    power = torque * rpm * (2 * np.pi / 60)
    if power > 9000 or power < 3500:
        reasons.append(f"⚠️ **ELECTRICAL OUT-OF-BOUNDS:** Power draws **{power:.0f} W** which is outside nominal rating limits (3500W-9000W). Risk of electrical circuit breaker failure.")
        
    if not reasons:
        if prob > 0.5:
            reasons.append("🧬 **MULTIVARIATE AI FLAG:** No single sensor exceeds standard limits, but the cross-parameter pattern indicates early signs of impending failure.")
        else:
            reasons.append("💚 **NOMINAL STATE:** All diagnostic streams are within safe engineering operational medians.")
    return reasons

def predict_failure_modes(air_temp, proc_temp, rpm, torque, tool_wear, type_str):
    type_enc = TYPE_MAP[type_str]
    features = engineer_features(air_temp, proc_temp, rpm, torque, tool_wear, type_enc)
    X_scaled = scaler.transform(np.array([features]))
    
    probs = {}
    for sub in SUB_TARGETS:
        if sub in sub_models:
            probs[sub] = float(sub_models[sub].predict_proba(X_scaled)[0, 1])
    return probs

def prepare_batch(df_raw):
    df = df_raw.copy()
    needed = ["Type", "Air temperature [K]", "Process temperature [K]",
              "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return None, f"Missing columns: {missing}"
    type_enc = {"L": 0, "M": 1, "H": 2}
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
        lambda p: "🔴 CRITICAL" if p >= 65 else ("🟡 WARNING" if p >= 35 else "🟢 SAFE")
    )
    return df, None


# ─────────────────────────────────────────────
# HEADER & CONTROL BAR
# ─────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom: 20px;'>
  <h1 style='color:#3b82f6; margin:0; font-size:2.2rem; font-weight:700; letter-spacing:-0.5px;'>🏭 predictive_maint_system_v3.0</h1>
  <p style='color:#64748b; margin:6px 0 0 0; font-size:0.95rem; font-family:"JetBrains Mono", monospace;'>
    ENGINE: XGBoost + Random Forest &nbsp;|&nbsp; DATASET: AI4I 2020 &nbsp;|&nbsp; STATUS: <span class="pulse-indicator"></span> ONLINE
  </p>
</div>
""", unsafe_allow_html=True)


if not models_loaded:
    st.error("🚨 **ML models / pipeline file not found.** Please verify the directory path and run model training script.")
    st.stop()


# ─────────────────────────────────────────────
# SIDEBAR CONTROL
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h3 style='margin-top:0;'>🛠️ CALIBRATION CENTER</h3>", unsafe_allow_html=True)
    st.markdown("Calibrate alert thresholds for the facility's risk posture.")
    st.divider()
    
    op_mode = st.radio(
        "Sensitivity Preset",
        ["⚖️ Balanced Mode (Optimal F1)", "🛡️ Safety-First (90% Recall)"],
        index=0,
        help="🛡️ Safety-First Mode drops threshold to catch early, low-probability signs of catastrophe."
    )
    
    if "Balanced" in op_mode:
        active_thresh = xgb_thresh_f1
        st.markdown(f"""
        <div style='background:rgba(59,130,246,0.1); border-left:3px solid #3b82f6; padding:12px; border-radius:4px;'>
          <span style='color:#93c5fd; font-size:0.75rem; font-weight:600; text-transform:uppercase;'>Balanced Target</span><br/>
          <span style='color:#fff; font-size:0.9rem;'>Threshold Limit: <b>{active_thresh*100:.0f}%</b></span>
        </div>
        """, unsafe_allow_html=True)
    else:
        active_thresh = xgb_thresh_safety
        st.markdown(f"""
        <div style='background:rgba(239,68,68,0.1); border-left:3px solid #ef4444; padding:12px; border-radius:4px;'>
          <span style='color:#fca5a5; font-size:0.75rem; font-weight:600; text-transform:uppercase;'>Safety Focus</span><br/>
          <span style='color:#fff; font-size:0.9rem;'>Threshold Limit: <b>{active_thresh*100:.0f}%</b></span>
        </div>
        """, unsafe_allow_html=True)
        
    st.divider()
    st.markdown("#### 📡 System Status")
    st.caption("Primary Model: XGBoost v3.3.0")
    st.caption("Scaler Matrix: standard_scaler")
    st.caption("Sub-models: 5 Multi-Output Units")
    st.divider()
    st.markdown("#### 📊 RF Constrained Model Metrics")
    st.markdown("""
    <div style='background:rgba(17,24,39,0.6); border:1px solid rgba(255,255,255,0.06);
                border-radius:12px; padding:14px; font-size:0.82rem; line-height:2;'>
      <div style='display:flex; justify-content:space-between;'>
        <span style='color:#94a3b8;'>Accuracy</span>
        <span style='color:#10b981; font-weight:700;'>94.60%</span>
      </div>
      <div style='display:flex; justify-content:space-between;'>
        <span style='color:#94a3b8;'>F1-Score</span>
        <span style='color:#10b981; font-weight:700;'>0.5345</span>
      </div>
      <div style='display:flex; justify-content:space-between;'>
        <span style='color:#94a3b8;'>Recall</span>
        <span style='color:#10b981; font-weight:700;'>91.18%</span>
      </div>
      <div style='display:flex; justify-content:space-between;'>
        <span style='color:#94a3b8;'>Precision</span>
        <span style='color:#f59e0b; font-weight:700;'>37.80%</span>
      </div>
      <div style='display:flex; justify-content:space-between;'>
        <span style='color:#94a3b8;'>PR-AUC</span>
        <span style='color:#10b981; font-weight:700;'>0.7213</span>
      </div>
      <div style='display:flex; justify-content:space-between;'>
        <span style='color:#94a3b8;'>Train/Test Gap</span>
        <span style='color:#10b981; font-weight:700;'>0.0492 (OK)</span>
      </div>
      <div style='display:flex; justify-content:space-between;'>
        <span style='color:#94a3b8;'>False Positives</span>
        <span style='color:#f87171; font-weight:700;'>107</span>
      </div>
      <div style='display:flex; justify-content:space-between;'>
        <span style='color:#94a3b8;'>False Negatives</span>
        <span style='color:#f87171; font-weight:700;'>6</span>
      </div>
      <div style='margin-top:8px; padding-top:8px; border-top:1px solid rgba(255,255,255,0.06);
                  color:#64748b; font-size:0.75rem;'>Threshold: 0.50 &nbsp;|&nbsp; Test: 2,000 rows</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.caption("Confidential. Internal plant monitoring systems only.")


# ─────────────────────────────────────────────
# TABS SYSTEM
# ─────────────────────────────────────────────
tab1, tab_ttf, tab_registry, tab2, tab3 = st.tabs([
    "🔬  Digital Twin Simulator",
    "🕐  Time-to-Failure Forecast",
    "🏭  Plant Asset Registry",
    "📦  Fleet Batch Analyzer",
    "📊  Facility Data Explorer",
])


# ═══════════════════════════════════════════════════════════
# TAB 1 — SINGLE MACHINE
# ═══════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">⚙️ Telemetry Controls & AI Assessment</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("#### 🎛️ Live Parameter Override")
        product_type = st.selectbox("Component Grade Variant", ["L (Low)", "M (Medium)", "H (High)"], key="t1_type")
        air_temp  = st.slider("Outside Intake Temperature (K)",  295.0, 305.0, 300.0, 0.1, key="t1_air")
        proc_temp = st.slider("Internal Chamber Temperature (K)",305.0, 315.0, 310.0, 0.1, key="t1_proc")
        rpm       = st.slider("Agitator / Rotational Speed (RPM)", 1168,  2886,  1500,  10,  key="t1_rpm")
        torque    = st.slider("Torque Output Load (Nm)",          3.8,   76.6,  40.0,  0.1, key="t1_torque")
        tool_wear = st.slider("Component Run-Time Wear (min)",    0,     253,   108,   1,   key="t1_wear")

        st.markdown("---")
        st.markdown("##### 🧪 Stress Load Test Profiles")
        sc1, sc2, sc3 = st.columns(3)
        if sc1.button("🔴 Heat Lock Failure (HDF)", key="btn_hdf", width="stretch"):
            air_temp, proc_temp, rpm, torque, tool_wear, product_type = 298.5, 307.0, 1340, 48.0, 50, "L (Low)"
        if sc2.button("🟡 High Power Stress (PWF)", key="btn_pwf", width="stretch"):
            air_temp, proc_temp, rpm, torque, tool_wear, product_type = 300.0, 310.0, 2700, 10.0, 80, "M (Medium)"
        if sc3.button("🟢 Standard Stable Load", key="btn_safe", width="stretch"):
            air_temp, proc_temp, rpm, torque, tool_wear, product_type = 300.0, 310.0, 1550, 38.0, 80, "H (High)"

    with col_right:
        type_enc = TYPE_MAP[product_type]
        features = engineer_features(air_temp, proc_temp, rpm, torque, tool_wear, type_enc)
        X_scaled = scaler.transform(np.array([features]))

        # Calculate prediction
        xgb_prob = float(xgb_model.predict_proba(X_scaled)[0, 1])
        avg_prob = xgb_prob

        # Radial Custom Ring
        st.components.v1.html(make_premium_gauge(avg_prob, active_thresh), height=270)

        # Risk Banner
        is_failing = avg_prob >= active_thresh
        if is_failing:
            st.markdown(
                f'<div class="alert-critical"><h2>🚨 HIGH FAILURE SHUTDOWN RISK 🚨</h2>'
                f'<p style="color:#eee;margin:6px 0 0;">Anomalous load patterns exceed safe operating limit of {active_thresh*100:.0f}%</p></div>',
                unsafe_allow_html=True,
            )
            # Audio trigger
            st.markdown("""
                <script>
                (function() {
                    try {
                        const ctx = new (window.AudioContext || window.webkitAudioContext)();
                        const playPulse = (delay, freq) => {
                            const osc = ctx.createOscillator();
                            const gain = ctx.createGain();
                            osc.type = 'sawtooth';
                            osc.frequency.setValueAtTime(freq, ctx.currentTime + delay);
                            gain.gain.setValueAtTime(0.12, ctx.currentTime + delay);
                            gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + delay + 0.35);
                            osc.connect(gain); gain.connect(ctx.destination);
                            osc.start(ctx.currentTime + delay); osc.stop(ctx.currentTime + delay + 0.4);
                        };
                        playPulse(0, 950); playPulse(0.35, 950);
                    } catch(e) {}
                })();
                </script>
            """, unsafe_allow_html=True)
        elif avg_prob >= 0.35:
            st.markdown(
                f'<div class="alert-warning"><h2>⚠️ MAINTENANCE ADVISORY</h2>'
                f'<p style="color:#eee;margin:6px 0 0;">Operating threshold is warning (<b>{avg_prob*100:.1f}%</b>). Inspections required.</p></div>',
                unsafe_allow_html=True,
            )
            # Single alert pulse
            st.markdown("""
                <script>
                (function() {
                    try {
                        const ctx = new (window.AudioContext || window.webkitAudioContext)();
                        const osc = ctx.createOscillator();
                        const gain = ctx.createGain();
                        osc.type = 'sine'; osc.frequency.setValueAtTime(580, ctx.currentTime);
                        gain.gain.setValueAtTime(0.08, ctx.currentTime);
                        gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.2);
                        osc.connect(gain); gain.connect(ctx.destination);
                        osc.start(ctx.currentTime); osc.stop(ctx.currentTime + 0.22);
                    } catch(e) {}
                })();
                </script>
            """, unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="alert-safe"><h2>🟢 COMPONENT HEALTH STABLE</h2>'
                f'<p style="color:#eee;margin:6px 0 0;">Nominal risk factor (<b>{avg_prob*100:.1f}%</b>)</p></div>',
                unsafe_allow_html=True,
            )

    # Computed metrics
    st.markdown('<div class="section-header">📐 Computed Operational Signals</div>', unsafe_allow_html=True)
    temp_delta    = proc_temp - air_temp
    power_W       = torque * rpm * (2 * np.pi / 60)
    torque_x_wear = torque * tool_wear
    
    mc1, mc2, mc3, mc4 = st.columns(4)
    # Using our custom premium card markup
    mc1.markdown(custom_metric_card("Thermodynamic Delta", f"{temp_delta:.2f}", "Kelvin (K)", "#0284c7"), unsafe_allow_html=True)
    mc2.markdown(custom_metric_card("Rotational Power", f"{power_W:.0f}", "Watts (W)", "#22c55e" if 3500<=power_W<=9000 else "#ef4444"), unsafe_allow_html=True)
    mc3.markdown(custom_metric_card("Overstrain Index", f"{torque_x_wear:.0f}", "N-m-Min", "#8b5cf6"), unsafe_allow_html=True)
    mc4.markdown(custom_metric_card("Life Wear Factor", f"{(tool_wear/253)*100:.1f}", "Percent (%)", "#14b8a6"), unsafe_allow_html=True)

    # Deviation
    st.markdown('<div class="section-header">📈 Parameter Deviation Analysis (Compared to Healthy Fleet Medians)</div>', unsafe_allow_html=True)
    dev_data = []
    comparison_features = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", "temp_delta", "power_W"]
    feat_display_names = {
        "Air temperature [K]": "Ambient Temp",
        "Process temperature [K]": "Process Temp",
        "Rotational speed [rpm]": "Spindle RPM",
        "Torque [Nm]": "Torque Load",
        "Tool wear [min]": "Cumulative Wear",
        "temp_delta": "Thermal Delta",
        "power_W": "Rotary Power"
    }
    current_vals = {
        "Air temperature [K]": air_temp,
        "Process temperature [K]": proc_temp,
        "Rotational speed [rpm]": rpm,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
        "temp_delta": temp_delta,
        "power_W": power_W
    }
    
    for f_name in comparison_features:
        median_val = fleet_medians.get(f_name, 1.0)
        curr_val = current_vals[f_name]
        pct_dev = ((curr_val - median_val) / median_val) * 100
        dev_data.append({
            "Parameters": feat_display_names[f_name],
            "Current": round(curr_val, 1),
            "Healthy Median": round(median_val, 1),
            "Deviation (%)": round(pct_dev, 1)
        })
    df_dev = pd.DataFrame(dev_data)
    
    # Styled Deviation Plot
    fig_dev = px.bar(
        df_dev, x="Deviation (%)", y="Parameters", orientation="h",
        color="Deviation (%)",
        color_continuous_scale=["#10b981", "#334155", "#ef4444"],
        template="plotly_dark",
    )
    fig_dev.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=30, b=10), height=260,
        coloraxis_showscale=False,
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)")
    )
    
    col_d1, col_d2 = st.columns([3, 2], gap="large")
    with col_d1:
        st.plotly_chart(fig_dev, width="stretch")
    with col_d2:
        st.markdown("<h5 style='margin-top:10px;'>📊 Out-of-Spec Diagnostic Flags</h5>", unsafe_allow_html=True)
        anomalies_found = False
        for idx, row in df_dev.iterrows():
            dev = row["Deviation (%)"]
            sign = "+" if dev >= 0 else ""
            if abs(dev) > 20:
                st.markdown(f"<span style='color:#ef4444;'>🛑</span> **{row['Parameters']}**: {sign}{dev}% deviation! (Current: {row['Current']} vs Base: {row['Healthy Median']})")
                anomalies_found = True
            elif abs(dev) > 10:
                st.markdown(f"<span style='color:#f59e0b;'>⚠️</span> **{row['Parameters']}**: {sign}{dev}% deviation. (Current: {row['Current']} vs Base: {row['Healthy Median']})")
                anomalies_found = True
        if not anomalies_found:
            st.markdown("<span style='color:#10b981;'>✅</span> All telemetry signals match historical baselines.")

    # Multi-Model breakdown
    st.markdown('<div class="section-header">🤖 Failure Mode Probabilities (Sub-Model Outputs)</div>', unsafe_allow_html=True)
    st.caption("Continuous predictions computed by 5 specialized classifiers trained specifically on target breakdowns.")
    
    sub_probs = predict_failure_modes(air_temp, proc_temp, rpm, torque, tool_wear, product_type)
    df_subs = pd.DataFrame({
        "Failure Mode": [f"{sub} Classifier" for sub in SUB_TARGETS],
        "Probability (%)": [round(sub_probs[sub]*100, 1) for sub in SUB_TARGETS]
    }).sort_values("Probability (%)", ascending=False)
    
    fig_subs = px.bar(
        df_subs, x="Probability (%)", y="Failure Mode", orientation="h",
        color="Probability (%)", color_continuous_scale="Reds",
        template="plotly_dark"
    )
    fig_subs.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False, height=220,
        margin=dict(l=0, r=0, t=10, b=10),
        yaxis=dict(autorange="reversed"),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)")
    )
    
    col_s1, col_s2 = st.columns([3, 2], gap="large")
    with col_s1:
        st.plotly_chart(fig_subs, width="stretch")
    with col_s2:
        st.markdown("<h5 style='margin-top:10px;'>⚡ Model Diagnostics</h5>", unsafe_allow_html=True)
        max_mode = df_subs.iloc[0]["Failure Mode"].split()[0]
        max_prob = df_subs.iloc[0]["Probability (%)"]
        
        if max_prob >= 35:
            st.error(f"**ML DIAGNOSIS:** AI predicts a high risk of **{max_mode}** at **{max_prob}%**.")
            st.markdown("Ensure corresponding coolant levels, lubrication pressures, and output torque speeds are immediately audited.")
        else:
            st.success("**ML DIAGNOSIS:** Stable state. All target failure mode probabilities are below warning threshold (35%).")

    # Explanations
    st.markdown('<div class="section-header">💡 Facility Operator Action Guidelines</div>', unsafe_allow_html=True)
    for r in explain_prediction(air_temp, proc_temp, rpm, torque, tool_wear, avg_prob, op_mode):
        st.markdown(f'<div class="insight-box">{r}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# TAB 2 — BATCH ANALYSIS
# ═══════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">📦 Fleet telemetry Batch Processing</div>', unsafe_allow_html=True)

    col_up, col_info = st.columns([2, 1], gap="large")

    with col_info:
        st.markdown("#### Facility Data Rules")
        st.code("Type\nAir temperature [K]\nProcess temperature [K]\nRotational speed [rpm]\nTorque [Nm]\nTool wear [min]")
        st.info("💡 You can upload your raw CSV export. The system will encode and engineer all features dynamically.")

    with col_up:
        uploaded = st.file_uploader("Upload CSV export file", type=["csv"], key="batch_uploader")
        use_builtin = st.checkbox("📂 Simulate Live Factory Stream (Analyze all 10,000 active records)", value=False)

    raw_df = None
    if use_builtin:
        dp = os.path.join(BASE_DIR, "ai4i2020.csv")
        if os.path.exists(dp):
            raw_df = pd.read_csv(dp, encoding="utf-8-sig")
            st.success(f"✅ Loaded builtin records database: **{len(raw_df):,} machines**")
        else:
            st.error("ai4i2020.csv database file missing.")
    elif uploaded is not None:
        raw_df = pd.read_csv(uploaded)
        st.success(f"✅ Loaded target telemetry database: **{len(raw_df):,} machines**")

    if raw_df is not None:
        with st.spinner("Processing batch analysis..."):
            result_df, err = prepare_batch(raw_df)

        if err:
            st.error(f"Error: {err}")
        else:
            # Calibrate threshold predictions
            result_df["Predicted Failure"] = (result_df["Failure Probability (%)"] >= (active_thresh * 100)).astype(int)
            result_df["Risk Level"] = result_df["Failure Probability (%)"].apply(
                lambda p: "🔴 CRITICAL" if p >= 65 else ("🟡 WARNING" if p >= 35 else "🟢 SAFE")
            )
            
            total      = len(result_df)
            n_critical = (result_df["Risk Level"] == "🔴 CRITICAL").sum()
            n_warning  = (result_df["Risk Level"] == "🟡 WARNING").sum()
            n_safe     = (result_df["Risk Level"] == "🟢 SAFE").sum()
            n_predicted = int(result_df["Predicted Failure"].sum())

            # KPIs Custom
            st.markdown('<div class="section-header">📈 Live Fleet Health Telemetry</div>', unsafe_allow_html=True)
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.markdown(custom_metric_card("Active Fleet Size", f"{total:,}", "Units", "#3b82f6"), unsafe_allow_html=True)
            k2.markdown(custom_metric_card("🔴 Critical Risks", f"{n_critical:,}", f"{n_critical/total*100:.1f}%", "#ef4444"), unsafe_allow_html=True)
            k3.markdown(custom_metric_card("🟡 Warning Risks", f"{n_warning:,}", f"{n_warning/total*100:.1f}%", "#f59e0b"), unsafe_allow_html=True)
            k4.markdown(custom_metric_card("🟢 Stable Units", f"{n_safe:,}", f"{n_safe/total*100:.1f}%", "#10b981"), unsafe_allow_html=True)
            k5.markdown(custom_metric_card("Triggered Alarms", f"{n_predicted:,}", f"Limit: {active_thresh*100:.0f}%", "#ef4444" if n_predicted>0 else "#10b981"), unsafe_allow_html=True)

            # ── LIVE EVALUATION METRICS ON EXTERNAL DATA ──────────────────────────────
            if "Machine failure" in result_df.columns:
                from sklearn.metrics import (
                    accuracy_score, precision_score, recall_score,
                    f1_score, average_precision_score, confusion_matrix
                )
                y_true  = result_df["Machine failure"].astype(int)
                y_pred  = result_df["Predicted Failure"].astype(int)
                y_proba = result_df["Failure Probability (%)"] / 100.0

                ext_acc   = accuracy_score(y_true, y_pred)
                ext_prec  = precision_score(y_true, y_pred, zero_division=0)
                ext_rec   = recall_score(y_true, y_pred, zero_division=0)
                ext_f1    = f1_score(y_true, y_pred, zero_division=0)
                ext_prauc = average_precision_score(y_true, y_proba)
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

                st.markdown('<div class="section-header">📐 Model Accuracy on Your Uploaded Data</div>',
                            unsafe_allow_html=True)
                st.caption("Metrics computed against the **Machine failure** ground-truth labels in your file — not the training dataset.")

                m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
                m1.markdown(custom_metric_card("Accuracy",  f"{ext_acc*100:.2f}", "%",     "#3b82f6"), unsafe_allow_html=True)
                m2.markdown(custom_metric_card("Precision", f"{ext_prec:.4f}",    "score", "#10b981" if ext_prec >= 0.5 else "#f59e0b"), unsafe_allow_html=True)
                m3.markdown(custom_metric_card("Recall",    f"{ext_rec:.4f}",     "score", "#10b981" if ext_rec  >= 0.7 else "#f59e0b"), unsafe_allow_html=True)
                m4.markdown(custom_metric_card("F1-Score",  f"{ext_f1:.4f}",      "score", "#10b981" if ext_f1   >= 0.5 else "#f59e0b"), unsafe_allow_html=True)
                m5.markdown(custom_metric_card("PR-AUC",    f"{ext_prauc:.4f}",   "score", "#10b981" if ext_prauc>= 0.6 else "#f59e0b"), unsafe_allow_html=True)
                m6.markdown(custom_metric_card("False Pos", f"{fp}",              "rows",  "#f59e0b"), unsafe_allow_html=True)
                m7.markdown(custom_metric_card("False Neg", f"{fn}",              "rows",  "#ef4444"), unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                fig_pie = go.Figure(go.Pie(
                    labels=["Critical", "Warning", "Safe"],
                    values=[n_critical, n_warning, n_safe],
                    hole=0.55,
                    marker_colors=["#ef4444", "#f59e0b", "#10b981"],
                    textfont_color="white",
                ))
                fig_pie.update_layout(
                    title="System Fleet Risk Levels",
                    paper_bgcolor="rgba(0,0,0,0)", font_color="white",
                    legend=dict(font=dict(color="white")),
                    margin=dict(t=40, b=10, l=10, r=10), height=340,
                )
                st.plotly_chart(fig_pie, width="stretch")

            with c2:
                fig_hist = px.histogram(
                    result_df, x="Failure Probability (%)", nbins=40,
                    color_discrete_sequence=["#3b82f6"], template="plotly_dark",
                    title="Risk Factor Distribution Spectrum",
                )
                fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,39,0.5)",
                                       margin=dict(t=40, b=10, l=10, r=10), height=340)
                fig_hist.add_vline(x=active_thresh*100, line_dash="dash", line_color="#ef4444", 
                                   annotation_text="Alarm Limit", annotation_position="top right")
                st.plotly_chart(fig_hist, width="stretch")

            # High-risk table
            st.markdown('<div class="section-header">🚨 Fleet Risk Status & Warning List</div>',
                        unsafe_allow_html=True)
            
            show_all = st.checkbox("🔍 Show all machines in fleet (uncheck to show warned/flagged machines only)", value=True)
            
            if show_all:
                hr = result_df.sort_values("Failure Probability (%)", ascending=False)
            else:
                hr = result_df[result_df["Failure Probability (%)"] >= (active_thresh * 100)].sort_values(
                    "Failure Probability (%)", ascending=False
                )
                
            show_cols = [c for c in ["UDI", "Type", "Air temperature [K]",
                                     "Process temperature [K]", "Rotational speed [rpm]",
                                     "Torque [Nm]", "Tool wear [min]",
                                     "Failure Probability (%)", "Risk Level"]
                         if c in hr.columns]
            if not hr.empty:
                st.dataframe(hr[show_cols].head(300).reset_index(drop=True), width="stretch", height=400)
                st.download_button(
                    "⬇️ Export Fleet Risk List (CSV)",
                    data=hr[show_cols].to_csv(index=False),
                    file_name="fleet_risk_report.csv",
                    mime="text/csv",
                )
            else:
                st.success("✅ Fleet is stable. Zero alarm overrides triggered.")


# ═══════════════════════════════════════════════════════════
# TAB 3 — DATASET EXPLORER
# ═══════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">📊 Facility Historical Metrics Analysis</div>', unsafe_allow_html=True)

    dp = os.path.join(BASE_DIR, "ai4i2020.csv")
    if not os.path.exists(dp):
        st.error("ai4i2020.csv missing.")
        st.stop()

    @st.cache_data
    def load_eda():
        return pd.read_csv(dp, encoding="utf-8-sig")

    df_eda = load_eda()
    total  = len(df_eda)
    fails  = int(df_eda["Machine failure"].sum())

    # KPIs custom
    e1, e2, e3, e4 = st.columns(4)
    e1.markdown(custom_metric_card("Historical Database Size", f"{total:,}", "Records", "#3b82f6"), unsafe_allow_html=True)
    e2.markdown(custom_metric_card("Total Fault Log Entries", f"{fails:,}", "Faults", "#ef4444"), unsafe_allow_html=True)
    e3.markdown(custom_metric_card("Nominal Facility Fail Rate", f"{fails/total*100:.2f}", "Percent (%)", "#f59e0b"), unsafe_allow_html=True)
    e4.markdown(custom_metric_card("Unbalanced Class Ratio", f"{int((total-fails)/fails)}:1", "Healthy:Failing", "#10b981"), unsafe_allow_html=True)

    # Type distribution
    st.markdown('<div class="section-header">📦 Grade Breakdown & Reliability Index</div>', unsafe_allow_html=True)
    ct1, ct2 = st.columns(2)
    type_vc = df_eda["Type"].value_counts().reset_index()
    type_vc.columns = ["Type", "Count"]
    with ct1:
        fig_tc = px.bar(type_vc, x="Type", y="Count", color="Type",
                        color_discrete_map={"L":"#f87171","M":"#fbbf24","H":"#60a5fa"},
                        template="plotly_dark", title="Grade Volume Breakdown")
        fig_tc.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,39,0.5)",
                             showlegend=False, margin=dict(t=40,b=10,l=10,r=10), height=320)
        st.plotly_chart(fig_tc, width="stretch")

    with ct2:
        fbt = df_eda.groupby("Type")["Machine failure"].agg(["sum","count"]).reset_index()
        fbt["fail_pct"] = fbt["sum"] / fbt["count"] * 100
        fig_fbt = px.bar(fbt, x="Type", y="fail_pct", color="Type",
                         color_discrete_map={"L":"#f87171","M":"#fbbf24","H":"#60a5fa"},
                         template="plotly_dark", title="Breakdown Index (%) by Grade")
        fig_fbt.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,39,0.5)",
                              showlegend=False, margin=dict(t=40,b=10,l=10,r=10), height=320)
        st.plotly_chart(fig_fbt, width="stretch")

    # Distributions
    st.markdown('<div class="section-header">🌡️ Critical Sensor Threshold Visualizer</div>', unsafe_allow_html=True)
    num_cols = ["Air temperature [K]", "Process temperature [K]",
                "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
    sel = st.selectbox("Choose Metric to Chart", num_cols, key="eda_feat")
    fig_dist = px.histogram(
        df_eda, x=sel, color=df_eda["Machine failure"].astype(str),
        barmode="overlay", nbins=60, template="plotly_dark",
        color_discrete_map={"0": "#3b82f6", "1": "#ef4444"},
        labels={"color": "Failure"},
        title=f"Historical {sel} Signature Analysis -- Blue = Stable | Red = Breakdown Event",
    )
    fig_dist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,39,0.5)",
                           margin=dict(t=40,b=10,l=10,r=10), height=380)
    st.plotly_chart(fig_dist, width="stretch")

    # Failure modes
    st.markdown('<div class="section-header">🔥 Mechanical Breakdown Category Log</div>', unsafe_allow_html=True)
    modes_map = {"TWF":"Tool Wear Failure (TWF)","HDF":"Heat Dissipation Failure (HDF)","PWF":"Power Failure (PWF)","OSF":"Overstrain Failure (OSF)","RNF":"Random Fault (RNF)"}
    mode_counts = {v: int(df_eda[k].sum()) for k, v in modes_map.items()}
    fig_mode = px.bar(
        x=list(mode_counts.values()), y=list(mode_counts.keys()),
        orientation="h", template="plotly_dark",
        color=list(mode_counts.values()),
        color_continuous_scale=["#1e3a8a","#3b82f6","#ef4444"],
        title="Diagnostic Fault Count Breakdown",
    )
    fig_mode.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,39,0.5)",
                           showlegend=False, coloraxis_showscale=False,
                           margin=dict(t=40,b=10,l=10,r=10), height=320,
                           yaxis=dict(autorange="reversed"),
                           xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"))
    st.plotly_chart(fig_mode, width="stretch")

    # Tool wear vs failure rate
    st.markdown('<div class="section-header">🔧 Cumulative Run-Time Wear Risk Exponential Curve</div>', unsafe_allow_html=True)
    df_eda["wear_bin"] = (df_eda["Tool wear [min]"] // 50 * 50).astype(int)
    wear_grp = df_eda.groupby("wear_bin")["Machine failure"].agg(["mean","count"]).reset_index()
    wear_grp.columns = ["wear_bin","fail_rate","count"]
    wear_grp["fail_pct"] = (wear_grp["fail_rate"] * 100).round(2)
    wear_grp["label"] = wear_grp["wear_bin"].astype(str) + "–" + (wear_grp["wear_bin"]+49).astype(str)
    fig_wear = px.bar(wear_grp, x="label", y="fail_pct",
                      color="fail_pct",
                      color_continuous_scale=["#10b981","#f59e0b","#ef4444"],
                      template="plotly_dark",
                      title="Exponential failure risk as component wear crosses limits",
                      labels={"label":"Cumulative Operating Minutes","fail_pct":"Historical Failure Rate (%)"})
    fig_wear.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,39,0.5)",
                           coloraxis_showscale=False,
                           margin=dict(t=40,b=10,l=10,r=10), height=360,
                           xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"))
    st.plotly_chart(fig_wear, width="stretch")


# ═══════════════════════════════════════════════════════════
# TAB TTF — TIME-TO-FAILURE FORECAST
# ═══════════════════════════════════════════════════════════
with tab_ttf:
    st.markdown('<div class="section-header">🕐 Time-to-Failure Forecasting Engine</div>', unsafe_allow_html=True)
    st.caption("Simulates how current parameter trends will evolve month-by-month and predicts exactly when this machine will cross the failure threshold.")

    # ── Left column: collect all inputs ──────────────────────
    ttf_col_left, ttf_col_right = st.columns([1, 1], gap="large")

    with ttf_col_left:
        st.markdown("#### ⚙️ Machine Parameters")
        ttf_type    = st.selectbox("Component Grade", ["L (Low)", "M (Medium)", "H (High)"], key="ttf_type")
        ttf_air     = st.slider("Air Temperature (K)",       295.0, 305.0, 300.0, 0.1, key="ttf_air")
        ttf_proc    = st.slider("Process Temperature (K)",   305.0, 315.0, 310.0, 0.1, key="ttf_proc")
        ttf_rpm     = st.slider("Rotational Speed (RPM)",    1168,  2886,  1500,  10,  key="ttf_rpm")
        ttf_torque  = st.slider("Torque (Nm)",               3.8,   76.6,  38.0,  0.1, key="ttf_torque")
        ttf_wear    = st.slider("Current Tool Wear (min)",   0,     253,   80,    1,   key="ttf_wear")
        ttf_horizon = st.slider("Forecast Horizon (months)", 3,     24,    12,    1,   key="ttf_horizon")

        st.markdown("#### 📉 Degradation Rates (per month)")
        ttf_wear_rate  = st.number_input("Wear accumulation (min/month)",  value=12.0, min_value=0.0, max_value=50.0, step=0.5, key="ttf_wear_rate")
        ttf_rpm_drift  = st.number_input("RPM drift (RPM/month)",          value=-25.0, min_value=-200.0, max_value=200.0, step=5.0, key="ttf_rpm_drift")
        ttf_torq_drift = st.number_input("Torque drift (Nm/month)",        value=1.2,  min_value=-5.0, max_value=10.0, step=0.1, key="ttf_torq_drift")

        st.markdown("#### 🔭 What-If Scenario Override")
        st.caption("Apply an immediate one-time adjustment to see how it shifts the forecast.")
        sc_rpm    = st.number_input("RPM adjustment (Δ)",         value=0.0,  step=10.0, key="sc_rpm")
        sc_torque = st.number_input("Torque adjustment (Δ Nm)",   value=0.0,  step=1.0,  key="sc_torque")
        sc_wear   = st.number_input("Tool wear adjustment (Δ min)",value=0.0, step=5.0,  key="sc_wear")

    # ── Compute OUTSIDE columns so errors are visible ─────────
    ttf_type_enc = TYPE_MAP[ttf_type]
    ttf_deg = {
        "wear_rate_per_month":       ttf_wear_rate,
        "rpm_drift_per_month":       ttf_rpm_drift,
        "torque_drift_per_month":    ttf_torq_drift,
        "air_temp_drift_per_month":  0.05,
        "proc_temp_drift_per_month": 0.1,
    }
    ttf_thresh = min(float(xgb_thresh_safety), 0.35)

    ttf_error = None
    result_base = None
    result_scen = None
    try:
        result_base = run_ttf_forecast(
            xgb_model=xgb_model,
            scaler=scaler,
            air_temp=float(ttf_air),
            proc_temp=float(ttf_proc),
            rpm=float(ttf_rpm),
            torque=float(ttf_torque),
            tool_wear=float(ttf_wear),
            type_enc=ttf_type_enc,
            active_threshold=ttf_thresh,
            horizon_months=int(ttf_horizon),
            degradation=ttf_deg,
            scenario_overrides={"rpm": 0.0, "torque": 0.0, "tool_wear": 0.0}
        )
        any_override = sc_rpm != 0.0 or sc_torque != 0.0 or sc_wear != 0.0
        if any_override:
            result_scen = run_ttf_forecast(
                xgb_model=xgb_model,
                scaler=scaler,
                air_temp=float(ttf_air),
                proc_temp=float(ttf_proc),
                rpm=float(ttf_rpm),
                torque=float(ttf_torque),
                tool_wear=float(ttf_wear),
                type_enc=ttf_type_enc,
                active_threshold=ttf_thresh,
                horizon_months=int(ttf_horizon),
                degradation=ttf_deg,
                scenario_overrides={"rpm": sc_rpm, "torque": sc_torque, "tool_wear": sc_wear}
            )
    except Exception as _e:
        import sys
        import traceback
        traceback.print_exc(file=sys.stderr)
        ttf_error = str(_e)

    # ── Right column: display results ─────────────────────────
    with ttf_col_right:
        if ttf_error:
            st.error(f"⚠️ Forecast error: {ttf_error}")
        elif result_base is None:
            st.warning("Adjust parameters on the left to generate a forecast.")
        else:
            fm = result_base.predicted_fail_month
            cur_prob = result_base.probabilities[0] * 100
            end_prob = result_base.probabilities[-1] * 100
            delta_prob = end_prob - cur_prob

            # Live metric cards
            pr1, pr2, pr3 = st.columns(3)
            pr1.markdown(custom_metric_card("Current Risk", f"{cur_prob:.1f}", "%",
                "#ef4444" if cur_prob >= 65 else ("#f59e0b" if cur_prob >= 35 else "#10b981")
            ), unsafe_allow_html=True)
            pr2.markdown(custom_metric_card(f"Risk @ Month {ttf_horizon}", f"{end_prob:.1f}", "%",
                "#ef4444" if end_prob >= 65 else ("#f59e0b" if end_prob >= 35 else "#10b981")
            ), unsafe_allow_html=True)
            if fm is not None:
                pr3.markdown(custom_metric_card("Est. Failure", f"Month {fm}", "Predicted", "#ef4444"), unsafe_allow_html=True)
            else:
                pr3.markdown(custom_metric_card("Est. Failure", "Safe", f"> {ttf_horizon} mo", "#10b981"), unsafe_allow_html=True)

            # --- Status Alert Banner ---
            if fm == 0:
                st.markdown(
                    f'<div class="alert-critical"><h2>🚨 IMMEDIATE INSPECTION REQUIRED 🚨</h2>'
                    f'<p style="color:#eee;margin:6px 0 0;">Machine failure probability is currently at <b>{cur_prob:.1f}%</b>, exceeding the safety threshold.<br/>'
                    f'<b>Maintenance Suggestion:</b> Shut down machine immediately. Swap tools, inspect bearings, and recalibrate load parameters.</p></div>',
                    unsafe_allow_html=True,
                )
            elif fm is not None:
                inv_msg = "Perform general mechanical maintenance and reduce peak torque load."
                if result_base.interventions:
                    top_inv = result_base.interventions[0]
                    inv_msg = f"{top_inv['action']} by {top_inv['adjustment']} ({top_inv['gain_label']})."
                
                alert_class = "alert-critical" if fm <= 3 else "alert-warning"
                alert_icon = "🚨" if fm <= 3 else "⚠️"
                st.markdown(
                    f'<div class="{alert_class}"><h2>{alert_icon} FAILS IN {fm} MONTHS {alert_icon}</h2>'
                    f'<p style="color:#eee;margin:6px 0 0;">Projected to cross safety threshold in Month {fm} under current degradation rates.<br/>'
                    f'<b>Maintenance Suggestion:</b> {inv_msg}</p></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="alert-safe"><h2>🟢 COMPONENT HEALTH STABLE</h2>'
                    f'<p style="color:#eee;margin:6px 0 0;">Machine is predicted safe for the full {ttf_horizon}-month horizon.<br/>'
                    f'<b>Maintenance Suggestion:</b> Continue standard schedule. Periodic sensor monitoring and standard calibration active.</p></div>',
                    unsafe_allow_html=True,
                )

            # Initialize Plotly Figure
            fig_ttf = go.Figure()

            # Confidence band
            fig_ttf.add_trace(go.Scatter(
                x=result_base.months + result_base.months[::-1],
                y=[v*100 for v in result_base.confidence_upper] + [v*100 for v in result_base.confidence_lower[::-1]],
                fill='toself', fillcolor='rgba(59,130,246,0.12)',
                line=dict(color='rgba(255,255,255,0)'), name='Uncertainty Band', showlegend=True
            ))

            # Baseline trajectory
            fig_ttf.add_trace(go.Scatter(
                x=result_base.months, y=[p*100 for p in result_base.probabilities],
                mode='lines+markers', name='Baseline Trajectory',
                line=dict(color='#3b82f6', width=3),
                marker=dict(size=6, color='#3b82f6', line=dict(color='white', width=1))
            ))

            # Scenario overlay
            if result_scen:
                fig_ttf.add_trace(go.Scatter(
                    x=result_scen.months, y=[p*100 for p in result_scen.probabilities],
                    mode='lines+markers', name='What-If Scenario',
                    line=dict(color='#10b981', width=3, dash='dash'),
                    marker=dict(size=6)
                ))

            # TTF threshold line
            fig_ttf.add_hline(
                y=ttf_thresh*100,
                line_dash="dot", line_color="#ef4444", line_width=2,
                annotation_text=f"TTF Alert ({ttf_thresh*100:.0f}%)",
                annotation_position="bottom right"
            )

            # Failure month marker
            if fm and fm > 0:
                fig_ttf.add_vline(
                    x=fm, line_dash="dash", line_color="#f59e0b", line_width=2,
                    annotation_text=f"⚠️ Failure: Month {fm}", annotation_position="top left"
                )
            if result_scen and result_scen.predicted_fail_month:
                sfm = result_scen.predicted_fail_month
                if sfm != fm:
                    fig_ttf.add_vline(
                        x=sfm, line_dash="dot", line_color="#10b981", line_width=2,
                        annotation_text=f"✅ Scenario: Month {sfm}", annotation_position="top right"
                    )

            # Dynamic y_max scaling
            all_probs = result_base.probabilities
            if result_scen:
                all_probs = all_probs + result_scen.probabilities
            max_prob = max(all_probs) * 100
            y_max = max(105.0, max_prob + 10.0)

            fig_ttf.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,39,0.5)",
                font_color="white", height=380,
                margin=dict(l=10, r=10, t=20, b=10),
                xaxis=dict(title="Month", showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                           tickmode='linear', dtick=1),
                yaxis=dict(title="Failure Probability (%)",
                           range=[0, y_max],          # auto-scaled to actual data range
                           showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
            )
            st.plotly_chart(fig_ttf, use_container_width=True)

    if result_base is not None:
        # ── Intervention Recommendations ──────────────────────────
        st.markdown('<div class="section-header">🛠️ Intervention Recommendations — Delay Failure</div>', unsafe_allow_html=True)
        interventions = result_base.interventions
        if interventions:
            int_cols = st.columns(min(len(interventions), 3))
            for idx, inv in enumerate(interventions[:3]):
                col = int_cols[idx % 3]
                gain = inv["gain_months"]
                color = "#10b981" if gain and gain >= 3 else "#f59e0b"
                card_html = f"""
                <div style="background:rgba(17,24,39,0.6); border:1px solid rgba(255,255,255,0.06);
                            border-top:3px solid {color}; border-radius:12px; padding:18px; margin:6px 0;">
                    <p style="margin:0; color:#94a3b8; font-size:0.75rem; text-transform:uppercase; letter-spacing:1.5px; font-weight:600;">Intervention {idx+1}</p>
                    <h3 style="margin:8px 0 4px; color:#fff; font-size:1.1rem;">{inv['action']}</h3>
                    <p style="margin:0; color:#cbd5e1; font-size:0.9rem;">Adjust <b>{inv['parameter']}</b> by <b>{'+' if inv['adjustment']>=0 else ''}{inv['adjustment']}</b> ({'+' if inv['pct_change']>=0 else ''}{inv['pct_change']}%)</p>
                    <p style="margin:8px 0 0; color:{color}; font-size:0.85rem; font-weight:600;">{inv['gain_label']}</p>
                </div>
                """
                col.markdown(card_html, unsafe_allow_html=True)
            if len(interventions) > 3:
                with st.expander(f"Show {len(interventions)-3} more interventions"):
                    for inv in interventions[3:]:
                        st.markdown(f"• **{inv['action']}**: adjust {inv['parameter']} by {inv['adjustment']:+} → {inv['gain_label']}")
        else:
            st.success("✅ No interventions needed — machine is predicted safe within the horizon.")

        # ── Conditional Safe Analysis ──────────────────────────────
        st.markdown('<div class="section-header">🔐 Conditional Safety Analysis</div>', unsafe_allow_html=True)
        cond_result = conditional_safe_forecast(
            xgb_model, scaler,
            ttf_air, ttf_proc, ttf_rpm, ttf_torque, ttf_wear,
            ttf_type_enc, ttf_thresh, ttf_horizon, ttf_deg
        )
        if cond_result["safe_conditions"]:
            st.info(f"Machine will remain safe for {ttf_horizon} months **IF** any of the following conditions hold:")
            for cond in cond_result["safe_conditions"]:
                st.markdown(f'<div class="insight-box">{cond}</div>', unsafe_allow_html=True)
        else:
            st.warning("No single parameter freeze prevents failure within the forecast horizon. Combined interventions are required.")

        # ── Projected Parameters Table ─────────────────────────────
        st.markdown('<div class="section-header">📋 Month-by-Month Projected Parameters</div>', unsafe_allow_html=True)
        proj_df = pd.DataFrame(result_base.projected_params)
        proj_df["prob_pct"] = (proj_df["prob"] * 100).round(1)
        proj_df["Risk"] = proj_df["prob"].apply(
            lambda p: "🔴 CRITICAL" if p >= 0.65 else ("🟡 WARNING" if p >= 0.35 else "🟢 SAFE")
        )
        proj_df = proj_df.rename(columns={
            "month": "Month", "air_temp": "Air Temp (K)", "proc_temp": "Proc Temp (K)",
            "rpm": "RPM", "torque": "Torque (Nm)", "tool_wear": "Wear (min)", "prob_pct": "Failure Prob (%)"
        })
        show_proj = ["Month", "Air Temp (K)", "Proc Temp (K)", "RPM", "Torque (Nm)", "Wear (min)", "Failure Prob (%)", "Risk"]
        st.dataframe(proj_df[show_proj], use_container_width=True, height=320)


# ═══════════════════════════════════════════════════════════
# TAB REGISTRY — PLANT ASSET REGISTRY
# ═══════════════════════════════════════════════════════════
with tab_registry:
    st.markdown('<div class="section-header">🏭 Dynamic Plant Asset Registry</div>', unsafe_allow_html=True)
    st.caption("Register any machine type in your chemical facility. Each asset gets its own TTF forecast and failure mode analysis. Heat exchangers include physics-based per-unit ranking.")

    # Initialize registry
    if "registry" not in st.session_state:
        st.session_state.registry = MachineRegistry(BASE_DIR)
    registry: MachineRegistry = st.session_state.registry

    reg_tab_add, reg_tab_fleet, reg_tab_hx = st.tabs([
        "➕ Add / Edit Asset", "📋 Fleet Overview", "♨️ Heat Exchanger Specialist"
    ])

    # ── Sub-tab: Add Asset ──────────────────────────────────────
    with reg_tab_add:
        st.markdown("#### Register a New Asset")
        ra1, ra2, ra3 = st.columns([1, 1, 1], gap="medium")
        with ra1:
            new_type = st.selectbox("Machine Type", MACHINE_TYPES, key="reg_new_type")
        with ra2:
            new_id   = st.text_input("Asset ID (leave blank for auto)", placeholder="e.g. PUMP-201", key="reg_new_id")
        with ra3:
            new_loc  = st.text_input("Location / Unit", placeholder="e.g. Unit 3 - Reactor Feed", key="reg_new_loc")
        new_desc = st.text_input("Description (optional)", key="reg_new_desc")

        st.markdown("##### 📡 Sensor Readings")
        template = registry.build_default_asset(new_type, new_id.strip() or "", new_loc)
        sensor_fields = SENSOR_FIELDS_BY_TYPE.get(new_type, [])
        sensor_cols = st.columns(min(3, len(sensor_fields)) or 1)
        sensor_vals = {}
        for si, field_name in enumerate(sensor_fields):
            default_val = template["sensors"].get(field_name, 0.0)
            if isinstance(default_val, str):
                sensor_vals[field_name] = sensor_cols[si % len(sensor_cols)].text_input(
                    field_name, value=default_val, key=f"reg_s_{field_name}"
                )
            else:
                sensor_vals[field_name] = sensor_cols[si % len(sensor_cols)].number_input(
                    field_name, value=float(default_val), key=f"reg_s_{field_name}"
                )

        if new_type in ML_COMPATIBLE_TYPES:
            st.markdown("##### 📉 Degradation Rates (per month)")
            dc1, dc2, dc3 = st.columns(3)
            deg_wear  = dc1.number_input("Wear rate (min/mo)",   value=5.0,   step=0.5,  key="reg_deg_wear")
            deg_rpm   = dc2.number_input("RPM drift (RPM/mo)",   value=-10.0, step=5.0,  key="reg_deg_rpm")
            deg_torq  = dc3.number_input("Torque drift (Nm/mo)", value=0.5,   step=0.1,  key="reg_deg_torq")
        else:
            deg_wear, deg_rpm, deg_torq = 5.0, -10.0, 0.5

        if new_type == "Heat Exchanger":
            st.markdown("##### 🔧 HX Design Parameters")
            hx1, hx2, hx3 = st.columns(3)
            hx_design_u   = hx1.number_input("Design U (W/m²K)",   value=1000.0, step=50.0, key="reg_hx_u")
            hx_area        = hx2.number_input("Heat Area (m²)",      value=50.0,  step=5.0,  key="reg_hx_area")
            hx_fluid_type  = hx3.selectbox("Fluid Type (for fouling limit)",
                                            list(FOULING_LIMITS.keys()), index=1, key="reg_hx_fluid")
            sensor_vals["design_U"]    = hx_design_u
            sensor_vals["heat_area_m2"]= hx_area
            sensor_vals["fluid_type"]  = hx_fluid_type

        if st.button("✅ Register Asset", key="reg_add_btn", type="primary"):
            new_asset = template.copy()
            new_asset["id"]          = new_id.strip() or template["id"]
            new_asset["description"] = new_desc
            new_asset["sensors"].update(sensor_vals)
            new_asset["degradation"].update({
                "wear_rate_per_month":    deg_wear,
                "rpm_drift_per_month":    deg_rpm,
                "torque_drift_per_month": deg_torq,
            })
            assigned_id = registry.add_asset(new_asset)
            st.success(f"✅ Asset **{assigned_id}** registered successfully!")
            st.rerun()

        # Edit / Delete existing
        if registry.count() > 0:
            st.markdown("---")
            st.markdown("#### 🗑️ Remove Asset")
            del_id = st.selectbox("Select asset to delete", [a["id"] for a in registry.get_all()], key="reg_del_sel")
            if st.button("Delete Selected Asset", key="reg_del_btn"):
                registry.delete_asset(del_id)
                st.success(f"Asset {del_id} removed.")
                st.rerun()

    # ── Sub-tab: Fleet Overview ─────────────────────────────────
    with reg_tab_fleet:
        st.markdown("#### 📋 Registered Assets")
        all_assets = registry.get_all()

        if not all_assets:
            st.info("No assets registered yet. Use the **Add / Edit Asset** tab to register machines.")
        else:
            # Summary metrics
            total_assets = len(all_assets)
            hx_count     = len(registry.get_hx_units())
            ml_count     = len(registry.get_ml_compatible())

            fm1, fm2, fm3 = st.columns(3)
            fm1.markdown(custom_metric_card("Total Assets", str(total_assets), "units", "#3b82f6"), unsafe_allow_html=True)
            fm2.markdown(custom_metric_card("Heat Exchangers", str(hx_count), "units", "#f59e0b"), unsafe_allow_html=True)
            fm3.markdown(custom_metric_card("ML-Forecast Ready", str(ml_count), "units", "#10b981"), unsafe_allow_html=True)

            # Per-asset cards with TTF
            st.markdown("---")
            horizon_fleet = st.slider("Forecast Horizon for All Assets (months)", 3, 24, 12, 1, key="fleet_horizon")

            for asset in all_assets:
                atype = asset.get("type", "Custom")
                aid   = asset["id"]
                aloc  = asset.get("location", "—")
                adesc = asset.get("description", "")
                sensors = asset.get("sensors", {})
                deg     = asset.get("degradation", {})

                with st.expander(f"{'♨️' if atype=='Heat Exchanger' else '⚙️'} {aid} — {atype} | {aloc}", expanded=False):
                    exp1, exp2 = st.columns([1, 1], gap="medium")

                    with exp1:
                        st.markdown(f"**Description:** {adesc or '—'}")
                        st.markdown(f"**Failure Modes:** {', '.join(asset.get('failure_modes', [])[:3])}")
                        sensor_rows = [f"`{k}` = {v}" for k, v in sensors.items() if not isinstance(v, str)][:6]
                        st.markdown("**Current Readings:**")
                        for row in sensor_rows:
                            st.markdown(f"  • {row}")

                    with exp2:
                        if atype in ML_COMPATIBLE_TYPES:
                            # Run TTF for this asset
                            try:
                                a_air   = float(sensors.get("air_temp",  300.0))
                                a_proc  = float(sensors.get("proc_temp", 310.0))
                                a_rpm   = float(sensors.get("rpm",       1500.0))
                                a_torq  = float(sensors.get("torque",    38.0))
                                a_wear  = float(sensors.get("tool_wear", 80.0))
                                grade   = asset.get("product_grade", "M (Medium)")
                                a_enc   = TYPE_MAP.get(grade, 1)

                                a_result = run_ttf_forecast(
                                    xgb_model, scaler,
                                    a_air, a_proc, a_rpm, a_torq, a_wear,
                                    a_enc, active_thresh, horizon_fleet, deg
                                )
                                afm = a_result.predicted_fail_month
                                if afm == 0:
                                    st.markdown('<div class="alert-critical"><h2>🚨 ALREADY CRITICAL</h2></div>', unsafe_allow_html=True)
                                elif afm is not None:
                                    st.markdown(f'<div class="alert-warning"><h2>⚠️ Fails: Month {afm}</h2><p style="color:#eee;margin:4px 0 0;">If parameters continue current trend.</p></div>', unsafe_allow_html=True)
                                else:
                                    st.markdown(f'<div class="alert-safe"><h2>🟢 Safe for {horizon_fleet} months</h2></div>', unsafe_allow_html=True)

                                # Mini sparkline
                                fig_mini = go.Figure(go.Scatter(
                                    x=a_result.months, y=[p*100 for p in a_result.probabilities],
                                    mode='lines', fill='tozeroy',
                                    line=dict(color='#3b82f6', width=2),
                                    fillcolor='rgba(59,130,246,0.15)'
                                ))
                                fig_mini.add_hline(y=active_thresh*100, line_dash="dot", line_color="#ef4444")
                                fig_mini.update_layout(
                                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,39,0.4)",
                                    height=160, margin=dict(l=5, r=5, t=5, b=5),
                                    xaxis=dict(showgrid=False, showticklabels=True, title="Month"),
                                    yaxis=dict(showgrid=False, title="Risk %", range=[0, 105]),
                                    font=dict(color="white", size=10),
                                    showlegend=False,
                                )
                                st.plotly_chart(fig_mini, use_container_width=True)
                            except Exception as e:
                                st.warning(f"TTF error: {e}")
                        elif atype == "Heat Exchanger":
                            st.info("🔄 Use the **Heat Exchanger Specialist** tab for detailed HX analysis.")
                        else:
                            st.info(f"⚙️ {atype} — sensor monitoring active. TTF forecasting not applicable for this machine type.")

    # ── Sub-tab: Heat Exchanger Specialist ─────────────────────
    with reg_tab_hx:
        st.markdown("#### ♨️ Heat Exchanger Specialist Panel")
        st.caption("Physics-based fouling analysis using effectiveness-NTU method and LMTD. Identifies which specific HX unit requires cleaning or bypass next.")

        hx_assets = registry.get_hx_units()

        # Allow adding quick HX units
        with st.expander("➕ Quick-Add Heat Exchanger Unit", expanded=(len(hx_assets) == 0)):
            st.markdown("**Add a Heat Exchanger unit for analysis:**")
            hqc1, hqc2, hqc3 = st.columns(3)
            hq_id    = hqc1.text_input("Unit ID", placeholder="HX-101", key="hq_id")
            hq_desc  = hqc2.text_input("Description", placeholder="Reactor Feed Preheater", key="hq_desc")
            hq_fluid = hqc3.selectbox("Shell-side fluid", list(FOULING_LIMITS.keys()), index=1, key="hq_fluid")

            hrc1, hrc2, hrc3, hrc4 = st.columns(4)
            hq_s_in  = hrc1.number_input("Shell Temp In (°C)",  value=120.0, key="hq_sin")
            hq_s_out = hrc2.number_input("Shell Temp Out (°C)", value=95.0,  key="hq_sout")
            hq_t_in  = hrc3.number_input("Tube Temp In (°C)",   value=60.0,  key="hq_tin")
            hq_t_out = hrc4.number_input("Tube Temp Out (°C)",  value=80.0,  key="hq_tout")

            hfc1, hfc2, hfc3, hfc4 = st.columns(4)
            hq_sflo  = hfc1.number_input("Shell Flow (kg/s)",   value=10.0, key="hq_sflo")
            hq_tflo  = hfc2.number_input("Tube Flow (kg/s)",    value=12.0, key="hq_tflo")
            hq_u_des = hfc3.number_input("Design U (W/m²K)",    value=1000.0, step=50.0, key="hq_udes")
            hq_area  = hfc4.number_input("Heat Area (m²)",      value=50.0, step=5.0, key="hq_area")

            hq_rf    = st.number_input("Measured Fouling Factor Rf (m²K/W) — enter 0 to auto-calculate", value=0.0, step=0.00005, format="%.6f", key="hq_rf")
            hq_days  = st.number_input("Days since last cleaning", value=0, step=1, key="hq_days")

            if st.button("Add HX Unit", key="hq_add_btn", type="primary"):
                auto_id = hq_id.strip() or f"HX-{101 + len(hx_assets)}"
                new_hx = registry.build_default_asset("Heat Exchanger", auto_id, "")
                new_hx["description"] = hq_desc
                new_hx["sensors"].update({
                    "shell_temp_in":   hq_s_in,  "shell_temp_out":  hq_s_out,
                    "tube_temp_in":    hq_t_in,  "tube_temp_out":   hq_t_out,
                    "shell_flow_kg_s": hq_sflo,  "tube_flow_kg_s":  hq_tflo,
                    "design_U":        hq_u_des, "heat_area_m2":    hq_area,
                    "fluid_type":      hq_fluid, "fouling_factor":  hq_rf,
                    "days_since_last_clean": hq_days,
                })
                registry.add_asset(new_hx)
                st.success(f"✅ {auto_id} added!")
                st.rerun()

        hx_assets = registry.get_hx_units()

        if not hx_assets:
            st.info("No heat exchangers registered yet. Add units using the panel above.")
        else:
            # Build HXUnit objects
            hx_units = []
            for ha in hx_assets:
                s = ha.get("sensors", {})
                hxu = HXUnit(
                    unit_id=ha["id"],
                    description=ha.get("description", ""),
                    fluid_type=str(s.get("fluid_type", "process_liquid")),
                    shell_temp_in=float(s.get("shell_temp_in", 120)),
                    shell_temp_out=float(s.get("shell_temp_out", 95)),
                    shell_flow_kg_s=float(s.get("shell_flow_kg_s", 10)),
                    tube_temp_in=float(s.get("tube_temp_in", 60)),
                    tube_temp_out=float(s.get("tube_temp_out", 80)),
                    tube_flow_kg_s=float(s.get("tube_flow_kg_s", 12)),
                    design_U=float(s.get("design_U", 1000)),
                    heat_area_m2=float(s.get("heat_area_m2", 50)),
                    fouling_factor=float(s.get("fouling_factor", 0)),
                    days_since_last_clean=int(s.get("days_since_last_clean", 0)),
                )
                hx_units.append(hxu)

            report = analyze_hx_fleet(hx_units)

            # Fleet verdict
            if report.most_critical:
                risk = report.most_critical.risk_level
                if risk == "CRITICAL":
                    st.markdown(f'<div class="alert-critical"><h2>🔴 HX FLEET: CRITICAL ALERT</h2><p style="color:#eee">{report.summary_message}</p></div>', unsafe_allow_html=True)
                elif risk == "WARNING":
                    st.markdown(f'<div class="alert-warning"><h2>⚠️ HX FLEET: WARNING</h2><p style="color:#eee">{report.summary_message}</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="alert-safe"><h2>🟢 HX FLEET: ALL HEALTHY</h2><p style="color:#eee">{report.summary_message}</p></div>', unsafe_allow_html=True)

            # Ranked unit table
            st.markdown('<div class="section-header">🏆 Unit Ranking — Most Critical First</div>', unsafe_allow_html=True)

            rank_data = []
            for rk, u in enumerate(report.ranked, 1):
                icon = "🔴" if u.risk_level == "CRITICAL" else ("🟡" if u.risk_level == "WARNING" else "🟢")
                dtc_str = f"{int(u.days_to_critical)} days" if u.days_to_critical is not None else "—"
                rank_data.append({
                    "Rank": rk,
                    "Unit": u.unit_id,
                    "Description": u.description or "—",
                    "Status": f"{icon} {u.risk_level}",
                    "Health Score": f"{u.health_score:.1f} / 100",
                    "Fouling %": f"{u.fouling_pct:.1f}%",
                    "Effectiveness": f"{u.eff_ratio*100:.1f}% of design",
                    "Temp Pinch (°C)": u.temperature_pinch,
                    "Days to Limit": dtc_str,
                    "Heat Duty (kW)": u.heat_duty_kw,
                })
            st.dataframe(pd.DataFrame(rank_data), use_container_width=True, height=300)

            # Per-unit detail cards
            st.markdown('<div class="section-header">🔍 Unit-by-Unit Detailed Diagnostics</div>', unsafe_allow_html=True)
            for u in report.ranked:
                risk_color = {"CRITICAL": "#ef4444", "WARNING": "#f59e0b", "SAFE": "#10b981"}[u.risk_level]
                with st.expander(f"{u.unit_id} — {u.risk_level} | Health: {u.health_score:.1f}/100 | Fouling: {u.fouling_pct:.1f}%", expanded=(u.risk_level == "CRITICAL")):
                    hd1, hd2, hd3, hd4 = st.columns(4)
                    hd1.markdown(custom_metric_card("Health Score",   f"{u.health_score:.1f}", "/ 100",           risk_color), unsafe_allow_html=True)
                    hd2.markdown(custom_metric_card("Fouling Used",   f"{u.fouling_pct:.1f}",  "% of TEMA limit", risk_color), unsafe_allow_html=True)
                    hd3.markdown(custom_metric_card("Effectiveness",  f"{u.eff_ratio*100:.1f}","% of design",     "#3b82f6"), unsafe_allow_html=True)
                    hd4.markdown(custom_metric_card("Temp Pinch",     f"{u.temperature_pinch}","°C",               "#8b5cf6"), unsafe_allow_html=True)

                    hm1, hm2 = st.columns(2)
                    hm1.markdown(custom_metric_card("Heat Duty",      f"{u.heat_duty_kw:.1f}", "kW",   "#14b8a6"), unsafe_allow_html=True)
                    hm2.markdown(custom_metric_card("LMTD",           f"{u.lmtd:.2f}",          "°C",   "#0284c7"), unsafe_allow_html=True)

                    st.markdown("**Diagnostic Messages:**")
                    for d in u.diagnostics:
                        st.markdown(f'<div class="insight-box">{d}</div>', unsafe_allow_html=True)

                    # Fouling progress bar
                    fouling_bar_color = risk_color
                    fouling_pct_capped = min(100, u.fouling_pct)
                    st.markdown(f"""
                    <div style="margin:12px 0 4px; color:#94a3b8; font-size:0.8rem; text-transform:uppercase; letter-spacing:1px;">Fouling Progress to TEMA Limit</div>
                    <div style="background:rgba(255,255,255,0.05); border-radius:8px; height:20px; width:100%; overflow:hidden;">
                        <div style="background:{fouling_bar_color}; height:100%; width:{fouling_pct_capped:.1f}%;
                                    border-radius:8px; transition:width 0.5s ease;
                                    box-shadow:0 0 8px {fouling_bar_color}88;"></div>
                    </div>
                    <div style="color:#64748b; font-size:0.75rem; margin-top:4px;">{u.fouling_pct:.1f}% — Limit: {u.fouling_limit:.2e} m²K/W | Current: {u.fouling_calc:.2e} m²K/W</div>
                    """, unsafe_allow_html=True)

            # HX Comparison Chart
            st.markdown('<div class="section-header">📊 Fleet Comparison — Fouling & Effectiveness</div>', unsafe_allow_html=True)
            hx_chart_df = pd.DataFrame([
                {"Unit": u.unit_id, "Fouling %": u.fouling_pct, "Effectiveness %": u.eff_ratio*100, "Risk": u.risk_level}
                for u in report.ranked
            ])
            color_map = {"CRITICAL": "#ef4444", "WARNING": "#f59e0b", "SAFE": "#10b981"}
            fig_hx = go.Figure()
            for _, row in hx_chart_df.iterrows():
                fig_hx.add_trace(go.Bar(
                    x=[row["Unit"]], y=[row["Fouling %"]],
                    name=row["Unit"],
                    marker_color=color_map.get(row["Risk"], "#3b82f6"),
                    showlegend=False,
                ))
            fig_hx.add_hline(y=100, line_dash="dot", line_color="#ef4444",
                             annotation_text="TEMA Fouling Limit", annotation_position="top right")
            fig_hx.add_hline(y=60, line_dash="dash", line_color="#f59e0b",
                             annotation_text="Warning Threshold (60%)", annotation_position="top left")
            fig_hx.update_layout(
                title="Fouling % of TEMA Limit per HX Unit",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,39,0.5)",
                font_color="white", height=320,
                yaxis=dict(range=[0, 120], title="Fouling % of TEMA Limit"),
                xaxis_title="Heat Exchanger Unit",
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_hx, use_container_width=True)
