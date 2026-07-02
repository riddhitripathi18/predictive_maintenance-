"""
ttf_engine.py
Time-to-Failure (TTF) Forecasting Engine
==========================================
Simulates future sensor states month-by-month using linear drift rates,
runs each projected state through the trained XGBoost model, and locates
the first month where failure probability crosses the active alert threshold.

DESIGN NOTE: _compute_interventions does NOT call run_ttf_forecast.
It runs its own minimal forward-simulation loop to avoid any recursion.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ─────────────────────────────────────────────────────────────
# Default degradation rates per month
# ─────────────────────────────────────────────────────────────
DEFAULT_DEGRADATION = {
    "wear_rate_per_month":       12.0,   # min/month
    "rpm_drift_per_month":      -25.0,   # RPM/month (negative = slowing down)
    "torque_drift_per_month":    1.2,    # Nm/month (positive = creeping up)
    "air_temp_drift_per_month":  0.05,   # K/month
    "proc_temp_drift_per_month": 0.1,    # K/month
}


@dataclass
class TTFResult:
    """Full TTF forecast result for a single machine."""
    months:               List[int]
    probabilities:        List[float]
    predicted_fail_month: Optional[int]
    threshold:            float
    current_params:       Dict
    projected_params:     List[Dict]
    fail_message:         str
    confidence_upper:     List[float]
    confidence_lower:     List[float]
    interventions:        List[Dict]


# ─────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────

def _engineer_features(air_temp, proc_temp, rpm, torque, tool_wear, type_enc):
    temp_delta    = proc_temp - air_temp
    power_W       = torque * rpm * (2 * np.pi / 60)
    torque_x_wear = torque * tool_wear
    wear_pct      = tool_wear / 253.0
    high_torque   = 1 if torque > 55 else 0
    high_wear     = 1 if tool_wear > 200 else 0
    return [type_enc, air_temp, proc_temp, rpm, torque, tool_wear,
            temp_delta, power_W, torque_x_wear, wear_pct, high_torque, high_wear]


def _clamp(air_temp, proc_temp, rpm, torque, tool_wear):
    air_temp  = max(293.0, min(310.0, air_temp))
    proc_temp = max(air_temp + 5.0, min(320.0, proc_temp))
    rpm       = max(800.0,  min(3500.0, rpm))
    torque    = max(1.0,    min(90.0,   torque))
    tool_wear = max(0.0,    min(300.0,  tool_wear))
    return air_temp, proc_temp, rpm, torque, tool_wear


def _simulate_trajectory(xgb_model, scaler, air0, proc0, rpm0, torq0, wear0,
                          type_enc, deg, horizon):
    """
    Runs the forward simulation and returns a list of (month, prob) tuples.
    This is a pure data function — no recursion, no side effects.
    """
    results = []
    for m in range(horizon + 1):
        a = air0  + deg["air_temp_drift_per_month"]  * m
        p = proc0 + deg["proc_temp_drift_per_month"] * m
        r = rpm0  + deg["rpm_drift_per_month"]       * m
        t = torq0 + deg["torque_drift_per_month"]    * m
        w = wear0 + deg["wear_rate_per_month"]        * m
        a, p, r, t, w = _clamp(a, p, r, t, w)
        feats = _engineer_features(a, p, r, t, w, type_enc)
        X_s   = scaler.transform(np.array([feats]))
        prob  = float(xgb_model.predict_proba(X_s)[0, 1])
        results.append((m, prob, {"month": m, "air_temp": round(a,2), "proc_temp": round(p,2),
                                   "rpm": round(r,1), "torque": round(t,2),
                                   "tool_wear": round(w,1), "prob": round(prob,4)}))
    return results


def _find_fail_month(trajectory, threshold):
    for m, prob, _ in trajectory:
        if prob >= threshold:
            return m
    return None


def _compute_interventions(xgb_model, scaler, air0, proc0, rpm0, torq0, wear0,
                            type_enc, threshold, horizon, deg, original_fail_month):
    """
    Compute intervention recommendations WITHOUT calling run_ttf_forecast.
    Uses _simulate_trajectory directly to avoid any recursion.
    """
    if original_fail_month is None:
        return []

    interventions = []

    param_scenarios = [
        ("Reduce Torque",              "torque_drift_per_month",  None,  "torque",   [-5, -10, -15, -20]),
        ("Increase RPM",               "rpm_drift_per_month",     None,  "rpm",      [+100, +200, +300]),
        ("Reduce Tool Wear (early swap)", None,                   "wear_rate_per_month", "tool_wear", [-4, -8, -12]),
        ("Reduce Process Temp",        "proc_temp_drift_per_month", None, "proc_temp", [-0.05, -0.1]),
    ]

    for label, drift_key, rate_key, param_name, adjustments in param_scenarios:
        for adj in adjustments:
            # Build modified degradation dict
            test_deg = dict(deg)
            if drift_key:
                test_deg[drift_key] = deg.get(drift_key, 0.0) + adj
            elif rate_key:
                test_deg[rate_key] = max(0.0, deg.get(rate_key, 0.0) + adj)

            traj = _simulate_trajectory(xgb_model, scaler,
                                        air0, proc0, rpm0, torq0, wear0,
                                        type_enc, test_deg, horizon)
            new_fail = _find_fail_month(traj, threshold)

            if new_fail is None:
                gain = horizon - (original_fail_month or 0)
                gain_label = "Prevents failure entirely within horizon"
            elif new_fail > (original_fail_month or 0):
                gain = new_fail - original_fail_month
                gain_label = f"Delays failure by {gain} month{'s' if gain != 1 else ''}"
            else:
                continue  # makes things worse

            base_val = abs(deg.get(drift_key or rate_key, 1.0)) or 1.0
            pct = (adj / base_val) * 100 if base_val != 0 else 0.0

            interventions.append({
                "action":         label,
                "parameter":      param_name,
                "adjustment":     adj,
                "pct_change":     round(pct, 1),
                "new_fail_month": new_fail,
                "gain_months":    gain,
                "gain_label":     gain_label,
            })
            break  # smallest effective adjustment per parameter

    interventions.sort(key=lambda x: x["gain_months"], reverse=True)
    return interventions[:5]


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def run_ttf_forecast(
    xgb_model,
    scaler,
    air_temp:         float,
    proc_temp:        float,
    rpm:              float,
    torque:           float,
    tool_wear:        float,
    type_enc:         int,
    active_threshold: float,
    horizon_months:   int = 24,
    degradation:      Optional[Dict] = None,
    scenario_overrides: Optional[Dict] = None,
) -> TTFResult:
    """
    Run a TTF forecast. No recursion — uses _simulate_trajectory internally.
    """
    deg = {**DEFAULT_DEGRADATION, **(degradation or {})}

    # Apply scenario overrides to starting state
    s_air  = air_temp  + (scenario_overrides or {}).get("air_temp",  0.0)
    s_proc = proc_temp + (scenario_overrides or {}).get("proc_temp", 0.0)
    s_rpm  = rpm       + (scenario_overrides or {}).get("rpm",       0.0)
    s_torq = torque    + (scenario_overrides or {}).get("torque",    0.0)
    s_wear = tool_wear + (scenario_overrides or {}).get("tool_wear", 0.0)

    traj = _simulate_trajectory(xgb_model, scaler,
                                s_air, s_proc, s_rpm, s_torq, s_wear,
                                type_enc, deg, horizon_months)

    months_list  = [r[0] for r in traj]
    probs_list   = [r[1] for r in traj]
    params_list  = [r[2] for r in traj]
    probs_upper  = [min(1.0, p * 1.10) for p in probs_list]
    probs_lower  = [max(0.0, p * 0.90) for p in probs_list]

    predicted_fail = _find_fail_month(traj, active_threshold)

    # Human-readable verdict
    if predicted_fail == 0:
        fail_msg = "⚠️ Machine is ALREADY above threshold. Immediate inspection required."
    elif predicted_fail is not None:
        fail_msg = (
            f"🔴 Predicted failure in Month {predicted_fail} "
            f"({predicted_fail} month{'s' if predicted_fail != 1 else ''} from now) "
            f"if current parameter trends continue unchanged."
        )
    else:
        fail_msg = (
            f"🟢 Machine is predicted SAFE for the full {horizon_months}-month "
            f"horizon under current degradation trajectory."
        )

    # Intervention recommendations (no recursion — uses _simulate_trajectory directly)
    interventions = _compute_interventions(
        xgb_model, scaler,
        s_air, s_proc, s_rpm, s_torq, s_wear,
        type_enc, active_threshold, horizon_months, deg, predicted_fail
    )

    return TTFResult(
        months=months_list,
        probabilities=probs_list,
        predicted_fail_month=predicted_fail,
        threshold=active_threshold,
        current_params={
            "air_temp":  round(air_temp, 2),
            "proc_temp": round(proc_temp, 2),
            "rpm":       round(rpm, 1),
            "torque":    round(torque, 2),
            "tool_wear": round(tool_wear, 1),
        },
        projected_params=params_list,
        fail_message=fail_msg,
        confidence_upper=probs_upper,
        confidence_lower=probs_lower,
        interventions=interventions,
    )


def conditional_safe_forecast(
    xgb_model, scaler,
    air_temp, proc_temp, rpm, torque, tool_wear,
    type_enc, active_threshold, horizon_months,
    degradation=None,
) -> Dict:
    """
    Returns which parameters, if held constant (zero drift), keep the
    machine safe for the full horizon. No recursion.
    """
    deg = {**DEFAULT_DEGRADATION, **(degradation or {})}
    conditions = []

    param_tests = [
        ("Torque",    "torque_drift_per_month"),
        ("RPM",       "rpm_drift_per_month"),
        ("Tool Wear", "wear_rate_per_month"),
    ]
    for label, drift_key in param_tests:
        test_deg = {**deg, drift_key: 0.0}
        traj = _simulate_trajectory(xgb_model, scaler,
                                    air_temp, proc_temp, rpm, torque, tool_wear,
                                    type_enc, test_deg, horizon_months)
        if _find_fail_month(traj, active_threshold) is None:
            conditions.append(f"✅ **{label}** stays constant (no further degradation)")

    return {"safe_conditions": conditions, "horizon": horizon_months}
