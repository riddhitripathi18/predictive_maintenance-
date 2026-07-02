"""
heat_exchanger_analyzer.py
Heat Exchanger (HX) Failure Prediction & Ranking Engine
==========================================================
Uses physics-based methods (effectiveness-NTU, fouling factor, temperature
pinch analysis) to:
  • Score each registered HX unit for current health
  • Rank all units from most-to-least critical
  • Estimate days until mandatory cleaning/bypass/replacement

No ML model required — pure thermal engineering first-principles.

Terminology
-----------
U  : Overall heat transfer coefficient (W/m²K)
A  : Heat transfer area (m²)
Rf : Fouling resistance (m²K/W) — increases over time
Q  : Actual heat duty (kW)
ε  : Effectiveness = Q_actual / Q_max
NTU: Number of Transfer Units = UA / C_min
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# ─────────────────────────────────────────────────────────────
# Typical fouling resistance limits per industry standard
# (TEMA / HEDH standards)
# Beyond these limits, cleaning is mandatory.
# ─────────────────────────────────────────────────────────────
FOULING_LIMITS = {
    "cooling_water":   0.000176,  # m²K/W  (TEMA R)
    "process_liquid":  0.000352,  # m²K/W
    "crude_oil":       0.000528,  # m²K/W
    "steam":           0.000088,  # m²K/W
    "refrigerant":     0.000176,  # m²K/W
    "custom":          0.000300,  # m²K/W  (user-defined default)
}

# Effectiveness degradation thresholds
EFF_WARNING   = 0.80   # Below 80% of design effectiveness → WARNING
EFF_CRITICAL  = 0.65   # Below 65% of design effectiveness → CRITICAL

# Days-to-clean estimation constants
# Assumed linear fouling accumulation unless user provides actual rate
DEFAULT_FOULING_RATE_PER_DAY = 5e-7  # m²K/W per day (conservative)


@dataclass
class HXUnit:
    """Represents a single heat exchanger unit with its current measurements."""
    unit_id:            str            # e.g., "HX-101"
    description:        str = ""       # e.g., "Reactor Feed Preheater"
    fluid_type:         str = "process_liquid"  # key into FOULING_LIMITS

    # Shell side (hot or cold fluid)
    shell_temp_in:      float = 120.0  # °C
    shell_temp_out:     float = 95.0   # °C
    shell_flow_kg_s:    float = 10.0   # kg/s
    shell_cp:           float = 4.18   # kJ/(kg·K) — specific heat

    # Tube side (other fluid)
    tube_temp_in:       float = 60.0   # °C
    tube_temp_out:      float = 80.0   # °C
    tube_flow_kg_s:     float = 12.0   # kg/s
    tube_cp:            float = 4.18   # kJ/(kg·K)

    # Design parameters
    design_U:           float = 1000.0 # W/(m²·K) — design overall HTC
    heat_area_m2:       float = 50.0   # m²
    design_effectiveness: float = 0.0  # auto-calculated if 0

    # Current measured parameters
    current_U:          float = 0.0    # W/(m²·K) — from latest inspection (0 = auto-estimate)
    fouling_factor:     float = 0.0    # m²K/W — measured or estimated (0 = auto-calc)
    days_since_last_clean: int = 0
    fouling_rate_per_day: float = DEFAULT_FOULING_RATE_PER_DAY

    # Computed fields (filled in by analyze())
    effectiveness:      float = field(default=0.0, init=False)
    design_eff:         float = field(default=0.0, init=False)
    eff_ratio:          float = field(default=1.0, init=False)
    heat_duty_kw:       float = field(default=0.0, init=False)
    lmtd:               float = field(default=0.0, init=False)
    current_U_calc:     float = field(default=0.0, init=False)
    fouling_calc:       float = field(default=0.0, init=False)
    fouling_limit:      float = field(default=0.0, init=False)
    fouling_pct:        float = field(default=0.0, init=False)
    days_to_critical:   Optional[float] = field(default=None, init=False)
    risk_level:         str   = field(default="SAFE", init=False)
    health_score:       float = field(default=100.0, init=False)  # 100 = perfect, 0 = failed
    diagnostics:        List[str] = field(default_factory=list, init=False)
    temperature_pinch:  float = field(default=0.0, init=False)


@dataclass
class HXAnalysisReport:
    """Full report for a fleet of HX units."""
    units:           List[HXUnit]
    ranked:          List[HXUnit]   # sorted critical → safe
    most_critical:   Optional[HXUnit]
    summary_message: str


def analyze_hx_unit(unit: HXUnit) -> HXUnit:
    """
    Run full physics analysis on a single HX unit.
    Populates all computed fields in-place and returns the unit.
    """
    diag = []

    # ── 1. Heat duty from shell side ──────────────────────────
    dT_shell = abs(unit.shell_temp_in - unit.shell_temp_out)
    dT_tube  = abs(unit.tube_temp_out - unit.tube_temp_in)
    Q_shell  = unit.shell_flow_kg_s * unit.shell_cp * dT_shell   # kW
    Q_tube   = unit.tube_flow_kg_s  * unit.tube_cp  * dT_tube    # kW
    unit.heat_duty_kw = round((Q_shell + Q_tube) / 2.0, 2)

    # ── 2. LMTD (counter-current assumed) ──────────────────────
    # ΔT₁ = hot_in − cold_out,  ΔT₂ = hot_out − cold_in
    hot_in   = max(unit.shell_temp_in,  unit.tube_temp_in)
    hot_out  = min(unit.shell_temp_out, unit.tube_temp_out)
    cold_in  = min(unit.shell_temp_in,  unit.tube_temp_in)
    cold_out = max(unit.shell_temp_out, unit.tube_temp_out)

    dT1 = hot_in  - cold_out
    dT2 = hot_out - cold_in
    dT1 = max(dT1, 0.01)
    dT2 = max(dT2, 0.01)

    if abs(dT1 - dT2) < 0.001:
        lmtd = dT1
    else:
        lmtd = (dT1 - dT2) / math.log(dT1 / dT2)
    unit.lmtd = round(lmtd, 3)

    # ── 3. Temperature pinch ────────────────────────────────────
    # If pinch < 5°C, HX is effectively at thermal equilibrium → critical
    unit.temperature_pinch = round(min(dT1, dT2), 2)
    if unit.temperature_pinch < 5.0:
        diag.append(f"🔴 TEMPERATURE PINCH: only {unit.temperature_pinch:.1f}°C — thermal crossover imminent.")

    # ── 4. Current U from Q = U·A·LMTD ─────────────────────────
    if lmtd > 0.001 and unit.heat_area_m2 > 0:
        # Q in kW → convert to W for U calculation
        unit.current_U_calc = round((unit.heat_duty_kw * 1000.0) / (unit.heat_area_m2 * lmtd), 2)
    else:
        unit.current_U_calc = unit.design_U

    # Use measured U if provided, else use calculated
    U_actual = unit.current_U if unit.current_U > 0 else unit.current_U_calc

    # ── 5. Fouling factor: Rf = 1/U_actual − 1/U_design ────────
    if unit.fouling_factor > 0:
        Rf = unit.fouling_factor
    elif unit.design_U > 0 and U_actual > 0:
        Rf = max(0.0, (1.0 / U_actual) - (1.0 / unit.design_U))
    else:
        Rf = 0.0
    unit.fouling_calc = round(Rf, 8)

    # ── 6. Fouling limit & percentage ───────────────────────────
    Rf_limit = FOULING_LIMITS.get(unit.fluid_type, FOULING_LIMITS["custom"])
    unit.fouling_limit = Rf_limit
    unit.fouling_pct   = round(min(100.0, (Rf / Rf_limit) * 100.0), 1) if Rf_limit > 0 else 0.0

    # ── 7. Effectiveness ─────────────────────────────────────────
    C_shell  = unit.shell_flow_kg_s * unit.shell_cp * 1000   # W/K
    C_tube   = unit.tube_flow_kg_s  * unit.tube_cp  * 1000   # W/K
    C_min    = min(C_shell, C_tube)
    C_max    = max(C_shell, C_tube)
    T_hot_in = max(unit.shell_temp_in, unit.tube_temp_in)
    T_cold_in= min(unit.shell_temp_in, unit.tube_temp_in)
    Q_max    = C_min * (T_hot_in - T_cold_in)

    Q_actual_W = unit.heat_duty_kw * 1000.0
    unit.effectiveness = round(Q_actual_W / Q_max, 4) if Q_max > 0 else 0.0

    # Design effectiveness (using design U)
    if C_min > 0 and C_max > 0:
        NTU_design = (unit.design_U * unit.heat_area_m2) / C_min
        Cr = C_min / C_max
        # Counter-current effectiveness formula
        if abs(Cr - 1.0) < 0.001:
            eff_design = NTU_design / (1.0 + NTU_design)
        else:
            num = 1.0 - math.exp(-NTU_design * (1.0 - Cr))
            den = 1.0 - Cr * math.exp(-NTU_design * (1.0 - Cr))
            eff_design = num / den if den != 0 else 0.0
        unit.design_eff = round(eff_design, 4)
    else:
        unit.design_eff = unit.design_effectiveness or 0.85

    unit.eff_ratio = round(unit.effectiveness / unit.design_eff, 4) if unit.design_eff > 0 else 1.0

    # ── 8. Days to critical ──────────────────────────────────────
    Rf_remaining = max(0.0, Rf_limit - Rf)
    if unit.fouling_rate_per_day > 0 and Rf_remaining > 0:
        unit.days_to_critical = round(Rf_remaining / unit.fouling_rate_per_day)
    elif Rf >= Rf_limit:
        unit.days_to_critical = 0
    else:
        unit.days_to_critical = None  # can't estimate

    # ── 9. Risk classification ────────────────────────────────────
    score = 100.0
    score -= unit.fouling_pct * 0.4          # fouling contributes up to 40 pts
    score -= max(0, (1.0 - unit.eff_ratio) * 100) * 0.4  # effectiveness drop up to 40 pts
    if unit.temperature_pinch < 10:
        score -= (10 - unit.temperature_pinch) * 2       # pinch penalty up to 20 pts
    unit.health_score = round(max(0.0, min(100.0, score)), 1)

    if unit.fouling_pct >= 90 or unit.eff_ratio < EFF_CRITICAL or unit.temperature_pinch < 5:
        unit.risk_level = "CRITICAL"
        diag.append(f"🔴 CRITICAL: Fouling at {unit.fouling_pct:.0f}% of limit. Effectiveness at {unit.eff_ratio*100:.0f}% of design.")
    elif unit.fouling_pct >= 60 or unit.eff_ratio < EFF_WARNING or (unit.days_to_critical and unit.days_to_critical < 30):
        unit.risk_level = "WARNING"
        diag.append(f"⚠️ WARNING: Fouling at {unit.fouling_pct:.0f}% of limit. Consider scheduling cleaning.")
    else:
        unit.risk_level = "SAFE"
        diag.append(f"✅ HEALTHY: Fouling at {unit.fouling_pct:.0f}% of limit. Effectiveness maintained at {unit.eff_ratio*100:.0f}% of design.")

    if unit.days_to_critical == 0:
        diag.append("🚨 FOULING LIMIT EXCEEDED — Mandatory bypass/cleaning required NOW.")
    elif unit.days_to_critical is not None:
        diag.append(f"📅 Estimated {unit.days_to_critical} days until fouling limit is reached.")

    if unit.fouling_pct > 30:
        diag.append(f"🔧 Calculated fouling resistance: {Rf:.2e} m²K/W (Limit: {Rf_limit:.2e} m²K/W).")

    unit.diagnostics = diag
    return unit


def analyze_hx_fleet(units: List[HXUnit]) -> HXAnalysisReport:
    """
    Analyze all HX units and produce a ranked fleet report.
    """
    analyzed = [analyze_hx_unit(u) for u in units]

    # Sort: CRITICAL first, then by health score ascending (worst first)
    risk_order = {"CRITICAL": 0, "WARNING": 1, "SAFE": 2}
    ranked = sorted(analyzed, key=lambda u: (risk_order.get(u.risk_level, 3), u.health_score))

    most_critical = ranked[0] if ranked else None

    if most_critical:
        if most_critical.risk_level == "CRITICAL":
            msg = (
                f"🔴 **{most_critical.unit_id}** requires IMMEDIATE attention — "
                f"fouling at {most_critical.fouling_pct:.0f}% of limit, "
                f"effectiveness at {most_critical.eff_ratio*100:.0f}% of design."
            )
        elif most_critical.risk_level == "WARNING":
            dtc = most_critical.days_to_critical
            msg = (
                f"⚠️ **{most_critical.unit_id}** is the most critical unit — "
                + (f"estimated {dtc} days to fouling limit." if dtc else "schedule inspection.")
            )
        else:
            msg = f"✅ All {len(analyzed)} HX units are operating within healthy parameters."
    else:
        msg = "No heat exchanger units registered."

    return HXAnalysisReport(
        units=analyzed,
        ranked=ranked,
        most_critical=most_critical,
        summary_message=msg,
    )
