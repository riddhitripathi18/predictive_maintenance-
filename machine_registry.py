"""
machine_registry.py
Dynamic Plant Asset Registry
==============================
Manages a JSON-persisted registry of all plant assets (pumps, compressors,
agitators, heat exchangers, reactors, etc.).

Each asset stores:
  • Identity: id, type, location, description
  • Current sensor readings
  • Degradation rates (used by TTF engine)
  • Asset-type-specific metadata (e.g., HX parameters)

Persistence: machine_registry.json in the project directory.
"""

import json
import os
from typing import Dict, List, Optional
from datetime import datetime

# Supported machine types
MACHINE_TYPES = [
    "Centrifugal Pump",
    "Reciprocating Pump",
    "Compressor",
    "Heat Exchanger",
    "Agitator / Mixer",
    "Reactor",
    "Blower / Fan",
    "Conveyor",
    "Turbine",
    "Motor / Drive",
    "Valve Actuator",
    "Custom",
]

# Failure modes per machine type
FAILURE_MODES_BY_TYPE = {
    "Centrifugal Pump":    ["Bearing Failure", "Seal Leak", "Cavitation", "Impeller Wear", "Overheating"],
    "Reciprocating Pump":  ["Valve Failure", "Piston Ring Wear", "Rod Seal Leak", "Crankshaft Damage"],
    "Compressor":          ["Bearing Failure", "Valve Failure", "Overheating", "Surge", "Seal Degradation"],
    "Heat Exchanger":      ["Tube Fouling", "Shell Fouling", "Tube Leak", "Tube Vibration", "Bypass Failure"],
    "Agitator / Mixer":    ["Seal Failure", "Bearing Damage", "Shaft Deflection", "Blade Wear", "Motor Overload"],
    "Reactor":             ["Cooling Jacket Failure", "Agitator Failure", "Temperature Runaway", "Pressure Buildup"],
    "Blower / Fan":        ["Blade Erosion", "Bearing Failure", "Motor Overload", "Surge"],
    "Conveyor":            ["Belt Wear", "Roller Failure", "Motor Overload", "Misalignment"],
    "Turbine":             ["Blade Erosion", "Bearing Failure", "Seal Degradation", "Vibration"],
    "Motor / Drive":       ["Winding Insulation Failure", "Bearing Failure", "Overheating", "Voltage Unbalance"],
    "Valve Actuator":      ["Actuator Failure", "Stem Seal Leak", "Spring Fatigue", "Positioner Drift"],
    "Custom":              ["Tool Wear Failure", "Heat Dissipation Failure", "Power Failure", "Overstrain Failure", "Random Fault"],
}

# Default sensor fields per machine type
SENSOR_FIELDS_BY_TYPE = {
    "Centrifugal Pump":    ["rpm", "torque", "discharge_pressure_bar", "suction_pressure_bar", "bearing_temp_c", "vibration_mm_s"],
    "Reciprocating Pump":  ["rpm", "torque", "discharge_pressure_bar", "crank_temp_c", "flow_lpm"],
    "Compressor":          ["rpm", "torque", "discharge_pressure_bar", "suction_pressure_bar", "discharge_temp_c", "vibration_mm_s"],
    "Heat Exchanger":      ["shell_temp_in", "shell_temp_out", "tube_temp_in", "tube_temp_out", "shell_flow_kg_s", "tube_flow_kg_s"],
    "Agitator / Mixer":    ["rpm", "torque", "power_kw", "bearing_temp_c", "vibration_mm_s"],
    "Reactor":             ["rpm", "torque", "jacket_temp_in_c", "jacket_temp_out_c", "reactor_temp_c", "pressure_bar"],
    "Blower / Fan":        ["rpm", "torque", "inlet_pressure_pa", "outlet_pressure_pa", "vibration_mm_s"],
    "Conveyor":            ["belt_speed_mps", "motor_current_a", "bearing_temp_c", "tension_n"],
    "Turbine":             ["rpm", "torque", "inlet_temp_c", "outlet_temp_c", "vibration_mm_s"],
    "Motor / Drive":       ["rpm", "current_a", "voltage_v", "bearing_temp_c", "power_kw"],
    "Valve Actuator":      ["position_pct", "supply_pressure_bar", "stem_temp_c", "actuator_current_a"],
    "Custom":              ["rpm", "torque", "air_temp", "proc_temp", "tool_wear"],
}

# Which machine types use the primary ML model (need rpm, torque, air_temp, proc_temp, tool_wear)
ML_COMPATIBLE_TYPES = {
    "Centrifugal Pump",
    "Reciprocating Pump",
    "Compressor",
    "Agitator / Mixer",
    "Motor / Drive",
    "Custom",
}

REGISTRY_FILENAME = "machine_registry.json"


class MachineRegistry:
    """
    In-memory + JSON-persisted registry of plant assets.
    """

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.registry_path = os.path.join(base_dir, REGISTRY_FILENAME)
        self.assets: List[Dict] = []
        self._load()

    # ──────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────
    def _load(self):
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, "r") as f:
                    data = json.load(f)
                    self.assets = data.get("assets", [])
            except (json.JSONDecodeError, IOError):
                self.assets = []

    def _save(self):
        with open(self.registry_path, "w") as f:
            json.dump({"assets": self.assets, "last_updated": datetime.now().isoformat()}, f, indent=2)

    # ──────────────────────────────────────────────────────────
    # CRUD
    # ──────────────────────────────────────────────────────────
    def add_asset(self, asset_data: Dict) -> str:
        """Add a new asset. Returns the assigned ID."""
        asset_id = asset_data.get("id") or self._generate_id(asset_data.get("type", "ASSET"))
        asset_data["id"] = asset_id
        asset_data["created_at"] = datetime.now().isoformat()
        asset_data["updated_at"] = datetime.now().isoformat()
        self.assets.append(asset_data)
        self._save()
        return asset_id

    def update_asset(self, asset_id: str, updates: Dict):
        """Update fields on an existing asset."""
        for i, a in enumerate(self.assets):
            if a["id"] == asset_id:
                self.assets[i].update(updates)
                self.assets[i]["updated_at"] = datetime.now().isoformat()
                self._save()
                return True
        return False

    def delete_asset(self, asset_id: str) -> bool:
        before = len(self.assets)
        self.assets = [a for a in self.assets if a["id"] != asset_id]
        if len(self.assets) < before:
            self._save()
            return True
        return False

    def get_asset(self, asset_id: str) -> Optional[Dict]:
        for a in self.assets:
            if a["id"] == asset_id:
                return a
        return None

    def get_all(self) -> List[Dict]:
        return self.assets

    def get_by_type(self, machine_type: str) -> List[Dict]:
        return [a for a in self.assets if a.get("type") == machine_type]

    def get_hx_units(self) -> List[Dict]:
        return self.get_by_type("Heat Exchanger")

    def get_ml_compatible(self) -> List[Dict]:
        return [a for a in self.assets if a.get("type") in ML_COMPATIBLE_TYPES]

    def count(self) -> int:
        return len(self.assets)

    # ──────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────
    def _generate_id(self, machine_type: str) -> str:
        prefix_map = {
            "Centrifugal Pump":   "PUMP",
            "Reciprocating Pump": "RPUMP",
            "Compressor":         "COMP",
            "Heat Exchanger":     "HX",
            "Agitator / Mixer":   "AGT",
            "Reactor":            "RXTR",
            "Blower / Fan":       "BLW",
            "Conveyor":           "CONV",
            "Turbine":            "TURB",
            "Motor / Drive":      "MOT",
            "Valve Actuator":     "VLV",
            "Custom":             "ASSET",
        }
        prefix = prefix_map.get(machine_type, "ASSET")
        existing_ids = [a["id"] for a in self.assets if a["id"].startswith(prefix)]
        nums = []
        for eid in existing_ids:
            try:
                nums.append(int(eid.replace(prefix + "-", "")))
            except ValueError:
                pass
        next_num = (max(nums) + 1) if nums else 101
        return f"{prefix}-{next_num}"

    def build_default_asset(self, machine_type: str, asset_id: str = "", location: str = "") -> Dict:
        """Build a template asset dict with default values for a given machine type."""
        default_sensors = {field: 0.0 for field in SENSOR_FIELDS_BY_TYPE.get(machine_type, [])}
        
        # Provide sensible defaults for ML-compatible types
        if machine_type in ML_COMPATIBLE_TYPES:
            default_sensors.update({
                "rpm":       1500.0,
                "torque":    38.0,
                "air_temp":  300.0,
                "proc_temp": 310.0,
                "tool_wear": 80.0,
            })

        if machine_type == "Heat Exchanger":
            default_sensors.update({
                "shell_temp_in":  120.0,
                "shell_temp_out": 95.0,
                "tube_temp_in":   60.0,
                "tube_temp_out":  80.0,
                "shell_flow_kg_s": 10.0,
                "tube_flow_kg_s":  12.0,
                "design_U":        1000.0,
                "heat_area_m2":    50.0,
                "fluid_type":      "process_liquid",
                "fouling_factor":  0.0,
                "days_since_last_clean": 0,
            })

        return {
            "id":          asset_id or self._generate_id(machine_type),
            "type":        machine_type,
            "location":    location,
            "description": "",
            "product_grade": "M (Medium)",
            "sensors":     default_sensors,
            "degradation": {
                "wear_rate_per_month":        5.0,
                "rpm_drift_per_month":       -10.0,
                "torque_drift_per_month":     0.5,
                "air_temp_drift_per_month":   0.05,
                "proc_temp_drift_per_month":  0.1,
            },
            "failure_modes": FAILURE_MODES_BY_TYPE.get(machine_type, []),
            "created_at":  datetime.now().isoformat(),
            "updated_at":  datetime.now().isoformat(),
        }
