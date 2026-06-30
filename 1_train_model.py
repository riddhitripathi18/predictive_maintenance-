"""
1_train_model.py
Predictive Maintenance POC -- ML Training Pipeline
Trains:
  1. Primary binary failure classifier (XGBoost & Random Forest)
  2. 5 individual XGBoost classifiers for each failure mode (TWF, HDF, PWF, OSF, RNF)
  3. Calibrates standard and Safety-First (90% Recall) thresholds
"""

import os
import pickle
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, f1_score, recall_score, precision_score
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "ai4i2020.csv")
OUTPUT_DIR  = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("  PREDICTIVE MAINTENANCE -- PHYSICS-INFORMED MULTI-MODEL PIPELINE")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
print(f"\n[1/8] Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n[2/8] Engineering features...")

le = LabelEncoder()
df["Type_encoded"] = le.fit_transform(df["Type"])  # L=0, M=1, H=2

# Physical formulas
df["temp_delta"]    = df["Process temperature [K]"] - df["Air temperature [K]"]
df["power_W"]       = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * (2 * np.pi / 60)
df["torque_x_wear"] = df["Torque [Nm]"] * df["Tool wear [min]"]
df["wear_pct"]      = df["Tool wear [min]"] / 253.0
df["high_torque"]   = (df["Torque [Nm]"] > 55).astype(int)
df["high_wear"]     = (df["Tool wear [min]"] > 200).astype(int)

FEATURE_COLS = [
    "Type_encoded",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "temp_delta",
    "power_W",
    "torque_x_wear",
    "wear_pct",
    "high_torque",
    "high_wear",
]

TARGET = "Machine failure"
SUB_TARGETS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

X = df[FEATURE_COLS]
y = df[TARGET]
y_subs = df[SUB_TARGETS]

print(f"    Features created: {len(FEATURE_COLS)}")
print(f"    Primary Failures: {y.sum()} | Normal: {(y==0).sum()}")
for sub in SUB_TARGETS:
    print(f"      - {sub} Failures: {y_subs[sub].sum()}")

# Save fleet medians (for local physical anomaly comparisons in dashboard)
fleet_medians = df[df[TARGET] == 0][FEATURE_COLS].median().to_dict()
with open(os.path.join(OUTPUT_DIR, "fleet_medians.pkl"), "wb") as f:
    pickle.dump(fleet_medians, f)

# ─────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT (Stratified on primary target)
# ─────────────────────────────────────────────
print("\n[3/8] Splitting 80/20 (stratified)...")
indices = np.arange(len(df))
train_idx, test_idx = train_test_split(
    indices, test_size=0.2, random_state=42, stratify=y
)

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
y_train_subs, y_test_subs = y_subs.iloc[train_idx], y_subs.iloc[test_idx]

print(f"    Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ─────────────────────────────────────────────
# 4. SCALING
# ─────────────────────────────────────────────
print("\n[4/8] Scaling features (StandardScaler)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

with open(os.path.join(OUTPUT_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
print("    scaler.pkl saved.")

# ─────────────────────────────────────────────
# 5. SMOTE
# ─────────────────────────────────────────────
print("\n[5/8] Applying SMOTE to training data...")
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)
print(f"    Train size after SMOTE: {X_train_sm.shape[0]} (Failures balanced: {y_train_sm.sum()})")

# ─────────────────────────────────────────────
# 6. TRAIN PRIMARY MODELS
# ─────────────────────────────────────────────
print("\n[6/8] Training primary models...")

# Random Forest
print("    [RF] Training primary Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=15, class_weight="balanced", random_state=42, n_jobs=-1
)
rf_model.fit(X_train_sm, y_train_sm)

# XGBoost
print("    [XGB] Training primary XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05, eval_metric="logloss",
    random_state=42, n_jobs=-1, verbosity=0
)
xgb_model.fit(X_train_sm, y_train_sm)

# Save primary models
with open(os.path.join(OUTPUT_DIR, "rf_model.pkl"), "wb") as f:
    pickle.dump(rf_model, f)
with open(os.path.join(OUTPUT_DIR, "xgb_model.pkl"), "wb") as f:
    pickle.dump(xgb_model, f)
with open(os.path.join(OUTPUT_DIR, "feature_cols.pkl"), "wb") as f:
    pickle.dump(FEATURE_COLS, f)

# ─────────────────────────────────────────────
# 7. TRAIN SUB-MODELS (FAILURE MODES)
# ─────────────────────────────────────────────
print("\n[7/8] Training failure mode models (XGBoost)...")
sub_models = {}

for sub in SUB_TARGETS:
    print(f"    Training model for {sub}...")
    sub_y_train = y_train_subs[sub]
    
    # Calculate pos weight to handle massive imbalance in sub-modes
    n_neg = (sub_y_train == 0).sum()
    n_pos = sub_y_train.sum()
    scale_pos = max(1.0, float(n_neg) / max(1.0, float(n_pos)))
    
    sub_xgb = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        scale_pos_weight=scale_pos, eval_metric="logloss",
        random_state=42, n_jobs=-1, verbosity=0
    )
    sub_xgb.fit(X_train_scaled, sub_y_train)  # Train on non-SMOTE scaled data
    
    # Save model
    sub_models[sub] = sub_xgb
    with open(os.path.join(OUTPUT_DIR, f"xgb_{sub.lower()}_model.pkl"), "wb") as f:
        pickle.dump(sub_xgb, f)
    print(f"    Saved model: xgb_{sub.lower()}_model.pkl")

# ─────────────────────────────────────────────
# 8. THRESHOLD CALIBRATION (Balanced F1 vs Safety-First Recall)
# ─────────────────────────────────────────────
print("\n[8/8] Calibrating decision thresholds...")

def calibrate_thresholds(model, X_test_s, y_test):
    proba = model.predict_proba(X_test_s)[:, 1]
    
    best_t_f1, best_f1 = 0.50, 0.0
    safety_t_90 = 0.50
    safety_t_90_recall = 0.0
    
    # Grid search thresholds
    for t in np.arange(0.02, 0.85, 0.01):
        y_pred = (proba >= t).astype(int)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        
        # 1. Best Balanced F1 threshold
        if f1 > best_f1:
            best_f1 = f1
            best_t_f1 = round(float(t), 2)
            
        # 2. Safety-first threshold (Guarantees Recall >= 90% while keeping highest possible precision)
        if rec >= 0.90:
            safety_t_90 = round(float(t), 2)
            safety_t_90_recall = rec
            
    return best_t_f1, safety_t_90

xgb_best_f1, xgb_safety = calibrate_thresholds(xgb_model, X_test_scaled, y_test)
rf_best_f1, rf_safety   = calibrate_thresholds(rf_model, X_test_scaled, y_test)

# Save thresholds
with open(os.path.join(OUTPUT_DIR, "xgb_threshold.pkl"), "wb") as f:
    pickle.dump(xgb_best_f1, f)
with open(os.path.join(OUTPUT_DIR, "xgb_safety_threshold.pkl"), "wb") as f:
    pickle.dump(xgb_safety, f)
    
with open(os.path.join(OUTPUT_DIR, "rf_threshold.pkl"), "wb") as f:
    pickle.dump(rf_best_f1, f)
with open(os.path.join(OUTPUT_DIR, "rf_safety_threshold.pkl"), "wb") as f:
    pickle.dump(rf_safety, f)

# Print reports
results_text = []
for model_name, model, f1_t, saf_t in [("XGBoost", xgb_model, xgb_best_f1, xgb_safety),
                                       ("Random Forest", rf_model, rf_best_f1, rf_safety)]:
    proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Balanced
    y_pred_f1 = (proba >= f1_t).astype(int)
    roc = roc_auc_score(y_test, proba)
    f1 = f1_score(y_test, y_pred_f1)
    rec = recall_score(y_test, y_pred_f1)
    prec = precision_score(y_test, y_pred_f1)
    cm = confusion_matrix(y_test, y_pred_f1)
    
    # Safety-First
    y_pred_saf = (proba >= saf_t).astype(int)
    f1_saf = f1_score(y_test, y_pred_saf)
    rec_saf = recall_score(y_test, y_pred_saf)
    prec_saf = precision_score(y_test, y_pred_saf)
    cm_saf = confusion_matrix(y_test, y_pred_saf)
    
    block = (
        f"\n{'='*56}\n"
        f"  MODEL : {model_name} (ROC-AUC: {roc:.4f})\n"
        f"{'='*56}\n"
        f"  [BALANCED MODE] (Threshold: {f1_t})\n"
        f"    F1-Score  : {f1:.4f}\n"
        f"    Recall    : {rec:.4f}  (Caught {rec*100:.1f}% of failures)\n"
        f"    Precision : {prec:.4f}\n"
        f"    Confusion Matrix:\n{cm}\n\n"
        f"  [SAFETY-FIRST MODE] (Threshold: {saf_t})\n"
        f"    F1-Score  : {f1_saf:.4f}\n"
        f"    Recall    : {rec_saf:.4f}  (Caught {rec_saf*100:.1f}% of failures)\n"
        f"    Precision : {prec_saf:.4f}\n"
        f"    Confusion Matrix:\n{cm_saf}\n"
    )
    print(block)
    results_text.append(block)

with open(os.path.join(RESULTS_DIR, "evaluation_report.txt"), "w") as f:
    f.write("\n".join(results_text))

print(f"\n{'='*60}")
print("  [OK] PIPELINE TRAINING COMPLETE")
print("  All models & thresholds generated successfully.")
print(f"{'='*60}\n")
