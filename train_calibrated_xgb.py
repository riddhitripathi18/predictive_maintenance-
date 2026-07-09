import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    precision_score, recall_score, f1_score, average_precision_score
)
import xgboost as xgb

# 1. DEFINE PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "ai4i2020_train_imbalanced.csv")
TEST_DATA_PATH = os.path.join(BASE_DIR, "ai4i2020_test_real.csv")

# 2. LOAD DATA
print("Loading imbalanced datasets...")
train_df = pd.read_csv(TRAIN_DATA_PATH)
test_df = pd.read_csv(TEST_DATA_PATH)

feature_cols = [
    "Type", "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"
]

X_train = train_df[feature_cols].copy()
y_train = train_df["Machine failure"].copy()
X_test = test_df[feature_cols].copy()
y_test = test_df["Machine failure"].copy()

# 3. PREPROCESSING & ENCODING
le = LabelEncoder()
X_train["Type"] = le.fit_transform(X_train["Type"])
X_test["Type"] = le.transform(X_test["Type"])

# Clean column names for XGBoost JSON-safe compatibility
def clean_cols(df):
    new_cols = [c.replace('[', '').replace(']', '').replace(' ', '_') for c in df.columns]
    df.columns = new_cols
    return df

X_train = clean_cols(X_train)
X_test = clean_cols(X_test)

# Calculate scale_pos_weight
n_neg = (y_train == 0).sum()
n_pos = y_train.sum()
scale_pos = float(n_neg) / float(n_pos)

# 4. TRAINING UNCALIBRATED MODEL
print("\nTraining uncalibrated XGBoost model (scale_pos_weight applied)...")
xgb_uncal = xgb.XGBClassifier(
    n_estimators=300,
    random_state=42,
    eval_metric="aucpr",
    scale_pos_weight=scale_pos,
    n_jobs=-1,
    verbosity=0
)
xgb_uncal.fit(X_train, y_train)

# 5. TRAINING CALIBRATED MODEL
print("Training calibrated XGBoost model (CalibratedClassifierCV with Platt scaling)...")
xgb_base = xgb.XGBClassifier(
    n_estimators=300,
    random_state=42,
    eval_metric="aucpr",
    scale_pos_weight=scale_pos,
    n_jobs=-1,
    verbosity=0
)
# Wrap with Platt scaling (sigmoid method) and 5-fold CV
xgb_cal = CalibratedClassifierCV(estimator=xgb_base, method="sigmoid", cv=5)
xgb_cal.fit(X_train, y_train)

# 6. INFERENCE & PROBABILITIES
uncal_probs = xgb_uncal.predict_proba(X_test)[:, 1]
cal_probs = xgb_cal.predict_proba(X_test)[:, 1]

# Predictions using default 0.5 threshold
uncal_preds = (uncal_probs >= 0.5).astype(int)
cal_preds = (cal_probs >= 0.5).astype(int)

# Calculate metrics
def get_metrics(y_true, y_pred, y_prob):
    return {
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "PR-AUC": average_precision_score(y_true, y_prob)
    }

uncal_metrics = get_metrics(y_test, uncal_preds, uncal_probs)
cal_metrics = get_metrics(y_test, cal_preds, cal_probs)

# 7. PRINT COMPARISONS
print("\n" + "="*60)
print("  SIDE-BY-SIDE MODEL EVALUATION (Threshold = 0.50)")
print("="*60)
print(f"{'Evaluation Metric':<20} | {'Uncalibrated XGB':<18} | {'Calibrated XGB':<18}")
print("-"*60)
for metric in ["Precision", "Recall", "F1-Score", "PR-AUC"]:
    print(f"{metric:<20} | {uncal_metrics[metric]:<18.4f} | {cal_metrics[metric]:<18.4f}")
print("="*60)

# Probability distribution breakdown helper
def print_distribution(title, probs):
    print(f"\n[Distribution] {title}")
    print(f"{'Probability Range':<20} | {'Count':<8} | {'Percentage':<10}")
    print("-"*44)
    total_samples = len(probs)
    for lower in range(0, 100, 10):
        upper = lower + 10
        lower_val = lower / 100.0
        upper_val = upper / 100.0
        if upper == 100:
            count = ((probs >= lower_val) & (probs <= upper_val)).sum()
        else:
            count = ((probs >= lower_val) & (probs < upper_val)).sum()
        pct = (count / total_samples) * 100
        print(f"{f'{lower}% - {upper}%':<20} | {count:<8} | {pct:<9.2f}%")

print_distribution("Uncalibrated XGBoost Model", uncal_probs)
print_distribution("Calibrated XGBoost Model", cal_probs)
