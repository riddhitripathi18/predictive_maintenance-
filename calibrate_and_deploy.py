import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb

# Define directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.pkl")
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "ai4i2020_train_imbalanced.csv")
TEST_DATA_PATH = os.path.join(BASE_DIR, "ai4i2020_test_real.csv")
SANITY_OUT_PATH = os.path.join(BASE_DIR, "sanity_test_100.csv")

print("="*60)
print("  STEP 1: CONFIRMING MODEL FILE PATH")
print("="*60)
print(f"Target model save path: {XGB_MODEL_PATH}")

# Load training data
train_df = pd.read_csv(TRAIN_DATA_PATH)
test_df = pd.read_csv(TEST_DATA_PATH)

feature_cols = [
    "Type", "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"
]

X_train = train_df[feature_cols].copy()
y_train = train_df["Machine failure"].copy()

# Preprocess & Encode
le = LabelEncoder()
X_train["Type"] = le.fit_transform(X_train["Type"])

# Clean column names for XGBoost
def clean_cols(df):
    new_cols = [c.replace('[', '').replace(']', '').replace(' ', '_') for c in df.columns]
    df.columns = new_cols
    return df

X_train = clean_cols(X_train)

# Calculate scale_pos_weight
n_neg = (y_train == 0).sum()
n_pos = y_train.sum()
scale_pos = float(n_neg) / float(n_pos)

print("\n" + "="*60)
print("  STEP 2: TRAINING CALIBRATED XGBOOST MODEL")
print("="*60)
xgb_base = xgb.XGBClassifier(
    n_estimators=300,
    random_state=42,
    eval_metric="aucpr",
    scale_pos_weight=scale_pos,
    n_jobs=-1,
    verbosity=0
)
xgb_cal = CalibratedClassifierCV(estimator=xgb_base, method="sigmoid", cv=5)
xgb_cal.fit(X_train, y_train)
print("Calibrated model training complete.")

# Overwrite xgb_model.pkl
print(f"Saving calibrated model to: {XGB_MODEL_PATH}")
with open(XGB_MODEL_PATH, "wb") as f:
    pickle.dump(xgb_cal, f)
print("Model file successfully overwritten.")

print("\n" + "="*60)
print("  STEP 3: VERIFYING MODEL LOAD BACK")
print("="*60)
with open(XGB_MODEL_PATH, "rb") as f:
    loaded_model = pickle.load(f)

print(f"Loaded model type: {type(loaded_model)}")
print(f"Is CalibratedClassifierCV: {isinstance(loaded_model, CalibratedClassifierCV)}")
if isinstance(loaded_model, CalibratedClassifierCV):
    print("Verification: SUCCESS! The model file is a CalibratedClassifierCV object.")
else:
    print("Verification: FAILED! Model is not calibrated.")

print("\n" + "="*60)
print("  STEP 4: CREATING SANITY CHECK FILE & VERIFYING WARNING ZONE")
print("="*60)
# Create a 100-row sample from test_real containing a realistic mixture of safe and failure rows
# (e.g. 90 safe and 10 failures to see both behaviors)
df_safe = test_df[test_df["Machine failure"] == 0].sample(n=90, random_state=42)
df_fail = test_df[test_df["Machine failure"] == 1].sample(n=10, random_state=42)
sanity_100 = pd.concat([df_safe, df_fail]).sample(frac=1.0, random_state=42).reset_index(drop=True)
sanity_100.to_csv(SANITY_OUT_PATH, index=False)
print(f"Generated 100-row sanity check file: {SANITY_OUT_PATH}")

# Run predictions on sanity_100
X_sanity = sanity_100[feature_cols].copy()
X_sanity["Type"] = le.transform(X_sanity["Type"])
X_sanity = clean_cols(X_sanity)

# Predict probabilities
probs = loaded_model.predict_proba(X_sanity)[:, 1]

# Count predictions in Warning Tier (30% - 60%)
warning_mask = (probs >= 0.30) & (probs <= 0.60)
warning_count = warning_mask.sum()
warning_rows = sanity_100[warning_mask]

print(f"Sanity Check batch predictions run on {len(X_sanity)} rows:")
print(f"  Safe (<30%): {(probs < 0.30).sum()}")
print(f"  Warning (30%-60%): {warning_count}")
print(f"  Critical (>60%): {(probs > 0.60).sum()}")

if warning_count > 0:
    print(f"\nVerification: SUCCESS! Found {warning_count} rows in the Warning zone.")
    print("Sample rows in Warning zone:")
    print(warning_rows[["UDI", "Type", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", "Machine failure"]].head(5).to_string(index=False))
else:
    print("\nVerification: FAILED! Warning zone is empty.")

# Touch main.py to trigger reload
print("\n" + "="*60)
print("  STEP 5: RELOADING THE BACKEND")
print("="*60)
main_py_path = os.path.join(BASE_DIR, "main.py")
if os.path.exists(main_py_path):
    with open(main_py_path, "a") as f:
        f.write("\n# Reload trigger\n")
    print("Touched main.py. FastAPI backend should reload the model automatically.")
else:
    print("main.py not found, skip reload.")
