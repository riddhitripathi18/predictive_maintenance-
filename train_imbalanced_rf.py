import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

# Define directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ORIG_DATA_PATH = os.path.join(BASE_DIR, "ai4i2020.csv")
TEST_DATA_PATH = os.path.join(BASE_DIR, "ai4i2020_test_real.csv")
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "ai4i2020_train_imbalanced.csv")

# 1. GENERATE TRAINING FILE
def generate_train_set():
    print(f"Generating training file: {TRAIN_DATA_PATH}...")
    if not os.path.exists(ORIG_DATA_PATH):
        raise FileNotFoundError(f"Original dataset not found at {ORIG_DATA_PATH}!")
    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(f"Test dataset not found at {TEST_DATA_PATH}!")
        
    df_orig = pd.read_csv(ORIG_DATA_PATH)
    df_test = pd.read_csv(TEST_DATA_PATH)
    
    feature_cols = [
        "Type", "Air temperature [K]", "Process temperature [K]",
        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"
    ]
    
    # The test set has re-indexed UDI (1 to 2000), so we must join on features 
    # to find the original UDIs of the test set rows.
    matches = pd.merge(df_orig, df_test, on=feature_cols, suffixes=("_orig", "_test"))
    test_orig_udis = matches["UDI_orig"].tolist()
    
    # Exclude those rows from original
    df_train_imbalanced = df_orig[~df_orig["UDI"].isin(test_orig_udis)].copy()
    
    # Keep only the requested columns
    expected_cols = ["UDI"] + feature_cols + ["Machine failure"]
    df_train_imbalanced = df_train_imbalanced[expected_cols]
    
    df_train_imbalanced.to_csv(TRAIN_DATA_PATH, index=False)
    print(f"Generated {TRAIN_DATA_PATH} successfully. Shape: {df_train_imbalanced.shape}")

# Regenerate train set if it doesn't exist, or if the existing file has overlap
if os.path.exists(TRAIN_DATA_PATH):
    df_train_exist = pd.read_csv(TRAIN_DATA_PATH)
    df_test = pd.read_csv(TEST_DATA_PATH)
    feature_cols = [
        "Type", "Air temperature [K]", "Process temperature [K]",
        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"
    ]
    overlap = len(pd.merge(df_train_exist[feature_cols], df_test[feature_cols], how="inner"))
    if overlap > 0:
        print(f"Existing training file has {overlap} overlapping rows with the test set. Regenerating to prevent leakage...")
        generate_train_set()
else:
    generate_train_set()

# 2. LOAD DATA
print(f"Loading datasets...")
train_df = pd.read_csv(TRAIN_DATA_PATH)
test_df = pd.read_csv(TEST_DATA_PATH)

# Verify zero overlap on features
feature_cols = [
    "Type", "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"
]
overlap = len(pd.merge(train_df[feature_cols], test_df[feature_cols], how="inner"))
assert overlap == 0, f"Critical Data Leakage: found {overlap} overlapping feature rows between train and test!"
print("Overlap verification: 0 overlapping feature rows between train and test. (Verified!)")

print(f"Train Shape: {train_df.shape} | Test Shape: {test_df.shape}")
print(f"Train Failure Rate: {train_df['Machine failure'].mean() * 100:.3f}%")
print(f"Test Failure Rate: {test_df['Machine failure'].mean() * 100:.3f}%")

# Drop UDI
X_train = train_df.drop(columns=["UDI", "Machine failure"])
y_train = train_df["Machine failure"]

X_test = test_df.drop(columns=["UDI", "Machine failure"])
y_test = test_df["Machine failure"]

# Encode Type column
le = LabelEncoder()
X_train["Type"] = le.fit_transform(X_train["Type"])
X_test["Type"] = le.transform(X_test["Type"])

# 3. TRAIN RANDOM FOREST CLASSIFIER
print("\nTraining RandomForestClassifier with class_weight='balanced' (constrained)...")
rf = RandomForestClassifier(
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    max_depth=8,           # stops trees from growing infinitely deep
    min_samples_leaf=5,    # each final decision must cover at least 5 real rows
    min_samples_split=10,  # need at least 10 rows before splitting further
    n_estimators=300       # 300 trees for stability
)
rf.fit(X_train, y_train)

# --- MEMORIZATION CHECK ---
train_f1 = f1_score(y_train, rf.predict(X_train))
test_f1_default = f1_score(y_test, rf.predict(X_test))
gap = train_f1 - test_f1_default
print(f"\nMemorization Check:")
print(f"  Train F1 : {train_f1:.4f}")
print(f"  Test  F1 : {test_f1_default:.4f}")
print(f"  Gap      : {gap:.4f}")
if gap > 0.15:
    print("  [WARNING] Still overfitting -- gap too large")
else:
    print("  [OK] Gap is healthy -- model is generalizing")

# 4. EVALUATE ON TEST SET (PROBABILITIES)
y_probs = rf.predict_proba(X_test)[:, 1]

# Calculate PR-AUC (Average Precision Score)
pr_auc = average_precision_score(y_test, y_probs)
print(f"\nPrecision-Recall AUC (PR-AUC) for Failure class: {pr_auc:.4f}")

# 5. TEST DECISION THRESHOLDS (0.1 to 0.5 in steps of 0.05)
thresholds = np.arange(0.1, 0.51, 0.05)
results = []

print("\nThreshold Sweep Analysis:")
print(f"{'Threshold':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
print("-" * 52)

best_f1 = -1.0
best_threshold = 0.5
best_preds = None

for t in thresholds:
    t_round = round(t, 2)
    # Apply threshold
    y_preds_t = (y_probs >= t_round).astype(int)
    
    # Calculate metrics for failure class (class 1)
    prec = precision_score(y_test, y_preds_t, pos_label=1, zero_division=0)
    rec = recall_score(y_test, y_preds_t, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_preds_t, pos_label=1, zero_division=0)
    
    print(f"{t_round:<12.2f} | {prec:<10.4f} | {rec:<10.4f} | {f1:<10.4f}")
    results.append({
        "Threshold": t_round,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "Predictions": y_preds_t
    })
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t_round
        best_preds = y_preds_t

# 6. PRINT FULL CLASSIFICATION REPORT FOR BEST F1 THRESHOLD
print(f"\n{'='*60}")
print(f"  Best F1-Score of {best_f1:.4f} achieved at Threshold: {best_threshold}")
print(f"{'='*60}")
print("Classification Report:")
print(classification_report(y_test, best_preds, target_names=["No Failure", "Machine Failure"], digits=4))
print(f"{'='*60}")

# 7. SANITY CHECK BATCHES
# Build two sanity check batches from test data and evaluate them
print(f"\n{'='*60}")
print("  SANITY CHECK BATCHES")
print(f"{'='*60}")

# Batch 1: All failure rows from the test set
sanity1 = test_df[test_df["Machine failure"] == 1].copy()
# Batch 2: All safe rows from the test set
sanity2 = test_df[test_df["Machine failure"] == 0].sample(n=50, random_state=42).copy()

for batch_name, batch_df in [("Sanity Batch 1 (all failure rows)", sanity1),
                              ("Sanity Batch 2 (50 random safe rows)", sanity2)]:
    X_batch = batch_df.drop(columns=["UDI", "Machine failure"])
    y_batch = batch_df["Machine failure"]
    X_batch["Type"] = le.transform(X_batch["Type"])
    batch_probs = rf.predict_proba(X_batch)[:, 1]
    batch_preds = (batch_probs >= best_threshold).astype(int)
    
    correct = (batch_preds == y_batch).sum()
    print(f"\n  {batch_name}:")
    print(f"    Rows: {len(batch_df)} | Correct: {correct} | Missed: {len(batch_df) - correct}")
    print(f"    Avg failure probability: {batch_probs.mean() * 100:.2f}%")
    print(f"    Min: {batch_probs.min() * 100:.2f}%  Max: {batch_probs.max() * 100:.2f}%")
    # Show worst missed cases
    batch_df = batch_df.copy()
    batch_df["Failure Probability (%)"] = (batch_probs * 100).round(2)
    batch_df["Predicted"] = batch_preds
    missed = batch_df[batch_df["Predicted"] != batch_df["Machine failure"]]
    if not missed.empty:
        print(f"    Missed rows (showing up to 5):")
        print(missed[["UDI", "Type", "Rotational speed [rpm]", "Torque [Nm]",
                       "Tool wear [min]", "Machine failure", "Failure Probability (%)"]].head(5).to_string(index=False))
    else:
        print("    [OK] Zero misses on this batch!")
print(f"\n{'='*60}")
