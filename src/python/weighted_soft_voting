"""
========================================================================
PART 8: LATE-FUSION ENSEMBLE (WEIGHTED SOFT VOTING)
Project: Automated HFO Detection with ML/DL Architecture

Description:
This script performs a weighted integration of the prediction 
probabilities from the 1D-CNN (Temporal Stream) and 2D-CNN (Spectral Stream). 
By applying class-specific scientific weights, the ensemble balances 
temporal morphology with time-frequency features to produce the 
final multi-class diagnostic output.

Key Features:
- Class-Specific Weighting: Optimizes importance based on model expertise.
- Global UID Syncing: Inner-join merging ensures temporal synchronization.
- Comprehensive Evaluation: Generates ROC-AUC, AUPRC, and Confusion Matrices.
- Signal Traceability: Preserves patient and time metadata for clinical review.
========================================================================
"""

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, average_precision_score, roc_curve, auc, precision_recall_curve
)
from sklearn.preprocessing import label_binarize

# ==========================================
# 1. DIRECTORY CONFIGURATION & WEIGHTS
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

# Defined relative paths within the project hierarchy
PATH_1D = os.path.join(project_root, "data", "processed", "RESULTS_1D", "probabilities")
PATH_2D = os.path.join(project_root, "data", "processed", "DL_RESULTS_2D_Stockwell", "probabilities")
OUTPUT_DIR = os.path.join(project_root, "data", "master", "ENSEMBLE_RESULTS_Voting_Stockwell")

if not os.path.exists(OUTPUT_DIR): 
    os.makedirs(OUTPUT_DIR)

CLASS_NAMES = ["Noise", "Ripple", "Fast Ripple", "FRandR"]

# SCIENTIFIC WEIGHTING STRATEGY: Class-specific importance factors
WEIGHTS = {
    "Noise":       {"1D": 0.3, "2D": 0.7}, # Spectral features are superior for noise rejection
    "Ripple":      {"1D": 0.6, "2D": 0.4}, # Temporal morphology is key for Ripples
    "Fast Ripple": {"1D": 0.5, "2D": 0.5},
    "FRandR":      {"1D": 0.4, "2D": 0.6}  # Spectral density is critical for complex oscillations
}

# ==========================================
# 2. DATA LOADING & SYNCHRONIZATION
# ==========================================
print("[INFO] Initializing Probability Loading...")

def load_parquets(folder):
    if not os.path.exists(folder):
        raise FileNotFoundError(f"[ERROR] Directory missing: {folder}")
    files = [f for f in os.listdir(folder) if f.endswith('.parquet')]
    if not files:
        print(f"[WARNING] No parquet files found in {folder}")
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(os.path.join(folder, f)) for f in files], ignore_index=True)

df_1d = load_parquets(PATH_1D)
df_2d = load_parquets(PATH_2D)

print(f"[STATUS] 1D-CNN: {len(df_1d)} samples | 2D-CNN: {len(df_2d)} samples")

# Merging both streams based on Global Unique Identifier (UID)
print("[INFO] Synchronizing models via Global_UID...")
cols_1d = ['Global_UID', 'Patient', 'True_Label'] + [c for c in df_1d.columns if "Prob_1D" in c]
cols_2d = ['Global_UID'] + [c for c in df_2d.columns if "Prob_2D" in c]

df_ensemble = pd.merge(df_1d[cols_1d], df_2d[cols_2d], on='Global_UID', how='inner')
print(f"[STATUS] Ensemble dataset ready: {len(df_ensemble)} matched events.")

# ==========================================
# 3. WEIGHTED SOFT VOTING EXECUTION
# ==========================================
print("[PROCESS] Executing Weighted Soft Voting...")

probs_1d = df_ensemble[[f"Prob_1D_{c}" for c in CLASS_NAMES]].values
probs_2d = df_ensemble[[f"Prob_2D_{c}" for c in CLASS_NAMES]].values

# Extraction of weight matrices
w_1d = np.array([WEIGHTS[c]["1D"] for c in CLASS_NAMES])
w_2d = np.array([WEIGHTS[c]["2D"] for c in CLASS_NAMES])

# Applying weighted average per class
weighted_sums = (probs_1d * w_1d) + (probs_2d * w_2d)
final_probs = weighted_sums / (w_1d + w_2d)

y_pred_class = np.argmax(final_probs, axis=1)
y_true = df_ensemble['True_Label'].values
y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3])

for i, c in enumerate(CLASS_NAMES):
    df_ensemble[f"Prob_Ensemble_{c}"] = final_probs[:, i]
df_ensemble['Pred_Label'] = y_pred_class

# ==========================================
# 4. SCIENTIFIC PERFORMANCE METRICS
# ==========================================
print("\n" + "="*80)
print("ENSEMBLE CLASSIFICATION REPORT")
print("="*80)

acc = accuracy_score(y_true, y_pred_class)
print(f"GLOBAL ENSEMBLE ACCURACY: {acc:.4f}")

metrics_list = []
cm = confusion_matrix(y_true, y_pred_class)

for i, class_name in enumerate(CLASS_NAMES):
    tp = cm[i, i]
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp
    tn = cm.sum() - (tp + fp + fn)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    roc_auc_val = roc_auc_score(y_true_bin[:, i], final_probs[:, i])
    au
