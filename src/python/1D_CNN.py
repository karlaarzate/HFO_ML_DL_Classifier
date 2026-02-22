"""
========================================================================
  PART 5: 1D-RESNET DEEP LEARNING CLASSIFIER (MULTICLASS)
  Project: Automated HFO Detection with ML/DL Architecture
  
  Description:
  This script implements a 1D Convolutional Neural Network based on the 
  Residual Network (ResNet) architecture for multiclass HFO classification. 
  The model classifies detected events into four categories: Noise, Ripples, 
  Fast Ripples, and mixed FR&R. It features a custom physiological data 
  augmentation layer to enhance morphological robustness.

  Key Features:
  - Architecture: 1D-ResNet with 3 residual blocks (64 to 256 filters).
  - Augmentation: Real-time temporal jitter (+/- 10ms) and amplitude scaling.
  - Training: Leave-One-Patient-Out (LOPO) cross-validation with 
    dynamic class weighting for oversampled physiological datasets.
  - Optimization: Early stopping and Global Average Pooling to prevent 
    overfitting in patient-specific validation folds.
=========================================================================
"""
import os
import gc
import shutil
import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, roc_auc_score, average_precision_score
)
from sklearn.preprocessing import label_binarize
from sklearn.utils import class_weight

# ==========================================
# 1. DIRECTORY SETUP AND HYPERPARAMETER CONFIGURATION
# ==========================================
HP = {
    "JITTER": 20,           # +/- 10ms temporal shift
    "SCALE": 0.05,          # +/- 5% amplitude scaling for morphology preservation
    "LR": 1e-3,             # Learning rate
    "BATCH_SIZE": 256,
    "EPOCHS": 40,
    "PATIENCE": 8,          # Early stopping patience
    "DROPOUT": 0.4
}

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

# --- INPUT PATHS ---
DRIVE_PATH = os.path.join(project_root, "data", "master", "CONSOLIDATE_CNN_DATASET_200ms_OVERSAMPLED25.parquet")
LOCAL_PATH = "/content/dataset_oversampled.parquet"

# --- OUTPUT PATHS ---
BASE_OUT = os.path.join(project_root, "data", "processed", "RESULTS_1D")
PROBS_DIR = os.path.join(BASE_OUT, "probabilities")
MODELS_DIR = os.path.join(BASE_OUT, "models")


for d in [PROBS_DIR, MODELS_DIR]:
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
        print(f"[INFO] Created directory: {d}")

# ==========================================
# 2. DATA INGESTION (HIGH-SPEED)
# ==========================================
if not os.path.exists(LOCAL_PATH):
    if os.path.exists(DRIVE_PATH):
        print(f"[INFO] Copying dataset to local VM: {DRIVE_PATH}")
        shutil.copy(DRIVE_PATH, LOCAL_PATH)
    else:
        raise FileNotFoundError(f"Dataset not found at {DRIVE_PATH}. Check your folder structure.")

df = pd.read_parquet(LOCAL_PATH)

# ==========================================
# 3. MORPHOLOGICAL AUGMENTATION LAYER
# ==========================================
class HFOAugmentation(tf.keras.layers.Layer):
    """
    Custom layer to apply jitter and scaling during training to improve generalization.
    """
    def __init__(self, jitter_range=20, scale_range=0.05, **kwargs):
        super().__init__(**kwargs)
        self.jitter_range = jitter_range
        self.scale_range = scale_range

    def call(self, inputs, training=None):
        if not training: return inputs
        batch_size = tf.shape(inputs)[0]
        
        # Circular Shift (Temporal Jitter)
        shift = tf.random.uniform([], -self.jitter_range, self.jitter_range, dtype=tf.int32)
        x = tf.roll(inputs, shift=shift, axis=1)
        
        # Amplitude Scaling
        scale = tf.random.uniform([batch_size, 1, 1], 1.0 - self.scale_range, 1.0 + self.scale_range)
        return x * scale

# ==========================================
# 4. ARCHITECTURE & PERFORMANCE METRICS
# ==========================================
def build_augmented_resnet(input_len=400, n_classes=4):
    """
    Builds a 1D-ResNet model with an integrated HFO augmentation layer.
    """
    inputs = tf.keras.Input(shape=(input_len, 1))
    x = HFOAugmentation(jitter_range=HP["JITTER"], scale_range=HP["SCALE"])(inputs)

    x = tf.keras.layers.Conv1D(64, 7, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # Increased capacity: progressing from 64 to 256 filters
    for f in [64, 128, 256]:
        shortcut = tf.keras.layers.Conv1D(f, 1, padding='same')(x) 
        x = tf.keras.layers.Conv1D(f, 3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv1D(f, 3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Add()([x, shortcut])
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(HP["DROPOUT"])(x)
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs)

def compute_fold_metrics(y_true, y_probs, class_names):
    """
    Calculates detailed metrics for each validation fold.
    """
    y_pred = np.argmax(y_probs, axis=1)
    y_true_bin = label_binarize(y_true, classes=[0,1,2,3])
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3])

    res = []
    for i, name in enumerate(class_names):
        tp = cm[i,i]; fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp; tn = cm.sum() - (tp + fp + fn)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2*(prec*sens)/(prec+sens) if (prec+sens) > 0 else 0
        auc = roc_auc_score(y_true_bin[:, i], y_probs[:, i])
        auprc = average_precision_score(y_true_bin[:, i], y_probs[:, i])
        res.append({"Class": name, "Prec": prec, "Sens": sens, "Spec": spec, "F1": f1, "AUC": auc, "AUPRC": auprc})
    return pd.DataFrame(res), acc, cm

# ==========================================
# 5. LOPO EXECUTION LOOP (RECOVERY ENABLED)
# ==========================================
CLASS_NAMES = ["Noise", "Ripple", "Fast Ripple", "FRandR"]
logo = LeaveOneGroupOut()

for fold, (train_idx, val_idx) in enumerate(logo.split(X_all, y_all, groups=groups)):
    test_patient = str(groups[val_idx][0])

    # Progress Check: Skip already processed folds
    p_path = os.path.join(PROBS_DIR, f"1D_Probs_Pat_{test_patient}.parquet")
    if os.path.exists(p_path):
        print(f"[SKIP] Fold {fold+1} for Patient {test_patient} already exists.")
        continue

    print(f"\n[EXECUTION] FOLD {fold+1} | TEST PATIENT: {test_patient}")

    # Dynamic class weight calculation for imbalance compensation
    y_train_fold = y_all[train_idx]
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_fold), y=y_train_fold)
    cw_dict = dict(enumerate(weights))

    model = build_augmented_resnet()
    model.compile(optimizer=tf.keras.optimizers.Adam(HP["LR"]),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    callbacks = [tf.keras.callbacks.EarlyStopping(patience=HP["PATIENCE"], restore_best_weights=True)]

    # Model training with fold-specific data
    model.fit(X_all[train_idx], y_train_fold,
              validation_data=(X_all[val_idx], y_all[val_idx]),
              batch_size=HP["BATCH_SIZE"], epochs=HP["EPOCHS"],
              callbacks=callbacks, class_weight=cw_dict, verbose=1)

    # Inference phase
    y_probs = model.predict(X_all[val_idx])

    # Result Persistence
    df_probs = pd.DataFrame(y_probs, columns=[f"Prob_1D_{c}" for c in CLASS_NAMES])
    df_probs['Global_UID'] = uids[val_idx]
    df_probs['True_Label'] = y_all[val_idx]
    df_probs['Patient'] = test_patient
    df_probs.to_parquet(p_path, index=False)

    model.save(os.path.join(MODELS_DIR, f"Model_1D_Pat_{test_patient}.keras"))

    # Memory cleanup
    del model, y_train_fold
    tf.keras.backend.clear_session()
    gc.collect()

# ==========================================
# 6. GLOBAL REPORT GENERATION (POST-LOPO)
# ==========================================
print("\n" + "="*85)
print("[INFO] LOPO TRAINING FINISHED. GENERATING GLOBAL REPORT...")
print("="*85)

all_files = [f for f in os.listdir(PROBS_DIR) if f.endswith('.parquet')]

if not all_files:
    print("[ERROR] No probability files found. Verify results directory.")
else:
    list_df = [pd.read_parquet(os.path.join(PROBS_DIR, f)) for f in all_files]
    df_all = pd.concat(list_df).reset_index(drop=True)
    print(f"[INFO] Successfully aggregated {len(all_files)} patient files.")
    print(f"[INFO] Total samples analyzed: {len(df_all):,}")

    def get_global_report(df, class_names):
        y_true = df['True_Label'].values
        prob_cols = [f"Prob_1D_{c}" for c in class_names]
        y_probs = df[prob_cols].values
        y_pred = np.argmax(y_probs, axis=1)

        y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3])
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

        results = []
        for i, name in enumerate(class_names):
            tp = cm[i, i]; fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp; tn = cm.sum() - (tp + fp + fn)

            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * (prec * sens) / (prec + sens) if (prec + sens) > 0 else 0
            auc = roc_auc_score(y_true_bin[:, i], y_probs[:, i])
            auprc = average_precision_score(y_true_bin[:, i], y_probs[:, i])

            results.append({
                "Class": name, "Precision": prec, "Recall (Sens)": sens,
                "Specificity": spec, "F1-Score": f1, "ROC-AUC": auc, "AUPRC": auprc
            })
        return pd.DataFrame(results), acc, cm

    report_df, total_acc, total_cm = get_global_report(df_all, CLASS_NAMES)

    # Print Final Results
    print("\n" + "="*85)
    print("GLOBAL 1D-CNN PERFORMANCE REPORT (AUGMENTED & OVERSAMPLED)")
    print("="*85)
    pd.options.display.float_format = '{:.4f}'.format
    print(f"OVERALL SYSTEM ACCURACY: {total_acc:.4f}")
    print("-" * 85)
    print(report_df.to_string(index=False))
    print("-" * 85)

    report_df.to_csv(os.path.join(BASE_OUT, "GLOBAL_1D_METRICS.csv"), index=False)
    print(f"[INFO] Final metrics exported to: {BASE_OUT}")
