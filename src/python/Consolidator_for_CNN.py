"""
========================================================================
PART 4: CNN DATASET CONSOLIDATION & SIGNAL HYDRATION
Project: Automated HFO Detection with ML/DL Architecture

Description:
This script synchronizes the XGBoost screening results with raw EEG 
snippets. It performs large-scale class balancing (Oversampling/Undersampling) 
to create a 1.2M sample dataset, extracts 200ms raw voltage traces, 
and exports a memory-optimized Parquet file for 1D-CNN training.

Key Features:
- Signal Hydration: Maps Parquet metadata to MAT-file raw snippets.
- Balanced Sampling: For all 4 classes.
- Memory Management: Chunk-based processing with PyArrow for large-scale I/O.
- Integrity Check: Validates 400-sample window length (200ms @ 2000Hz).
========================================================================
"""

import pandas as pd
import numpy as np
import scipy.io as sio
import h5py
import os
import sys
import re
import gc
from tqdm import tqdm

# ==========================================
# 1. PROJECT DIRECTORY & PATH CONFIGURATION
# ==========================================
# Get the absolute path of the current script (src/python/)
script_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate to the project root (HFO-Hybrid-Detection/)
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

# --- INPUT PATHS ---
# XGBoost Screening results (List of HFO candidates that passed the gate)
SURVIVOR_PATH = os.path.join(project_root, "data", "processed", "XGBOOST_SCREENER_RESULTS", "CNN_INPUT_LIST_200ms.parquet")

# Intermediate directories from MATLAB Preprocessing
FEATURES_DIR = os.path.join(project_root, "data", "processed", "FEATURES")
SNIPPETS_DIR = os.path.join(project_root, "data", "processed", "SNIPPETS")

# --- OUTPUT PATHS ---
# Final consolidated dataset for CNN training
OUTPUT_PATH = os.path.join(project_root, "data", "master", "CONSOLIDATE_CNN_DATASET_200ms_OVERSAMPLED25.parquet")

# --- SIGNAL PARAMETERS ---
EXPECTED_SAMPLES = 400   # 200ms @ 2000Hz
HARD_NOISE_THRESHOLD = 0.40

# Ensure master directory exists
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# ==========================================
# 2. DATASET INGESTION & UID GENERATION
# ==========================================
print(f"[INFO] Loading Survivor List: {SURVIVOR_PATH}")
df_survivors = pd.read_parquet(SURVIVOR_PATH)

# Generate Global Unique Identifier for signal traceability
df_survivors['Global_UID'] = (
    df_survivors['Patient'].astype(str) + "_" +
    df_survivors['SourceFile'].str.replace('.mat','') + "_" +
    df_survivors['TimeSec'].map('{:.5f}'.format) + "_" +
    df_survivors['Channel'].astype(str)
)

# ==========================================
# 3. CLASS BALANCING & TARGET SAMPLING
# ==========================================
print("\n" + "="*60)
print(f"STRATEGY: EQUALIZED TARGET SAMPLING (1.2M TOTAL)")
print("="*60)

TOTAL_SAMPLES = 1200000
target_per_class = TOTAL_SAMPLES // 4

# Stratify samples by clinical ground truth
d_noise = df_survivors[(df_survivors['yAnyHFO'] == 0) & (df_survivors['Screener_Prob'] > HARD_NOISE_THRESHOLD)]
d_r     = df_survivors[df_survivors['yR'] == 1]
d_fr    = df_survivors[df_survivors['yFR'] == 1]
d_frand = df_survivors[df_survivors['yFRandR'] == 1]

target_map = {
    "Noise":       target_per_class,
    "Ripple":      target_per_class,
    "Fast Ripple": target_per_class,
    "FRandR":      target_per_class
}

balanced_list = []

for df_sub, (name, target) in zip([d_noise, d_r, d_fr, d_frand], target_map.items()):
    if len(df_sub) == 0:
        print(f"[WARNING] Empty class detected: {name}")
        continue

    # Execute Hybrid Resampling:
    # - Undersampling if available data > target
    # - Oversampling (replace=True) if available data < target
    should_replace = len(df_sub) < target
    resampled = df_sub.sample(n=target, random_state=42, replace=should_replace)

    balanced_list.append(resampled)
    multiplier = target / len(df_sub)
    print(f" - {name:<12}: Target {target:,} | Available {len(df_sub):,} | Factor: {multiplier:.2f}x")

# Finalize and shuffle the extraction plan
df_to_extract = pd.concat(balanced_list).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"\n[INFO] Total Extraction Plan Finalized: {len(df_to_extract):,}")

# ==========================================
# 4. SIGNAL EXTRACTION ENGINE
# ==========================================
print("\n[INFO] Extracting voltage traces in memory-safe mode...")
final_data = []
skipped_count = 0
total_extracted = 0
CHUNK_SIZE = 10 
temp_chunks = []

# Index project files for rapid O(1) lookups
archivos_snippets = {f: os.path.join(SNIPPETS_DIR, f) for f in os.listdir(SNIPPETS_DIR) if f.endswith('.mat')}
archivos_features = {f: os.path.join(FEATURES_DIR, f) for f in os.listdir(FEATURES_DIR) if f.endswith('.parquet')}

df_to_extract['Lookup_Key'] = df_to_extract['TimeSec'].map('{:.5f}'.format) + "_" + df_to_extract['Channel'].astype(str)
grouped = list(df_to_extract.groupby(['SourceFile', 'Patient']))

for i, ((source_file, patient_id), group) in enumerate(tqdm(grouped, desc="Hydrating Signals")):

    clean_name = source_file.replace('.mat', '')

    # Flexible prefix matching to handle patient-specific naming conventions
    match_s = [path for name, path in archivos_snippets.items() if clean_name in name]
    match_f = [path for name, path in archivos_features.items() if clean_name in name]

    if not match_s or not match_f:
        skipped_count += len(group)
        continue

    s_path, f_path = match_s[0], match_f[0]

    try:
        # Determine MAT-file version (v7 vs v7.3/HDF5)
        try:
            mat = sio.loadmat(s_path)
            raw_snippets = mat['SNIP']['snippet'][0][0]
            use_h5 = False
        except:
            h5 = h5py.File(s_path, 'r')
            raw_snippets = h5['SNIP']['snippet']
            use_h5 = True

        df_feats = pd.read_parquet(f_path)
        df_feats['Lookup'] = df_feats['TimeSec'].map('{:.5f}'.format) + "_" + df_feats['Channel'].astype(str)
        lookup_dict = {key: idx for idx, key in enumerate(df_feats['Lookup'])}

        for _, row in group.iterrows():
            key = row['Lookup_Key']
            if key in lookup_dict:
                idx = lookup_dict[key]
                wave = h5[raw_snippets[idx][0]][:].flatten() if use_h5 else raw_snippets[0, idx].flatten()

                if len(wave) == EXPECTED_SAMPLES:
                    m_label = 3 if row['yFRandR'] else 2 if row['yFR'] else 1 if row['yR'] else 0
                    final_data.append({
                        'Global_UID': row['Global_UID'],
                        'Raw_Voltage': wave.astype(np.float32),
                        'Multiclass_Label': m_label,
                        'Screener_Prob': row['Screener_Prob'],
                        'Patient': str(patient_id),
                        'Original_Time': row['TimeSec'],
                        'Channel': int(row['Channel'])
                    })
                    total_extracted += 1

        if use_h5: h5.close()
    except Exception as e:
        print(f"[ERROR] Extraction failed for {clean_name}: {e}")

    # Memory Guard: Chunking interim results to disk
    if (i + 1) % CHUNK_SIZE == 0 or (i + 1) == len(grouped):
        if final_data:
            chunk_path = f"{OUTPUT_PATH}_part{len(temp_chunks)}.tmp"
            pd.DataFrame(final_data).to_parquet(chunk_path)
            temp_chunks.append(chunk_path)
            final_data = [] 
            gc.collect()    

# ==========================================
# 5. FINAL DATA CONSOLIDATION
# ==========================================
if temp_chunks:
    print("\n[INFO] Consolidating chunks into final master dataset...")

    import pyarrow.parquet as pq
    import pyarrow as pa

    try:
        # Initialize final Parquet stream with Schema definition
        first_chunk_path = temp_chunks[0]
        df_first = pd.read_parquet(first_chunk_path)
        table = pa.Table.from_pandas(df_first, preserve_index=False)
        
        writer = pq.ParquetWriter(OUTPUT_PATH, table.schema, compression='snappy')
        writer.write_table(table)

        total_samples = len(df_first)
        global_counts = df_first['Multiclass_Label'].value_counts()

        del df_first, table
        gc.collect()

        # Sequentially append chunks to disk
        for p in tqdm(temp_chunks[1:], desc="Appending Data"):
            df_chunk = pd.read_parquet(p)
            global_counts = global_counts.add(df_chunk['Multiclass_Label'].value_counts(), fill_value=0)
            total_samples += len(df_chunk)

            table_chunk = pa.Table.from_pandas(df_chunk, preserve_index=False)
            writer.write_table(table_chunk)

            del df_chunk, table_chunk
            gc.collect()

        writer.close()

        # Performance and Balance Report
        print("\n" + "="*60)
        print(" SUCCESS: CNN MASTER DATASET CREATED")
        print("="*60)
        class_map = {0: 'Noise', 1: 'Ripple (R)', 2: 'Fast Ripple (FR)', 3: 'FRandR'}
        print(f"Target: {OUTPUT_PATH}")
        print(f"Total Hydrated Samples: {total_samples:,}\n")
        print(f"{'Class':<20} | {'Count':<12} | {'Percentage':<10}")
        print("-" * 60)

        for cls_idx in sorted(global_counts.index):
            count = int(global_counts[cls_idx])
            name = class_map.get(int(cls_idx), f"Unknown ({cls_idx})")
            percentage = (count / total_samples) * 100
            print(f"{name:<20} | {count:>12,} | {percentage:>9.2f}%")
        print("-" * 60)

    except Exception as e:
        print(f"[CRITICAL ERROR] Consolidation failed: {e}")
    finally:
        # Environment Cleanup
        print("\n[CLEANUP] Removing temporary chunk files...")
        for p in temp_chunks:
            if os.path.exists(p): os.remove(p)
else:
    print(f"[FATAL] No data chunks generated. Process aborted.")
