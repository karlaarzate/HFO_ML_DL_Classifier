# Automated HFO Detection Pipeline: A Hybrid ML/DL Framework

This repository provides an end-to-end pipeline for the detection and classification of High-Frequency Oscillations (HFOs) in intracranial EEG (iEEG) data. The framework utilizes a hybrid approach combining classical signal processing, gradient-boosted trees (XGBoost), and Deep Learning (CNNs).

## Dataset & Citations

The intracranial EEG (iEEG) data used in this project was provided by the **Fedele et al.** study. If you use this pipeline or the processed data, please cite the original work:

> **Fedele T, Burnos S, Boran E, Krayenbühl N, Hilfiker P, Grunwald I, Sarnthein J. Resection of high frequency oscillation sources predicts seizure freedom in the individual patient. Scientific Reports. 2017;7(1):17046.**

This dataset includes multi-channel iEEG recordings with clinical ground truth labels for Ripples (80-250 Hz) and Fast Ripples (250-500 Hz), which are essential for the validation of the hybrid models implemented here.

## Pipeline Overview
1. **Preprocessing (MATLAB)**: Signal conditioning, artifact removal, and hybrid physics-based feature extraction.
2. **Screener Stage (Python/XGBoost)**: High-sensitivity candidate filtering.
3. **Deep Learning Stage (Python/TensorFlow)**: 1D-CNN temporal analysis and 2D-CNN Time-Frequency (CWT) classification.
4. **Ensemble Stage**: Final decision via a voting system.

---

## Stage 1: Preprocessing & Feature Engineering

### 1.1 MATLAB Signal Conditioning
The preprocessing stage (implemented in 'Preprocessing_HFO_Hybrid.m') prepares the raw iEEG data for machine learning. 
- **Window Length**: 200ms (400 samples at 2000Hz).
- **Filtering**: 50Hz Notch filter and 80-500Hz 4th-order Butterworth bandpass filter.
- **Montage**: Bipolar derivation based on clinical specifications.

### 1.2 Hybrid Feature Extraction
We extract a multi-dimensional feature space (28 metrics) that combines:
- **Temporal Metrics**: RMS, Line-Length, Kurtosis, Skewness, Zero-Crossing.
- **Spectral Metrics**: Peak Frequency, Band-specific Power (Ripple vs. Fast Ripple).
- **Advanced Physics**: 
  - **SNR_Burst**: Variance ratio between the signal core and baseline.
  - **GaborCorrelation**: Template matching using Gabor atoms for morphological validation.

### 1.3 Data Export Formats
The pipeline generates two primary outputs for each processed file:
- **'.parquet'**: Contains the feature matrix for XGBoost training. Includes traceability metadata (SourceFile, TimeSec).
- **'.mat' (v7.3)**: Contains the normalized 200ms snippets and clinical ground truth labels for CNN training.

---

## Stage 2: XGBoost Screener Results
The first classification stage uses an XGBoost model optimized for maximum sensitivity to act as a data screener.

### Validation Metrics
<img width="1277" height="690" alt="xgboost" src="https://github.com/user-attachments/assets/9efa1ecc-c313-48d1-a81f-d26b74caae5a" />

*Figure 1. Performance metrics and normalized confusion matrix for the XGBoost screening stage.*

## Installation & Requirements

This project requires both MATLAB and Python environments to execute the full pipeline.

### 1. MATLAB Environment (Preprocessing)
The preprocessing scripts were developed and tested in **MATLAB R2023b** (or later).

**Required Toolboxes:**
* Signal Processing Toolbox
* Statistics and Machine Learning Toolbox

**Setup:**
1. Clone this repository.
2. Add the 'MATLAB/' folder to your MATLAB path.
3. Ensure your raw data follows the structure expected by the 'Preprocessing_HFO_Hybrid.m' script.

### 2. Python Environment (ML/DL Pipeline)
The classification stages (XGBoost & CNNs) require **Python 3.10+**. We recommend using a virtual environment or Google Colab for GPU acceleration.

**Core Dependencies:**
* **Data Handling:** 'pandas', 'numpy', 'pyarrow' (for Parquet support).
* **Signal Processing:** 'PyWavelets', 'scipy'.
* **Machine Learning:** 'xgboost', 'scikit-learn'.
* **Deep Learning:** 'tensorflow' (Keras), 'joblib'.
* **Visualization:** 'matplotlib', 'seaborn', 'tqdm'.

**Quick Install:**
'pip install pandas numpy pywavelets scipy xgboost scikit-learn tensorflow joblib matplotlib seaborn tqdm'
