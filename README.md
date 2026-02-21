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

## Installation & Requirements

This project requires both MATLAB and Python environments to execute the full pipeline.

### 1. MATLAB Environment (Preprocessing)
The preprocessing scripts were developed and tested in **MATLAB R2023b** (or later).

**Required Toolboxes:**
* Signal Processing Toolbox
* Statistics and Machine Learning Toolbox

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
