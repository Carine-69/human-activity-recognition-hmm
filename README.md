# Modeling Human Activity States Using Hidden Markov Models

**Formative 2 Machine Learning Techniques II**  
African Leadership University | January Term 2026  
Carine Umugabekazi & Carine Ahishakiye

---

## Overview

This project implements a complete Hidden Markov Model (HMM) pipeline for Human Activity Recognition (HAR) using real smartphone inertial measurement unit (IMU) data. The system classifies four physical activities  **jumping, walking, standing, and still**  from accelerometer and gyroscope signals recorded at 100 Hz using the Sensor Logger application.

The primary use case is **physical rehabilitation monitoring**: an automated system that allows clinicians to remotely verify whether patients are performing prescribed movement exercises correctly, without requiring physical presence or relying on self-reporting.

---

## Group Members and Contributions

| Member | Activities Collected | Code Responsibilities |
|---|---|---|
| Carine Umugabekazi | Jumping, Standing | Data loading, zip parsing, feature extraction, HMM training (Baum–Welch), evaluation metrics, confusion matrix, HMM parameter visualizations |
| Carine Ahishakiye | Walking, Still | Z-score normalization, label encoding, Viterbi decoding, raw signal and FFT visualizations, report writing |

Both members contributed equally to report writing and final review.

---

## Results

The trained model achieved **98.58% overall accuracy** on 141 completely unseen test windows (2 held-out recordings per activity, never used during training).

| Activity | Samples | Sensitivity | Specificity | Overall Accuracy |
|---|:---:|:---:|:---:|:---:|
| Jumping | 42 | 0.9524 | 1.0000 | 0.9858 |
| Still | 25 | 1.0000 | 1.0000 | 1.0000 |
| Walking | 32 | 1.0000 | 1.0000 | 1.0000 |
| Standing | 42 | 1.0000 | 0.9798 | 0.9858 |

Three of four activities achieved perfect sensitivity and specificity. The minor confusion between jumping and standing arises from low-energy boundary windows at the start and end of recording sessions.

---

## Methodology

### 1. Data Collection
- 50 recordings collected across 4 activities (~10 seconds each)
- Total dataset: 49,300 sensor rows, 494.7 seconds of motion data
- Both phones configured at **100 Hz (10 ms intervals)** via Sensor Logger, confirmed from `Metadata.csv`
- No resampling required as sampling rates were identical across both devices

### 2. Preprocessing and Windowing
- Accelerometer and Gyroscope CSV files merged per recording using nearest-timestamp join (20 ms tolerance)
- Sliding window: **100 samples (1 second)** with **50% overlap (step = 50 samples)**
- Window size chosen to capture one full jump cycle (0.5–1.0 s)
- Total windows produced: **915** across all activities

### 3. Feature Extraction
26 features extracted per window across two domains:

**Time-domain:** mean, variance, standard deviation, RMS (per axis), Signal Magnitude Area (acc and gyro), axis correlations (xy, xz)

**Frequency-domain (FFT):** dominant frequency, spectral energy, spectral entropy (acc axes only)

All features normalized using **Z-score standardization** (StandardScaler fitted on training data only to prevent data leakage).

### 4. Model Architecture
One `GaussianHMM` trained per activity (4 models total):

| Component | Configuration |
|---|---|
| Hidden states | 4 per model |
| Emission model | Gaussian, diagonal covariance |
| Training algorithm | Baum–Welch EM (tol = 1e-4) |
| Initial/transition probs | Uniform initialization |
| Inference | Maximum log-likelihood across all 4 models |

### 5. Training Convergence (Baum–Welch)

| Activity | Training Windows | Iterations to Converge | Final Log-Likelihood |
|---|:---:|:---:|:---:|
| Jumping | 218 | 11 | −6,196.61 nats |
| Still | 167 | 6 | 18,151.66 nats |
| Walking | 159 | 16 | 3,882.05 nats |
| Standing | 230 | 11 | 18,005.07 nats |

Convergence was determined by log-likelihood improvement threshold (`tol = 1e-4`), not an arbitrary iteration cap.

### 6. Viterbi Decoding
The Viterbi algorithm (`model.decode(algorithm='viterbi')`) was applied to recover the most probable hidden state sequence for each test recording, revealing the internal sub-phase structure of each activity (e.g. crouch → push-off → airborne → landing within jumping).

---

## Repository Structure

```
human-activity-recognition-hmm/
│
├── data/
│   ├── raw/
│   │   ├── jumping/          ← 12 zip recordings (Carine Umugabekazi)
│   │   ├── still/            ← 12 zip recordings (Carine Umugabekazi)
│   │   ├── walking/          ← 13 zip recordings (Carine Ahishakiye)
│   │   └── standing/         ← 13 zip recordings (Carine Ahishakiye)
│   └── processed/
│       └── features_normalized.csv
│
├── notebooks/
│   └── activity_recognition_hmm.ipynb
│
├── models/
│   ├── hmm_models.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
│
├── figures/
│   ├── raw_signals.png
│   ├── fft_per_activity.png
│   ├── baum_welch_convergence.png
│   ├── transition_matrices.png
│   ├── initial_state_distribution.png
│   ├── emission_means_heatmap.png
│   ├── emission_covariance_heatmap.png
│   ├── confusion_matrix.png
│   ├── sensitivity_specificity.png
│   ├── decoded_sequence.png
│   ├── viterbi_state_paths.png
│   └── score_distribution.png
│
├── report/
│   └── Hidden_Markov_Models_Group_12_Report.pdf
│
├── requirements.txt
└── README.md
```

---

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/Carine-69/human-activity-recognition-hmm.git
cd human-activity-recognition-hmm
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the notebook**

Open and run all cells in `notebooks/activity_recognition_hmm.ipynb` from top to bottom. The notebook will:
- Load and validate all 50 recordings from `data/raw/`
- Extract features and normalize them
- Train one GaussianHMM per activity using Baum–Welch
- Decode test sequences using Viterbi
- Evaluate on unseen held-out recordings and generate all figures

---

## Dependencies

```
numpy
pandas
matplotlib
seaborn
scikit-learn
hmmlearn
scipy
```

Install all at once:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn hmmlearn scipy
```