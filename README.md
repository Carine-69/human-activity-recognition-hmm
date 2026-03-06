# Human Activity Recognition using Hidden Markov Models

A machine learning project that uses smartphone sensor data to recognise 
four human activities: jumping, standing, walking and still.

## Group Members
- Carine Umugabekazi — Jumping, Standing
- Carine Ahishakiye — Walking, Still

## Results
Overall accuracy: **98.58%**

| Activity | Sensitivity | Specificity |
|---|---|---|
| Jumping | 0.95 | 1.00 |
| Still | 1.00 | 1.00 |
| Walking | 1.00 | 1.00 |
| Standing | 1.00 | 0.98 |

## How it works
We recorded accelerometer and gyroscope data on our phones using Sensor Logger 
at 100 Hz. We extracted 26 features from 1-second windows then trained one 
Gaussian HMM per activity using Baum-Welch. Viterbi decoding was used to find 
the most likely hidden state sequence for each test recording.

## Folder Structure
```
HMM-Human-Activity-Recognition/
│
├── data/
│   ├── raw/
│   │   ├── jumping/        ← zip recordings (Person 1)
│   │   ├── still/          ← zip recordings (Person 1)
│   │   ├── walking/        ← zip recordings (Person 2)
│   │   └── standing/       ← zip recordings (Person 2)
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
│   └── HMM_Activity_Recognition_Report.pdf
│
├── requirements.txt
└── README.md

## Requirements
```
pip install numpy pandas matplotlib seaborn scikit-learn hmmlearn scipy
```

## How to run
1. Clone the repo
2. Add your zip files to the correct folder under `data/raw/`
3. Open and run `hmm_activity_recognition.ipynb` top to bottom

