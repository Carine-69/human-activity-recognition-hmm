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
human-activity-recognition-hmm/
  data/
    raw/
      jumping/
      standing/
      walking/
      still/
    processed/
      features_normalized.csv
  models/
    hmm_models.pkl
    scaler.pkl
    label_encoder.pkl
  figures/
  notebooks/
    hmm_activity_recognition.ipynb
```

## Requirements
```
pip install numpy pandas matplotlib seaborn scikit-learn hmmlearn scipy
```

## How to run
1. Clone the repo
2. Add your zip files to the correct folder under `data/raw/`
3. Open and run `hmm_activity_recognition.ipynb` top to bottom