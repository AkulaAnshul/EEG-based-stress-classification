# EEG Stress Classifier

A real-time EEG-based stress detection web app using machine learning.
Classifies EEG signals from the SAM-40 dataset as **Stressed** or **Relaxed**
using multiple ML models with a live waveform monitor.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Flask](https://img.shields.io/badge/Flask-2.x-lightgrey)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Models

| Model         | Accuracy |
|---------------|----------|
| SVM           | 70.3%    |
| Random Forest | 75.2%    |
| KNN           | 75.1%    |
| MLP           | 74.9%    |
| XGBoost       | 78.8%    |
| CNN-LSTM      | 78.4%    |

---

## Project Structure

eeg_stress_detection/
├── server.py # Flask backend API
├── dataset.py # Data loading utilities
├── features.py # Feature extraction (band power)
├── variables.py # Path and config variables
├── requirements.txt # Python dependencies
│
├── svm_train.py # Train SVM model
├── rf_train.py # Train Random Forest model
├── knn_train.py # Train KNN model
├── mlp_train.py # Train MLP model
├── coherence_xgb_train.py # Train XGBoost model (rich features)
├── cnn_lstm_train.py # Train CNN-LSTM model
├── master_train.py # Run all training + organize models
│
├── static/
│ ├── index.html # Frontend UI
│ └── style.css # Dark theme styles
│
├── models/ # Saved .pkl / .keras files (git ignored)
└── Data/ # SAM-40 .mat files (git ignored)
---

## Setup

### 1. Clone the repository

git clone https://github.com/YOUR_USERNAME/eeg-stress-classifier.git
cd eeg-stress-classifier

### 2. Install dependencies
pip install -r requirements.txt

### 3. Download the dataset
Download the SAM-40 dataset
and place the .mat files inside the Data/ folder as per the paths in variables.py.

### 4. Train all models
python master_train.py
This will train all 6 models and save them to the models/ folder automatically.

### 5. Run the app
python server.py
Open your browser at http://127.0.0.1:5000

Usage
Select a model from the dropdown (SVM, RF, KNN, MLP, XGBoost, CNN-LSTM)

Click Load Sample to load a random EEG epoch with live waveform

Watch the real-time scrolling EEG signal

Click Classify EEG Signal to get the prediction

See Prediction, Ground Truth, and Match result instantly

Dataset
SAM-40 — Stress and Affect Multi-Modal (EEG) Dataset

40 subjects, 3 stress-inducing tasks (Arithmetic, Stroop, Mirror Image) + Relax

32-channel EEG, 128 Hz sampling rate

Features: Band power (delta, theta, alpha, beta, gamma) across 32 channels

Requirements
Python 3.10+

Flask

NumPy, SciPy, scikit-learn

XGBoost

imbalanced-learn (SMOTE)

TensorFlow / Keras (for CNN-LSTM)

MNE, MNE-Features
