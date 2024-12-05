# Final-Project-BME3053C-New

This project focuses on classifying sleep stages using EEG signals. We aim to predict the stage of sleep based on spectral power density (PSD) values from different EEG frequency bands.

## Table of Contents
1. [Data Collection](#data-collection)
2. [Data Preprocessing](#data-preprocessing)
3. [Model Training](#model-training)
4. [Evaluation](#evaluation)
5. [Requirements](#requirements)
6. [Order of Scripts](#order-of-scripts)
7. [Installation](#installation)

## Data Collection
We use an open-source EEG dataset for this project from St. Vincent's University Hospital / University College Dublin Sleep Apnea Database which can be found at https://physionet.org/content/ucddb/1.0.0/. This specific dataset which contains data of different patients with sleep apnea. The dataset includes EEG signals sampled at consistent intervals.

### Features Extracted:
- **Delta** (0.5–4 Hz)
- **Theta** (4–8 Hz)
- **Alpha** (8–13 Hz)
- **Beta** (13–30 Hz)
- **Gamma** (>30 Hz)

## Data Preprocessing
Raw EEG signals are preprocessed to extract the spectral power density (PSD) for each frequency band. Key steps include:
- Normalization of the EEG signals
- Removal of noise and artifacts
- Extraction of PSD values using Fast Fourier Transform (FFT)

## Model Training
A machine learning classifier is trained to predict sleep stages based on the PSD values. The workflow includes:
- Splitting the dataset into training and testing sets
- Training a classifier using Random Forests
- Saving the trained model for evaluation and deployment

## Evaluation
The trained model is evaluated on the testing set using metrics such as:
- Accuracy
- Precision
- Recall
- F1 Score  

## Requirements
The following Python libraries are required:
- `pandas`
- `scikit-learn`
- `joblib`
- `numpy`
- `matplotlib`
- `scipy`

## Order of Scripts
To execute the project in the correct order, use the following sequence:
1. **`viewing_signal.py` (Optional):** Displays an image of the raw EEG signal.
2. **`edf_reading.py`:** Prepares the raw EEG data for analysis by converting the image into data.
3. **`feature_extraction.py`:** Prepares the raw EEG data for analysis by cleaning and extracting PSD values.
4. **`combine_features.py`:** Combines the EEG data into one data frame.
5. **`train_model.py`:** Trains the sleep stage classifier on the preprocessed data and predicts the sleep stage. 

## Installation
To install the necessary dependencies, use pip:
```bash
pip install pandas scikit-learn joblib numpy matplotlib scipy
