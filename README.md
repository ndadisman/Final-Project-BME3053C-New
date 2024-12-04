# Final-Project-BME3053C-New
This project focuses on classifying sleep stages using EEG signals. We aim to predict the stage of sleep based on spectral power density (PSD) values from different EEG frequency bands.
## Table of Contents
1. [Data Collection](#data-collection)
2. [Data Preprocessing](#data-preprocessing)
3. [Model Training](#model-training)
4. [Evaluation](#evaluation)
5. [Requirements](#requirements)
6. [Installation](#installation)
7. [Usage](#usage)
## Installation
To install the necessary dependencies, use pip:
```bash
pip install pandas scikit-learn joblib numpy matplotlib
## Usage
To run the project, place your EEG data CSV files in the appropriate directory. Then execute:
```bash
python train_model.py
## Data Collection
The dataset consists of EEG data files that contain PSD values for different frequency bands, labeled by their corresponding sleep stage.
## Data Preprocessing
Combine multiple CSV files into one dataset, and create new features such as `Dominant_Wave` based on the highest PSD value in each record.
## Model Training
The model used is a Random Forest Classifier, which is trained on the PSD values of EEG signals to classify sleep stages.
## Evaluation
The model's accuracy is measured, and a classification report is generated that includes precision, recall, and F1-score for the predicted sleep stages.
