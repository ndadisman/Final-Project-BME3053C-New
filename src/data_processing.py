import pandas as pd

def clean_data(data):
    # Remove any rows with missing values
    return data.dropna()

def transform_data(data):
    # Example transformation (e.g., scaling or normalizing data)
    return (data - data.min()) / (data.max() - data.min())
