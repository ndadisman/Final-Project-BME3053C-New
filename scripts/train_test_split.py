import os
import pandas as pd

features_folder = r"c:\Users\nldad\Documents\Final-Project-BME3053C\Final-Project-BME3053C-New\features"

dataframes = []
for file in os.listdir(features_folder):
    if file.endswith(".csv"):
        file_path = os.path.join(features_folder, file)
        print(f"Reading file: {file_path}")
        df = pd.read_csv(file_path)
        dataframes.append(df)

# Combine all dataframes into one
if dataframes:
    data = pd.concat(dataframes, ignore_index=True)
    print("Combined data:")
    print(data.head())
else:
    print("No CSV files found in the directory.")
