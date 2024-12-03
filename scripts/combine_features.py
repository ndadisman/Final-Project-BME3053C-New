import os
import pandas as pd

features_folder = r"c:\Users\nldad\Documents\Final-Project-BME3053C\Final-Project-BME3053C-New\features"
combined_csv_path =r"c:\Users\nldad\Documents\Final-Project-BME3053C\Final-Project-BME3053C-New\features_dataset.csv"

# Combine all CSV files into one DataFrame
dataframes = []
for file in os.listdir(features_folder):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(features_folder, file))
        # Optionally, add a 'Label' column here for supervised learning
        # df['Label'] = your_label_logic_here
        dataframes.append(df)

combined_df = pd.concat(dataframes, ignore_index=True)
combined_df.to_csv(combined_csv_path, index=False)
print(f"Combined dataset saved at {combined_csv_path}")
