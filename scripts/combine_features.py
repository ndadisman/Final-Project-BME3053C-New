import os
import pandas as pd

features_folder = r"c:\Users\nldad\Documents\Final-Project-BME3053C\Final-Project-BME3053C-New\features"
combined_csv_path = r"c:\Users\nldad\Documents\Final-Project-BME3053C\Final-Project-BME3053C-New\features_dataset.csv"

# Combine all CSV files into one DataFrame
dataframes = []
for file in os.listdir(features_folder):
    if file.endswith(".csv"):
        # Read the CSV file
        df = pd.read_csv(os.path.join(features_folder, file))
        
        # Assuming the label is in the filename, e.g., 'class1_data.csv' -> label 'class1'
        label = file.split('_')[0]  # Adjust this based on your filename convention
        df['Signal'] = label  # Add the label as a new column
        
        dataframes.append(df)

# Combine all dataframes into one
combined_df = pd.concat(dataframes, ignore_index=True)

# Save the combined dataset with the added label column
combined_df.to_csv(combined_csv_path, index=False)
print(f"Combined dataset saved at {combined_csv_path}")
