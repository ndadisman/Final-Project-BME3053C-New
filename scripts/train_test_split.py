import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Define the directory containing the feature CSV files
features_folder = r"c:\Users\nldad\Documents\Final-Project-BME3053C\Final-Project-BME3053C-New\features"

dataframes = []  # List to store dataframes
for file in os.listdir(features_folder):
    if file.endswith(".csv"):
        file_path = os.path.join(features_folder, file)
        print(f"Reading file: {file_path}")
        
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        
        # Extract label from the filename (assuming the label is the first part of the filename, e.g., 'class1_data.csv')
        label = file.split('_')[0]  # Adjust this based on your file naming convention
        df['Label'] = label  # Add the label as a new column in the DataFrame
        
        dataframes.append(df)

# Combine all dataframes into one large dataframe
if dataframes:
    data = pd.concat(dataframes, ignore_index=True)
    print("Combined data:")
    print(data.head())
    
    # Check if 'Label' column exists and proceed with feature extraction and splitting
    if 'Label' in data.columns:
        # Features: all columns except 'Label'
        X = data.drop('Label', axis=1)
        
        # Labels: the 'Label' column
        y = data['Label']
        
        # Split the data into training and testing sets (80-20 split)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Save the train and test splits for future use
        X_train.to_csv(r"c:\Users\nldad\Documents\Final-Project-BME3053C\train_features.csv", index=False)
        X_test.to_csv(r"c:\Users\nldad\Documents\Final-Project-BME3053C\test_features.csv", index=False)
        y_train.to_csv(r"c:\Users\nldad\Documents\Final-Project-BME3053C\train_labels.csv", index=False)
        y_test.to_csv(r"c:\Users\nldad\Documents\Final-Project-BME3053C\test_labels.csv", index=False)
        
        print("Train-test split completed and saved.")
    else:
        print("No 'Label' column found. Please check your data for labels.")
else:
    print("No CSV files found in the directory.")
