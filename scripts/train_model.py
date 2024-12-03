import pandas as pd
from sklearn.model_selection import train_test_split  # Correct import for train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib  # For saving the trained model
import os

# Path to the pre-combined dataset
combined_csv_path = r"c:\Users\nldad\Documents\Final-Project-BME3053C\Final-Project-BME3053C-New\features_dataset.csv"

# Read the pre-combined dataset into a DataFrame
data = pd.read_csv(combined_csv_path)
print("Data loaded:")
print(data.head())



    # Check if 'Label' column exists and proceed with feature extraction and splitting
if 'Label' in data.columns:
    # Features: all columns except 'Label'
    X = data.drop('Label', axis=1)
    
    # Labels: the 'Label' column
    y = data['Label']
    
    # Split the data into training and testing sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize the model (Random Forest as an example)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train the model on the training data
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model for future use
    model_filename = r"c:\Users\nldad\Documents\Final-Project-BME3053C\random_forest_model.pkl"
    joblib.dump(model, model_filename)
    print(f"Model saved as {model_filename}")
    
    # Save the train and test splits for future use
    X_train.to_csv(r"c:\Users\nldad\Documents\Final-Project-BME3053C\train_features.csv", index=False)
    X_test.to_csv(r"c:\Users\nldad\Documents\Final-Project-BME3053C\test_features.csv", index=False)
    y_train.to_csv(r"c:\Users\nldad\Documents\Final-Project-BME3053C\train_labels.csv", index=False)
    y_test.to_csv(r"c:\Users\nldad\Documents\Final-Project-BME3053C\test_labels.csv", index=False)
    
    print("Train-test split completed and saved.")
else:
    print("No 'Label' column found. Please check your data for labels.")

