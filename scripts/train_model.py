import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib  # For saving the trained model

# Path to the pre-combined dataset
combined_csv_path = r"c:\Users\nldad\Documents\Final-Project-BME3053C\Final-Project-BME3053C-New\features_dataset.csv"

# Read the pre-combined dataset into a DataFrame
data = pd.read_csv(combined_csv_path)
print("Data loaded:")
print(data.head(30))
print(data.describe())

# Ensure required PSD columns exist
required_columns = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
if set(required_columns).issubset(data.columns):
    # Add a new column to classify dominant wave (based on the maximum PSD value)
    data['Dominant_Wave'] = data[required_columns].idxmax(axis=1)

    # Features: PSD values
    X = data[required_columns]
    
    # Labels: Dominant wave type
    y = data['Dominant_Wave']
    
    # Split the data into training and testing sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize the classification model
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
    model_filename = r"c:\Users\nldad\Documents\Final-Project-BME3053C\trained_model.pkl"
    joblib.dump(model, model_filename)
    print(f"Model saved as {model_filename}")
else:
    print(f"Required PSD columns {required_columns} not found in the dataset.")
