import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib  # For saving the trained model
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import seaborn as sns

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
    # Apply the function to classify sleep stages for each row
    
    # Features: PSD values
    X = data[required_columns]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Labels: Sleep stage
    y = data['Sleep Stage']
    
    # Split the data into training and testing sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Initialize the classification model
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    
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



# Determine correct and incorrect predictions
correct_predictions = y_test == y_pred
incorrect_predictions = ~correct_predictions

# Count correct and incorrect predictions
correct_count = np.sum(correct_predictions)
incorrect_count = np.sum(incorrect_predictions)

# Prepare data for the bar chart
categories = ['Correct', 'Incorrect']
signal_count = [correct_count, incorrect_count]

# Create the bar chart
plt.figure(figsize=(8, 6))
plt.bar(categories, signal_count, color=['green', 'red'])
plt.xlabel('Prediction Categories')
plt.ylabel('Signal Count')
plt.title('Predicted vs. Classified: Correct vs. Incorrect Predictions')
plt.xticks(ticks=np.arange(len(categories)), labels=categories)
plt.show()


# Convert the classification report to a DataFrame
report_df = pd.DataFrame(model).transpose()

# Filter out the accuracy, macro avg, and weighted avg rows
class_metrics = report_df.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')

# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(class_metrics.iloc[:, :-1], annot=True, cmap="Blues", cbar=False)  # Exclude support column
plt.title("Classification Report Heatmap")
plt.xlabel("Metrics")
plt.ylabel("Classes")
plt.show()