from sklearn.model_selection import train_test_split
import pandas as pd

# Load the dataset
data = pd.read_csv(r"c:\Users\nldad\Documents\Final-Project-BME3053C\Final-Project-BME3053C-New\features")

# Separate features and labels
X = data.drop(columns=["Label"])  # Replace 'Label' with your label column name
y = data["Label"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training and testing sets created.")
