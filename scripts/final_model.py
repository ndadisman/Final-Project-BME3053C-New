import joblib

# Save the trained model to a file
joblib.dump(clf, 'train_model.pkl')
print("Model saved.")

# To load the model later:
# clf = joblib.load('trained_model.pkl')
