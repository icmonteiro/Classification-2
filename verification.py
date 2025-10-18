import pandas as pd
import numpy as np
from mymodel import predict
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score

# Load training data for verification (assuming Xtrain2.pkl and Ytrain2.npy are available)
X_test_raw = pd.read_pickle("Xtrain2.pkl")
Y_test = np.load("Ytrain2.npy")

# Make predictions
Y_pred = predict(X_test_raw)

# Print first 10 predictions
print("\nFirst 10 predictions:", Y_pred[:10])

# Quick metrics
print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("Balanced Accuracy:", balanced_accuracy_score(Y_test, Y_pred))
print("F1 macro:", f1_score(Y_test, Y_pred, average='macro'))

# Check shape
print("Predictions shape:", Y_pred.shape)
print("Expected shape: (14,) for 14 patients")

# Check unique patients
patients = sorted(X_test_raw['Patient_Id'].unique())
print("Number of unique patients:", len(patients))
print("Patient IDs:", patients)

# Verify no errors in prediction
print("All predictions are 0 or 1: \n", np.all(np.isin(Y_pred, [0, 1])))

# Note: For submission, X_test will have 115 rows (sequences) for 4 patients, predict should return (4,)
print("\nFor submission:")
print("- X_test will have 115 rows (sequences from 4 patients)")
print("- predict(X_test) should return shape (4,) - one prediction per patient")
print("- Patients are grouped by Patient_Id, features aggregated per patient")
