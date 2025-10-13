# mymodel.py
import joblib
import numpy as np

from utils import patient_features  # feature engineering function

# Load trained model once
model_data = joblib.load("classification_model.pkl")
model = model_data['model']

def predict(X_test):
    """
    Receives X_test: shape (N,132)
    Returns predictions: shape (N,)
    """
    X_features = patient_features(X_test)  # transform to 105 features
    return model.predict(X_features)
