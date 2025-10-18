# =================================================================
# Project: Identification of the impaired side - mymodel.py
# Instituto Superior Técnico - MEEC
#
# Students:
#   Inês Monteiro (ist1113307)
#   Tiago Anastácio (ist1116348)
#
# Date: 18th October
# =================================================================

import joblib
import numpy as np

from utils_task2 import extract_features_from_patients

# Load trained model once
model_data = joblib.load("mymodel_task2_final.pkl")
model = model_data['model']

def predict(X_test):
    
    patients = sorted(X_test['Patient_Id'].unique())
    X_features = extract_features_from_patients(patients, X_test)
    return model.predict(X_features)
