# =========================================================================
# Project: Identification of the impaired side - mycode.py
# Instituto Superior Técnico - MEEC
#
# Students:
#   Inês Monteiro (ist1113307)
#   Tiago Anastácio (ist1116348)
#
# Date: 18th October
# =========================================================================

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (balanced_accuracy_score, accuracy_score, f1_score,
                             precision_score, recall_score, confusion_matrix,
                             classification_report)
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline

from utils_task2 import extract_features_from_patients

# ===============================================================
# 1. LOAD DATA
# ===============================================================
BASE_DIR = os.path.dirname(__file__)
data = pd.read_pickle(os.path.join(BASE_DIR, "Xtrain2.pkl"))
Y_train = np.load(os.path.join(BASE_DIR, "Ytrain2.npy"))
patients = sorted(data['Patient_Id'].unique())


# ===============================================================
# 2. DEFINE MODELS AND HYPERPARAMETER GRIDS
# ===============================================================

print("GROUP K-FOLD - 2 folds")
print(f"\nTotal patients: {len(patients)}")
print(f"Class distribution: {np.bincount(Y_train)} (Left=0, Right=1)")


models_and_grids = {
    "LogisticRegression": {
        "model": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        "params": {"classifier__C": [0.001, 0.01, 0.05, 0.1]}
    },
    "SVM_RBF": {
        "model": SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42),
        "params": {"classifier__C": [0.01, 0.1, 0.5], "classifier__gamma": [0.01, 0.05, 0.1]}
    },
    "RandomForest": {
        "model": RandomForestClassifier(class_weight='balanced', random_state=42),
        "params": {"classifier__n_estimators": [5, 10, 20], "classifier__max_depth": [5, 10]}
    },
    "GradientBoosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {"classifier__n_estimators": [50,100], "classifier__learning_rate": [0.001 ,0.01], "classifier__max_depth": [3, 5]}
    }
}

#  EXTRACT PATIENT-LEVEL FEATURES
X_all = extract_features_from_patients(patients, data)
patient_groups = np.arange(len(patients))

# 5. GROUP K-FOLD CROSS-VALIDATION WITH GRID SEARCH
gkf = GroupKFold(n_splits=2, shuffle=True, random_state=42)
best_model_name = None
best_model_pipeline = None
best_cv_score = 0

for name, mg in models_and_grids.items():
    print(f"\nEvaluating model: {name}")
    pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", mg["model"])])
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=mg["params"],
        cv=gkf.split(patients, Y_train, groups=patient_groups),
        scoring='balanced_accuracy',
        n_jobs=-1
    )
    grid.fit(X_all, Y_train)
    print(f"  Best params: {grid.best_params_}")
    print(f"  CV Balanced Accuracy: {grid.best_score_:.4f}")

    if grid.best_score_ > best_cv_score:
        best_cv_score = grid.best_score_
        best_model_name = name
        best_model_pipeline = grid.best_estimator_

# ===============================================================
# 3. TRAIN FINAL MODEL ON ALL PATIENTS
# ===============================================================
final_pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", models_and_grids[best_model_name]["model"])])
final_grid = GridSearchCV(
    estimator=final_pipeline,
    param_grid=models_and_grids[best_model_name]["params"],
    cv=gkf.split(patients, Y_train, groups=patient_groups),
    scoring='balanced_accuracy',
    n_jobs=-1
)
final_grid.fit(X_all, Y_train)
final_model_pipeline = final_grid.best_estimator_

# ===============================================================
# 7. PATIENT-LEVEL EVALUATION
# ===============================================================
y_pred_patients = final_model_pipeline.predict(X_all)

ticks = ['Left', 'Right']
print("\nClassification Report (patient-level):")
print(classification_report(Y_train, y_pred_patients, target_names=ticks, digits=4))

# Additional metrics
ba_all = balanced_accuracy_score(Y_train, y_pred_patients)
acc_all = accuracy_score(Y_train, y_pred_patients)
f1_macro = f1_score(Y_train, y_pred_patients, average='macro')
f1_micro = f1_score(Y_train, y_pred_patients, average='micro')
prec = precision_score(Y_train, y_pred_patients, zero_division=0)
rec = recall_score(Y_train, y_pred_patients)
cm = confusion_matrix(Y_train, y_pred_patients)

print("\nAdditional Metrics (patient-level):")
print(f"Balanced Accuracy: {best_cv_score:.4f}, Balanced Accuracy (all): {ba_all:.4f}")
print(f"Accuracy: {acc_all:.4f}")
print(f"F1 Macro: {f1_macro:.4f}")
print(f"F1 Micro: {f1_micro:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print("Confusion Matrix:")
print(cm)


# ===============================================================
# 9. SAVE FINAL MODEL
# ===============================================================
model_data = {
    'model': final_model_pipeline,
    'model_name': best_model_name,
    'cv_balanced_accuracy': best_cv_score,
    'final_balanced_accuracy_all_patients': ba_all
}

joblib.dump(model_data, "mymodel_task2_final.pkl")
print(f"\nBest model saved as 'mymodel_task2_final.pkl' ({best_model_name})")
