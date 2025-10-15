# ===============================================================
# REALISTIC EVALUATION + FINAL MODEL TRAINING (SUBMISSION VERSION)
# ===============================================================

import os
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from utils_task2 import extract_patient_features_improved


# ===============================================================
# 1. LOAD DATA
# ===============================================================
BASE_DIR = os.path.dirname(__file__)
data = pd.read_pickle(os.path.join(BASE_DIR, "Xtrain2.pkl"))
Y_train = np.load(os.path.join(BASE_DIR, "Ytrain2.npy"))
patients = sorted(data['Patient_Id'].unique())

print("="*70)
print("REALISTIC EVALUATION: MANUAL HOLDOUT")
print("="*70)
print(f"\nTotal: {len(patients)} patients")
print(f"Class distribution: {np.bincount(Y_train)} (Left=0, Right=1)")


# ===============================================================
# 2. MANUAL HOLDOUT SPLIT
# ===============================================================
print("\n" + "="*70)
print("MANUAL HOLDOUT STRATEGY")
print("="*70)

left_patients = [p for p, y in zip(patients, Y_train) if y == 0]
right_patients = [p for p, y in zip(patients, Y_train) if y == 1]

# ✏️ Choose how many holdout patients to use
# For example: 2 total (1 left + 1 right), 3 total (2 left + 1 right), 4 total (2 left + 2 right)
np.random.seed(42)  # Comment out to make the split random every run
holdout_left = np.random.choice(left_patients, size=2, replace=False)
holdout_right = np.random.choice(right_patients, size=2, replace=False)
holdout_patients = list(holdout_left) + list(holdout_right)

train_patients = [p for p in patients if p not in holdout_patients]

print(f"\nHoldout (test) patients: {sorted(holdout_patients)}")
print(f"Training patients: {sorted(train_patients)} ({len(train_patients)} patients)")


# ===============================================================
# 3. FEATURE EXTRACTION
# ===============================================================
def extract_features(patient_list, dataset):
    feats = []
    for pid in patient_list:
        seqs = dataset[dataset['Patient_Id'] == pid]['Skeleton_Sequence'].values
        ex_ids = dataset[dataset['Patient_Id'] == pid]['Exercise_Id'].values
        f = extract_patient_features_improved(seqs, ex_ids)
        feats.append(f)
    return np.array(feats)

print("\n" + "="*70)
print("EXTRACTING FEATURES")
print("="*70)

X_train = extract_features(train_patients, data)
X_test = extract_features(holdout_patients, data)
print(f"Training features: {X_train.shape}")
print(f"Test features:     {X_test.shape}")


# ===============================================================
# 4. SCALE FEATURES
# ===============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ===============================================================
# 5. GRID SEARCH (MODEL SELECTION)
# ===============================================================
print("\n" + "="*70)
print("TRAINING MODELS (Grid Search on training patients)")
print("="*70)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC())
])

param_grid = [
    {'clf': [LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)],
     'clf__C': [0.01, 0.1, 1.0]},
    {'clf': [SVC(kernel='rbf', class_weight='balanced', random_state=42)],
     'clf__C': [0.1, 1.0]},
    {'clf': [SVC(kernel='linear', class_weight='balanced', random_state=42)],
     'clf__C': [0.1]},
    {'clf': [RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42)],
     'clf__max_depth': [2, 3]}
]

grid = GridSearchCV(pipe, param_grid, scoring='balanced_accuracy', cv=3, n_jobs=-1)
grid.fit(X_train_scaled, [Y_train[patients.index(p)] for p in train_patients])

best_model = grid.best_estimator_
best_name = type(best_model.named_steps['clf']).__name__
best_params = grid.best_params_
best_cv_ba = grid.best_score_

print(f"\nBest model from GridSearchCV: {best_name} {best_params}")
print(f"Cross-val Balanced Accuracy: {best_cv_ba:.4f}")


# ===============================================================
# 6. EVALUATE ON HOLDOUT PATIENTS
# ===============================================================
y_test_split = np.array([Y_train[patients.index(p)] for p in holdout_patients])
y_pred = best_model.predict(X_test_scaled)

holdout_ba = balanced_accuracy_score(y_test_split, y_pred)
holdout_acc = accuracy_score(y_test_split, y_pred)

print("\n" + "="*70)
print(f"BEST MODEL ON HOLDOUT: {best_name}")
print(f"Holdout Balanced Accuracy: {holdout_ba:.4f}")
print(f"Holdout Accuracy:          {holdout_acc:.4f}")
print("="*70)

print("\nHoldout predictions (each patient):")
for pid, true, pred in zip(holdout_patients, y_test_split, y_pred):
    true_lbl = 'Left' if true == 0 else 'Right'
    pred_lbl = 'Left' if pred == 0 else 'Right'
    mark = '✓' if true == pred else '✗'
    print(f"  Patient {pid}: True={true_lbl:5s}, Pred={pred_lbl:5s} {mark}")

cm = confusion_matrix(y_test_split, y_pred, labels=[0,1])
print(f"\nConfusion Matrix (Holdout):")
print(f"                Pred Left  Pred Right")
print(f"  True Left        {cm[0,0]:3d}        {cm[0,1]:3d}")
print(f"  True Right       {cm[1,0]:3d}        {cm[1,1]:3d}")

print("\n" + classification_report(y_test_split, y_pred,
                                    labels=[0,1],
                                    target_names=['Left', 'Right'],
                                    digits=4,
                                    zero_division=0))


# ===============================================================
# 7. RETRAIN BEST MODEL ON ALL PATIENTS (FINAL SUBMISSION)
# ===============================================================
print("\n" + "="*70)
print("RETRAINING BEST MODEL ON ALL PATIENTS")
print("="*70)

# Extract all features
X_all = extract_features(patients, data)
scaler_final = StandardScaler()
X_all_scaled = scaler_final.fit_transform(X_all)

# Retrain final model
final_model = type(best_model.named_steps['clf'])(**best_model.named_steps['clf'].get_params())
final_model.fit(X_all_scaled, Y_train)

# Training performance
train_ba = balanced_accuracy_score(Y_train, final_model.predict(X_all_scaled))
train_acc = accuracy_score(Y_train, final_model.predict(X_all_scaled))

print(f"Training BA:  {train_ba:.4f}")
print(f"Training Acc: {train_acc:.4f}")


# ===============================================================
# 8. SAVE FINAL MODEL (SUBMISSION)
# ===============================================================
joblib.dump({
    'model': final_model,
    'scaler': scaler_final
}, "mymodel_task2_final.pkl")

print("\n" + "="*70)
print("MODEL SAVED")
print("="*70)
print(f"✓ Saved: mymodel_task2_final.pkl")
print(f"  Best model: {best_name}")
print(f"  Holdout BA (realistic test estimate): {holdout_ba:.4f}")
print(f"  Training BA (all patients): {train_ba:.4f}")
print("="*70)
