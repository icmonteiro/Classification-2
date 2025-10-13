# ===============================================================
# Project: Classification of Rehabilitation Exercises - Part 2
# Instituto Superior Técnico - MEEC
# ===============================================================

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')  # Suppress sklearn warnings

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, LeaveOneOut, cross_val_score, cross_val_predict
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix, classification_report

from utils_task2 import extract_patient_features, extract_patient_features_improved  # your feature extraction functions

# ===============================================================
# 1. LOAD DATA
# ===============================================================
BASE_DIR = os.path.dirname(__file__)
data = pd.read_pickle(os.path.join(BASE_DIR, "Xtrain2.pkl"))
Y_train = np.load(os.path.join(BASE_DIR, "Ytrain2.npy"))

patients = sorted(data['Patient_Id'].unique())

print("="*70)
print("TASK 2: IMPAIRED SIDE CLASSIFICATION")
print(f"Loaded {len(patients)} patients and {len(data)} sequences")
print(f"Class distribution: {np.bincount(Y_train)}")

# ===============================================================
# 2. EXTRACT FEATURES PER PATIENT
# ===============================================================
print("="*70)
print("EXTRACTING TEMPORAL FEATURES (stroke-relevant asymmetry + movement dynamics)")

# Analyze exercise distribution per patient
print("\nExercise distribution per patient:")
print("Bilateral exercises (E3, E4): Face washing, Putting on socks - should show clear asymmetry")
print("Unilateral exercises (E1, E2, E5): Single-hand tasks - less discriminative")
print()

exercise_analysis = {}
for pid in patients:
    patient_data = data[data['Patient_Id'] == pid]
    exercises = sorted(patient_data['Exercise_Id'].unique())
    n_sequences = len(patient_data)
    
    # Count bilateral vs unilateral
    bilateral_count = len(patient_data[patient_data['Exercise_Id'].isin(['E3', 'E4'])])
    unilateral_count = len(patient_data[patient_data['Exercise_Id'].isin(['E1', 'E2', 'E5'])])
    
    exercise_analysis[pid] = {'exercises': exercises, 'n_sequences': n_sequences}
    print(f"  Patient {pid}: {exercises} ({n_sequences} sequences)")
    print(f"    Bilateral (E3,E4): {bilateral_count}, Unilateral (E1,E2,E5): {unilateral_count}")

X_patients = []
for pid in patients:
    patient_data = data[data['Patient_Id'] == pid]
    seqs = patient_data['Skeleton_Sequence'].values
    exercise_ids = patient_data['Exercise_Id'].values
    
    # Pass exercise IDs for exercise-aware feature extraction
    feats = extract_patient_features_improved(seqs, exercise_ids)
    X_patients.append(feats)
X_patients = np.array(X_patients)
print(f"\nFinal feature matrix: {X_patients.shape}")
print(f"Features per patient: {X_patients.shape[1]} (enhanced: weighted bilateral + range + unilateral asymmetry)")

# ===============================================================
# 3. SCALE FEATURES
# ===============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_patients)

# ===============================================================
# 4. MODEL COMPARISON WITH GROUPKFOLD
# ===============================================================
print("="*70)
print("MODEL COMPARISON (GroupKFold Cross-Validation)")
print("="*70)

# Use GroupKFold with k=3 (4-5 patients per fold) - more stable than LOPO
group_kfold = GroupKFold(n_splits=3)
groups = np.array([data.loc[data['Patient_Id'] == pid].index[0] for pid in patients])

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Cross-validation strategy
group_kfold = GroupKFold(n_splits=3)
print("Cross-validation strategy: GroupKFold with k=3")
print("This respects patient boundaries and is more stable than LOPO for 14 patients")

# Pipeline with optional scaler
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # SVM & Logistic Regression benefit from scaling
    ('clf', SVC())                 # placeholder
])

# Parameter grid using your models dictionary
param_grid = [
    {'clf': [SVC(kernel='rbf', gamma='scale', class_weight='balanced', random_state=42)], 
     'clf__C': [0.005, 0.01, 0.02]},
    
    {'clf': [SVC(kernel='linear', class_weight='balanced', random_state=42)], 
     'clf__C': [0.005, 0.01]},
    
    {'clf': [LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)], 
     'clf__C': [0.005, 0.01]},
    
    {'clf': [RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)], 
     'clf__max_depth': [2, 3, 4]}
]

# Grid search with GroupKFold
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    scoring='balanced_accuracy',
    cv=group_kfold,
    n_jobs=-1
)

# Fit to your data
grid_search.fit(X_scaled, Y_train, groups=groups)

# Best model
best_model = grid_search.best_estimator_
best_name = type(best_model.named_steps['clf']).__name__
best_ba = grid_search.best_score_

print(f"\nBEST MODEL: {best_name}")
print(f"GroupKFold Balanced Accuracy: {best_ba:.4f}")


# Train final model
best_model.fit(X_scaled, Y_train)
Y_pred_cv = cross_val_predict(best_model, X_scaled, Y_train, cv=group_kfold, groups=groups)

# ===============================================================
# 5. EVALUATION & REPORT
# ===============================================================
print("="*70)
print("EVALUATION RESULTS")
print("="*70)

# Confusion matrix
cm = confusion_matrix(Y_train, Y_pred_cv, labels=[0,1])
print("\nConfusion Matrix (GroupKFold CV):")
print(f"                Pred Left  Pred Right")
print(f"  True Left        {cm[0,0]:3d}        {cm[0,1]:3d}")
print(f"  True Right       {cm[1,0]:3d}        {cm[1,1]:3d}")

# Classification report
print("\nClassification Report:")
print(classification_report(Y_train, Y_pred_cv,
                            labels=[0,1],
                            target_names=['Left', 'Right'],
                            digits=4,
                            zero_division=0))

# Misclassified patients analysis
print("\nMisclassified patients:")
for i, (true, pred) in enumerate(zip(Y_train, Y_pred_cv)):
    if true != pred:
        true_label = 'Left' if true == 0 else 'Right'
        pred_label = 'Left' if pred == 0 else 'Right'
        print(f"  Patient {patients[i]}: True={true_label}, Predicted={pred_label}")

# Training performance (for overfitting check)
Y_pred_train = best_model.predict(X_scaled)
train_ba = balanced_accuracy_score(Y_train, Y_pred_train)
print(f"\nTraining Balanced Accuracy: {train_ba:.4f}")
print(f"CV Balanced Accuracy:      {best_ba:.4f}")

if train_ba - best_ba > 0.1:
    print("⚠ WARNING: Large gap between training and CV - possible overfitting!")
else:
    print("✓ Good generalization (small gap between training and CV)")


# ===============================================================
# 6. VISUALIZATION
# ===============================================================
def plot_mediapipe_skeleton(skeleton, ax, title="Skeleton", color='blue'):
    """
    Plot MediaPipe skeleton with proper connections (stick figure)
    
    MediaPipe Pose keypoint connections:
    - Face: 0-1-2-3-4-5-6-7-8-9-10
    - Torso: 11-12, 11-23, 12-24, 23-24
    - Left arm: 11-13-15-17-19-21
    - Right arm: 12-14-16-18-20-22
    - Left leg: 23-25-27-29-31
    - Right leg: 24-26-28-30-32
    """
    
    # Define MediaPipe Pose connections
    connections = [
        # Face outline
        (0, 1), (1, 2), (2, 3), (3, 7),  # nose to left ear
        (0, 4), (4, 5), (5, 6), (6, 8),  # nose to right ear
        (9, 10),  # mouth
        
        # Torso
        (11, 12),  # shoulders
        (11, 23),  # left shoulder to left hip
        (12, 24),  # right shoulder to right hip
        (23, 24),  # hips
        
        # Left arm
        (11, 13),  # left shoulder to left elbow
        (13, 15),  # left elbow to left wrist
        (15, 17),  # left wrist to left pinky
        (15, 19),  # left wrist to left index
        (15, 21),  # left wrist to left thumb
        
        # Right arm
        (12, 14),  # right shoulder to right elbow
        (14, 16),  # right elbow to right wrist
        (16, 18),  # right wrist to right pinky
        (16, 20),  # right wrist to right index
        (16, 22),  # right wrist to right thumb
        
        # Left leg
        (23, 25),  # left hip to left knee
        (25, 27),  # left knee to left ankle
        (27, 29),  # left ankle to left heel
        (27, 31),  # left ankle to left foot index
        
        # Right leg
        (24, 26),  # right hip to right knee
        (26, 28),  # right knee to right ankle
        (28, 30),  # right ankle to right heel
        (28, 32),  # right ankle to right foot index
    ]
    
    # Plot keypoints
    ax.scatter(skeleton[:, 0], skeleton[:, 1], c=color, s=30, alpha=0.8)
    
    # Plot connections (stick figure)
    for start_idx, end_idx in connections:
        if start_idx < len(skeleton) and end_idx < len(skeleton):
            start_point = skeleton[start_idx]
            end_point = skeleton[end_idx]
            ax.plot([start_point[0], end_point[0]], 
                   [start_point[1], end_point[1]], 
                   color=color, linewidth=2, alpha=0.7)
    
    # Highlight important keypoints for stroke analysis
    important_kps = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]  # shoulders, elbows, wrists, hips, knees, ankles
    for kp_idx in important_kps:
        if kp_idx < len(skeleton):
            ax.scatter(skeleton[kp_idx, 0], skeleton[kp_idx, 1], 
                      c='red', s=50, alpha=0.9, marker='o', edgecolors='black')
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.invert_yaxis()  # MediaPipe coordinates (origin at top-left)
    ax.grid(True, alpha=0.3)
    
    # Add keypoint labels for important points
    for kp_idx in [11, 12, 15, 16]:  # shoulders and wrists (most important for stroke)
        if kp_idx < len(skeleton):
            ax.annotate(f'{kp_idx}', (skeleton[kp_idx, 0], skeleton[kp_idx, 1]), 
                       xytext=(5, 5), textcoords='offset points', 
                       fontsize=8, color='red', fontweight='bold')


def plot_patient_skeletons(patient_id, data, max_exercises=6):
    """Plot skeleton visualizations for a patient with proper stick figures"""
    patient_data = data[data['Patient_Id'] == patient_id]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (_, row) in enumerate(patient_data.iterrows()):
        if i >= max_exercises: break
        
        seq = row['Skeleton_Sequence']
        exercise = row['Exercise_Id']  # Use actual exercise ID (E1, E2, E3, E4, E5)
        
        # Plot first frame
        skeleton = seq[0].reshape(33, 2)
        
        plot_mediapipe_skeleton(skeleton, axes[i], 
                               f'Patient {patient_id} - {exercise}', 
                               color='blue')
    
    # Hide unused subplots
    for i in range(len(patient_data), max_exercises):
        axes[i].set_visible(False)
    
    plt.suptitle(f'Patient {patient_id} - Skeleton Visualizations\n(Red dots: Key stroke-relevant keypoints)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'patient_{patient_id}_skeletons.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Skeleton visualization saved: patient_{patient_id}_skeletons.png")

# Plot skeletons for misclassified patients
print("\n" + "="*70)
print("VISUALIZATION")
print("="*70)

# Print MediaPipe keypoint mapping for reference
print("\nMediaPipe Pose Keypoints:")
keypoint_names = [
    "0: nose", "1: left eye (inner)", "2: left eye", "3: left eye (outer)",
    "4: right eye (inner)", "5: right eye", "6: right eye (outer)",
    "7: left ear", "8: right ear", "9: mouth (left)", "10: mouth (right)",
    "11: left shoulder", "12: right shoulder", "13: left elbow", "14: right elbow",
    "15: left wrist", "16: right wrist", "17: left pinky", "18: right pinky",
    "19: left index", "20: right index", "21: left thumb", "22: right thumb",
    "23: left hip", "24: right hip", "25: left knee", "26: right knee",
    "27: left ankle", "28: right ankle", "29: left heel", "30: right heel",
    "31: left foot index", "32: right foot index"
]

for i in range(0, len(keypoint_names), 4):
    print("  " + " | ".join(f"{kp:20s}" for kp in keypoint_names[i:i+4]))

print("\nRed highlighted keypoints in plots: 11,12 (shoulders), 15,16 (wrists), 23,24 (hips), 25,26 (knees), 27,28 (ankles)")
print("These are the most important for stroke asymmetry detection.")

misclassified_patients = []
for i, (true, pred) in enumerate(zip(Y_train, Y_pred_cv)):
    if true != pred:
        misclassified_patients.append(patients[i])

if misclassified_patients:
    print(f"Plotting skeletons for misclassified patients: {misclassified_patients}")
    for pid in misclassified_patients[:2]:  # Plot first 2 misclassified
        plot_patient_skeletons(pid, data)
else:
    print("No misclassified patients to visualize!")

# ===============================================================
# 7. SAVE MODEL
# ===============================================================
model_data = {
    'model': best_model,
    'scaler': scaler,
    'model_name': best_name,
    'balanced_accuracy': best_ba,
    'patients': patients,
    'feature_names': ['Bilateral_Arm_Asym_Weighted', 'Bilateral_Hand_Asym_Weighted', 'Bilateral_Arm_Std', 'Bilateral_Hand_Std', 'Bilateral_Arm_Range', 'Bilateral_Hand_Range', 'Unilateral_Arm_Asym', 'Unilateral_Hand_Asym']
}
joblib.dump(model_data, "mymodel_task2.pkl")

print("\n" + "="*70)
print("MODEL SAVED")
print("="*70)
print(f"✓ Model saved: mymodel_task2.pkl")
print(f"  Best model: {best_name}")
print(f"  Expected Balanced Accuracy: {best_ba:.4f}")
print(f"  Features: {X_patients.shape[1]} per patient")
print("="*70)
