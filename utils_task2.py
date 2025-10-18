# ===============================================================
# FILE: Best_Model/utils.py
# ===============================================================
import numpy as np

def compute_patient_features(seqs, ex_ids):

    # Extracts features for a patient from multiple sequences.
    # Computes amplitude and speed metrics for the limbs.

    all_feats = []

    for seq in seqs:
        seq = np.array(seq)
        if seq.ndim == 2:
            seq = seq.reshape(-1, 33, 2)

        center = (seq[:, 23, :] + seq[:, 24, :]) / 2
        seq_centered = seq - center[:, None, :]

        diffs = np.diff(seq_centered, axis=0)
        speed = np.linalg.norm(diffs, axis=2)

        left_arm = [11, 13, 15]
        right_arm = [12, 14, 16]
        left_leg = [23, 25, 27]
        right_leg = [24, 26, 28]

        feats = [
            np.mean(np.ptp(seq_centered[:, left_arm, :], axis=0)),
            np.mean(np.ptp(seq_centered[:, right_arm, :], axis=0)),
            np.mean(np.ptp(seq_centered[:, left_leg, :], axis=0)),
            np.mean(np.ptp(seq_centered[:, right_leg, :], axis=0)),
            np.mean(speed[:, left_arm]),
            np.mean(speed[:, right_arm]),
            np.mean(speed[:, left_leg]),
            np.mean(speed[:, right_leg])
        ]
        all_feats.append(feats)

    # Average of the features across all sequences of the patien
    return np.mean(all_feats, axis=0)

def extract_features_from_patients(patient_list, dataset):

    #Extract features for a list of patients from the dataset.
    
    feats = []
    for pid in patient_list:
        seqs = dataset[dataset['Patient_Id'] == pid]['Skeleton_Sequence'].values
        ex_ids = dataset[dataset['Patient_Id'] == pid]['Exercise_Id'].values
        f = compute_patient_features(seqs, ex_ids)
        feats.append(f)
    return np.array(feats)
