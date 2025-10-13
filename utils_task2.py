"""
IMPROVED Utils for Task 2 with better asymmetry features
"""

import numpy as np

def extract_sequence_features(sequence):
    """
    Extract asymmetry features from ONE sequence.
    Focus on LEFT vs RIGHT differences (impairment signature).
    
    Input: (T, 66) - T time steps, 33 keypoints × 2
    Output: 20 features (asymmetry-focused)
    """
    T = sequence.shape[0]
    seq = sequence.reshape(T, 33, 2)

    # Temporal statistics
    mean_pos = seq.mean(axis=0)  # (33, 2)
    std_pos = seq.std(axis=0)    # (33, 2) - movement amplitude
    
    # Left vs Right keypoint indices
    left_kps = np.array([1,2,3,7,11,13,15,17,19,21,23,25,27,29,31])
    right_kps = np.array([4,5,6,8,12,14,16,18,20,22,24,26,28,30,32])
    
    # ============================================================
    # PRIMARY ASYMMETRY FEATURES
    # ============================================================
    
    # 1. MOVEMENT ASYMMETRY (std = how much it moves)
    left_mov = std_pos[left_kps].mean()
    right_mov = std_pos[right_kps].mean()
    mov_ratio = left_mov / (right_mov + 1e-6)  # <1: left impaired, >1: right impaired
    mov_diff = left_mov - right_mov             # negative: left moves less
    
    # 2. HAND ASYMMETRY (most important!)
    left_wrist_mov = std_pos[15].mean()   # keypoint 15
    right_wrist_mov = std_pos[16].mean()  # keypoint 16
    wrist_ratio = left_wrist_mov / (right_wrist_mov + 1e-6)
    wrist_diff = left_wrist_mov - right_wrist_mov
    
    # 3. ARM ASYMMETRY
    left_elbow_mov = std_pos[13].mean()
    right_elbow_mov = std_pos[14].mean()
    elbow_ratio = left_elbow_mov / (right_elbow_mov + 1e-6)
    
    left_shoulder_mov = std_pos[11].mean()
    right_shoulder_mov = std_pos[12].mean()
    shoulder_ratio = left_shoulder_mov / (right_shoulder_mov + 1e-6)
    
    # 4. LEG ASYMMETRY
    left_knee_mov = std_pos[25].mean()
    right_knee_mov = std_pos[26].mean()
    knee_ratio = left_knee_mov / (right_knee_mov + 1e-6)
    
    # 5. SPATIAL ASYMMETRY (position bias)
    # Impaired side might be positioned differently
    left_center_x = mean_pos[left_kps, 0].mean()
    right_center_x = mean_pos[right_kps, 0].mean()
    spatial_x_asymmetry = left_center_x - right_center_x
    
    # 6. RANGE OF MOTION ASYMMETRY
    left_range = (seq[:, left_kps].max(axis=0) - seq[:, left_kps].min(axis=0)).mean()
    right_range = (seq[:, right_kps].max(axis=0) - seq[:, right_kps].min(axis=0)).mean()
    range_ratio = left_range / (right_range + 1e-6)
    
    # 7. VELOCITY ASYMMETRY (if enough frames)
    if T > 2:
        velocity = np.diff(seq, axis=0)
        vel_mag = np.linalg.norm(velocity, axis=2)  # (T-1, 33)
        left_vel = vel_mag[:, left_kps].mean()
        right_vel = vel_mag[:, right_kps].mean()
        vel_ratio = left_vel / (right_vel + 1e-6)
    else:
        vel_ratio = 1.0
    
    # 8. CONSISTENCY (how stable is the movement?)
    # Impaired side might be more erratic
    left_consistency = std_pos[left_kps].std()   # low = consistent
    right_consistency = std_pos[right_kps].std()
    consistency_ratio = left_consistency / (right_consistency + 1e-6)
    
    # ============================================================
    # COMBINE FEATURES
    # ============================================================
    
    features = np.array([
        # Movement ratios (PRIMARY)
        mov_ratio,           # 1 - global left/right
        wrist_ratio,         # 2 - MOST IMPORTANT
        elbow_ratio,         # 3
        shoulder_ratio,      # 4
        knee_ratio,          # 5
        
        # Movement differences
        mov_diff,            # 6
        wrist_diff,          # 7
        
        # Range and velocity
        range_ratio,         # 8
        vel_ratio,           # 9
        
        # Spatial
        spatial_x_asymmetry, # 10
        
        # Consistency
        consistency_ratio,   # 11
        
        # Absolute values (context)
        left_mov,            # 12
        right_mov,           # 13
        left_wrist_mov,      # 14
        right_wrist_mov,     # 15
        left_knee_mov,       # 16
        right_knee_mov,      # 17
        
        # Metadata
        T / 300.0,           # 18 - normalized duration
        np.log(T + 1)        # 19 - log duration
    ])
    
    return features


def extract_patient_features(patient_sequences):
    """
    Aggregate features from ALL sequences of ONE patient.
    
    Input: list of sequences (variable number, variable length)
    Output: patient-level feature vector
    
    Strategy: Extract from each sequence, then aggregate with
    mean, std, min, max, median
    """
    
    # Extract features from each sequence
    seq_features = []
    for seq in patient_sequences:
        feat = extract_sequence_features(seq)
        seq_features.append(feat)
    
    seq_features = np.array(seq_features)  # (n_sequences, n_features)
    
    # Aggregate with MULTIPLE statistics
    patient_feat = np.concatenate([
        seq_features.mean(axis=0),      # Average behavior
        seq_features.std(axis=0),       # Variability across exercises
        seq_features.min(axis=0),       # Minimum asymmetry
        seq_features.max(axis=0),       # Maximum asymmetry
        np.median(seq_features, axis=0) # Robust central tendency
    ])
    
    # Also add: number of sequences (patient activity level)
    n_sequences = len(patient_sequences)
    patient_feat = np.append(patient_feat, n_sequences / 50.0)  # normalized
    
    return patient_feat


# Alternative: Simple aggregation (less prone to overfitting)
def extract_patient_features_simple(patient_sequences):
    """
    Simpler aggregation - just mean (less overfitting risk)
    """
    seq_features = []
    for seq in patient_sequences:
        feat = extract_sequence_features(seq)
        seq_features.append(feat)
    
    seq_features = np.array(seq_features)
    
    # Just mean and std (40 features total instead of 96)
    patient_feat = np.concatenate([
        seq_features.mean(axis=0),
        seq_features.std(axis=0)
    ])
    
    return patient_feat


def extract_sequence_features_improved(sequence):
    """
    Extract stroke-relevant asymmetry features focusing on key body parts.
    
    Input: (T, 66) - skeleton sequence
    Output: 4 features (ultra-simple to prevent overfitting)
    """
    T = sequence.shape[0]
    seq = sequence.reshape(T, 33, 2)

    # Temporal statistics
    std_pos = seq.std(axis=0)    # (33, 2)
    
    # Most important keypoints for stroke impairment
    left_arm = [11, 13, 15]  # shoulder, elbow, wrist
    right_arm = [12, 14, 16]
    left_leg = [23, 25, 27]  # hip, knee, ankle
    right_leg = [24, 26, 28]
    
    # 1. ARM ASYMMETRY (most important for stroke)
    left_arm_mov = std_pos[left_arm].mean()
    right_arm_mov = std_pos[right_arm].mean()
    arm_total = left_arm_mov + right_arm_mov + 1e-6
    arm_asymmetry = (left_arm_mov - right_arm_mov) / arm_total
    
    # 2. HAND ASYMMETRY (most discriminative)
    left_hand_mov = std_pos[15].mean()  # left wrist
    right_hand_mov = std_pos[16].mean()  # right wrist
    hand_total = left_hand_mov + right_hand_mov + 1e-6
    hand_asymmetry = (left_hand_mov - right_hand_mov) / hand_total
    
    # 3. LEG ASYMMETRY
    left_leg_mov = std_pos[left_leg].mean()
    right_leg_mov = std_pos[right_leg].mean()
    leg_total = left_leg_mov + right_leg_mov + 1e-6
    leg_asymmetry = (left_leg_mov - right_leg_mov) / leg_total
    
    # 4. COORDINATION (variability within each side)
    left_variability = std_pos[left_arm + left_leg].std()
    right_variability = std_pos[right_arm + right_leg].std()
    coord_total = left_variability + right_variability + 1e-6
    coordination_asymmetry = (left_variability - right_variability) / coord_total
    
    return np.array([arm_asymmetry, hand_asymmetry, leg_asymmetry, coordination_asymmetry])


def extract_sequence_features_temporal(sequence):
    """
    Extract temporal asymmetry features that capture movement dynamics.
    
    Input: (T, 66) - skeleton sequence
    Output: 6 features (temporal + static)
    """
    T = sequence.shape[0]
    seq = sequence.reshape(T, 33, 2)

    # Static features (same as before)
    std_pos = seq.std(axis=0)
    left_arm = [11, 13, 15]  # shoulder, elbow, wrist
    right_arm = [12, 14, 16]
    
    # Static asymmetry
    left_arm_mov = std_pos[left_arm].mean()
    right_arm_mov = std_pos[right_arm].mean()
    arm_total = left_arm_mov + right_arm_mov + 1e-6
    static_arm_asymmetry = (left_arm_mov - right_arm_mov) / arm_total
    
    left_hand_mov = std_pos[15].mean()
    right_hand_mov = std_pos[16].mean()
    hand_total = left_hand_mov + right_hand_mov + 1e-6
    static_hand_asymmetry = (left_hand_mov - right_hand_mov) / hand_total
    
    # Temporal features (movement dynamics)
    if T > 2:
        # Velocity (first derivative)
        velocity = np.diff(seq, axis=0)  # (T-1, 33, 2)
        vel_mag = np.linalg.norm(velocity, axis=2)  # (T-1, 33)
        
        # Velocity asymmetry
        left_arm_vel = vel_mag[:, left_arm].mean()
        right_arm_vel = vel_mag[:, right_arm].mean()
        vel_total = left_arm_vel + right_arm_vel + 1e-6
        temporal_arm_asymmetry = (left_arm_vel - right_arm_vel) / vel_total
        
        # Acceleration (second derivative) - smoothness
        if T > 3:
            acceleration = np.diff(velocity, axis=0)  # (T-2, 33, 2)
            acc_mag = np.linalg.norm(acceleration, axis=2)  # (T-2, 33)
            
            # Smoothness asymmetry (impaired side might be more jerky)
            left_arm_acc = acc_mag[:, left_arm].mean()
            right_arm_acc = acc_mag[:, right_arm].mean()
            acc_total = left_arm_acc + right_arm_acc + 1e-6
            smoothness_asymmetry = (left_arm_acc - right_arm_acc) / acc_total
        else:
            smoothness_asymmetry = 0.0
    else:
        temporal_arm_asymmetry = 0.0
        smoothness_asymmetry = 0.0
    
    # Movement consistency (how stable is the movement?)
    if T > 5:
        # Split sequence into segments and check consistency
        segment_size = T // 3
        segments = []
        for i in range(0, T-segment_size, segment_size):
            seg = seq[i:i+segment_size]
            seg_std = seg.std(axis=0)
            left_seg_mov = seg_std[left_arm].mean()
            right_seg_mov = seg_std[right_arm].mean()
            seg_asymmetry = (left_seg_mov - right_seg_mov) / (left_seg_mov + right_seg_mov + 1e-6)
            segments.append(seg_asymmetry)
        
        # Consistency = low variance across segments
        consistency = 1.0 / (np.std(segments) + 1e-6)  # Higher = more consistent
    else:
        consistency = 1.0
    
    return np.array([
        static_arm_asymmetry,      # 1 - Static arm asymmetry
        static_hand_asymmetry,     # 2 - Static hand asymmetry  
        temporal_arm_asymmetry,    # 3 - Movement speed asymmetry
        smoothness_asymmetry,      # 4 - Movement smoothness asymmetry
        consistency,               # 5 - Movement consistency
        T / 100.0                  # 6 - Normalized duration
    ])


def extract_sequence_features_ultra_simple(sequence):
    """
    Ultra-simple features to prevent overfitting with 14 patients.
    
    Input: (T, 66) - skeleton sequence
    Output: 2 features only (most discriminative)
    """
    T = sequence.shape[0]
    seq = sequence.reshape(T, 33, 2)

    # Temporal statistics
    std_pos = seq.std(axis=0)    # (33, 2)
    
    # Most important keypoints for stroke impairment
    left_arm = [11, 13, 15]  # shoulder, elbow, wrist
    right_arm = [12, 14, 16]
    
    # 1. ARM ASYMMETRY (most important for stroke)
    left_arm_mov = std_pos[left_arm].mean()
    right_arm_mov = std_pos[right_arm].mean()
    arm_total = left_arm_mov + right_arm_mov + 1e-6
    arm_asymmetry = (left_arm_mov - right_arm_mov) / arm_total
    
    # 2. HAND ASYMMETRY (most discriminative)
    left_hand_mov = std_pos[15].mean()  # left wrist
    right_hand_mov = std_pos[16].mean()  # right wrist
    hand_total = left_hand_mov + right_hand_mov + 1e-6
    hand_asymmetry = (left_hand_mov - right_hand_mov) / hand_total
    
    return np.array([arm_asymmetry, hand_asymmetry])


def extract_patient_features_improved(patient_sequences, exercise_ids=None):
    """
    Exercise-aware feature extraction focusing on bilateral vs unilateral tasks.
    
    Input: 
    - patient_sequences: list of sequences
    - exercise_ids: list of exercise IDs (E1, E2, E3, E4, E5)
    Output: patient-level feature vector (6 features)
    """
    if exercise_ids is None:
        # Fallback if no exercise info
        seq_features = []
        for seq in patient_sequences:
            feat = extract_sequence_features_ultra_simple(seq)
            seq_features.append(feat)
        seq_features = np.array(seq_features)
        
        if len(seq_features) == 0:
            return np.zeros(6)
        elif len(seq_features) == 1:
            median_feat = seq_features[0]
            std_feat = np.zeros(2)
        else:
            median_feat = np.median(seq_features, axis=0)
            std_feat = np.std(seq_features, axis=0)
        
        return np.concatenate([median_feat, std_feat, [0, 0]])  # Add 2 zeros for bilateral features
    
    # Exercise-aware analysis
    bilateral_exercises = ['E3', 'E4']  # Face washing, putting on socks
    unilateral_exercises = ['E1', 'E2', 'E5']  # Single-hand tasks
    
    bilateral_features = []
    unilateral_features = []
    
    for i, (seq, ex_id) in enumerate(zip(patient_sequences, exercise_ids)):
        feat = extract_sequence_features_ultra_simple(seq)
        
        if ex_id in bilateral_exercises:
            bilateral_features.append(feat)
        elif ex_id in unilateral_exercises:
            unilateral_features.append(feat)
    
    # Extract bilateral asymmetry (most important for stroke detection)
    if len(bilateral_features) > 0:
        bilateral_array = np.array(bilateral_features)
        bilateral_median = np.median(bilateral_array, axis=0)
        bilateral_std = np.std(bilateral_array, axis=0)
        bilateral_min = np.min(bilateral_array, axis=0)  # Worst asymmetry
        bilateral_max = np.max(bilateral_array, axis=0)  # Best asymmetry
    else:
        bilateral_median = np.zeros(2)
        bilateral_std = np.zeros(2)
        bilateral_min = np.zeros(2)
        bilateral_max = np.zeros(2)
    
    # Extract unilateral patterns (less discriminative)
    if len(unilateral_features) > 0:
        unilateral_array = np.array(unilateral_features)
        unilateral_median = np.median(unilateral_array, axis=0)
    else:
        unilateral_median = np.zeros(2)
    
    # Weight bilateral features more heavily (they're most discriminative)
    bilateral_weighted = bilateral_median * 1.5  # 50% more weight for bilateral
    
    # Add range of bilateral asymmetry (max - min) - shows variability in impairment
    bilateral_range = bilateral_max - bilateral_min
    
    # Combine features: weighted bilateral + bilateral variability + bilateral range + unilateral
    patient_feat = np.concatenate([
        bilateral_weighted,    # 2 features: weighted bilateral asymmetry
        bilateral_std,         # 2 features: bilateral consistency
        bilateral_range,       # 2 features: bilateral asymmetry range
        unilateral_median      # 2 features: unilateral patterns
    ])
    
    return patient_feat