"""
FACE PIPELINE FOR IMDB-WIKI BIOMETRIC PROJECT + AGE ANALYSIS
Uses Mediapipe + Evaluator to compute biometric performance
AND evaluates the system per age group.

FIXED VERSION: Uses geometric features (inter-landmark distances) instead of 
normalized raw coordinates for better discrimination.
"""

import os
import numpy as np
import cv2
from scipy.spatial.distance import euclidean, cdist

from load_img_data import get_images
from process_img_data import get_landmark_points
from evaluator import Evaluator



DATA_DIR = "IMDB"        
NUM_THRESHOLDS = 300     
MAX_IMAGES_PER_USER = 40 

# Define your age groups
AGE_GROUPS = {
    "18-25": (18, 25),
    "26-35": (26, 35),
    "36-50": (36, 50),
    "50+":   (51, 200)
}


#Data sampling

def sample(images, labels, ages, max_per_user=40):

    images = np.array(images, dtype=object)
    labels = np.array(labels)
    ages   = np.array(ages)

    users = np.unique(labels)

    out_imgs, out_labels, out_ages = [], [], []

    for user in users:
        idx = np.where(labels == user)[0]
        np.random.shuffle(idx)
        chosen = idx[:max_per_user]

        for c in chosen:
            out_imgs.append(images[c])
            out_labels.append(labels[c])
            out_ages.append(ages[c])

    print(f"\nSampling: Using {max_per_user} images per user")
    print(f"Users: {len(users)} | Total sampled: {len(out_imgs)}")

    return out_imgs, out_labels, out_ages


#FIXED: Extract geometric features from landmarks
def extract_face_features(landmarks):
    """
    Extract geometric features from face landmarks.
    Instead of raw coordinates, we compute:
    1. Pairwise distances between key landmarks
    2. This is scale-invariant when normalized by face size
    
    This approach creates more discriminative features than raw coordinates.
    """
    lm = np.array(landmarks)
    
    if len(lm) == 0:
        return np.zeros(100)  # Return zero vector if no landmarks
    
    # Select key landmark indices for feature extraction
    # For Mediapipe face mesh (468 landmarks), we use key facial points
    # If fewer landmarks, use all of them
    n_landmarks = len(lm)
    
    if n_landmarks >= 468:
        # Key facial landmarks for Mediapipe face mesh
        key_indices = [
            # Eyes
            33, 133, 362, 263,  # Eye corners
            159, 145, 386, 374,  # Eye top/bottom
            # Nose
            1, 4, 5, 6, 168,  # Nose bridge and tip
            # Mouth
            61, 291, 0, 17,  # Mouth corners and top/bottom
            # Face contour
            10, 152, 234, 454,  # Forehead, chin, cheeks
            # Eyebrows
            70, 63, 105, 107,
            336, 296, 334, 293,
        ]
        key_indices = [i for i in key_indices if i < n_landmarks]
    else:
        # Use all landmarks if fewer than 468
        key_indices = list(range(min(n_landmarks, 50)))
    
    # Extract key landmarks
    key_lm = lm[key_indices]
    
    # Compute pairwise distances between key landmarks
    # This creates a geometric descriptor that captures face shape
    n_key = len(key_lm)
    distances = []
    
    for i in range(n_key):
        for j in range(i + 1, n_key):
            dist = np.sqrt((key_lm[i][0] - key_lm[j][0])**2 + 
                          (key_lm[i][1] - key_lm[j][1])**2)
            distances.append(dist)
    
    features = np.array(distances)
    
    # Normalize by face size (max distance) for scale invariance
    if len(features) > 0 and np.max(features) > 0:
        features = features / np.max(features)
    
    return features


#FIXED: Use Euclidean distance converted to similarity score
def compute_similarity(feat1, feat2, eps=1e-12):
    """
    Compute similarity between two feature vectors.
    Uses Euclidean distance converted to similarity score in [0, 1].
    
    Lower distance = higher similarity
    """
    # Ensure same length
    min_len = min(len(feat1), len(feat2))
    f1 = feat1[:min_len]
    f2 = feat2[:min_len]
    
    # Euclidean distance
    dist = np.sqrt(np.sum((f1 - f2)**2))
    
    # Convert to similarity: exp(-distance) gives values in (0, 1]
    # Or use: 1 / (1 + distance) for range (0, 1]
    similarity = 1.0 / (1.0 + dist)
    
    return similarity


#FIXED: Compute scores using new similarity function
def compute_scores(features, labels):

    genuine = []
    impostor = []

    n = len(features)

    for i in range(n):
        for j in range(i + 1, n):
            
            sim = compute_similarity(features[i], features[j])

            if labels[i] == labels[j]:
                genuine.append(sim)
            else:
                impostor.append(sim)

    return np.array(genuine), np.array(impostor)


#Age group scores splitting

def compute_age_group_scores(features, labels, ages):

    groups = {}

    for age_group, (low, high) in AGE_GROUPS.items():

        indices = [i for i, a in enumerate(ages) if low <= a <= high]

        if len(indices) < 5:
            print(f"Skipping {age_group}: Not enough samples")
            continue

        f = [features[i] for i in indices]
        l = [labels[i] for i in indices]

        g, im = compute_scores(f, l)
        groups[age_group] = (g, im, len(indices))

    return groups


#Main Pipeline

def run_face_pipeline():

    print("\n========== FACE SYSTEM STARTED ==========\n")
    print("Loading IMDB cropped images...")

    images, labels, ages = get_images(DATA_DIR)

    print(f"Images loaded: {len(images)}")
    print(f"Users detected: {len(set(labels))}")

    images, labels, ages = sample(images, labels, ages, MAX_IMAGES_PER_USER)

   #Validate Images
    clean_imgs, clean_labels, clean_ages = [], [], []

    for img, lab, age in zip(images, labels, ages):
        if isinstance(img, np.ndarray) and img.dtype == np.uint8 and img.ndim == 2:
            clean_imgs.append(img)
            clean_labels.append(lab)
            clean_ages.append(age)

    print(f"Clean images retained: {len(clean_imgs)} / {len(images)}")

    #Landmarks extracted
    print("\nExtracting Mediapipe landmarks...")
    landmarks, new_labels = get_landmark_points(clean_imgs, clean_labels)

    print(f"\nExtracting geometric features from landmarks...")
    features = [extract_face_features(lm) for lm in landmarks]
    
    # Filter out any empty features
    valid_indices = [i for i, f in enumerate(features) if len(f) > 0]
    features = [features[i] for i in valid_indices]
    new_labels = np.array([new_labels[i] for i in valid_indices])
    new_ages = np.array([clean_ages[i] for i in valid_indices])

    print(f"Valid features extracted: {len(features)}")
    if len(features) > 0:
        print(f"Feature dimension: {len(features[0])}")

    #Performance
    print("\nComputing similarity scores...")
    genuine, impostor = compute_scores(features, new_labels)
    
    print(f"Genuine scores: {len(genuine)}, mean={np.mean(genuine):.4f}, std={np.std(genuine):.4f}")
    print(f"Impostor scores: {len(impostor)}, mean={np.mean(impostor):.4f}, std={np.std(impostor):.4f}")

    evaluator = Evaluator(
        num_thresholds=NUM_THRESHOLDS,
        genuine_scores=genuine,
        impostor_scores=impostor,
        plot_title="Face-Mediapipe-System"
    )

    FPR, FNR, TPR = evaluator.compute_rates()
    evaluator.plot_score_distribution()
    evaluator.plot_det_curve(FPR, FNR)
    evaluator.plot_roc_curve(FPR, TPR)

   #Age Groups
    
    print("\n========== AGE GROUP ANALYSIS ==========\n")

    age_groups = compute_age_group_scores(features, new_labels, new_ages)

    for group_name, (genuine_g, impostor_g, count) in age_groups.items():

        print(f"\n--- Age group {group_name} (n={count}) ---")

        eval_age = Evaluator(
            num_thresholds=NUM_THRESHOLDS,
            genuine_scores=genuine_g,
            impostor_scores=impostor_g,
            plot_title=f"Face_Age_{group_name}"
        )

        FPRg, FNRg, TPRg = eval_age.compute_rates()
        eval_age.plot_score_distribution()
        eval_age.plot_det_curve(FPRg, FNRg)
        eval_age.plot_roc_curve(FPRg, TPRg)

    print("\n========== FACE SYSTEM COMPLETE ==========\n")
    
    return genuine, impostor


# Export function for main.py
def run_face_system(directory="IMDB", num_users=200):
    """
    Simplified wrapper for main.py integration.
    Returns genuine and impostor scores.
    """
    global DATA_DIR
    DATA_DIR = directory
    
    genuine, impostor = run_face_pipeline()
    return genuine, impostor


if __name__ == "__main__":
    run_face_pipeline()
