"""
FACE PIPELINE FOR IMDB-WIKI BIOMETRIC PROJECT + AGE ANALYSIS
Uses Mediapipe + Evaluator to compute biometric performance
AND evaluates the system per age group.
"""
#
import os
import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity

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


#Landmark embedding

def extract_face_features(landmarks):

    lm = np.array(landmarks).flatten()
    norm = np.linalg.norm(lm)

    if norm == 0:
        return lm

    return lm / norm


#Similarity Scores - VECTORIZED (O(n²) memory but ~100x faster than nested loops)

def compute_scores(features, labels):
    """
    Compute genuine and impostor scores using vectorized matrix operations.
    
    Returns:
    - genuine: All intra-user (same person) scores
    - impostor: All inter-user (different person) scores
    - user_pair_scores: Dict mapping (user_i, user_j) -> mean similarity score
    """
    n = len(features)
    labels = np.array(labels)
    
    print(f"  Computing {n}x{n} similarity matrix ({n*(n-1)//2:,} comparisons)...")
    
    # Step 1: Compute full similarity matrix at once (uses optimized BLAS)
    sim_matrix = cosine_similarity(features)
    
    # Step 2: Create label match matrix (True where labels match)
    label_match = labels[:, None] == labels[None, :]
    
    # Step 3: Get upper triangle indices (i < j) to avoid duplicates
    upper_tri_indices = np.triu_indices(n, k=1)
    
    # Step 4: Extract scores from upper triangle
    all_scores = sim_matrix[upper_tri_indices]
    is_genuine = label_match[upper_tri_indices]
    
    # Step 5: Separate genuine and impostor scores
    genuine = all_scores[is_genuine]
    impostor = all_scores[~is_genuine]
    
    # Step 6: Compute USER-PAIR mean scores (for chimeric fusion)
    unique_users = sorted(list(set(labels)))
    user_to_idx = {u: i for i, u in enumerate(unique_users)}
    num_users = len(unique_users)
    
    # Create user-level similarity matrix
    user_sim_sum = np.zeros((num_users, num_users))
    user_sim_count = np.zeros((num_users, num_users))
    
    sample_to_user = np.array([user_to_idx[l] for l in labels])
    
    for idx, (i, j) in enumerate(zip(*upper_tri_indices)):
        ui, uj = sample_to_user[i], sample_to_user[j]
        if ui > uj:
            ui, uj = uj, ui
        user_sim_sum[ui, uj] += all_scores[idx]
        user_sim_count[ui, uj] += 1
    
    # Build user_pair_scores dictionary
    user_pair_scores = {}
    for i, user_i in enumerate(unique_users):
        for j, user_j in enumerate(unique_users):
            if i < j and user_sim_count[i, j] > 0:
                user_pair_scores[(user_i, user_j)] = user_sim_sum[i, j] / user_sim_count[i, j]
    
    print(f"  Genuine: {len(genuine):,} | Impostor: {len(impostor):,}")
    print(f"  User pairs: {len(user_pair_scores):,} | Users: {num_users}")
    
    return genuine, impostor, user_pair_scores, unique_users


#Age group scores splitting

def compute_age_group_scores(features, labels, ages):

    groups = {}

    for age_group, (low, high) in AGE_GROUPS.items():

        indices = [i for i, a in enumerate(ages) if low <= a <= high]

        if len(indices) < 5:
            print(f"Skipping {age_group}: Not enough samples")
            continue

        f = features[indices]
        l = labels[indices]

        # compute_scores now returns 4 values - ignore user_pair data for age groups
        g, im, _, _ = compute_scores(f, l)
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

    features = np.array([extract_face_features(lm) for lm in landmarks])
    new_ages = np.array(clean_ages)

    print(f"Landmarks extracted: {len(features)}")

    #Performance
    print("\nComputing similarity scores...")
    genuine, impostor, user_pair_scores, user_list = compute_scores(features, new_labels)

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

    print("\n========== FACE SYSTEM COMPLETE ==========\n")
    
    # Return scores AND user-pair data for chimeric fusion
    return genuine, impostor, user_pair_scores, user_list


def run_face_system(directory="IMDB", num_users=200):
    """
    Wrapper function for main.py integration.
    Returns genuine and impostor scores.
    """
    global DATA_DIR
    DATA_DIR = directory
    
    return run_face_pipeline()


if __name__ == "__main__":
    run_face_pipeline()
