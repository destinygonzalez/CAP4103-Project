"""
FACE PIPELINE FOR IMDB-WIKI BIOMETRIC PROJECT + AGE ANALYSIS
Uses Mediapipe + Evaluator to compute biometric performance
AND evaluates the system per age group.
"""

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


#Similarity Scores 

def compute_scores(features, labels):

    genuine = []
    impostor = []

    n = len(features)

    for i in range(n):
        for j in range(i + 1, n):

            sim = cosine_similarity(
                features[i].reshape(1, -1),
                features[j].reshape(1, -1)
            )[0][0]

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

        f = features[indices]
        l = labels[indices]

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

    features = np.array([extract_face_features(lm) for lm in landmarks])
    new_ages = np.array(clean_ages)

    print(f"Landmarks extracted: {len(features)}")

    #Performance
    print("\nComputing similarity scores...")
    genuine, impostor = compute_scores(features, new_labels)

    evaluator = Evaluator(
        num_thresholds=NUM_THRESHOLDS,
        genuine_scores=genuine,
        impostor_scores=impostor,
        plot_title="Overall Face-Mediapipe System"
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
            plot_title=f"Age Group {group_name}"
        )

        FPRg, FNRg, TPRg = eval_age.compute_rates()
        eval_age.plot_score_distribution()
        eval_age.plot_det_curve(FPRg, FNRg)
        eval_age.plot_roc_curve(FPRg, TPRg)

    print("\n========== FACE SYSTEM COMPLETE ==========\n")


if __name__ == "__main__":
    run_face_pipeline()
