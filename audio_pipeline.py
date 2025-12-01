"""
AUDIO PIPELINE FOR BIOMETRIC PROJECT
Uses MFCC features + Evaluator to compute biometric performance

FIXED VERSION: Uses Euclidean distance-based similarity and enhanced MFCC features
for better speaker discrimination.
"""

import os
import numpy as np
import librosa
from scipy.io import wavfile
from evaluator import Evaluator


DATA_DIR = "AudioMNIST"
NUM_THRESHOLDS = 300
MAX_FILES_PER_SPEAKER = 20     # Fast loading


#FIXED: Use Euclidean distance converted to similarity
def compute_similarity(a, b, eps=1e-12):
    """
    Compute similarity between two feature vectors using Euclidean distance.
    Returns value in range (0, 1] where 1 = identical.
    """
    # Euclidean distance
    dist = np.sqrt(np.sum((a - b)**2))
    
    # Convert to similarity: 1 / (1 + distance)
    # This gives values in (0, 1] where smaller distance = higher similarity
    similarity = 1.0 / (1.0 + dist)
    
    return float(similarity)


#FIXED: Enhanced MFCC Extraction with delta features
def fast_mfcc(path):
    """
    Extract enhanced MFCC features including:
    - 13 MFCC coefficients
    - 13 delta (first derivative) coefficients
    - 13 delta-delta (second derivative) coefficients
    Total: 39 features for better speaker discrimination
    """
    sr, sig = wavfile.read(path)

    # Convert int16 -> float32
    if sig.dtype != np.float32:
        sig = sig.astype(np.float32) / 32768.0

    # Extract 13 MFCCs
    mfcc = librosa.feature.mfcc(y=sig, sr=sr, n_mfcc=13)
    
    # Compute delta (first derivative) features
    mfcc_delta = librosa.feature.delta(mfcc)
    
    # Compute delta-delta (second derivative) features  
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    
    # Concatenate and take mean over time
    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.mean(mfcc_delta, axis=1),
        np.mean(mfcc_delta2, axis=1)
    ])
    
    # Also add standard deviation for more robust features
    features_std = np.concatenate([
        np.std(mfcc, axis=1),
        np.std(mfcc_delta, axis=1),
        np.std(mfcc_delta2, axis=1)
    ])
    
    # Combine mean and std features
    all_features = np.concatenate([features, features_std])
    
    return all_features


#Load AudioMNIST quickly
def load_audio_mnist(root="AudioMNIST"):
    data_path = os.path.join(root, "data")

    speakers = sorted(os.listdir(data_path))
    X = []
    y_id = []
    y_age = []

    for spk in speakers:
        spk_folder = os.path.join(data_path, spk)
        if not os.path.isdir(spk_folder):
            continue

        count = 0

        for file in os.listdir(spk_folder):
            if not file.endswith(".wav"):
                continue
            if count >= MAX_FILES_PER_SPEAKER:
                break

            parts = file.split("_")
            age = int(parts[0])
            speaker_id = int(parts[1])

            path = os.path.join(spk_folder, file)
            mfcc_vec = fast_mfcc(path)

            X.append(mfcc_vec)
            y_id.append(speaker_id)
            y_age.append(age)

            count += 1

    return np.array(X), np.array(y_id), np.array(y_age)


#Feature extraction
def extract_features(mfcc_vec):
    """
    Optional additional feature processing.
    Currently returns raw MFCC features.
    """
    return mfcc_vec


#Emotional variation (more realistic)
def augment_pitch(mfcc):
    """
    Simulate emotional variation by adding noise and scaling.
    This creates a more realistic variation than simple multiplication.
    """
    # Add small random perturbation
    noise = np.random.randn(*mfcc.shape) * 0.1
    
    # Scale certain frequency bands differently
    scaled = mfcc.copy()
    scaled[:13] *= 0.95  # Slightly reduce static MFCCs
    scaled[13:26] *= 1.05  # Slightly increase delta MFCCs
    
    return scaled + noise


#FIXED: Score generation using new similarity function
def compute_scores(features, labels):
    genuine = []
    impostor = []

    print("Computing similarity scores...")

    n = len(features)
    for i in range(n):
        for j in range(i + 1, n):
            sim = compute_similarity(features[i], features[j])

            if labels[i] == labels[j]:
                genuine.append(sim)
            else:
                impostor.append(sim)

    return np.array(genuine), np.array(impostor)


#Plot wrapper
def evaluate_and_plot(genuine, impostor, title):
    print(f"\n=== Generating curves for: {title} ===")
    print(f"Genuine: n={len(genuine)}, mean={np.mean(genuine):.4f}, std={np.std(genuine):.4f}")
    print(f"Impostor: n={len(impostor)}, mean={np.mean(impostor):.4f}, std={np.std(impostor):.4f}")

    evaluator = Evaluator(
        num_thresholds=NUM_THRESHOLDS,
        genuine_scores=genuine,
        impostor_scores=impostor,
        plot_title=title
    )

    FPR, FNR, TPR = evaluator.compute_rates()

    evaluator.plot_score_distribution()
    evaluator.plot_det_curve(FPR, FNR)
    evaluator.plot_roc_curve(FPR, TPR)
    
    return genuine, impostor


#Main pipeline
def run_audio_pipeline():

    print("\n======= AUDIO SYSTEM STARTED =======\n")

    #Load
    mfccs, speakers, ages = load_audio_mnist(DATA_DIR)
    print(f"Loaded samples: {len(mfccs)}")
    if len(mfccs) > 0:
        print(f"Feature dimension: {len(mfccs[0])}")

    #Features
    print("Extracting clean MFCC features...")
    features = np.array([extract_features(m) for m in mfccs])

    #Clean Scores
    genuine_clean, impostor_clean = compute_scores(features, speakers)
    print(f"Clean: Genuine={len(genuine_clean)}, Impostor={len(impostor_clean)}")

    #Emotional Condition
    print("\nApplying emotional variation...")
    emotional_features = np.array([augment_pitch(f) for f in features])

    genuine_em, impostor_em = compute_scores(emotional_features, speakers)
    print(f"Emotional: Genuine={len(genuine_em)}, Impostor={len(impostor_em)}")

    #Age groups
    print("\nGrouping by age...")

    age_groups = {
        "Young_18_30": [],
        "Mid_31_50": [],
        "Old_51_plus": []
    }

    for i, a in enumerate(ages):
        if a <= 30:
            age_groups["Young_18_30"].append(i)
        elif a <= 50:
            age_groups["Mid_31_50"].append(i)
        else:
            age_groups["Old_51_plus"].append(i)

    #Output the plots

    # 1. baseline clean
    evaluate_and_plot(genuine_clean, impostor_clean, "Audio_Clean")

    # 2. emotional
    evaluate_and_plot(genuine_em, impostor_em, "Audio_Emotional")

    # 3. age group plots
    for group, idxs in age_groups.items():
        if len(idxs) < 5:
            print(f"Skipping {group}: only {len(idxs)} samples")
            continue

        print(f"\nEvaluating group: {group} ({len(idxs)} samples)")

        f_sub = features[idxs]
        s_sub = speakers[idxs]

        gen_g, imp_g = compute_scores(f_sub, s_sub)
        evaluate_and_plot(gen_g, imp_g, f"Audio_{group}")

    print("\n======= AUDIO SYSTEM COMPLETE =======\n")
    
    return genuine_clean, impostor_clean


# Export function for main.py
def run_voice_system(directory="AudioMNIST/data", num_users=50):
    """
    Simplified wrapper for main.py integration.
    Returns genuine and impostor scores.
    """
    global DATA_DIR
    DATA_DIR = directory.replace("/data", "")
    
    genuine, impostor = run_audio_pipeline()
    return genuine, impostor


def main():
    run_audio_pipeline()

if __name__ == "__main__":
    main()
