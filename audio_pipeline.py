import os
import numpy as np
import librosa
from scipy.io import wavfile
from evaluator import Evaluator


DATA_DIR = "AudioMNIST"
NUM_THRESHOLDS = 300
MAX_FILES_PER_SPEAKER = 20     # Fast loading


#Cosine similarity

def cosine_sim(a, b, eps=1e-12):
    num = np.dot(a, b)
    den = (np.linalg.norm(a) * np.linalg.norm(b)) + eps
    return float(num / den)


#MFCC Extraction
def fast_mfcc(path):
    sr, sig = wavfile.read(path)

    # Convert int16 → float32
    if sig.dtype != np.float32:
        sig = sig.astype(np.float32) / 32768.0

    mfcc = librosa.feature.mfcc(y=sig, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)


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
    return mfcc_vec


#Emotional variation (pitch)

def augment_pitch(mfcc):
    return mfcc * 0.95     # lightweight emotional simulation


#Score generation

def compute_scores(features, labels):
    genuine = []
    impostor = []

    print("Computing similarity scores...")

    n = len(features)
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_sim(features[i], features[j])

            if labels[i] == labels[j]:
                genuine.append(sim)
            else:
                impostor.append(sim)

    return np.array(genuine), np.array(impostor)


#Plot wrapper

def evaluate_and_plot(genuine, impostor, title):
    print(f"\n=== Generating curves for: {title} ===")

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


#Main pipeline

def run_audio_pipeline():

    print("\n======= AUDIO SYSTEM STARTED =======\n")

    #Load
    mfccs, speakers, ages = load_audio_mnist(DATA_DIR)
    print(f"Loaded samples: {len(mfccs)}")

    #Features
    print("Extracting clean MFCC features...")
    features = np.array([extract_features(m) for m in mfccs])

    #Clean Scorews
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

    #Output the 15 plots

    # 1. baseline clean
    evaluate_and_plot(genuine_clean, impostor_clean, "Audio_Clean")

    # 2. emotional
    evaluate_and_plot(genuine_em, impostor_em, "Audio_Emotional")

    # 3. age group plots
    for group, idxs in age_groups.items():
        if len(idxs) < 5:
            continue

        print(f"\nEvaluating group: {group} ({len(idxs)} samples)")

        f_sub = features[idxs]
        s_sub = speakers[idxs]

        gen_g, imp_g = compute_scores(f_sub, s_sub)
        evaluate_and_plot(gen_g, imp_g, f"Audio_{group}")

    print("\n======= AUDIO SYSTEM COMPLETE =======\n")


def main():
    run_audio_pipeline()

if __name__ == "__main__":
    main()
