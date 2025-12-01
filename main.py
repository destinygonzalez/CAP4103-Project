# main.py

from face_pipeline import run_face_system
from audio_pipeline import run_voice_system
from fusion import score_level_fusion
from evaluator import Evaluator


def main():

    # FACE RECOGNITION (RQ1)
    face_genuine, face_impostor = run_face_system(
        directory="./IMDB-crop",
        num_users=200
    )

    # VOICE RECOGNITION (RQ2)
    voice_genuine, voice_impostor = run_voice_system(
        directory="./AudioMNIST/data",
        num_users=50
    )

    # SCORE LEVEL FUSION
    fused_genuine, fused_impostor = score_level_fusion(
        face_genuine, face_impostor,
        voice_genuine, voice_impostor
    )

    # RUN EVALUATOR FOR EACH SYSTEM
    systems = {
        "Face": (face_genuine, face_impostor),
        "Voice": (voice_genuine, voice_impostor),
        "Fusion": (fused_genuine, fused_impostor)
    }

    for name, (g, i) in systems.items():
        evaluator = Evaluator(
            num_thresholds=500,
            genuine_scores=g,
            impostor_scores=i,
            plot_title=name
        )

        FPR, FNR, TPR = evaluator.compute_rates()
        evaluator.plot_score_distribution()
        evaluator.plot_det_curve(FPR, FNR)
        evaluator.plot_roc_curve(FPR, TPR)


if __name__ == "__main__":
    main()
