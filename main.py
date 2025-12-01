# main.py
#
"""
Multimodal Biometric System - Face + Voice Fusion

This system uses the CHIMERIC USER approach for fusion:
- IMDB dataset provides face biometrics
- AudioMNIST dataset provides voice biometrics
- Since these datasets contain DIFFERENT people, we create virtual "chimeric users"
  by pairing Face_ID_k with Voice_ID_k

ChimericUser_0 = (IMDB_Face_0, AudioMNIST_Voice_0)
ChimericUser_1 = (IMDB_Face_1, AudioMNIST_Voice_1)
...

When comparing ChimericUser_i vs ChimericUser_j:
  - face_score = similarity(Face_i, Face_j)
  - voice_score = similarity(Voice_i, Voice_j)
  - fused_score = 0.6 * face_score + 0.4 * voice_score

This ensures the SAME virtual user pair is compared across both modalities.
"""

import numpy as np
from face_pipeline import run_face_system
from audio_pipeline import run_voice_system
from fusion import score_level_fusion
from evaluator import Evaluator


def main():

    # 1. RUN FACE SYSTEM
    print("\n" + "="*60)
    print("RUNNING FACE BIOMETRIC SYSTEM (IMDB Dataset)")
    print("="*60)
    
    # Now returns: genuine, impostor, user_pair_scores, user_list
    print("DEBUG: About to call run_face_system...", flush=True)
    result = run_face_system(
        directory="IMDB",
        num_users=200
    )
    print(f"DEBUG: run_face_system returned {type(result)}, length={len(result) if hasattr(result, '__len__') else 'N/A'}", flush=True)
    
    face_genuine, face_impostor, face_user_pairs, face_users = result
    print("DEBUG: Unpacked face results successfully", flush=True)
    
    print(f"\nFace system results:")
    print(f"  Genuine scores: {len(face_genuine):,}")
    print(f"  Impostor scores: {len(face_impostor):,}")
    print(f"  Unique users: {len(face_users)}")
    print(f"  User pairs: {len(face_user_pairs)}")

    # 2. RUN VOICE SYSTEM
    print("\n" + "="*60)
    print("RUNNING VOICE BIOMETRIC SYSTEM (AudioMNIST Dataset)")
    print("="*60, flush=True)
    
    # Now returns: genuine, impostor, user_pair_scores, user_list
    print("DEBUG: About to call run_voice_system...", flush=True)
    voice_result = run_voice_system(
        directory="AudioMNIST/data",
        num_users=50
    )
    print(f"DEBUG: run_voice_system returned", flush=True)
    
    voice_genuine, voice_impostor, voice_user_pairs, voice_users = voice_result
    print("DEBUG: Unpacked voice results successfully", flush=True)
    
    print(f"\nVoice system results:")
    print(f"  Genuine scores: {len(voice_genuine):,}")
    print(f"  Impostor scores: {len(voice_impostor):,}")
    print(f"  Unique users: {len(voice_users)}")
    print(f"  User pairs: {len(voice_user_pairs)}")
    
    # 3. RUN FUSION (CHIMERIC USER MODEL)
    print("\n" + "="*60)
    print("RUNNING CHIMERIC USER FUSION")
    print("="*60, flush=True)
    
    # Pass user pair data for proper chimeric fusion
    print("DEBUG: About to call score_level_fusion...", flush=True)
    print(f"DEBUG: face_user_pairs type={type(face_user_pairs)}, len={len(face_user_pairs)}", flush=True)
    print(f"DEBUG: voice_user_pairs type={type(voice_user_pairs)}, len={len(voice_user_pairs)}", flush=True)
    
    fused_genuine, fused_impostor = score_level_fusion(
        face_genuine, face_impostor,
        voice_genuine, voice_impostor,
        face_user_pair_scores=face_user_pairs,
        face_users=face_users,
        voice_user_pair_scores=voice_user_pairs,
        voice_users=voice_users
    )
    
    print(f"\nFusion results:")
    print(f"  Fused genuine scores: {len(fused_genuine):,}")
    print(f"  Fused impostor scores: {len(fused_impostor):,}")

    # 4. EVALUATE ALL SYSTEMS
    print("\n" + "="*60)
    print("GENERATING FINAL EVALUATION PLOTS")
    print("="*60)
    
    # Only evaluate systems that have scores
    systems = {}
    
    if len(face_genuine) > 0 and len(face_impostor) > 0:
        systems["Face_Final"] = (face_genuine, face_impostor)
    
    if len(voice_genuine) > 0 and len(voice_impostor) > 0:
        systems["Voice_Final"] = (voice_genuine, voice_impostor)
    
    if len(fused_genuine) > 0 and len(fused_impostor) > 0:
        systems["Fusion_Chimeric"] = (fused_genuine, fused_impostor)
    else:
        print("\nWARNING: Fusion has no valid scores - skipping evaluation")

    for name, (g, i) in systems.items():
        print(f"\n--- Evaluating {name} system ---")
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

    print("\n" + "="*60)
    print("ALL SYSTEMS COMPLETED SUCCESSFULLY")
    print("="*60)
    print("\nNOTE: Fusion uses Chimeric User model.")
    print("ChimericUser_k = (IMDB_Face_k, AudioMNIST_Voice_k)")
    print("This is valid for methodology demonstration,")
    print("but represents virtual users, not real multimodal data.")


if __name__ == "__main__":
    main()
