# fusion.py
"""
Chimeric User Fusion Module

Since IMDB (face) and AudioMNIST (voice) contain DIFFERENT people,
we create "Chimeric Users" by pairing one face identity with one voice identity.

ChimericUser_k = (FaceID_k, VoiceID_k)

For fusion to be valid:
- When comparing ChimericUser_i vs ChimericUser_j:
  - face_score = similarity(FaceID_i, FaceID_j)
  - voice_score = similarity(VoiceID_i, VoiceID_j)
  - fused_score = w_face * face_score + w_voice * voice_score

This ensures the SAME virtual user pair is being compared across both modalities.
"""

import numpy as np


def chimeric_fusion(face_user_pair_scores, face_users, 
                    voice_user_pair_scores, voice_users,
                    face_genuine, face_impostor,
                    voice_genuine, voice_impostor,
                    w_face=0.6, w_voice=0.4):
    """
    Perform proper chimeric user fusion.
    
    Creates virtual users by pairing Face ID k with Voice ID k,
    then computes fused scores for each chimeric user pair.
    
    Parameters:
    -----------
    face_user_pair_scores : dict
        {(face_id_i, face_id_j): similarity} for all face user pairs
    face_users : list
        List of unique face user IDs
    voice_user_pair_scores : dict
        {(voice_id_i, voice_id_j): similarity} for all voice user pairs
    voice_users : list
        List of unique voice user IDs
    face_genuine, face_impostor : array
        Original face genuine/impostor scores (for fallback)
    voice_genuine, voice_impostor : array
        Original voice genuine/impostor scores (for fallback)
    w_face, w_voice : float
        Fusion weights (must sum to 1)
    
    Returns:
    --------
    fused_genuine : np.array
        Fused genuine scores (from per-sample arrays, normalized and aligned)
    fused_impostor : np.array
        Fused impostor scores (from chimeric user pairs)
    """
    
    # Determine number of chimeric users
    num_chimeric = min(len(face_users), len(voice_users))
    
    print(f"\n========== CHIMERIC USER FUSION ==========")
    print(f"Face users available: {len(face_users)}")
    print(f"Voice users available: {len(voice_users)}")
    print(f"Creating {num_chimeric} chimeric users")
    print(f"Weights: face={w_face}, voice={w_voice}")
    print()
    
    # Create chimeric user mapping
    # ChimericUser_k -> (FaceUser[k], VoiceUser[k])
    print("Chimeric User Mapping (first 10):")
    for k in range(min(10, num_chimeric)):
        print(f"  Chimeric_{k}: Face={face_users[k]}, Voice={voice_users[k]}")
    if num_chimeric > 10:
        print(f"  ... and {num_chimeric - 10} more")
    print()
    
    # Compute IMPOSTOR scores for chimeric user pairs
    # Chimeric_i vs Chimeric_j where i != j
    chimeric_impostor = []
    matched_pairs = 0
    
    for i in range(num_chimeric):
        for j in range(i + 1, num_chimeric):
            # Get face user IDs for this chimeric pair
            face_i, face_j = face_users[i], face_users[j]
            # Get voice user IDs for this chimeric pair
            voice_i, voice_j = voice_users[i], voice_users[j]
            
            # Look up face score (ensure i < j in the key)
            if face_i < face_j:
                face_key = (face_i, face_j)
            else:
                face_key = (face_j, face_i)
            
            # Look up voice score (ensure i < j in the key)
            if voice_i < voice_j:
                voice_key = (voice_i, voice_j)
            else:
                voice_key = (voice_j, voice_i)
            
            # Only include if both scores exist
            if face_key in face_user_pair_scores and voice_key in voice_user_pair_scores:
                f_score = face_user_pair_scores[face_key]
                v_score = voice_user_pair_scores[voice_key]
                
                # ALWAYS normalize cosine similarity [-1, 1] -> [0, 1]
                # This ensures consistent scale for both modalities
                f_norm = (f_score + 1) / 2
                v_norm = (v_score + 1) / 2
                
                # Fuse
                fused = w_face * f_norm + w_voice * v_norm
                chimeric_impostor.append(fused)
                matched_pairs += 1
    
    print(f"Chimeric impostor pairs created: {matched_pairs}")
    
    # For GENUINE scores, we use the per-sample genuine arrays
    # Since each modality's genuine scores are intra-user (same person),
    # we align by index under the chimeric assumption
    
    # ALWAYS normalize cosine similarity [-1, 1] -> [0, 1]
    def normalize(scores):
        scores = np.array(scores)
        if len(scores) == 0:
            return scores
        # Always normalize to ensure consistent scale
        return (scores + 1) / 2
    
    face_gen_norm = normalize(face_genuine)
    voice_gen_norm = normalize(voice_genuine)
    
    # Align by truncating to minimum length
    n_gen = min(len(face_gen_norm), len(voice_gen_norm))
    
    if n_gen > 0:
        chimeric_genuine = (w_face * face_gen_norm[:n_gen] + 
                           w_voice * voice_gen_norm[:n_gen])
    else:
        chimeric_genuine = np.array([])
        print("WARNING: No genuine scores available!")
    
    print(f"Chimeric genuine scores: {len(chimeric_genuine)}")
    print(f"Chimeric impostor scores: {len(chimeric_impostor)}")
    print(f"==========================================\n")
    
    return np.array(chimeric_genuine), np.array(chimeric_impostor)


def score_level_fusion(face_genuine, face_impostor, voice_genuine, voice_impostor,
                       face_user_pair_scores=None, face_users=None,
                       voice_user_pair_scores=None, voice_users=None):
    """
    Main fusion entry point.
    
    If user_pair_scores are provided, uses proper chimeric fusion.
    Otherwise, falls back to simple index-aligned fusion with a warning.
    """
    
    # If we have user pair data, do proper chimeric fusion
    if (face_user_pair_scores is not None and 
        voice_user_pair_scores is not None and
        face_users is not None and 
        voice_users is not None):
        
        return chimeric_fusion(
            face_user_pair_scores, face_users,
            voice_user_pair_scores, voice_users,
            face_genuine, face_impostor,
            voice_genuine, voice_impostor,
            w_face=0.6, w_voice=0.4
        )
    
    # Fallback: simple fusion with warning
    print("\n" + "="*60)
    print("WARNING: Using simple index-aligned fusion!")
    print("This is NOT semantically valid for cross-dataset fusion.")
    print("For valid fusion, pass user_pair_scores from pipelines.")
    print("="*60 + "\n")
    
    return simple_fusion(face_genuine, face_impostor, 
                         voice_genuine, voice_impostor)


def simple_fusion(face_genuine, face_impostor, voice_genuine, voice_impostor,
                  w_face=0.6, w_voice=0.4):
    """
    Simple index-aligned fusion (fallback method).
    
    WARNING: Only valid if scores at the same index represent
    the same comparison pair. This is NOT the case when fusing
    scores from different datasets!
    """
    
    # ALWAYS normalize cosine similarity [-1, 1] -> [0, 1]
    def normalize(scores):
        scores = np.array(scores)
        if len(scores) == 0:
            return scores
        # Always normalize to ensure consistent scale
        return (scores + 1) / 2
    
    face_genuine = normalize(face_genuine)
    face_impostor = normalize(face_impostor)
    voice_genuine = normalize(voice_genuine)
    voice_impostor = normalize(voice_impostor)
    
    # Truncate to minimum length
    n_gen = min(len(face_genuine), len(voice_genuine))
    n_imp = min(len(face_impostor), len(voice_impostor))
    
    if n_gen > 0:
        fused_genuine = (w_face * face_genuine[:n_gen] + 
                        w_voice * voice_genuine[:n_gen])
    else:
        fused_genuine = np.array([])
    
    if n_imp > 0:
        fused_impostor = (w_face * face_impostor[:n_imp] + 
                         w_voice * voice_impostor[:n_imp])
    else:
        fused_impostor = np.array([])
    
    print(f"Simple fusion: {n_gen} genuine, {n_imp} impostor")
    
    return fused_genuine, fused_impostor
