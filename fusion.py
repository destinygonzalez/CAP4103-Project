# fusion.py

import numpy as np

def score_level_fusion(face_genuine, face_impostor, voice_genuine, voice_impostor):
    """
    Simple weighted average fusion.
    """

    w_face = 0.6
    w_voice = 0.4

    fused_genuine = (w_face*face_genuine[:len(voice_genuine)]
                     + w_voice*voice_genuine)

    fused_impostor = (w_face*face_impostor[:len(voice_impostor)]
                      + w_voice*voice_impostor)

    return fused_genuine, fused_impostor
