# matching.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def cosine_sim(a, b):
    a = a.reshape(1, -1)
    b = b.reshape(1, -1)
    return float(cosine_similarity(a, b)[0, 0])

def compute_scores_for_embeddings(embeddings_by_user):
    """
    embeddings_by_user : dict[int -> list[np.array]]
    Returns:
        genuine_scores, impostor_scores  (both np arrays in [0,1])
    """
    users = list(embeddings_by_user.keys())
    genuine, impostor = [], []

    # Genuine: same user
    for u in users:
        embs = embeddings_by_user[u]
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                s = cosine_sim(embs[i], embs[j])
                genuine.append((s + 1) / 2.0)

    # Impostor: different users
    for i in range(len(users)):
        for j in range(i + 1, len(users)):
            ei = embeddings_by_user[users[i]]
            ej = embeddings_by_user[users[j]]
            for vi in ei:
                for vj in ej:
                    s = cosine_sim(vi, vj)
                    impostor.append((s + 1) / 2.0)

    return np.array(genuine), np.array(impostor)
