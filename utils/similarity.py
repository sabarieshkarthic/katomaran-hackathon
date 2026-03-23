"""
Similarity Matching Engine
Cosine similarity search against face DB and body DB embeddings.
"""

import numpy as np
from typing import Optional, Tuple


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalized vectors."""
    if a is None or b is None:
        return 0.0
    # Both should already be L2-normalized, dot product = cosine similarity
    sim = float(np.dot(a, b))
    return max(0.0, min(1.0, sim))


def fused_face_body_score(
    face_emb: np.ndarray,
    body_emb: np.ndarray,
    stored_face_emb: np.ndarray,
    stored_body_emb: np.ndarray,
    face_weight: float = 0.7,
    body_weight: float = 0.3,
) -> float:
    """
    Weighted fusion of face similarity and body similarity.
    Score = face_weight * face_sim + body_weight * body_sim
    If either embedding is missing, uses the available one at full weight.
    """
    face_sim = cosine_similarity(face_emb, stored_face_emb) if (face_emb is not None and stored_face_emb is not None) else None
    body_sim = cosine_similarity(body_emb, stored_body_emb) if (body_emb is not None and stored_body_emb is not None) else None

    if face_sim is not None and body_sim is not None:
        return face_weight * face_sim + body_weight * body_sim
    elif face_sim is not None:
        return face_sim
    elif body_sim is not None:
        return body_sim
    return 0.0


def find_best_face_match(
    face_emb: np.ndarray,
    body_emb: np.ndarray,
    db_faces: list,
    face_threshold: float = 0.7,
    face_weight: float = 0.7,
    body_weight: float = 0.3,
) -> Tuple[Optional[str], float]:
    """
    Search face DB for best match using fused face+body similarity.
    Returns (face_id, score) or (None, 0.0) if no match above threshold.

    db_faces: list of dicts from get_all_faces()
    """
    best_id = None
    best_score = 0.0

    for record in db_faces:
        if record['face_embedding'] is None:
            continue
        score = fused_face_body_score(
            face_emb,
            body_emb,
            record['face_embedding'],
            record.get('body_embedding'),
            face_weight,
            body_weight,
        )
        if score > best_score:
            best_score = score
            best_id = record['face_id']

    if best_score >= face_threshold:
        return best_id, best_score
    return None, best_score


def find_best_body_match(
    body_emb: np.ndarray,
    db_bodies: list,
    body_threshold: float = 0.4,
) -> Tuple[Optional[str], float]:
    """
    Search body DB for best match using cosine similarity.
    Returns (body_id, score) or (None, 0.0) if no match above threshold.

    db_bodies: list of dicts from get_all_bodies()
    """
    best_id = None
    best_score = 0.0

    for record in db_bodies:
        if record['body_embedding'] is None:
            continue
        if record['linked_face_id'] is not None:
            # Already upgraded to face — skip (will be found in face DB)
            continue
        score = cosine_similarity(body_emb, record['body_embedding'])
        if score > best_score:
            best_score = score
            best_id = record['body_id']

    if best_score >= body_threshold:
        return best_id, best_score
    return None, best_score
