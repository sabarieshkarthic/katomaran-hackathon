"""
Face Quality Checker
Evaluates whether a detected face crop is usable for embedding.
Returns CLEAR (True) or NOT CLEAR (False) with reason.
"""

import cv2
import numpy as np


def check_face_quality(face_crop: np.ndarray, min_face_size: int = 60) -> tuple[bool, str]:
    """
    Evaluate face quality.
    Returns (is_clear: bool, reason: str)

    Quality fails if:
    - Face is None or empty
    - Face is smaller than min_face_size
    - Face is too blurry (Laplacian variance < threshold)
    - Face appears to be a side/profile view (landmark heuristic)
    """
    if face_crop is None or face_crop.size == 0:
        return False, "no_face"

    h, w = face_crop.shape[:2]

    # 1. Size check
    if h < min_face_size or w < min_face_size:
        return False, "small_face"

    # 2. Motion blur check via Laplacian variance
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if len(face_crop.shape) == 3 else face_crop
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < 40.0:
        return False, "motion_blur"

    # 3. Aspect ratio check - very wide or very tall faces are likely profile views
    aspect = w / max(h, 1)
    if aspect < 0.5 or aspect > 2.0:
        return False, "side_angle"

    # 4. Brightness / occlusion check - very dark means occluded or backside
    mean_brightness = np.mean(gray)
    if mean_brightness < 30:
        return False, "occlusion_or_backside"

    return True, "clear"


def compute_blur_score(face_crop: np.ndarray) -> float:
    """Returns Laplacian variance as a blur metric. Higher = sharper."""
    if face_crop is None or face_crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if len(face_crop.shape) == 3 else face_crop
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
