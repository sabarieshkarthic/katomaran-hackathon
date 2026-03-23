"""
Logger — per-video event log + cropped image saver
===================================================
Issue 1 fix : Each video gets its own log file: logs/<video_name>/events.log
              Images go to logs/<video_name>/entries/ and logs/<video_name>/exits/
Issue 2 fix : Face detection notes logged (face_clear / body_only / backside)
Issue 5 fix : Correct crop saved per case:
                Case A (face clear)      → face crop saved
                Case B (face not clear)  → face crop if available, else body crop
                Case C (backside/body)   → full body crop saved
"""

import os
import cv2
import numpy as np
from datetime import datetime


# ── image case labels for filenames ──────────────────────────────────
CASE_FACE   = "face"    # Case A: clear face
CASE_NOCLEAR = "noclear" # Case B: face found but not clear
CASE_BODY   = "body"    # Case C: no face / backside


def _session_dirs(session_logs_dir: str, event_type: str) -> str:
    """Return and create path:  <session_logs_dir>/<entries|exits>/"""
    path = os.path.join(session_logs_dir, event_type)
    os.makedirs(path, exist_ok=True)
    return path


def _log_path(session_logs_dir: str) -> str:
    os.makedirs(session_logs_dir, exist_ok=True)
    return os.path.join(session_logs_dir, "events.log")


def _save_image(directory: str, person_id: str, crop: np.ndarray, case: str) -> str:
    """Save crop to directory. Returns saved path or empty string."""
    if crop is None or crop.size == 0:
        return ""
    safe_id = person_id.replace("-", "")[:16]
    ts_str  = datetime.now().strftime("%H%M%S%f")[:10]
    filename = "{}_{}_{}.jpg".format(safe_id, case, ts_str)
    img_path = os.path.join(directory, filename)
    cv2.imwrite(img_path, crop)
    return img_path


def save_entry(
    session_logs_dir: str,
    person_id: str,
    face_crop,          # np.ndarray or None
    body_crop,          # np.ndarray or None
    face_clear: bool,
    face_found: bool,   # True if InsightFace found ANY face (even unclear)
    timestamp: str,
) -> str:
    """
    Save entry image and append to events.log.
    Issue 5 fix — crop selection:
      Case A (face_clear=True)           → save face_crop
      Case B (face_found=True, not clear) → save face_crop (blurry face)
      Case C (face_found=False)           → save body_crop (backside/occluded)
    Returns saved image path.
    """
    directory = _session_dirs(session_logs_dir, "entries")

    if face_clear and face_crop is not None:
        case = CASE_FACE
        crop = face_crop
    elif face_found and face_crop is not None:
        case = CASE_NOCLEAR
        crop = face_crop
    else:
        case = CASE_BODY
        crop = body_crop

    img_path = _save_image(directory, person_id, crop, case)

    log_path = _log_path(session_logs_dir)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("[{}] ENTRY | ID={} | case={} | image={}\n".format(
            timestamp, person_id, case, img_path))

    return img_path


def save_exit(
    session_logs_dir: str,
    person_id: str,
    face_crop,
    body_crop,
    face_clear: bool,
    face_found: bool,
    timestamp: str,
) -> str:
    """
    Save exit image and append to events.log.
    Same crop-selection logic as save_entry.
    """
    directory = _session_dirs(session_logs_dir, "exits")

    if face_clear and face_crop is not None:
        case = CASE_FACE
        crop = face_crop
    elif face_found and face_crop is not None:
        case = CASE_NOCLEAR
        crop = face_crop
    else:
        case = CASE_BODY
        crop = body_crop

    img_path = _save_image(directory, person_id, crop, case)

    log_path = _log_path(session_logs_dir)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("[{}] EXIT  | ID={} | case={} | image={}\n".format(
            timestamp, person_id, case, img_path))

    return img_path


def read_events_log(session_logs_dir: str, last_n: int = 100):
    """Read last N lines from this session's events.log."""
    log_path = _log_path(session_logs_dir)
    if not os.path.exists(log_path):
        return []
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return [l.strip() for l in lines[-last_n:]]


def make_session_logs_dir(base_logs_dir: str, video_name: str) -> str:
    """
    Issue 1 fix: Create a per-video session folder.
    e.g.  logs/myvideo_20250322_143012/
    Returns the session logs directory path.
    """
    safe_name = os.path.splitext(os.path.basename(video_name))[0]
    # Keep only alphanumeric + underscore
    safe_name = "".join(c if c.isalnum() or c == "_" else "_" for c in safe_name)[:40]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(base_logs_dir, "{}_{}".format(safe_name, ts))
    os.makedirs(os.path.join(session_dir, "entries"), exist_ok=True)
    os.makedirs(os.path.join(session_dir, "exits"),   exist_ok=True)
    return session_dir


def ensure_base_logs_dir(base_logs_dir: str):
    os.makedirs(base_logs_dir, exist_ok=True)