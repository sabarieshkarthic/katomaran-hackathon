# katomaran-hackathon
# Intelligent Face Tracker — Auto-Registration & Unique Visitor Counting

## Stack
| Component | Library |
|---|---|
| Person detection | YOLOv8 (`ultralytics`) |
| Face detection | InsightFace built-in detector |
| Face embedding | InsightFace ArcFace (512-dim) |
| Body embedding | ResNet50 penultimate layer (2048-dim) |
| Tracking | ByteTrack |
| Frontend | Streamlit |
| Database | SQLite (2 tables: `face`, `body`) |

---

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

YOLOv8 weights (`yolov8n.pt`) and InsightFace models (`buffalo_l`) download automatically on first run.

---

## Project Structure

```
face_tracker/
├── app.py                    # Streamlit frontend
├── frame_processor.py        # Per-frame pipeline (YOLO + ByteTrack + embeddings)
├── identity_manager.py       # Core brain: state machine, matching, logging
├── config.json               # All tunable parameters
├── requirements.txt
├── database/
│   └── db_manager.py         # SQLite CRUD (face + body tables)
├── models/
│   └── embedders.py          # FaceEmbedder (InsightFace) + BodyEmbedder (ResNet50)
└── utils/
    ├── face_quality.py       # Blur, size, angle quality checks
    ├── similarity.py         # Cosine similarity + fused matching
    └── logger.py             # Entry/exit image saving + events.log
```

---

## Configuration (`config.json`)

| Parameter | Default | Description |
|---|---|---|
| `frame_skip` | 5 | Run YOLO every N frames; DeepSORT Kalman-predicts between |
| `face_threshold` | 0.7 | Cosine similarity cutoff for face matching (fused face+body) |
| `body_threshold` | 0.6 | Cosine similarity cutoff for body-only matching |
| `exit_frames` | 30 | Frames a person must be missing before marked EXITED |
| `min_face_size` | 40 | Minimum face bounding-box pixels for quality pass |
| `face_fusion_weight` | 0.7 | Weight of face score in fused similarity |
| `body_fusion_weight` | 0.3 | Weight of body score in fused similarity |

---

## Three Processing Cases

### Case 1 — Face clear from entry
`YOLOv8` → face detected → quality ✓ → InsightFace embedding (512-dim) + ResNet50 embedding (2048-dim) → fused cosine search in `face` table → match (≥0.7): known person reactivated; no match: new `face_id` inserted → ByteTrack tracks → on exit: `exit_timestamp` written, unique counter updated.

### Case 2 — Face unclear initially, clears later
Phase 1: body-only ResNet50 embedding → search `body` table (threshold 0.4) → new `temp_id` in `body` table → ByteTrack tracks + per-frame face quality recheck.  
Phase 2 (face clears): InsightFace embedding → search `face` table → if match: link `body` row to existing `face_id`; if no match: create new `face_id` with **original entry timestamp** from temp period → `body.linked_face_id` updated.

### Case 3 — Face never clear (backside / always occluded)
Entire lifetime tracked by `body_id` (temp_id). ResNet50 body embeddings only. On exit: `body` table row gets `exit_timestamp`. Unique count = `body` rows with `exit_timestamp IS NOT NULL AND linked_face_id IS NULL`.

---

## Unique Visitor Count Logic

```sql
-- Persons identified by face
SELECT COUNT(*) FROM face WHERE exit_timestamp IS NOT NULL

-- Persons tracked body-only (never upgraded to face)
+ SELECT COUNT(*) FROM body WHERE exit_timestamp IS NOT NULL AND linked_face_id IS NULL
```

Re-entries are detected by similarity match against IDs that already have `exit_timestamp`. They do **not** increment the counter.

---


```

- **NEW**: First appearance. Entry logged once (image + timestamp + DB row).
- **ACTIVE**: Detected/tracked normally.
- **LOST**: Missing from frame. Counter increments each frame. Reverts to ACTIVE if redetected within `exit_frames`.
- **EXITED**: `lost_frames >= exit_frames`. Exit logged once. Unique counter updated.

---

## Log Structure

```
logs/
├── entries/
│   └── 2025-01-15/
│       └── <id>_<timestamp>.jpg
├── exits/
│   └── 2025-01-15/
│       └── <id>_<timestamp>.jpg
└── events.log
```

`events.log` format:
```
[2025-01-15T09:12:33] ENTRY | ID=a3f1b2c4-... | image=logs/entries/...
[2025-01-15T09:13:01] EXIT  | ID=a3f1b2c4-... | image=logs/exits/...
```

---

## Hardware Notes
- **CPU**: Works but slow. Expect ~2–3 fps with frame_skip=5.
- **GPU (CUDA)**: Replace `onnxruntime` with `onnxruntime-gpu` in requirements.txt and set `ctx_id=0` in InsightFace. Expect 15–25 fps.
- ResNet50 automatically uses CUDA if available via PyTorch.

---
# “This project is a part of a hackathon run by https://katomaran.com ” 
