# katomaran-hackathon
# Intelligent Face Tracker with Auto-Registration & Unique Visitor Counting

A hybrid identity pipeline that detects, tracks, recognises, and counts unique visitors from video. Uses face embeddings as the primary signal and body embeddings (OSNet) as fallback. Every person gets exactly one entry log and one exit log. Re-entries are detected and do **not** re-increment the unique counter.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Person detection | YOLOv8 |
| Face detection & embedding | InsightFace (ArcFace backbone) |
| Body re-ID embedding | ResNet50 |
| Multi-object tracker | ByteTrack |
| Database | SQLite (2 tables) |
| Frontend | Streamlit |
| Config | `config.json` |

---

## Architecture Diagram

```
                        ┌─────────────────────────────────┐
                        │     Video input (Streamlit)      │
                        └────────────────┬────────────────┘
                                         │
                        ┌────────────────▼────────────────┐
                        │         Frame processor          │
                        │   frame_count % frame_skip == 0  │
                        └─────────┬──────────────┬─────────┘
                                  │              │
                       detect frame            tracking frame
                                  │              │
              ┌───────────────────▼──┐    ┌──────▼──────────────────┐
              │     YOLOv8           │    │      ByteTrack           │
              │  person + face detect│    │  Kalman prediction only  │
              └────────────┬─────────┘    └──────────────────────────┘
                           │
              ┌────────────▼─────────────┐
              │    Face quality check     │
              │  blur · size · angle      │
              │  occlusion · backside     │
              └────────┬────────┬─────────┘
                       │        │
                    CLEAR    NOT CLEAR
                       │        │
         ┌─────────────▼──┐  ┌──▼──────────────────┐
         │  InsightFace    │  │  ResNet body embed    │
         │  face embed     │  │  only                │
         │  + ResNet body   │  └──────────┬───────────┘
         └────────┬────────┘             │
                  │                      │
         fused score                body similarity
         face×0.7 + body×0.3        threshold 0.4
         threshold 0.7                   │
                  │                      │
         ┌────────▼──────────────────────▼─────────┐
         │             Identity manager             │
         │   match · assign · merge · state machine │
         └────┬──────────┬───────────────┬──────────┘
              │          │               │
          face_id    temp_id        re-entry
          (face DB)  (body DB)    (no recount)
              │          │
         ┌────▼──────────▼──────┐
         │    State machine      │
         │  NEW → ACTIVE →       │
         │  LOST → EXITED        │
         └────┬──────────────────┘
              │
    ┌─────────┼─────────────┐
    │         │             │
┌───▼───┐ ┌──▼────┐ ┌──────▼──────┐
│SQLite │ │ Logs  │ │  Streamlit  │
│  DB   │ │ files │ │   live UI   │
└───────┘ └───────┘ └─────────────┘
```

---

## Component Reference

### C1 — `config.json`

Controls all tunable parameters. Loaded once at startup.

```json
{
  "frame_skip": 5,
  "face_threshold": 0.7,
  "body_threshold": 0.4,
  "exit_frames": 30,
  "min_face_size": 40
}
```

| Key | Purpose |
|---|---|
| `frame_skip` | Run YOLOv8 only every Nth frame; ByteTrack fills between |
| `face_threshold` | Cosine similarity cutoff for face match (higher = stricter) |
| `body_threshold` | Cosine similarity cutoff for body-only match |
| `exit_frames` | Consecutive missing frames before EXITED state is set |
| `min_face_size` | Minimum face crop size (px) for quality gate |

---

### C2 — Frame Processor & Frame Skipping

```
frame_count % frame_skip == 0  →  detection frame (YOLOv8 runs)
                            else  →  tracking frame (ByteTrack Kalman only)
```

Detection is expensive. Tracking is cheap. YOLOv8 initialises and refreshes tracks; ByteTrack maintains identity and position between detections.

---

### C3 — YOLOv8 Person + Face Detection

On detection frames:

1. YOLOv8 detects **person** bounding boxes across the full frame.
2. A secondary pass (InsightFace built-in detector) detects **faces** within each person crop.

Output per person: `body_bbox` + `face_bbox` (or `None`).

---

### C4 — Face Quality Gate

A face is `NOT CLEAR` if any condition holds:

| Condition | Check |
|---|---|
| Motion blur | Laplacian variance below threshold |
| Small face | Bounding box < `min_face_size` |
| Side/profile | Landmark angle check |
| Occlusion | Partial face coverage |
| Backside | No face detected in person crop |

Output: `CLEAR` or `NOT CLEAR` — single boolean that splits the entire pipeline.

---

### C5 — Identity Manager (core brain)

Maintains in-memory mapping: `track_id → face_id | temp_id`.

Responsibilities:
- Match new embeddings against the database
- Assign new `face_id` (clear face) or `temp_id` (body only) when no match found
- Trigger the **temp → face upgrade** when a person's face clears mid-track
- Drive the state machine per track

---

### C6 — Face Embedding (InsightFace / ArcFace)

Runs **only when face quality is CLEAR**.

- Model: ArcFace backbone via InsightFace
- Output: 512-dimensional unit vector
- Discriminative: same person ≈ cosine similarity > 0.9; different people < 0.5

---

### C7 — Body Embedding (OSNet)

Runs **for all persons**, regardless of face quality.

- Model: OSNet (Omni-Scale Network, person re-ID trained)
- Encodes: clothing colour, texture, body shape
- Used as sole signal when face is not available; as secondary signal (weight 0.3) when face is clear
- Less reliable than face embedding — clothing can change between visits

---

### C8 — Similarity Matching Engine

**When face is CLEAR:**

```
fused_score = face_cosine × 0.7 + body_cosine × 0.3

fused_score < face_threshold (0.7)  →  match found, return existing face_id
fused_score ≥ 0.7                  →  new person, assign new face_id
```

**When face is NOT CLEAR:**

```
body_cosine < body_threshold (0.4)  →  match found, return existing temp_id
body_cosine ≥ 0.4                  →  new person, assign new temp_id
```

---

### C9 — State Machine (per tracked person)

```
NEW ──────────────► ACTIVE ──────► LOST ──────────► EXITED
 │                    ▲              │ reappears        │
 │ entry logged once  │              └──────────────────┘
 │                    │                  (< exit_frames)
 └────────────────────┘
                                    exit logged once on EXITED
                                    unique counter increments
                                    if no prior exit_timestamp
```

The **LOST buffer** absorbs brief detection failures (occlusion, motion blur) without falsely logging an exit. Duration controlled by `exit_frames`.

---

### C10 — ByteTrack

Multi-object tracker replacing DeepSort.

**Key differences from DeepSort:**

| Feature | DeepSort | ByteTrack |
|---|---|---|
| Matching strategy | Appearance + IoU (one pass) | Two-pass: high-conf detections first, then low-conf |
| Low-confidence detections | Discarded | Used in second association pass to recover lost tracks |
| Appearance model | Required (deep features) | Optional — IoU-only mode available |
| Track recovery | Limited | Better recovery of briefly-occluded persons |
| Speed | Moderate | Faster (simpler association logic) |

ByteTrack processes high-confidence detections first, associates them to existing tracks, then attempts to recover unmatched tracks using low-confidence detections. This makes it more robust to partial occlusion and crowded scenes.

`track_id` assigned by ByteTrack is the real-time key the Identity Manager maps to `face_id` or `temp_id`.

---

### C11 — Identity Merger (temp → face upgrade)

Runs every frame for every person currently tracked under a `temp_id`.

```
Each frame:
  if face quality is now CLEAR:
    generate face_embedding
    search face DB
    if match found  →  use existing face_id, merge temp history
    if no match     →  create new face_id with ORIGINAL entry_timestamp
    update body table: set linked_face_id = face_id
    update tracker mapping: track_id → face_id (replacing temp_id)
```

The original `entry_timestamp` from the `temp_id` is **always preserved** — the person physically entered when they first appeared, not when their face became identifiable.

---

### C12 — Logging System

```
logs/
├── entries/
│   └── YYYY-MM-DD/
│       └── face_id__HH-MM-SS.jpg    ← cropped face or body crop
├── exits/
│   └── YYYY-MM-DD/
│       └── face_id__HH-MM-SS.jpg
└── events.log
```

**On ENTRY** (state = NEW):
- Save crop image to `logs/entries/YYYY-MM-DD/`
- Append to `events.log`: `[timestamp] ENTRY | ID=<id> | path=<img>`
- Insert row in SQLite with `entry_timestamp`

**On EXIT** (state = EXITED):
- Save crop image to `logs/exits/YYYY-MM-DD/`
- Append to `events.log`: `[timestamp] EXIT | ID=<id> | path=<img>`
- Update SQLite row: set `exit_timestamp`

Invariant: **exactly one ENTRY and one EXIT per ID**, enforced by the state machine.

---

### C13 — SQLite Database

**`face` table** — persons with at least one clear face detection:

| Column | Type | Notes |
|---|---|---|
| `face_id` | TEXT PK | UUID |
| `entry_timestamp` | REAL | Unix epoch, when first seen |
| `exit_timestamp` | REAL | NULL until exit |
| `face_embedding` | BLOB | 512-dim ArcFace vector |

**`body` table** — persons tracked by appearance only:

| Column | Type | Notes |
|---|---|---|
| `body_id` | TEXT PK | UUID |
| `entry_timestamp` | REAL | When first seen |
| `exit_timestamp` | REAL | NULL until exit |
| `linked_face_id` | TEXT FK | NULL until temp→face upgrade |
| `body_embedding` | BLOB | OSNet vector |

---

### C14 — Unique Counter

A person is counted **exactly once** — on their first exit.

```sql
-- Unique visitor total
SELECT COUNT(*) FROM face WHERE exit_timestamp IS NOT NULL
UNION ALL
SELECT COUNT(*) FROM body
  WHERE exit_timestamp IS NOT NULL
    AND linked_face_id IS NULL;
```

The `linked_face_id IS NULL` clause prevents double-counting a body row that was later promoted to a face row.

Re-entries are detected by running similarity search against IDs that already have `exit_timestamp IS NOT NULL`. A match → reuse the existing ID, no increment.

---

### C15 — Streamlit Frontend

- Video upload widget
- Live annotated frame display (bounding boxes, ID labels, state indicators)
- Real-time sidebar: unique visitors, active count, total entries, total exits
- Scrollable `events.log` viewer
- Tabular SQLite viewer (face + body tables)

---

## Three Identity Cases

### Case 1 — Face clear from the start

```
YOLOv8 detect → face quality CLEAR
→ InsightFace face_emb + OSNet body_emb
→ fused similarity search (face×0.7 + body×0.3)
→ score < 0.7 → NEW face_id, log entry
→ score ≥ 0.7 → existing face_id, continue tracking
→ person exits → log exit, increment unique count (if first exit)
```

### Case 2 — Face unclear initially, clears later

```
Phase 1: face NOT CLEAR
→ OSNet body_emb only
→ body similarity < 0.4 → NEW temp_id, log entry with entry_timestamp
→ ByteTrack tracks person, face quality re-checked every frame

Phase 2: face becomes CLEAR
→ InsightFace face_emb generated
→ face DB searched
  → match found → merge into existing face_id (preserve entry_timestamp)
  → no match   → new face_id (use original entry_timestamp from temp_id)
→ body table updated: linked_face_id = face_id
→ tracking continues under face_id
```

### Case 3 — Face never clear

```
→ OSNet body_emb only, every detection frame
→ body DB searched each frame
→ new temp_id assigned, entry logged
→ state machine runs full cycle: NEW → ACTIVE → LOST → EXITED
→ exit logged on body table row
→ unique count updated (body_id with no linked_face_id)

Re-entry in Case 3:
→ body similarity match against exited temp_id → same person, no recount
→ if face is clear on re-entry → upgrade to face_id, merge, still no recount
```

---

## Re-entry Handling

```
Person re-enters frame
→ generate embedding (face or body, whichever is available)
→ similarity search including IDs where exit_timestamp IS NOT NULL
→ high similarity match → re-entry confirmed
  → reuse existing ID (face_id or temp_id)
  → DO NOT increment unique counter
  → log new entry event against same ID
→ no match → new person, new ID, new count on exit
```

If a body-only person re-enters with a now-visible face:
1. Generate face embedding
2. Upgrade `temp_id` → `face_id`
3. Merge entry/exit history
4. Counter still not incremented (already counted on first exit)

---

## End-to-End Working Flow

```
1.  config.json loaded → thresholds, frame_skip, exit_frames set
2.  SQLite initialised → face and body tables created if absent
3.  Streamlit: user uploads video
4.  Frame extracted
5.  Frame processor: frame_count % frame_skip == 0?
      YES → step 6
      NO  → ByteTrack Kalman update, skip to step 12
6.  YOLOv8: person bboxes + face bboxes per person
7.  ByteTrack: assign / update track_ids
8.  For each track:
      Face quality check → CLEAR or NOT CLEAR
9.  CLEAR  → InsightFace face_emb + OSNet body_emb
              fused similarity search, threshold 0.7
    NOT CLEAR → OSNet body_emb only
                body similarity search, threshold 0.4
10. Identity Manager:
      match found → existing face_id or temp_id
      no match    → new face_id or temp_id, insert SQLite row
      temp_id + face now clear → trigger Identity Merger
11. State machine update per track:
      NEW     → log ENTRY (image + timestamp → filesystem + SQLite + events.log)
      ACTIVE  → continue
      LOST    → increment lost_frame counter
      EXITED  → log EXIT, increment unique count if no prior exit_timestamp
12. Annotate frame: bboxes, ID labels, state markers
13. Draw HUD: unique count, active persons, entries, exits
14. Write annotated frame to output video
15. Update Streamlit UI → st.image(), sidebar stats
16. Repeat from step 4
17. On video end: flush all ACTIVE/LOST tracks → force EXITED, log exits
```

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

```

---

## Requirements

```
python>=3.9
ultralytics          # YOLOv8
insightface          # ArcFace face embedding
torchreid            # OSNet body embedding
opencv-python
numpy
scipy
sqlite3              # stdlib
streamlit
```

ByteTrack is integrated directly from source (no separate pip install needed — include `bytetrack/` in the repo).

---

## Configuration Quick Reference

```json
{
  "frame_skip": 5,
  "face_threshold": 0.70,
  "body_threshold": 0.40,
  "exit_frames": 30,
  "min_face_size": 40,
  "face_body_weight": [0.7, 0.3],
  "bytetrack": {
    "track_thresh": 0.5,
    "track_buffer": 30,
    "match_thresh": 0.8,
    "min_box_area": 10
  }
}
```

`bytetrack.track_thresh` — confidence threshold separating high and low detection pools for the two-pass association.  
`bytetrack.track_buffer` — frames to keep a lost track alive before discarding (should match `exit_frames`).  
`bytetrack.match_thresh` — IoU threshold for track-detection association.

---

## Requirement Coverage

| Requirement | Component |
|---|---|
| YOLOv8 person + face detection | C3 |
| InsightFace embeddings | C6 |
| OSNet body fallback | C7 |
| Auto-registration of new persons | C5 |
| Tracking across frames | C10 (ByteTrack) |
| Frame skip config | C1 + C2 |
| Exactly one entry per person | C9 state machine |
| Exactly one exit per person | C9 state machine |
| Cropped image in logs | C12 |
| Timestamp logging | C12 + C13 |
| `events.log` | C12 |
| `logs/entries/` + `logs/exits/` | C12 |
| SQLite face + body tables | C13 |
| Unique visitor count | C14 |
| Re-entry no double-count | C14 + C5 |
| Streamlit frontend | C15 |
| `config.json` | C1 |
---
# “This project is a part of a hackathon run by https://katomaran.com ” 
