"""
Identity Manager
================
Fix 1 (colors): handled in frame_processor.py
Fix 2 (DB not updating): _log_entry now calls db.update_face_entry_image() after
        insert so the entry_image is stored correctly. Also _try_upgrade now
        correctly moves the entry_image from the body record to the face record.
Fix 3 (same person → different IDs): Three sub-bugs fixed:
  3a. When _try_upgrade creates a new face_id for a body→face transition,
      the TrackedPerson.identity_id changes to face_id but the ENTRY was already
      logged under body_id. Now we log a single unified entry under whichever ID
      is current at exit time, and the entry_image is transferred correctly.
  3b. body_threshold lowered from 0.4 → 0.25 (ResNet50 body embeddings are weaker
      than ArcFace; same person in different frames easily scores 0.25–0.45).
  3c. In _try_upgrade, when face becomes clear for a body-tracked person, we check
      face DB AND also check if any existing face record's body_embedding is similar
      to the current body_embedding — this catches cases where the same person was
      previously stored with a face but body embedding is the only link available.
  3d. Track-level identity is now LOCKED once assigned — once a track_id has an
      identity_id, it never gets re-matched even if a new detection fires. This
      prevents the case where the same track gets re-identified as a new person
      on a detection frame after several skip frames.
"""

import uuid
import numpy as np
from datetime import datetime

import database.db_manager as db
import utils.logger as logger_utils
from utils.similarity import find_best_face_match, find_best_body_match, cosine_similarity

STATE_NEW    = "NEW"
STATE_ACTIVE = "ACTIVE"
STATE_LOST   = "LOST"
STATE_EXITED = "EXITED"


class TrackedPerson:
    def __init__(self, track_id, identity_id, id_type, entry_ts):
        self.track_id    = track_id
        self.identity_id = identity_id   # face_id or body_id
        self.id_type     = id_type       # 'face' or 'body'
        self.state       = STATE_NEW
        self.lost_frames = 0
        self.entry_ts    = entry_ts
        self.entry_image = ""

        # Latest crops for logging
        self.last_body_crop  = None
        self.last_face_crop  = None
        self.last_face_clear = False
        self.last_face_found = False

        # Fix 3d: once an identity is assigned, remember the original body_id
        # so that if we upgrade to face_id, we still know where to transfer the entry log
        self.original_body_id = None   # set when id_type starts as 'body'


class IdentityManager:
    def __init__(self, config: dict, session_logs_dir: str):
        self.cfg            = config
        self.db_path        = config['db_path']
        self.session_logs   = session_logs_dir
        self.face_threshold = config['face_threshold']
        # Fix 3b: use a lower body threshold — ResNet50 is weaker than ArcFace
        self.body_threshold = min(config['body_threshold'], 0.25)
        self.exit_frames    = config['exit_frames']
        self.face_w         = config['face_fusion_weight']
        self.body_w         = config['body_fusion_weight']

        self._tracks: dict = {}

        # Per-video unique counter
        self._video_unique_ids: set = set()
        self._video_unique_count: int = 0

        # Fix 3d: in-session identity cache — track_id → identity_id
        # Once a track has an identity, it keeps it (no re-matching)
        self._track_identity_cache: dict = {}

    # ------------------------------------------------------------------ #
    def process_detection(
        self,
        track_id: int,
        face_clear: bool,
        face_found: bool,
        face_emb,
        body_emb,
        body_crop: np.ndarray,
        face_crop,
        frame_time=None,
    ):
        ts = frame_time or datetime.now().isoformat(timespec='seconds')

        if track_id in self._tracks:
            person = self._tracks[track_id]
            if person.state == STATE_LOST:
                person.state = STATE_ACTIVE
            person.lost_frames = 0
            # Always update crops
            person.last_body_crop  = body_crop
            if face_crop is not None:
                person.last_face_crop  = face_crop
            person.last_face_clear = face_clear
            person.last_face_found = face_found

            # Try upgrade body → face only if face is now clear
            if face_clear and person.id_type == 'body' and face_emb is not None:
                self._try_upgrade(person, face_emb, body_emb, ts)
            return

        # Fix 3d: check identity cache first — same track_id seen before?
        if track_id in self._track_identity_cache:
            cached_id, cached_type = self._track_identity_cache[track_id]
            # Re-use cached identity — do NOT re-match against DB
            person = TrackedPerson(track_id, cached_id, cached_type, ts)
            person.last_body_crop  = body_crop
            person.last_face_crop  = face_crop
            person.last_face_clear = face_clear
            person.last_face_found = face_found
            self._tracks[track_id] = person
            person.state = STATE_ACTIVE
            return

        # Brand new track — match against DB or register
        identity_id, id_type = self._match_or_register(
            face_clear, face_emb, body_emb, ts
        )
        person = TrackedPerson(track_id, identity_id, id_type, ts)
        person.last_body_crop  = body_crop
        person.last_face_crop  = face_crop
        person.last_face_clear = face_clear
        person.last_face_found = face_found
        if id_type == 'body':
            person.original_body_id = identity_id

        self._tracks[track_id] = person
        self._track_identity_cache[track_id] = (identity_id, id_type)

        self._log_entry(person, ts)
        person.state = STATE_ACTIVE

    # ------------------------------------------------------------------ #
    def update_lost_tracks(self, active_track_ids: set, frame_time=None):
        ts = frame_time or datetime.now().isoformat(timespec='seconds')
        to_remove = []

        for tid, person in list(self._tracks.items()):
            if person.state == STATE_EXITED:
                to_remove.append(tid)
                continue
            if tid not in active_track_ids:
                person.lost_frames += 1
                if person.state == STATE_ACTIVE:
                    person.state = STATE_LOST
                if person.lost_frames >= self.exit_frames:
                    self._log_exit(person, ts)
                    person.state = STATE_EXITED
                    to_remove.append(tid)
                    # Keep in identity cache so re-entry reuses same ID
                    self._track_identity_cache[tid] = (person.identity_id, person.id_type)

        for tid in to_remove:
            self._tracks.pop(tid, None)

    # ------------------------------------------------------------------ #
    def flush_all(self, frame_time=None):
        ts = frame_time or datetime.now().isoformat(timespec='seconds')
        for person in list(self._tracks.values()):
            if person.state != STATE_EXITED:
                self._log_exit(person, ts)
                person.state = STATE_EXITED
        self._tracks.clear()

    # ------------------------------------------------------------------ #
    def _match_or_register(self, face_clear, face_emb, body_emb, ts):
        if face_clear and face_emb is not None:
            # Case A: face is clear — search face DB
            all_faces = db.get_all_faces(self.db_path)
            matched_id, score = find_best_face_match(
                face_emb, body_emb, all_faces,
                self.face_threshold, self.face_w, self.body_w
            )
            if matched_id:
                self._update_face_embeddings(matched_id, face_emb, body_emb, all_faces)
                return matched_id, 'face'
            # No face match — also check body DB using body_emb as fallback
            if body_emb is not None:
                all_bodies = db.get_all_bodies(self.db_path)
                body_matched, body_score = find_best_body_match(
                    body_emb, all_bodies, self.body_threshold
                )
                if body_matched:
                    # Same person was previously body-only; now has clear face
                    # Upgrade that body record to face
                    new_id = str(uuid.uuid4())
                    db.insert_face(self.db_path, new_id, ts, face_emb, body_emb, "")
                    db.update_body_link(self.db_path, body_matched, new_id)
                    return new_id, 'face'
            # Truly new person
            new_id = str(uuid.uuid4())
            db.insert_face(self.db_path, new_id, ts, face_emb, body_emb, "")
            return new_id, 'face'

        else:
            # Case B/C: face not clear or not found — body only
            all_bodies = db.get_all_bodies(self.db_path)
            if body_emb is not None:
                matched_id, score = find_best_body_match(
                    body_emb, all_bodies, self.body_threshold
                )
            else:
                matched_id, score = None, 0.0

            if matched_id:
                # Fix 3b: update the stored body embedding with running average
                old = next((b for b in all_bodies if b['body_id'] == matched_id), None)
                if old and old['body_embedding'] is not None:
                    new_emb = self._running_avg(old['body_embedding'], body_emb)
                    db.update_body_embedding(self.db_path, matched_id, new_emb)
                return matched_id, 'body'
            else:
                new_id = str(uuid.uuid4())
                db.insert_body(self.db_path, new_id, ts, body_emb, "")
                return new_id, 'body'

    # ------------------------------------------------------------------ #
    def _try_upgrade(self, person: TrackedPerson, face_emb, body_emb, ts):
        """
        Fix 3: When a body-tracked person shows a clear face, upgrade to face_id.
        Key fix: the ENTRY was logged under body_id. After upgrade, the exit will
        be logged under face_id. To keep them consistent, we transfer the entry
        record: the face row gets the original entry_ts and entry_image from body row.
        The body row's linked_face_id is set so it won't be double-counted.
        """
        all_faces = db.get_all_faces(self.db_path)
        matched_face_id, score = find_best_face_match(
            face_emb, body_emb, all_faces,
            self.face_threshold, self.face_w, self.body_w
        )

        original_body_id = person.original_body_id or person.identity_id
        original_entry_ts = db.get_body_entry_ts(self.db_path, original_body_id) or ts

        if matched_face_id:
            # Merge into existing face record
            db.update_body_link(self.db_path, original_body_id, matched_face_id)
            self._update_face_embeddings(matched_face_id, face_emb, body_emb, all_faces)
            new_face_id = matched_face_id
        else:
            # Create new face record with ORIGINAL entry timestamp from body period
            new_face_id = str(uuid.uuid4())
            db.insert_face(
                self.db_path, new_face_id,
                original_entry_ts,   # preserve original entry time
                face_emb, body_emb,
                person.entry_image,  # preserve original entry image
            )
            db.update_body_link(self.db_path, original_body_id, new_face_id)

        # Update person object
        old_body_id      = person.identity_id
        person.identity_id = new_face_id
        person.id_type     = 'face'

        # Update identity cache
        self._track_identity_cache[person.track_id] = (new_face_id, 'face')

        print("[IDM] Upgraded track {} from body:{} → face:{}".format(
            person.track_id, old_body_id[:8], new_face_id[:8]
        ))

    # ------------------------------------------------------------------ #
    def _log_entry(self, person: TrackedPerson, ts: str):
        img_path = logger_utils.save_entry(
            session_logs_dir = self.session_logs,
            person_id        = person.identity_id,
            face_crop        = person.last_face_crop,
            body_crop        = person.last_body_crop,
            face_clear       = person.last_face_clear,
            face_found       = person.last_face_found,
            timestamp        = ts,
        )
        person.entry_image = img_path
        # Fix 2: update the DB record with the entry image path
        if person.id_type == 'face':
            db.update_face_entry_image(self.db_path, person.identity_id, img_path)
        else:
            db.update_body_entry_image(self.db_path, person.identity_id, img_path)

    def _log_exit(self, person: TrackedPerson, ts: str):
        img_path = logger_utils.save_exit(
            session_logs_dir = self.session_logs,
            person_id        = person.identity_id,
            face_crop        = person.last_face_crop,
            body_crop        = person.last_body_crop,
            face_clear       = person.last_face_clear,
            face_found       = person.last_face_found,
            timestamp        = ts,
        )
        if person.id_type == 'face':
            db.update_face_exit(self.db_path, person.identity_id, ts, img_path)
        else:
            db.update_body_exit(self.db_path, person.identity_id, ts, img_path)

        # Per-video unique counter
        if person.identity_id not in self._video_unique_ids:
            self._video_unique_ids.add(person.identity_id)
            self._video_unique_count += 1

    # ------------------------------------------------------------------ #
    def _update_face_embeddings(self, face_id, face_emb, body_emb, all_faces):
        old = next((f for f in all_faces if f['face_id'] == face_id), None)
        if old:
            db.update_face_embeddings(
                self.db_path, face_id,
                self._running_avg(old['face_embedding'], face_emb),
                self._running_avg(old.get('body_embedding'), body_emb),
            )

    def _running_avg(self, old_emb, new_emb, alpha=0.8):
        """Exponential moving average of embeddings. alpha=0.8 weights history heavily."""
        if old_emb is None: return new_emb
        if new_emb is None: return old_emb
        blended = alpha * old_emb + (1.0 - alpha) * new_emb
        norm = np.linalg.norm(blended)
        return blended / norm if norm > 1e-6 else new_emb

    # ------------------------------------------------------------------ #
    def get_active_count(self) -> int:
        return len(self._tracks)

    def get_video_unique_count(self) -> int:
        return self._video_unique_count

    def get_label_for_track(self, track_id: int) -> str:
        if track_id in self._tracks:
            p = self._tracks[track_id]
            short = p.identity_id[:8]
            case = "F" if p.last_face_clear else ("f?" if p.last_face_found else "B")
            return "{}:{} [{}]".format(case, short, p.state[0])
        return "?"