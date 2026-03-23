"""
Frame Processor  —  YOLOv8 + ByteTrack + InsightFace + ResNet50
================================================================
Fix 1: Box colors changed to bold, highly visible colors.
       GREEN  (0,200,60)  → face clear (Case A)
       ORANGE (0,120,255) → face found but unclear (Case B)  
       RED    (60,60,255) → body only / backside (Case C)
       GREY   (140,140,140) → skip frame
Fix 3: face_found flag returned from _detect_face_in_crop and passed to IDM.
"""

import cv2
import numpy as np
from datetime import datetime

from models.embedders import FaceEmbedder, BodyEmbedder
from utils.face_quality import check_face_quality
from identity_manager import IdentityManager

# Fix 1: dark, saturated colors (BGR)
COLOR_FACE_CLEAR  = (0,  200,  60)   # bold green  — Case A: face clear
COLOR_FACE_BLURRY = (0,  120, 255)   # bold orange — Case B: face found, not clear
COLOR_BODY_ONLY   = (60,  60, 255)   # bold red    — Case C: backside / no face
COLOR_SKIP        = (140, 140, 140)  # grey        — skip frame
FONT              = cv2.FONT_HERSHEY_SIMPLEX
BOX_THICK         = 2
_MIN_CROP_DIM     = 320


class FrameProcessor:
    def __init__(self, config: dict, identity_manager: IdentityManager):
        self.cfg        = config
        self.idm        = identity_manager
        self.frame_skip = config['frame_skip']
        self.min_face   = config['min_face_size']
        self.conf       = config.get('model_confidence', 0.3)
        self.iou_thr    = config.get('iou_threshold', 0.45)

        self._frame_count = 0
        self._yolo        = None
        self._face_emb    = FaceEmbedder()
        self._body_emb    = BodyEmbedder()
        self._last_tracks = []   # (tid, x1,y1,x2,y2, color) cache for skip frames

    def _ensure_yolo(self):
        if self._yolo is None:
            from ultralytics import YOLO
            self._yolo = YOLO("yolov8n.pt")
            print("[FrameProcessor] YOLOv8n loaded.")

    def process_frame(self, frame: np.ndarray) -> tuple:
        self._ensure_yolo()
        self._frame_count += 1
        is_det = (self._frame_count == 1) or (self._frame_count % self.frame_skip == 0)
        return self._detection_and_track(frame) if is_det else self._skip_frame_annotate(frame)

    # ------------------------------------------------------------------
    def _detection_and_track(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        ts   = datetime.now().isoformat(timespec='seconds')

        results = self._yolo.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=self.conf,
            iou=self.iou_thr,
            imgsz=640,
            verbose=False,
        )

        boxes_obj = results[0].boxes
        has_boxes = (
            boxes_obj is not None
            and boxes_obj.id is not None
            and len(boxes_obj) > 0
        )
        xyxy_all = boxes_obj.xyxy.cpu().numpy()           if has_boxes else np.empty((0,4))
        ids_all  = boxes_obj.id.cpu().numpy().astype(int) if has_boxes else np.empty((0,), dtype=int)

        annotated  = frame.copy()
        active_ids = set()
        self._last_tracks = []

        for box, tid in zip(xyxy_all, ids_all):
            x1 = max(0, int(box[0]));  y1 = max(0, int(box[1]))
            x2 = min(w, int(box[2]));  y2 = min(h, int(box[3]))
            if x2 <= x1 or y2 <= y1:
                continue

            tid_int = int(tid)
            active_ids.add(tid_int)

            body_crop = frame[y1:y2, x1:x2].copy()
            face_crop, face_clear, face_found = self._detect_face_in_crop(body_crop)

            face_emb = None
            if face_clear and face_crop is not None:
                face_emb = self._face_emb.get_embedding_from_crop(face_crop)
                if face_emb is None:
                    face_clear = False

            body_emb = self._body_emb.get_embedding(body_crop)

            self.idm.process_detection(
                track_id   = tid_int,
                face_clear = face_clear,
                face_found = face_found,
                face_emb   = face_emb,
                body_emb   = body_emb,
                body_crop  = body_crop,
                face_crop  = face_crop,
                frame_time = ts,
            )

            # Fix 1: pick color based on case
            if face_clear:
                color = COLOR_FACE_CLEAR    # Case A
            elif face_found:
                color = COLOR_FACE_BLURRY   # Case B
            else:
                color = COLOR_BODY_ONLY     # Case C

            self._last_tracks.append((tid_int, x1, y1, x2, y2, color))
            label = self.idm.get_label_for_track(tid_int)
            self._draw_box(annotated, x1, y1, x2, y2, label, color)

        self.idm.update_lost_tracks(active_ids, ts)
        self._draw_hud(annotated, self._frame_count, True, len(active_ids))
        return annotated, {'active_tracks': len(active_ids), 'detection_frame': True}

    # ------------------------------------------------------------------
    def _skip_frame_annotate(self, frame: np.ndarray):
        annotated = frame.copy()
        for (tid, x1, y1, x2, y2, color) in self._last_tracks:
            label = self.idm.get_label_for_track(tid)
            # Use same color as detection frame but slightly dimmer
            dim = tuple(max(0, c - 40) for c in color)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), dim, 1)
            cv2.putText(annotated, label, (x1, max(y1 - 4, 12)), FONT, 0.42, dim, 1)
        self._draw_hud(annotated, self._frame_count, False, len(self._last_tracks))
        return annotated, {'active_tracks': self.idm.get_active_count(), 'detection_frame': False}

    # ------------------------------------------------------------------
    def _detect_face_in_crop(self, body_crop: np.ndarray):
        """
        Returns (face_crop, face_clear, face_found).
        face_found = InsightFace found ANY face in crop
        face_clear = face_found AND quality check passed
        """
        try:
            self._face_emb._ensure_loaded()
            app = self._face_emb._app
            if app is None:
                return None, False, False

            crop_h, crop_w = body_crop.shape[:2]
            scale = 1.0
            if crop_h < _MIN_CROP_DIM or crop_w < _MIN_CROP_DIM:
                scale  = max(_MIN_CROP_DIM / crop_h, _MIN_CROP_DIM / crop_w)
                search = cv2.resize(body_crop,
                                    (int(crop_w * scale), int(crop_h * scale)),
                                    interpolation=cv2.INTER_LINEAR)
            else:
                search = body_crop

            faces = app.get(search)
            if not faces:
                return None, False, False   # Case C

            best = max(faces, key=lambda f: f.det_score)
            bx   = best.bbox.astype(float)
            fx1  = max(0,      int(bx[0] / scale))
            fy1  = max(0,      int(bx[1] / scale))
            fx2  = min(crop_w, int(bx[2] / scale))
            fy2  = min(crop_h, int(bx[3] / scale))

            if fx2 <= fx1 or fy2 <= fy1:
                return None, False, False

            face_crop = body_crop[fy1:fy2, fx1:fx2].copy()
            is_clear, _ = check_face_quality(face_crop, self.min_face)
            return face_crop, is_clear, True   # face_found always True here

        except Exception as e:
            print("[_detect_face_in_crop] {}".format(e))
            return None, False, False

    # ------------------------------------------------------------------
    def _draw_box(self, img, x1, y1, x2, y2, label, color):
        # Thick border
        cv2.rectangle(img, (x1, y1), (x2, y2), color, BOX_THICK)
        # White inner border for contrast on any background
        cv2.rectangle(img, (x1+1, y1+1), (x2-1, y2-1), (255,255,255), 1)
        # Label background
        (lw, lh), _ = cv2.getTextSize(label, FONT, 0.52, 1)
        label_y = max(lh + 8, y1)
        cv2.rectangle(img, (x1, label_y - lh - 6), (x1 + lw + 4, label_y), color, -1)
        cv2.putText(img, label, (x1 + 2, label_y - 3), FONT, 0.52, (255, 255, 255), 1)

    def _draw_hud(self, img, frame_idx, is_det, n_tracks):
        h = img.shape[0]
        mode = "DET" if is_det else "skip"
        text = "F:{}[{}] Tracks:{} Unique(vid):{}".format(
            frame_idx, mode, n_tracks, self.idm.get_video_unique_count()
        )
        # Black shadow then white text for readability
        cv2.putText(img, text, (11, h-9),  FONT, 0.45, (0,0,0), 2)
        cv2.putText(img, text, (10, h-10), FONT, 0.45, (220,220,220), 1)