"""
Embedding Modules
- FaceEmbedder: InsightFace ArcFace → 512-dim face embedding
- BodyEmbedder: ResNet50 (torchvision) → 2048-dim body embedding (L2-normalized)
"""

import numpy as np
import cv2

# Lazy imports to avoid crash if not installed
_insightface_app = None
_body_model = None
_body_transform = None
_torch = None


def _get_insightface():
    global _insightface_app
    if _insightface_app is None:
        import insightface
        from insightface.app import FaceAnalysis
        _insightface_app = FaceAnalysis(
            name='buffalo_l',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        _insightface_app.prepare(ctx_id=0, det_size=(640, 640))
    return _insightface_app


def _get_body_model():
    global _body_model, _body_transform, _torch
    if _body_model is None:
        import torch
        import torchvision.models as models
        import torchvision.transforms as T
        _torch = torch

        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Remove final classification layer — use penultimate (2048-dim) as embedding
        model = torch.nn.Sequential(*list(model.children())[:-1])
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        _body_model = model

        _body_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),   # Standard Re-ID input size (H x W)
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
    return _body_model, _body_transform, _torch


class FaceEmbedder:
    """
    Wraps InsightFace FaceAnalysis to extract ArcFace 512-dim embeddings.
    Input: full frame (BGR numpy array) + face bounding box [x1,y1,x2,y2]
    Output: 512-dim L2-normalized numpy float32 vector, or None
    """

    def __init__(self):
        self._app = None

    def _ensure_loaded(self):
        if self._app is None:
            self._app = _get_insightface()

    def get_embedding(self, frame: np.ndarray, face_box=None) -> np.ndarray:
        """
        Run InsightFace on the full frame. If face_box is given,
        restrict to the face region for efficiency.
        Returns normalized 512-dim vector or None.
        """
        self._ensure_loaded()
        try:
            if face_box is not None:
                x1, y1, x2, y2 = [int(v) for v in face_box]
                # Add padding
                pad = 20
                h, w = frame.shape[:2]
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(w, x2 + pad)
                y2 = min(h, y2 + pad)
                region = frame[y1:y2, x1:x2]
            else:
                region = frame

            if region.size == 0:
                return None

            faces = self._app.get(region)
            if not faces:
                return None

            # Pick face with highest detection score
            best = max(faces, key=lambda f: f.det_score)
            emb = best.embedding
            if emb is None:
                return None

            # L2 normalize
            norm = np.linalg.norm(emb)
            if norm < 1e-6:
                return None
            return (emb / norm).astype(np.float32)

        except Exception as e:
            return None

    def get_embedding_from_crop(self, face_crop: np.ndarray) -> np.ndarray:
        """Get embedding directly from a face crop (BGR numpy array)."""
        self._ensure_loaded()
        try:
            if face_crop is None or face_crop.size == 0:
                return None
            # Ensure minimum size
            if face_crop.shape[0] < 40 or face_crop.shape[1] < 40:
                face_crop = cv2.resize(face_crop, (112, 112))
            faces = self._app.get(face_crop)
            if not faces:
                return None
            best = max(faces, key=lambda f: f.det_score)
            emb = best.embedding
            if emb is None:
                return None
            norm = np.linalg.norm(emb)
            if norm < 1e-6:
                return None
            return (emb / norm).astype(np.float32)
        except Exception:
            return None


class BodyEmbedder:
    """
    ResNet50-based body embedding (penultimate layer = 2048-dim).
    Input: body crop (BGR numpy array)
    Output: 2048-dim L2-normalized numpy float32 vector, or None
    """

    def __init__(self):
        self._model = None
        self._transform = None
        self._torch = None

    def _ensure_loaded(self):
        if self._model is None:
            self._model, self._transform, self._torch = _get_body_model()

    def get_embedding(self, body_crop: np.ndarray) -> np.ndarray:
        """
        Extract ResNet50 body embedding from a body crop.
        Returns 2048-dim L2-normalized float32 vector or None.
        """
        self._ensure_loaded()
        try:
            if body_crop is None or body_crop.size == 0:
                return None

            h, w = body_crop.shape[:2]
            if h < 32 or w < 16:
                return None

            # BGR → RGB for torchvision
            rgb = cv2.cvtColor(body_crop, cv2.COLOR_BGR2RGB)
            tensor = self._transform(rgb).unsqueeze(0)

            import torch
            device = next(self._model.parameters()).device
            tensor = tensor.to(device)

            with torch.no_grad():
                feat = self._model(tensor)         # (1, 2048, 1, 1)
                feat = feat.squeeze()              # (2048,)
                feat = feat.cpu().numpy().astype(np.float32)

            norm = np.linalg.norm(feat)
            if norm < 1e-6:
                return None
            return feat / norm

        except Exception as e:
            return None
