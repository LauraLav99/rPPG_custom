from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2, mediapipe as mp, numpy as np


def build_face_detector(model_path):
    opts = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        output_face_blendshapes=False, output_facial_transformation_matrixes=False, num_faces=1)
    return vision.FaceLandmarker.create_from_options(opts)

# ROI geometry
# ---------------------------------------------------------------------------
def compute_rois(landmarks, active, shape, ROI_POLYGONS,shrink=0.9):
    """Poligoni [(nome, array Nx2 int32)] delle ROI attive, per i landmark di questo frame."""
    h, w = shape[:2]
    result = []
    for name in active:
        idx = ROI_POLYGONS[name]
        if max(idx) >= len(landmarks):
            continue
        pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in idx], dtype=np.float32)
        hull = cv2.convexHull(pts).reshape(-1, 2)
        c = hull.mean(axis=0)
        poly = np.round(c + (hull - c) * shrink).astype(np.int32)
        poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
        poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
        result.append((name, poly))
    return result

def draw_rois(frame, rois, detected=True):
    """Anteprima: poligoni semitrasparenti + contorno."""
    overlay = frame.copy()
    for name, poly in rois:
        cv2.fillPoly(overlay, [poly.reshape(-1, 1, 2)], ROI_COLORS.get(name, (255, 255, 0)))
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, dst=frame)
    for name, poly in rois:
        col = ROI_COLORS.get(name, (255, 255, 0))
        cv2.polylines(frame, [poly.reshape(-1, 1, 2)], True, col, 2)
        x, y = poly.min(axis=0)
        cv2.putText(frame, name, (int(x), max(int(y) - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    if not detected:
        cv2.putText(frame, "VOLTO NON RILEVATO (uso ultime ROI)", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
