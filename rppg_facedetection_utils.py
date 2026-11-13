from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def build_face_detector(model_path: str):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)

# ROI geometry
# ---------------------------------------------------------------------------
def compute_rois(face_landmarks, active_roi_names, frame_shape,ROI_LANDMARKS,MIN_DIM_PX,MAX_DIM_REL):
    """
    Converte i landmark MediaPipe (coordinate normalizzate 0-1)
    in rettangoli pixel (name, rx, ry, rw, rh).
    """
    h, w, _ = frame_shape
    result = []
    for name in active_roi_names:
        if name not in ROI_LANDMARKS:
            continue
        xs, ys = [], []
        for idx in ROI_LANDMARKS[name]:
            if idx < len(face_landmarks):
                lm = face_landmarks[idx]
                xs.append(int(lm.x * w))
                ys.append(int(lm.y * h))
        if not xs:
            continue

        rx, ry = min(xs), min(ys)
        rw = max(xs) - rx
        rh = max(ys) - ry

        # dimensione minima garantita, centrata sul baricentro
        if rw < MIN_DIM_PX or rh < MIN_DIM_PX:
            cx, cy = rx + rw // 2, ry + rh // 2
            rw = max(rw, MIN_DIM_PX)
            rh = max(rh, MIN_DIM_PX)
            rx, ry = cx - rw // 2, cy - rh // 2

        # tetto massimo
        rw = min(rw, int(w * MAX_DIM_REL))
        rh = min(rh, int(h * MAX_DIM_REL))

        # clipping ai bordi del frame
        rx = max(0, min(rx, w - 1))
        ry = max(0, min(ry, h - 1))
        rw = max(10, min(rw, w - rx))
        rh = max(10, min(rh, h - ry))

        result.append((name, rx, ry, rw, rh))
    return result


