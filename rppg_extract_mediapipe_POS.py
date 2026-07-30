#!/usr/bin/env python3
"""
rPPG Extractor con MediaPipe + POS_WANG
----------------------------------------
Usa MediaPipe FaceLandmarker per tracciare le ROI sul volto,
estrae le medie RGB per ogni frame e applica l'algoritmo POS_WANG
(Wang et al. 2017) per ricavare il segnale PPG.

Il video non viene mai scritto su disco: per ogni frame vengono
tenute solo 3 medie scalari (R, G, B) per ROI, poi il frame è scartato.

Uso:
    python rppg_extract_mediapipe.py --mode face --duration 30 --plot
    python rppg_extract_mediapipe.py --mode face --roi forehead+cheeks --input video.mp4 --plot
    python rppg_extract_mediapipe.py --mode finger --duration 20

Richiede:
    pip install opencv-python mediapipe numpy scipy matplotlib
    # Scarica il modello da:
    # https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
"""

import argparse
import json
import os
import time
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.signal import butter, filtfilt
from scipy import io as scio
from scipy import linalg
from scipy import signal
from scipy import sparse
from skimage.util import img_as_float
from sklearn.metrics import mean_squared_error

# ---------------------------------------------------------------------------
# ROI: indici dei landmark MediaPipe (478 punti, modello face_landmarker)
# ---------------------------------------------------------------------------
ROI_LANDMARKS = {
    "forehead": [67, 109, 10, 338],
    "cheek_l":  [116, 123, 147, 50],
    "cheek_r":  [345, 346, 347, 348],
}

ROI_PRESETS = {
    "forehead":        ["forehead"],
    "cheeks":          ["cheek_l", "cheek_r"],
    "forehead+cheeks": ["forehead", "cheek_l", "cheek_r"],
}

ROI_COLORS = {          # BGR per OpenCV
    "forehead": (0,   0,   255),
    "cheek_l":  (0,   255,  0 ),
    "cheek_r":  (255,  0,   0 ),
}

MIN_DIM_PX = 40   # dimensione minima ROI in pixel
MAX_DIM_REL = 0.20  # dimensione massima ROI come frazione del frame


# ---------------------------------------------------------------------------
# MediaPipe: inizializzazione
# ---------------------------------------------------------------------------
def build_face_detector(model_path: str):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


# ---------------------------------------------------------------------------
# ROI geometry
# ---------------------------------------------------------------------------
def compute_rois(face_landmarks, active_roi_names, frame_shape):
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


def extract_rgb_means(frame, rois):
    """
    Ritorna la media RGB (in ordine R, G, B) su tutti i pixel
    di tutte le ROI attive, pesata per area.
    Restituisce np.array([R, G, B]) o None se nessuna ROI valida.
    """
    r_vals, g_vals, b_vals = [], [], []
    for (_, rx, ry, rw, rh) in rois:
        region = frame[ry : ry + rh, rx : rx + rw]   # BGR
        if region.size == 0:
            continue
        b_vals.append(region[:, :, 0].ravel())
        g_vals.append(region[:, :, 1].ravel())
        r_vals.append(region[:, :, 2].ravel())

    if not r_vals:
        return None

    R = float(np.mean(np.concatenate(r_vals)))
    G = float(np.mean(np.concatenate(g_vals)))
    B = float(np.mean(np.concatenate(b_vals)))
    return np.array([R, G, B], dtype=np.float64)


def extract_rgb_finger(frame):
    """Media RGB sull'intero frame (dito sull'obiettivo)."""
    b = float(np.mean(frame[:, :, 0]))
    g = float(np.mean(frame[:, :, 1]))
    r = float(np.mean(frame[:, :, 2]))
    return np.array([r, g, b], dtype=np.float64)

def detrend(input_signal, lambda_value):
    signal_length = input_signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = sparse.spdiags(diags_data, diags_index,
                (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    return filtered_signal
# ---------------------------------------------------------------------------
# POS_WANG  (Wang et al. 2017, adattato da ubicomplab/rPPG-Toolbox)
#
# Input atteso:
#   rgb  — np.ndarray shape (N, 3), media RGB per frame già calcolata
#            (R=col 0, G=col 1, B=col 2)
#   fs   — frequenza di campionamento in Hz
#
# La funzione originale nel toolbox riceve i frame grezzi (N, H, W, 3)
# e ne calcola la media spaziale internamente. Poiché noi la calcoliamo
# frame per frame durante la lettura (per non tenere i frame in RAM),
# passiamo direttamente le medie — il risultato è identico.
# ---------------------------------------------------------------------------
def POS_WANG(rgb: np.ndarray, fs: float) -> np.ndarray:
    """
    Algoritmo POS per l'estrazione del segnale rPPG.

    Parametri
    ---------
    rgb : (N, 3) float array  — medie [R, G, B] per ogni frame
    fs  : float               — frequenza di campionamento (Hz)

    Ritorna
    -------
    H : (N,) float array — segnale PPG grezzo
    """
    N = rgb.shape[0]
    H = np.zeros(N, dtype=np.float64)
    l = max(int(np.ceil(1.6 * fs)), 2)   # finestra ~1.6 s

    # Proiezione POS (matrice 2×3 da Wang et al.)
    P = np.array([[0., 1., -1.],
                  [-2., 1.,  1.]])

    for n in range(N):
        m = max(0, n - l + 1)
        window = rgb[m : n + 1, :]            # (l', 3)

        # Normalizzazione temporale: divide per la media di finestra
        mu = np.mean(window, axis=0)          # (3,)
        mu[mu == 0] = 1e-6                    # evita divisione per zero
        Cn = window / mu                      # (l', 3)  — shape OK per matmul

        S = (P @ Cn.T)                        # (2, l')

        std0 = np.std(S[0], ddof=1) if S.shape[1] > 1 else 0.0
        std1 = np.std(S[1], ddof=1) if S.shape[1] > 1 else 0.0
        alpha = std0 / std1 if std1 > 1e-9 else 1.0

        h = S[0] + alpha * S[1]              # (l',)
        H[m : n + 1] += h - np.mean(h)
    BVP=H
    BVP = detrend(np.asmatrix(BVP).H, 100)
    BVP = np.asarray(np.transpose(BVP))[0]
    b, a = signal.butter(1, [0.75 / fs * 2, 3 / fs * 2], btype='bandpass')
    BVP = signal.filtfilt(b, a, BVP.astype(np.double))
    return BVP


# ---------------------------------------------------------------------------
# Segnale: filtraggio e stima HR
# ---------------------------------------------------------------------------
def bandpass_filter(signal, fs, low=0.7, high=4.0, order=3):
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, min(high / nyq, 0.99)], btype="band")
    return filtfilt(b, a, signal)


def estimate_heart_rate(signal, fs):
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / fs)
    fft_vals = np.abs(np.fft.rfft(signal - np.mean(signal)))
    mask = (freqs >= 0.7) & (freqs <= 4.0)
    if not np.any(mask):
        return None
    return float(freqs[mask][np.argmax(fft_vals[mask])] * 60.0)


# ---------------------------------------------------------------------------
# Anteprima
# ---------------------------------------------------------------------------
def draw_rois(frame, face_box, rois, landmarks=None):
    fx, fy, fw, fh = face_box
    cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (200, 200, 200), 1)

    if landmarks is not None:
        h, w, _ = frame.shape
        for idx in [33, 263]:   # angoli interni occhi
            if idx < len(landmarks):
                lm = landmarks[idx]
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1)

    for (name, rx, ry, rw, rh) in rois:
        color = ROI_COLORS.get(name, (255, 255, 0))
        cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), color, 2)
        cv2.putText(frame, name, (rx, max(ry - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="rPPG Extractor — MediaPipe + POS_WANG")
    parser.add_argument("--mode", choices=["face", "finger"], required=True)
    parser.add_argument("--roi", choices=list(ROI_PRESETS.keys()), default="forehead",
                        help="ROI del volto (solo --mode face)")
    parser.add_argument("--input", type=str, default=None,
                        help="File video da analizzare. Se omesso usa la webcam.")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Durata in secondi (ignorato con --input)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Indice webcam (ignorato con --input)")
    parser.add_argument("--model", type=str, default="face_landmarker.task",
                        help="Percorso del modello MediaPipe FaceLandmarker")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--folder", type=str, default="OUTPUT",
                        help="Cartella di salvataggio (default: OUTPUT)")
    args = parser.parse_args()
    os.makedirs(args.folder, exist_ok=True)
    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = os.path.splitext(os.path.basename(args.input))[0] if args.input else "webcam"
        args.output = f"ppg_{args.mode}_{fname}_{ts}"

    active_roi_names = ROI_PRESETS[args.roi]

    # ---- Apertura sorgente ----
    using_file = args.input is not None
    if using_file:
        if not os.path.isfile(args.input):
            raise FileNotFoundError(f"File video non trovato: {args.input}")
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            raise RuntimeError(f"Impossibile aprire: {args.input}")
    else:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            raise RuntimeError(f"Impossibile aprire la webcam {args.camera}")

    # ---- fs dai metadati del video (disponibile subito dopo VideoCapture) ----
    # Per i file è esatta; per la webcam è la fps dichiarata dal driver
    # (potrebbe differire leggermente dalla fps reale, ma è la miglior stima a priori).
    fs_from_meta = cap.get(cv2.CAP_PROP_FPS)
    if fs_from_meta <= 0:
        fs_from_meta = 30.0   # fallback
        print(f"[WARN] FPS non leggibili dai metadati, assumo {fs_from_meta} fps")
    else:
        print(f"FPS da metadati: {fs_from_meta:.3f}")

    if using_file:
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Sorgente: file '{args.input}' | Frame totali: {n_frames} | "
              f"Durata attesa: {n_frames / fs_from_meta:.1f}s")
    else:
        print(f"Sorgente: webcam {args.camera} | Durata: {args.duration}s")

    # ---- Inizializza MediaPipe solo per --mode face ----
    face_detector = None
    if args.mode == "face":
        if not os.path.isfile(args.model):
            raise FileNotFoundError(
                f"Modello MediaPipe non trovato: '{args.model}'\n"
                "Scaricalo da:\n"
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                "face_landmarker/float16/1/face_landmarker.task"
            )
        face_detector = build_face_detector(args.model)
        print(f"ROI attive: {', '.join(active_roi_names)}")
        print("Legenda: fronte=rosso | guancia sx=verde | guancia dx=blu")

    if not using_file:
        print("Premi 'q' per interrompere.")

    # ---- Raccolta dati ----
    # Accumuliamo solo 3 scalari per frame (medie R, G, B),
    # NON i frame stessi — memoria O(N) non O(N*H*W*3).
    rgb_buffer = []       # lista di np.array([R,G,B]) — shape finale (N, 3)
    timestamps = []
    last_face_box = None
    last_rois = None
    last_landmarks = None
    start_time = time.time()
    frame_index = 0
# rppgtoolbox calcola pow_wang su chaunk di n frmae. POS fa una normaliyyayione temporale del segnale,
# il risultato piu' smmoth potrebbe essere dovuto a questo. Il segnale di mediapipe non è piu' smooth è piè frtatagliato
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if using_file:
                now = frame_index / fs_from_meta
            else:
                now = time.time() - start_time
                if now > args.duration:
                    break

            frame_index += 1
            rgb_val = None

            if args.mode == "face":
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = face_detector.detect(mp_image)

                if result.face_landmarks:
                    landmarks = result.face_landmarks[0]
                    last_landmarks = landmarks

                    # Calcolo face_box dal bounding box dei landmark
                    h_f, w_f, _ = frame.shape
                    xs = [int(lm.x * w_f) for lm in landmarks]
                    ys = [int(lm.y * h_f) for lm in landmarks]
                    last_face_box = (min(xs), min(ys),
                                     max(xs) - min(xs), max(ys) - min(ys))
                    last_rois = compute_rois(landmarks, active_roi_names, frame.shape)

                elif last_landmarks is not None:
                    # Volto non rilevato: usa gli ultimi landmark noti
                    last_rois = compute_rois(last_landmarks, active_roi_names, frame.shape)

                if last_rois:
                    rgb_val = extract_rgb_means(frame, last_rois)

            else:
                rgb_val = extract_rgb_finger(frame)

            if rgb_val is not None:
                rgb_buffer.append(rgb_val)
                timestamps.append(now)

            if not args.no_preview:
                if args.mode == "face" and last_face_box and last_rois:
                    draw_rois(frame, last_face_box, last_rois, last_landmarks)
                cv2.putText(frame, f"t={now:.1f}s  campioni={len(rgb_buffer)}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.imshow("rPPG - anteprima (NON salvata)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            # 'frame' esce di scope qui e viene deallocato dal GC

    finally:
        if face_detector:
            face_detector.close()
        cap.release()
        cv2.destroyAllWindows()

    if len(rgb_buffer) < 30:
        print("Troppi pochi campioni: segnale non salvato.")
        return

    # ---- Post-processing ----
    rgb_array = np.stack(rgb_buffer, axis=0)   # (N, 3)
    timestamps = np.array(timestamps)
    N = len(rgb_buffer)
    duration_real = timestamps[-1] - timestamps[0]

    # fs finale: per file usiamo quella dei metadati (più precisa);
    # per webcam confrontiamo con quella misurata e prendiamo la misurata
    # (più fedele alla realtà del frame-rate effettivo durante la sessione).
    fs_measured = (N - 1) / duration_real if duration_real > 0 else fs_from_meta
    if using_file:
        fs = fs_from_meta          # i frame del file hanno timing esatto
        print(f"fs usata: {fs:.3f} Hz (dai metadati del file)")
    else:
        fs = fs_measured           # la webcam potrebbe non rispettare il dichiarato
        print(f"fs usata: {fs:.3f} Hz (misurata; dichiarata: {fs_from_meta:.3f} Hz)")

    # POS_WANG — opera sull'intero buffer in una sola chiamata
    print("Esecuzione POS_WANG...")
    ppg_pos = POS_WANG(rgb_array, fs)          # (N,)

    # Filtraggio passa-banda
    try:
        ppg_filtered = bandpass_filter(ppg_pos, fs)
    except Exception as e:
        print(f"[WARN] Filtraggio non riuscito ({e}): uso segnale POS grezzo.")
        ppg_filtered = ppg_pos

    hr = estimate_heart_rate(ppg_filtered, fs)

    # Segnale verde grezzo (per confronto / debug)
    green_raw = rgb_array[:, 1]

    # ---- Salvataggio: SOLO il segnale, mai il video ----
    csv_path  = os.path.join(args.folder, args.output + ".csv")
    json_path = os.path.join(args.folder, args.output + ".json")

    with open(csv_path, "w") as f:
        f.write("time_s,green_raw,ppg_pos,ppg_filtered\n")
        for t, g, p, pf in zip(timestamps, green_raw, ppg_pos, ppg_filtered):
            f.write(f"{t:.4f},{g:.4f},{p:.6f},{pf:.6f}\n")

    meta = {
        "mode": args.mode,
        "algorithm": "POS_WANG",
        "roi": args.roi if args.mode == "face" else None,
        "active_rois": active_roi_names if args.mode == "face" else None,
        "sampling_rate_hz": float(fs),
        "sampling_rate_from_meta_hz": float(fs_from_meta),
        "duration_s": float(duration_real),
        "n_samples": N,
        "estimated_heart_rate_bpm": hr,
        "acquired_at": datetime.now().isoformat(),
    }
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSegnale PPG salvato in: {csv_path}")
    print(f"Metadati salvati in:    {json_path}")
    if hr is not None:
        print(f"Frequenza cardiaca stimata: {hr:.1f} bpm")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

            axes[0].plot(timestamps, green_raw, color="green", alpha=0.7)
            axes[0].set_ylabel("Verde grezzo")
            axes[0].set_title("Canale verde (confronto)")

            axes[1].plot(timestamps, ppg_pos, color="steelblue", alpha=0.7)
            axes[1].set_ylabel("POS grezzo")
            axes[1].set_title("POS_WANG (segnale grezzo)")

            axes[2].plot(timestamps, ppg_filtered, color="crimson", linewidth=1.5)
            axes[2].set_ylabel("POS filtrato")
            axes[2].set_xlabel("Tempo (s)")
            title = "POS_WANG filtrato (0.7–4 Hz)"
            if hr:
                title += f"   |   HR stimata: {hr:.1f} bpm"
            axes[2].set_title(title)

            plt.tight_layout()
            plot_path = args.output + ".png"
            plt.savefig(plot_path, dpi=150)
            print(f"Grafico salvato in:        {plot_path}")
        except ImportError:
            print("matplotlib non installato (pip install matplotlib).")


if __name__ == "__main__":
    main()
