#!/usr/bin/env python3
"""
rPPG Extractor - prototipo
---------------------------
Registra video dalla webcam, estrae in tempo reale il segnale PPG
(viso o dito sull'obiettivo) e salva SOLO il segnale estratto.
Il video non viene mai scritto su disco né accumulato in memoria:
ogni frame viene letto, usato per calcolare un valore scalare, e
scartato immediatamente.

Uso:
    python rppg_extractor.py --mode face --duration 30 --plot
    python rppg_extractor.py --mode face --roi forehead+cheeks --duration 30 --plot
    python rppg_extractor.py --mode face --input mio_video.mp4 --roi cheeks --plot
    python rppg_extractor.py --mode finger --duration 20

ROI disponibili per --mode face:
    forehead          solo fronte (default)
    cheeks            guancia sinistra + guancia destra
    forehead+cheeks   fronte + entrambe le guance

Output:
    ppg_<mode>_<timestamp>.csv   -> time_s, ppg_raw, ppg_filtered
    ppg_<mode>_<timestamp>.json  -> metadati (fs, durata, HR stimata, roi usate)
    ppg_<mode>_<timestamp>.png   -> grafico (solo con --plot)
"""

import argparse
import json
import os
import time
from datetime import datetime
import cv2
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from unsupervised_methods import utils
from unsupervised_methods.methods.POS_WANG import POS_WANG
#from unsupervised_methods.unsupervised_predictor import unsupervised_predict

FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# ---------------------------------------------------------------------------
# Definizione delle regioni del volto
# Coordinate relative al bounding box del volto rilevato:
#   (x_rel, y_rel, w_rel, h_rel) — frazioni di larghezza/altezza del box
#
#   Puoi modificare questi valori per spostare o ridimensionare le regioni:
#   - aumenta x_rel per spostare a destra, y_rel per spostare in basso
#   - aumenta w_rel/h_rel per allargare/allungare la regione
# ---------------------------------------------------------------------------
ROI_DEFINITIONS = {
    #              x_rel  y_rel  w_rel  h_rel
    "forehead": (0.25,  0.05,  0.50,  0.22),   # fronte centrale
    "cheek_l":  (0.05,  0.45,  0.25,  0.25),   # guancia sinistra
    "cheek_r":  (0.70,  0.45,  0.25,  0.25),   # guancia destra
}

# Combinazioni selezionabili con --roi
ROI_PRESETS = {
    "forehead":        ["forehead"],
    "cheeks":          ["cheek_l", "cheek_r"],
    "forehead+cheeks": ["forehead", "cheek_l", "cheek_r"],
}

# Colori anteprima per ogni regione (BGR)
ROI_COLORS = {
    "forehead": (0,   0,   255),   # rosso
    "cheek_l":  (0,   255,  0 ),   # verde
    "cheek_r":  (255, 0,    0 ),   # blu
}


# ---------------------------------------------------------------------------
# Funzioni ROI
# ---------------------------------------------------------------------------
def compute_rois(face_box, active_roi_names):
    """
    Calcola le coordinate pixel assolute per ogni ROI attiva.
    Ritorna lista di (nome, x, y, w, h).
    """
    fx, fy, fw, fh = face_box
    result = []
    for name in active_roi_names:
        xr, yr, wr, hr = ROI_DEFINITIONS[name]
        rx = fx + int(fw * xr)
        ry = fy + int(fh * yr)
        rw = int(fw * wr)
        rh = int(fh * hr)
        result.append((name, rx, ry, rw, rh))
    return result


def get_face_box(frame, face_cascade):
    """Rileva il volto più grande; ritorna (x, y, w, h) o None."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)
    )
    if len(faces) == 0:
        return None
    return tuple(max(faces, key=lambda f: f[2] * f[3]))


def extract_value_face(frame, rois,method="default"):
    """
    Media del canale verde su tutti i pixel di tutte le ROI attive,
    pesata per area (più pixel = più peso).
    """
    pixels = []
    for (_, rx, ry, rw, rh) in rois:
        region = frame[ry : ry + rh, rx : rx + rw]
        if region.size > 0:
            pixels.append(region[:, :, 1].astype(np.float64).ravel())
        if not pixels:
            return None
        if method == "default":
            return float(np.mean(np.concatenate(pixels)))
        elif method == "POS_WANG":
            return float(np.concatenate(POS_WANG(region[:, :, :3].reshape(-1, 3)).ravel()).mean())


def draw_rois(frame, face_box, rois):
    """Disegna il box del volto e i rettangoli delle ROI sull'anteprima."""
    fx, fy, fw, fh = face_box
    cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (200, 200, 200), 1)
    for (name, rx, ry, rw, rh) in rois:
        color = ROI_COLORS.get(name, (255, 255, 0))
        cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), color, 2)
        cv2.putText(frame, name, (rx, max(ry - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def extract_value_finger(frame):
    """Media del canale rosso sull'intero frame."""
    return float(np.mean(frame[:, :, 2].astype(np.float64)))


# ---------------------------------------------------------------------------
# Elaborazione del segnale
# ---------------------------------------------------------------------------
def bandpass_filter(signal, fs, low=0.7, high=4.0, order=3):
    """Filtro passa-banda 0.7–4 Hz (42–240 bpm)."""
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, min(high / nyq, 0.99)], btype="band")
    return filtfilt(b, a, signal)


def estimate_heart_rate(signal, fs):
    """Stima HR individuando il picco FFT dominante tra 0.7 e 4 Hz."""
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / fs)
    fft_vals = np.abs(np.fft.rfft(signal - np.mean(signal)))
    mask = (freqs >= 0.7) & (freqs <= 4.0)
    if not np.any(mask):
        return None
    return float(freqs[mask][np.argmax(fft_vals[mask])] * 60.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Estrattore PPG da webcam (prototipo rPPG)")
    parser.add_argument("--mode", choices=["face", "finger"], required=True,
                        help="face (viso) o finger (dito sull'obiettivo)")
    parser.add_argument("--roi", choices=list(ROI_PRESETS.keys()), default="forehead",
                        help="Regioni del volto da usare (solo con --mode face). "
                             "Valori: forehead | cheeks | forehead+cheeks  [default: forehead]")
    parser.add_argument("--input", type=str, default=None,
                        help="File video da analizzare (mp4, avi, mov...). Se omesso usa la webcam.")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Durata registrazione in secondi (ignorato con --input)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Indice webcam (ignorato con --input)")
    parser.add_argument("--output", type=str, default=None,
                        help="Nome base dei file di output (senza estensione)")
    parser.add_argument("--no-preview", action="store_true",
                        help="Disattiva la finestra di anteprima")
    parser.add_argument("--plot", action="store_true",
                        help="Salva un grafico PNG del segnale")
    args = parser.parse_args()

    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"ppg_{args.mode}_{ts}"

    active_roi_names = ROI_PRESETS[args.roi]

    # --- Apertura sorgente video ---
    using_file = args.input is not None
    if using_file:
        if not os.path.isfile(args.input):
            raise FileNotFoundError(f"File video non trovato: {args.input}")
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            raise RuntimeError(f"Impossibile aprire il file: {args.input}")
        file_fps = cap.get(cv2.CAP_PROP_FPS)
        n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print(f"Sorgente: file '{args.input}' | FPS: {file_fps:.2f} | Frame: {int(n_frames)}")
    else:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            raise RuntimeError(f"Impossibile aprire la webcam {args.camera}")
        file_fps = None
        print(f"Sorgente: webcam {args.camera} | Durata: {args.duration}s")

    face_cascade = None
    if args.mode == "face":
        face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        if face_cascade.empty():
            raise RuntimeError("Impossibile caricare il classificatore volto")
        print(f"ROI attive: {', '.join(active_roi_names)}")
        print("Legenda anteprima -> fronte: rosso | guancia sx: verde | guancia dx: blu")

    if not using_file:
        print("Premi 'q' per interrompere prima della durata impostata.")

    timestamps, values = [], []
    last_face_box = None
    last_rois = None
    start_time = time.time()
    frame_index = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if using_file:
                now = frame_index / file_fps if file_fps and file_fps > 0 else frame_index / 30.0
            else:
                now = time.time() - start_time
                if now > args.duration:
                    break

            frame_index += 1
            value = None

            if args.mode == "face":
                face_box = get_face_box(frame, face_cascade)
                if face_box is not None:
                    last_face_box = face_box
                    last_rois = compute_rois(face_box, active_roi_names)
                elif last_face_box is not None:
                    # Volto perso: riusa l'ultima posizione nota
                    last_rois = compute_rois(last_face_box, active_roi_names)

                if last_rois:
                    value = extract_value_face(frame, last_rois,"POS_WANG")
            else:
                value = extract_value_finger(frame)

            if value is not None:
                timestamps.append(now)
                values.append(value)

            if not args.no_preview:
                if args.mode == "face" and last_face_box and last_rois:
                    draw_rois(frame, last_face_box, last_rois)
                cv2.putText(frame, f"t={now:.1f}s  campioni={len(values)}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.imshow("rPPG - anteprima (NON salvata)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            # frame scartato qui — non viene mai salvato su disco

    finally:
        cap.release()
        cv2.destroyAllWindows()

    if len(values) < 10:
        print("Troppi pochi campioni: segnale non salvato.")
        return

    timestamps = np.array(timestamps)
    values = np.array(values)
    duration_real = timestamps[-1] - timestamps[0]
    fs = max(round((len(timestamps) - 1) / duration_real), 5)

    t_uniform = np.arange(0, duration_real, 1.0 / fs)
    raw_uniform = interp1d(timestamps - timestamps[0], values,
                           kind="linear", fill_value="extrapolate")(t_uniform)

    try:
        filtered = bandpass_filter(raw_uniform, fs)
    except Exception as e:
        print(f"Filtraggio non riuscito ({e}): uso segnale grezzo.")
        filtered = raw_uniform

    hr = estimate_heart_rate(filtered, fs)

    # --- Salvataggio: SOLO il segnale PPG ---
    csv_path = args.output + ".csv"
    json_path = args.output + ".json"

    with open(csv_path, "w") as f:
        f.write("time_s,ppg_raw,ppg_filtered\n")
        for t, r, flt in zip(t_uniform, raw_uniform, filtered):
            f.write(f"{t:.4f},{r:.4f},{flt:.4f}\n")

    meta = {
        "mode": args.mode,
        "roi": args.roi if args.mode == "face" else None,
        "active_rois": active_roi_names if args.mode == "face" else None,
        "sampling_rate_hz": fs,
        "duration_s": float(duration_real),
        "n_samples": int(len(t_uniform)),
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

            plt.figure(figsize=(10, 5))
            plt.plot(t_uniform, raw_uniform, label="PPG grezzo", alpha=0.4)
            plt.plot(t_uniform, filtered, label="PPG filtrato", linewidth=1.5)
            plt.xlabel("Tempo (s)")
            plt.ylabel("Intensità canale verde (u.a.)")
            roi_label = args.roi if args.mode == "face" else "finger"
            title = f"Segnale PPG — {args.mode} / {roi_label}"
            if hr:
                title += f"   |   HR stimata: {hr:.1f} bpm"
            plt.title(title)
            plt.legend()
            plt.tight_layout()
            plot_path = args.output + ".png"
            plt.savefig(plot_path, dpi=150)
            print(f"Grafico salvato in:        {plot_path}")
        except ImportError:
            print("matplotlib non installato (pip install matplotlib).")


if __name__ == "__main__":
    main()
