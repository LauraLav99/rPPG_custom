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
import mediapipe as mp
import cv2
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Configurazione del FaceLandmarker (Sostituisce mp.solutions.face_mesh)
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
face_mesh_detector = vision.FaceLandmarker.create_from_options(options)
# Mappatura degli indici dei landmark di MediaPipe per ogni zona
# Ogni lista contiene gli indici che delimitano i contorni di quella specifica area della pelle

# Mappatura degli indici dei landmark di MediaPipe per ogni zona
# Ogni lista contiene gli indici che delimitano i contorni di quella specifica area della pelle
# ROI_LANDMARKS = {
#     "forehead": [10, 9, 151, 67, 103, 54, 21, 68, 69, 108, 109, 297, 332, 284, 251, 298, 299, 337, 338],
#     "cheek_l":  [116, 123, 147, 213, 192, 214, 135, 136, 138, 215, 187, 50, 227],
#     "cheek_r":  [346, 347, 348, 329, 371, 423, 425, 426, 427, 411, 352, 280, 447]
# }

ROI_LANDMARKS = {
    "forehead": [ 67, 109, 10, 338],
    "cheek_l":  [116, 123, 147, 50],
    "cheek_r":  [345,346, 347, 348]
}



ROI_PRESETS = {
    "forehead":        ["forehead"],
    "cheeks":          ["cheek_l", "cheek_r"],
    "forehead+cheeks": ["forehead", "cheek_l", "cheek_r"],
}

# I colori rimangono invariati (BGR)
ROI_COLORS = {
    "forehead": (0,   0,   255),   # rosso
    "cheek_l":  (0,   255,  0 ),   # verde
    "cheek_r":  (255, 0,    0 ),   # blu
}



# ---------------------------------------------------------------------------
# Funzioni ROI
# ---------------------------------------------------------------------------

def get_face_box_and_landmarks(frame):
    """
    Rileva il volto usando la nuova API MediaPipe Tasks FaceLandmarker.
    Ritorna (face_box, face_landmarks) o (None, None).
    """
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = face_mesh_detector.detect(mp_image)
    
    # IMPORTANTE: MediaPipe Tasks restituisce una lista di volti.
    # Controlliamo se la lista contiene almeno un volto rilevato.
    if not results.face_landmarks or len(results.face_landmarks) == 0:
        return None, None
        
    # Estraiamo esplicitamente la lista di punti del PRIMO volto [0]
    face_landmarks = results.face_landmarks[0]
    
    # Calcolo delle coordinate assolute del box bianco
    x_coords = [lm.x for lm in face_landmarks]
    y_coords = [lm.y for lm in face_landmarks]
    
    fx = max(0, int(min(x_coords) * w))
    fy = max(0, int(min(y_coords) * h))
    fw = min(w, int(max(x_coords) * w)) - fx
    fh = min(h, int(max(y_coords) * h)) - fy
    
    return (fx, fy, fw, fh), face_landmarks


# def compute_rois_from_landmarks(face_landmarks, active_roi_names, frame_shape):
#     """
#     Calcola le ROI adattando correttamente i decimali di MediaPipe (0-1)
#     ai pixel reali dell'immagine (w, h).
#     """
#     # OpenCV restituisce (altezza, larghezza, canali) -> h = altezza, w = larghezza
#     h, w, _ = frame_shape
#     result = []
    
#     for name in active_roi_names:
#         if name not in ROI_LANDMARKS:
#             continue
            
#         x_pixels = []
#         y_pixels = []
        
#         for idx in ROI_LANDMARKS[name]:
#             if idx >= len(face_landmarks):
#                 continue
#             lm = face_landmarks[idx]
            
#             # --- CORREZIONE ADATTAMENTO DECIMALI ---
#             # Moltiplichiamo IMMEDIATAMENTE i decimali (0.0 - 1.0) per le dimensioni reali del frame.
#             # lm.x va SEMPRE con la larghezza (w), lm.y va SEMPRE con l'altezza (h).
#             px = int(lm.x * w)
#             py = int(lm.y * h)
            
#             x_pixels.append(px)
#             y_pixels.append(py)
            
#         if not x_pixels or not y_pixels:
#             continue
            
#         # Ora che abbiamo i pixel reali e corretti, calcoliamo i confini della ROI
#         rx = min(x_pixels)
#         ry = min(y_pixels)
#         rw = max(x_pixels) - rx
#         rh = max(y_pixels) - ry
        
#         # Se i punti sono troppo vicini o se la ROI è un punto singolo,
#         # impostiamo una dimensione fissa in pixel proporzionata al viso
#         if rw <= 2 or rh <= 2:
#             rw = 30
#             rh = 30
#             # Ricentriamo il quadratino rispetto al punto
#             rx = rx - (rw // 2)
#             ry = ry - (rh // 2)
            
#         # Blocco di sicurezza antiricorsivo: impedisce fisicamente l'allungamento
#         # Se l'altezza calcolata supera il 15% dell'altezza del volto o è palesemente enorme, la limitiamo
#         if rh > (h * 0.15) or rh > 80:
#             rh = 30  # Forza una dimensione quadrata stabile per l'rPPG
#         if rw > (w * 0.15) or rw > 80:
#             rw = 30

#         # Clipping rigido finale all'interno dei bordi del video per evitare crash
#         rx = max(0, min(rx, w - 1))
#         ry = max(0, min(ry, h - 1))
#         rw = max(5, min(rw, w - rx))
#         rh = max(5, min(rh, h - ry))
        
#         result.append((name, rx, ry, rw, rh))
        
#     return result

def compute_rois_from_landmarks(face_landmarks, active_roi_names, frame_shape):
    """
    Calcola le ROI adattando i decimali di MediaPipe.
    Versione ampliata: ingrandisce i rettangoli per coprire più pelle.
    """
    h, w, _ = frame_shape
    result = []
    
    for name in active_roi_names:
        if name not in ROI_LANDMARKS:
            continue
            
        x_pixels = []
        y_pixels = []
        
        for idx in ROI_LANDMARKS[name]:
            if idx >= len(face_landmarks):
                continue
            lm = face_landmarks[idx]
            
            # Conversione corretta decimale -> pixel
            px = int(lm.x * w)
            py = int(lm.y * h)
            
            x_pixels.append(px)
            y_pixels.append(py)
            
        if not x_pixels or not y_pixels:
            continue
            
        # Calcolo dei confini naturali basati sui 4 punti di test
        rx = min(x_pixels)
        ry = min(y_pixels)
        rw = max(x_pixels) - rx
        rh = max(y_pixels) - ry
        
        # --- MODIFICA PER AMPLIARE LE FINESTRE ---
        # Se la ROI è troppo piccola o se i 4 punti sono molto vicini,
        # forziamo una dimensione minima più generosa (es. 45x45 o 50x50 pixel)
        MIN_DIM_W = 50  # Modifica questo valore per allargare (larghezza)
        MIN_DIM_H = 45  # Modifica questo valore per alzare (altezza)
        
        if rw < MIN_DIM_W or rh < MIN_DIM_H:
            # Calcoliamo il centro per espandere il quadrato simmetricamente
            cx = rx + (rw // 2)
            cy = ry + (rh // 2)
            rw = MIN_DIM_W
            rh = MIN_DIM_H
            rx = cx - (rw // 2)
            ry = cy - (rh // 2)
            
        # Rimosso il vecchio padding che stringeva i rettangoli.
        # Lasciamo invece un tetto massimo di sicurezza per non farli diventare enormi
        MAX_DIM_W = int(w * 0.20) # Massimo 20% della larghezza video
        MAX_DIM_H = int(h * 0.20) # Massimo 20% della altezza video
        
        rw = min(rw, MAX_DIM_W)
        rh = min(rh, MAX_DIM_H)

        # Clipping rigido finale per non uscire dai bordi del video
        rx = max(0, min(rx, w - 1))
        ry = max(0, min(ry, h - 1))
        rw = max(10, min(rw, w - rx))
        rh = max(10, min(rh, h - ry))
        
        result.append((name, rx, ry, rw, rh))
        
    return result



def extract_value_face(frame, rois):
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
    return float(np.mean(np.concatenate(pixels)))



def extract_value_finger(frame):
    """Media del canale rosso sull'intero frame."""
    return float(np.mean(frame[:, :, 2].astype(np.float64)))

# def draw_rois(frame, face_box, rois):
#     """Disegna il box del volto e i rettangoli delle ROI sull'anteprima."""
#     fx, fy, fw, fh = face_box
#     cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (200, 200, 200), 1)
#     for (name, rx, ry, rw, rh) in rois:
#         color = ROI_COLORS.get(name, (255, 255, 0))
#         cv2.rectangle(frame, (rx, ry), (rx + rw, rx + rh), color, 2)
#         cv2.putText(frame, name, (rx, max(ry - 4, 10)),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

# def draw_rois(frame, face_box, rois):
#     """
#     Versione di Debug forzata: impedisce visivamente ai rettangoli
#     di allungarsi sullo schermo.
#     """
#     fx, fy, fw, fh = face_box
#     # Disegna il box bianco del viso
#     cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (200, 200, 200), 1)
    
#     for (name, rx, ry, rw, rh) in rois:
#         color = ROI_COLORS.get(name, (255, 255, 0))
        
#         # --- FORZATURA VISIVA DI DEBUG ---
#         # Se il rettangolo arriva qui lungo e stretto, forziamo le dimensioni
#         # a un quadrato perfetto di 30x30 pixel direttamente prima del disegno.
#         visual_w = 30
#         visual_h = 30
        
#         # Disegnamo il rettangolo usando la coordinata di partenza (rx, ry)
#         # ma imponendo una larghezza e altezza fisse e piccolissime
#         cv2.rectangle(frame, (rx, ry), (rx + visual_w, ry + visual_h), color, 2)
        
#         # Scriviamo il nome della ROI
#         cv2.putText(frame, name, (rx, max(ry - 4, 10)),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
def draw_rois(frame, face_box, rois):
    """
    Funzione di disegno corretta per Opzione 2.
    Disegna il box del volto, i cerchietti sugli occhi e i rettangoli delle ROI.
    """
    h, w, _ = frame.shape
    fx, fy, fw, fh = face_box
    
    # 1. Disegna il box bianco del volto
    cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (200, 200, 200), 1)
    
    # Accesso sicuro alla variabile globale salvata nel ciclo principale
    global last_landmarks
    
    # 2. DISEGNO DEI PUNTI DEGLI OCCHI (Per verificare la distorsione di MediaPipe)
    if 'last_landmarks' in globals() and last_landmarks is not None:
        occhio_destro = 33
        occhio_sinistro = 263
        
        for idx in [occhio_destro, occhio_sinistro]:
            if idx < len(last_landmarks):
                # Nella nuova API (Opzione 2), last_landmarks è una lista diretta di oggetti
                lm = last_landmarks[idx]
                cx = int(lm.x * w)
                cy = int(lm.y * h)
                # Disegna un cerchio pieno verde acceso
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    # 3. DISEGNO DEI RETTANGOLI DELLE ROI (Fronte e Guance)
    for (name, rx, ry, rw, rh) in rois:
        color = ROI_COLORS.get(name, (255, 255, 0))
        cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), color, 2)
        cv2.putText(frame, name, (rx, max(ry - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


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

    #face_cascade = None
    if args.mode == "face":
        print(f"ROI attive: {', '.join(active_roi_names)}")
        print("Legenda anteprima -> forehead: rosso | cheek_l: verde | cheek_r: blu")

    if not using_file:
        print("Premi 'q' per interrompere prima della durata impostata.")

    timestamps, values = [], []
    last_face_box = None
    last_rois = None
    last_landmarks = None
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
                face_box,landmarks = get_face_box_and_landmarks(frame)
                if landmarks is not None:
                    last_face_box = face_box
                    last_landmarks = landmarks
                    last_rois = compute_rois_from_landmarks(landmarks, active_roi_names, frame.shape)
                elif last_landmarks is not None:
                    # Volto perso temporaneamente: riusa gli ultimi landmark noti sul frame corrente
                    last_rois = compute_rois_from_landmarks(last_landmarks, active_roi_names, frame.shape)

                if last_rois:
                    value = extract_value_face(frame, last_rois)
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
        # Rilascia le risorse di MediaPipe Tasks per evitare memory leak
        if 'face_mesh_detector' in locals() or 'face_mesh_detector' in globals():
            face_mesh_detector.close()
            
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
