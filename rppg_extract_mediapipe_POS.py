#!/usr/bin/env python3
"""rPPG Extractor — MediaPipe + POS_WANG (vedi docstring completo nel file)"""

import argparse, json, os, shutil, subprocess, time
from datetime import datetime, timedelta, timezone
import cv2, mediapipe as mp, numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy import sparse
from scipy.signal import butter, filtfilt, windows

# ---------------------------------------------------------------------------
# ROI = poligoni costruiti, frame per frame, sui landmark FaceMesh (478 punti).
# Per ogni regione si prende l'involucro convesso (convex hull) dei landmark
# elencati e lo si rimpicciolisce verso il centro (--roi-shrink) per stare
# lontani da capelli e sfondo. La media RGB e' calcolata solo sui pixel dentro
# il poligono. Occhi, sopracciglia, naso e bocca restano fuori per costruzione
# (verificato: nessuno dei loro landmark cade dentro i poligoni).
# Per cambiare una regione basta modificare l'elenco di indici.
#   cheek_l = lato SINISTRO dell'immagine (guancia destra del soggetto)
#   cheek_r = la sua riflessa (indici ottenuti per simmetria dal modello canonico)
# ---------------------------------------------------------------------------
ROI_POLYGONS = {
    "forehead": [103, 67, 109, 10, 338, 297, 332,  68, 104, 69, 108, 151, 337, 299, 333, 298],
    "cheek_l":  [117, 229, 230, 120, 100, 142, 203, 206, 216, 207, 187, 147, 123, 116],
    "cheek_r":  [346, 449, 450, 349, 329, 371, 423, 426, 436, 427, 411, 376, 352, 345],
}
ROI_PRESETS = {
    "forehead":        ["forehead"],
    "cheeks":          ["cheek_l", "cheek_r"],
    "forehead+cheeks": ["forehead", "cheek_l", "cheek_r"],
}
ROI_COLORS = {"forehead": (0, 0, 255), "cheek_l": (0, 255, 0), "cheek_r": (255, 0, 0)}   # BGR
MIN_ROI_PIXELS = 100    # sotto questa area (px) il frame viene scartato

def build_face_detector(model_path):
    opts = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        output_face_blendshapes=False, output_facial_transformation_matrixes=False, num_faces=1)
    return vision.FaceLandmarker.create_from_options(opts)

def compute_rois(landmarks, active, shape, shrink=0.9):
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

def extract_rgb_means(frame, rois):
    """
    Media [R,G,B] sui pixel dentro l'UNIONE dei poligoni (un pixel in due ROI conta una volta).
    Le ROI piu' grandi pesano di piu' in automatico. None se l'area e' troppo piccola.
    """
    if not rois:
        return None
    allp = np.vstack([p for _, p in rois])
    x0, y0 = allp.min(axis=0)
    x1, y1 = allp.max(axis=0) + 1
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)       # maschera solo sul riquadro utile
    for _, p in rois:
        cv2.fillPoly(mask, [(p - [x0, y0]).reshape(-1, 1, 2)], 255)
    if cv2.countNonZero(mask) < MIN_ROI_PIXELS:
        return None
    b, g, r, _ = cv2.mean(frame[y0:y1, x0:x1], mask=mask)
    return np.array([r, g, b], dtype=np.float64)

def extract_rgb_finger(frame):
    return np.array([np.mean(frame[:,:,2]), np.mean(frame[:,:,1]), np.mean(frame[:,:,0])])

def detrend(sig, lam):
    n = sig.shape[0]
    H = np.identity(n)
    D = sparse.spdiags(np.array([np.ones(n), -2*np.ones(n), np.ones(n)]),
                       [0,1,2], n-2, n).toarray()
    return np.dot((H - np.linalg.inv(H + lam**2 * D.T @ D)), sig)

def POS_WANG(rgb, fs):
    """
    POS_WANG — Wang et al. 2017.

    Input : rgb (N,3) medie [R,G,B] per frame  |  fs: Hz
    Output: (BVP, warmup_n)
      - BVP[n]    <-> frame n  <-> timestamps[n]  (allineamento 1:1, nessun delay)
      - warmup_n  : primo indice affidabile (finestra POS completa da qui in poi)
    """
    N = rgb.shape[0]
    H = np.zeros(N)
    l = max(int(np.ceil(1.6 * fs)), 2)
    P = np.array([[0.,1.,-1.],[-2.,1.,1.]])

    for n in range(N):
        m   = max(0, n - l + 1)
        win = rgb[m:n+1]
        mu  = np.mean(win, axis=0); mu[mu==0] = 1e-6
        S   = P @ (win / mu).T
        s0  = np.std(S[0], ddof=1) if S.shape[1]>1 else 0.
        s1  = np.std(S[1], ddof=1) if S.shape[1]>1 else 0.
        h   = S[0] + (s0/s1 if s1>1e-9 else 1.) * S[1]
        H[m:n+1] += h - np.mean(h)

    BVP = detrend(H, 100)
    b, a = butter(1, [0.75/fs*2, 3/fs*2], btype='bandpass')
    BVP = filtfilt(b, a, BVP.astype(np.float64))
    return BVP, l   # l = numero campioni di warm-up

def get_recording_start(path):
    """
    Legge la data/ora ORIGINALE di registrazione dai metadati interni del file
    (non dal file system, che cambia a ogni copia). Richiede ffprobe (ffmpeg).
    Ordine di preferenza:
      1. com.apple.quicktime.creationdate  (iPhone: include il fuso orario locale)
      2. creation_time del container / dello stream video (di norma in UTC)
    Ritorna (datetime aware | None, nome_tag | None).
    """
    if shutil.which("ffprobe") is None:
        print("[WARN] ffprobe non trovato: installa ffmpeg per leggere la data di registrazione")
        return None, None
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_entries", "format_tags:stream_tags", path],
            capture_output=True, text=True, timeout=30).stdout
        info = json.loads(out)
    except Exception as e:
        print(f"[WARN] lettura metadati fallita: {e}")
        return None, None

    tag_sets = [info.get("format", {}).get("tags", {})]
    tag_sets += [st.get("tags", {}) for st in info.get("streams", [])]
    for key in ("com.apple.quicktime.creationdate", "creation_time"):
        for tags in tag_sets:
            tags = {k.lower(): v for k, v in tags.items()}
            if key in tags:
                raw = tags[key].strip().replace("Z", "+00:00")
                try:
                    dt = datetime.fromisoformat(raw)
                except ValueError:
                    try:
                        dt = datetime.strptime(raw, "%Y-%m-%dT%H:%M:%S%z")
                    except ValueError:
                        continue
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                # Alcuni encoder scrivono 1970-01-01 / 1904 = "data non impostata"
                if dt.year < 1990:
                    continue
                return dt, key
    return None, None

def CHROM_DEHAAN(rgb, fs, low=0.7, high=2.5, win_sec=1.6):
    """
    CHROM — De Haan & Jeanne 2013 (stessa struttura di CHROME_DEHAAN in rPPG-Toolbox).

    Input : rgb (N,3) medie [R,G,B] per frame  |  fs: Hz
    Output: (BVP, warmup_n, valid_n)
      - BVP ha lunghezza N; BVP[n] <-> frame n <-> timestamps[n]
      - warmup_n: i primi campioni sono attenuati dalla finestra di Hann (win/2)
      - valid_n : l'overlap-add copre solo i primi valid_n campioni; la coda e' 0
                  (il chiamante la mette a NaN)
    Passi per ogni finestra (50% overlap):
      1. normalizza ogni canale per la sua media nella finestra
      2. Xs = 3R - 2G ,  Ys = 1.5R + G - 1.5B
      3. passa-banda (Butterworth ordine 3, low-high Hz) su Xs e Ys
      4. alpha = std(Xf)/std(Yf) ;  S = Xf - alpha*Yf
      5. pesa con finestra di Hann e somma con la finestra precedente
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    N = rgb.shape[0]
    nyq = fs / 2.0
    high = min(high, 0.95 * nyq)
    if low >= high:
        raise ValueError(f"fs={fs:.1f} Hz troppo bassa per la banda {low}-{high} Hz")
    b, a = butter(3, [low / nyq, high / nyq], btype="bandpass")

    win_l = int(np.ceil(win_sec * fs))
    if win_l % 2:
        win_l += 1
    half = win_l // 2
    if win_l <= 3 * max(len(a), len(b)):
        raise ValueError(f"finestra di {win_l} campioni troppo corta per filtfilt (fs troppo bassa)")
    n_win = (N - half) // half
    if n_win < 1:
        raise ValueError("segnale troppo corto per CHROM (serve almeno una finestra da 1.6 s)")

    total = half * (n_win + 1)
    S = np.zeros(total)
    hann = windows.hann(win_l)

    ws = 0
    for _ in range(n_win):
        wm, we = ws + half, ws + win_l
        seg = rgb[ws:we]
        base = seg.mean(axis=0)
        base[base == 0] = 1e-6
        norm = seg / base

        Xs = 3.0 * norm[:, 0] - 2.0 * norm[:, 1]
        Ys = 1.5 * norm[:, 0] + norm[:, 1] - 1.5 * norm[:, 2]
        Xf = filtfilt(b, a, Xs)
        Yf = filtfilt(b, a, Ys)

        sy = np.std(Yf)
        alpha = np.std(Xf) / sy if sy > 1e-12 else 1.0
        swin = hann * (Xf - alpha * Yf)

        S[ws:wm] += swin[:half]     # seconda meta' della finestra precedente + prima di questa
        S[wm:we] = swin[half:]
        ws = wm

    BVP = np.zeros(N)
    BVP[:total] = S
    return BVP, half, total


ALGO_NAMES = {"pos": "POS_WANG", "chrom": "CHROM_DEHAAN", "green": "GREEN",
              "multi": "MULTI(POS+CHROM+GREEN)", "fused": "FUSED"}

def run_algo(name, rgb, fs):
    """Ritorna (ppg[N], warmup_n, valid_n) per l'algoritmo scelto."""
    if name == "chrom":
        return CHROM_DEHAAN(rgb, fs)
    if name == "green":
        return GREEN_SIGNAL(rgb, fs)
    ppg, warmup_n = POS_WANG(rgb, fs)
    return ppg, warmup_n, len(ppg)

def GREEN_SIGNAL(rgb, fs, low=0.7, high=2.5):
    """
    Baseline: solo canale verde, normalizzato e passa-banda (fase zero).
    Ritorna (sig, warmup_n, valid_n).
    """
    g = np.asarray(rgb, dtype=np.float64)[:, 1]
    m = g.mean()
    x = g / (m if m != 0 else 1e-6) - 1.0
    nyq = fs / 2.0
    high = min(high, 0.95 * nyq)
    if low >= high:
        raise ValueError(f"fs={fs:.1f} Hz troppo bassa per la banda {low}-{high} Hz")
    b, a = butter(3, [low / nyq, high / nyq], btype="bandpass")
    if len(x) <= 3 * max(len(a), len(b)):
        raise ValueError("segnale troppo corto per il filtro")
    return filtfilt(b, a, x), 0, len(x)


def _psd(x, nfft):
    """Spettro di potenza con finestra di Hann e zero-padding."""
    x = x - x.mean()
    return np.abs(np.fft.rfft(x * np.hanning(len(x)), nfft)) ** 2


def _snr_at(f, P, f0, band=(0.7, 4.0), half_bw=0.1):
    """
    SNR lineare (De Haan): potenza entro +-half_bw Hz da f0 e 2*f0
    diviso la potenza restante nella banda cardiaca.
    """
    inband = (f >= band[0]) & (f <= band[1])
    sig = (np.abs(f - f0) <= half_bw) | (np.abs(f - 2 * f0) <= half_bw)
    ps = P[inband & sig].sum()
    pn = P[inband & ~sig].sum()
    return float(ps / max(pn, 1e-12))


def fuse_signals(signals, fs, win_sec=10.0, mode="weighted"):
    """
    Fonde piu' segnali rPPG (dict nome -> array(N), con NaN dove non validi).

    Per ogni finestra (win_sec, 50% overlap):
      1. z-score di ogni segnale (scale diverse, es. POS vs CHROM)
      2. frequenza cardiaca di CONSENSO = picco della somma degli spettri
         normalizzati; l'SNR di ciascun algoritmo e' misurato su quella
         frequenza, cosi' un picco pulito ma altrove (artefatto) pesa poco
      3. allineamento di polarita' rispetto al segnale con SNR massimo
         (POS/CHROM possono avere segno opposto)
      4. mode="weighted": media pesata per SNR | mode="best": solo il migliore
      5. continuita' di segno con le finestre gia' accumulate, poi overlap-add
         normalizzato (nessun taglio di ampiezza ai bordi)
    Un algoritmo con NaN nella finestra viene escluso da quella finestra.
    La polarita' del segnale fuso e' quella della prima finestra.

    Ritorna (fused[N] con NaN dove nessun algoritmo e' valido, info dict).
    """
    names = list(signals)
    X = np.stack([np.asarray(signals[k], dtype=np.float64) for k in names])
    K, N = X.shape
    W = min(int(round(win_sec * fs)), N)
    if W < int(4 * fs):
        raise ValueError("segnale troppo corto per la fusione (servono almeno 4 s)")
    hop = max(W // 2, 1)
    starts = list(range(0, N - W + 1, hop))
    if starts[-1] != N - W:
        starts.append(N - W)

    nfft = max(1024, 1 << int(np.ceil(np.log2(4 * W))))
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    inband = (f >= 0.7) & (f <= 4.0)
    peakband = (f >= 0.7) & (f <= 3.0)
    win = np.hanning(W + 2)[1:-1]            # strettamente positiva

    num = np.zeros(N)
    den = np.zeros(N)
    snr_hist, w_hist = [], []

    for s in starts:
        seg = X[:, s:s + W]
        ok = np.all(np.isfinite(seg), axis=1)
        z = np.zeros_like(seg)
        for k in range(K):
            if not ok[k]:
                continue
            sd = seg[k].std()
            if sd < 1e-12:
                ok[k] = False
                continue
            z[k] = (seg[k] - seg[k].mean()) / sd

        if not ok.any():
            snr_hist.append(np.full(K, np.nan))
            w_hist.append(np.zeros(K))
            continue

        P = {k: _psd(z[k], nfft) for k in np.flatnonzero(ok)}
        total = sum(P[k] / max(P[k][inband].sum(), 1e-12) for k in P)
        f0 = f[peakband][np.argmax(total[peakband])]
        snr = np.zeros(K)
        for k in P:
            snr[k] = _snr_at(f, P[k], f0)

        ref = int(np.argmax(np.where(ok, snr, -1.0)))
        for k in np.flatnonzero(ok):
            if k != ref and np.dot(z[k], z[ref]) < 0:
                z[k] = -z[k]

        if mode == "best":
            w = np.zeros(K)
            w[ref] = 1.0
        else:
            w = np.where(ok, snr, 0.0)
            w = w / w.sum() if w.sum() > 1e-9 else ok / ok.sum()
        fused_w = (w[:, None] * z).sum(axis=0)

        covered = den[s:s + W] > 1e-12
        if covered.any():
            prev = num[s:s + W][covered] / den[s:s + W][covered]
            if np.dot(fused_w[covered], prev) < 0:
                fused_w = -fused_w

        num[s:s + W] += win * fused_w
        den[s:s + W] += win
        snr_hist.append(np.where(ok, snr, np.nan))
        w_hist.append(w)

    fused = np.full(N, np.nan)
    m = den > 1e-12
    fused[m] = num[m] / den[m]

    snr_arr = np.array(snr_hist)
    w_arr = np.array(w_hist)
    with np.errstate(all="ignore"):
        snr_db = 10.0 * np.log10(snr_arr)
    info = {
        "names": names,
        "n_windows": len(starts),
        "window_s": W / fs,
        "median_snr_db": {k: (float(np.nanmedian(snr_db[:, i])) if np.isfinite(snr_db[:, i]).any() else None)
                          for i, k in enumerate(names)},
        "mean_weight": {k: float(w_arr[:, i].mean()) for i, k in enumerate(names)},
    }
    return fused, info


def compare_signals(signals, fs, start=0):
    """
    Confronta piu' algoritmi sulla porzione comune valida (dopo `start`).
    Ritorna: frequenza di consenso, SNR (dB) di ciascuno a quella frequenza
    e |correlazione| a coppie (a lag zero, segno ignorato).
    """
    names = list(signals)
    X = np.stack([np.asarray(signals[k], dtype=np.float64) for k in names])[:, start:]
    X = X[:, np.all(np.isfinite(X), axis=0)]
    if X.shape[1] < int(4 * fs):
        raise ValueError("porzione valida comune troppo corta per il confronto")
    Z = (X - X.mean(axis=1, keepdims=True)) / np.maximum(X.std(axis=1, keepdims=True), 1e-12)

    nfft = max(1024, 1 << int(np.ceil(np.log2(4 * Z.shape[1]))))
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    inband = (f >= 0.7) & (f <= 4.0)
    peakband = (f >= 0.7) & (f <= 3.0)
    P = [_psd(z, nfft) for z in Z]
    total = sum(p / max(p[inband].sum(), 1e-12) for p in P)
    f0 = float(f[peakband][np.argmax(total[peakband])])

    C = np.abs(np.corrcoef(Z))
    return {
        "common_samples": int(X.shape[1]),
        "consensus_hr_bpm": f0 * 60.0,
        "snr_db_at_consensus": {k: float(10 * np.log10(max(_snr_at(f, P[i], f0), 1e-12)))
                                for i, k in enumerate(names)},
        "abs_correlation": {f"{names[i]}-{names[j]}": float(C[i, j])
                            for i in range(len(names)) for j in range(i + 1, len(names))},
    }


def estimate_hr(sig, fs):
    freqs = np.fft.rfftfreq(len(sig), 1./fs)
    fft   = np.abs(np.fft.rfft(sig - sig.mean()))
    mask  = (freqs>=0.7)&(freqs<=4.)
    return float(freqs[mask][np.argmax(fft[mask])]*60.) if np.any(mask) else None

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

def main():
    os.chdir(r'C:\Users\annad\PycharmProjects\rppg_custom_github\rPPG_custom')
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["face","finger"], required=True)
    p.add_argument("--roi", choices=list(ROI_PRESETS.keys()), default="forehead")
    p.add_argument("--roi-shrink", type=float, default=0.9,
                   help="Riduzione dei poligoni verso il centro (0.5-1.0, default 0.9): "
                        "piu' basso = ROI piu' piccole e lontane da capelli/sfondo")
    p.add_argument("--algo", choices=["pos", "chrom", "green", "all", "multi"], default="pos",
                   help="pos | chrom | green | all (tre algoritmi, tre output separati + confronto) | multi (tre algoritmi + fusione pesata per SNR in un unico output)")
    p.add_argument("--fusion", choices=["weighted", "best"], default="weighted",
                   help="Solo con --algo multi: media pesata per SNR o solo il migliore per finestra")
    p.add_argument("--fusion-win", type=float, default=10.0,
                   help="Solo con --algo multi: durata delle finestre di fusione in secondi")
    p.add_argument("--input", type=str, default=None)
    p.add_argument("--duration", type=float, default=30.)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--model", type=str, default="face_landmarker.task")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--folder", type=str, default="OUTPUT")
    p.add_argument("--no-preview", action="store_true")
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    os.makedirs(args.folder, exist_ok=True)
    if args.output is None:
        ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = os.path.splitext(os.path.basename(args.input))[0] if args.input else "webcam"
        args.output = f"ppg_{args.mode}_{fname}_{ts}"

    using_file = args.input is not None
    cap = cv2.VideoCapture(args.input if using_file else args.camera)
    if not cap.isOpened():
        raise RuntimeError("Impossibile aprire la sorgente video")

    rec_start, rec_tag = None, None
    fs_meta = cap.get(cv2.CAP_PROP_FPS) or 30.
    print(f"FPS da metadati: {fs_meta:.3f}")
    if using_file:
        print(f"Sorgente: '{args.input}' | {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frame")
        print("Timestamp: CAP_PROP_POS_MSEC (OpenCV, letto dopo ogni cap.read)")
        rec_start, rec_tag = get_recording_start(args.input)
        if rec_start:
            print(f"Data registrazione (da metadati '{rec_tag}'): {rec_start.isoformat()}")
        else:
            print("[WARN] Nessuna data di registrazione nei metadati del file")
    else:
        print(f"Sorgente: webcam {args.camera} | Durata: {args.duration}s")
        print("Timestamp: wall clock (time.time)")

    face_detector = None
    if args.mode == "face":
        face_detector = build_face_detector(args.model)
        print(f"ROI: {', '.join(ROI_PRESETS[args.roi])}")

    rgb_buf, ts_buf = [], []
    last_rois = None
    n_missed = 0      # frame senza volto rilevato (si riusano le ultime ROI)
    n_seen = 0
    t0 = time.time()
    if not using_file:
        rec_start, rec_tag = datetime.fromtimestamp(t0).astimezone(), 'wall_clock'

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            # ---- Timestamp del frame ------------------------------------------------
            # File  : CAP_PROP_POS_MSEC = timestamp codificato nel container video,
            #         aggiornato da OpenCV dopo ogni cap.read(). Gestisce VFR e drop.
            # Webcam: wall clock, CAP_PROP_POS_MSEC non è affidabile live.
            if using_file:
                now = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            else:
                now = time.time() - t0
                if now > args.duration: break

            rgb_val = None
            n_seen += 1
            detected = True
            if args.mode == "face":
                res = face_detector.detect(
                    mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                detected = bool(res.face_landmarks)
                if detected:
                    last_rois = compute_rois(res.face_landmarks[0], ROI_PRESETS[args.roi],
                                             frame.shape, args.roi_shrink)
                else:
                    n_missed += 1     # si riusano le ultime ROI note
                if last_rois:
                    rgb_val = extract_rgb_means(frame, last_rois)
            else:
                rgb_val = extract_rgb_finger(frame)

            if rgb_val is not None:
                rgb_buf.append(rgb_val); ts_buf.append(now)

            if not args.no_preview:
                if args.mode=="face" and last_rois:
                    draw_rois(frame, last_rois, detected)
                cv2.putText(frame,f"t={now:.1f}s n={len(ts_buf)}",
                            (10,25),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)
                cv2.imshow("rPPG preview (non salvata)", frame)
                if cv2.waitKey(1)&0xFF==ord('q'): break
    finally:
        if face_detector: face_detector.close()
        cap.release(); cv2.destroyAllWindows()

    if args.mode == "face" and n_missed:
        print(f"[WARN] Volto non rilevato in {n_missed}/{n_seen} frame ({100*n_missed/n_seen:.1f}%): "
              f"in quei frame sono state riusate le ultime ROI valide")
    if len(ts_buf) < 30:
        print("Troppi pochi campioni."); return

    rgb_arr = np.stack(rgb_buf)       # (N, 3)
    ts_arr  = np.array(ts_buf)        # (N,)
    N       = len(ts_arr)
    dur     = ts_arr[-1] - ts_arr[0]

    # fs: metadati per file (timing esatto), misurata per webcam (più fedele al reale)
    fs = fs_meta if using_file else (N-1)/dur
    print(f"fs usata: {fs:.3f} Hz")

    def hr_of_sig(sig, warmup_n):
        x = sig[warmup_n:]
        x = x[np.isfinite(x)]
        return estimate_hr(x, fs) if len(x) >= int(4 * fs) else None

    def write_outputs(out_name, signals, main_name, warmup_n, info=None, algo_label=None):
        """Scrive CSV + JSON (+ PNG con --plot) per un insieme di segnali. Ritorna una sintesi."""
        algo_label = algo_label or ALGO_NAMES[main_name]
        names = list(signals)
        finite_idx = np.flatnonzero(np.isfinite(signals[main_name]))
        valid_n = int(finite_idx[-1] + 1) if finite_idx.size else 0
        if valid_n < N:
            print(f"[{algo_label}] Nota: ultimi {N-valid_n} campioni senza finestra completa -> NaN nel CSV")

        warmup_mask = np.zeros(N, dtype=int)
        warmup_mask[:warmup_n] = 1
        wu_dur = float(ts_arr[min(warmup_n, N) - 1] - ts_arr[0]) if warmup_n > 0 else 0.0
        hr = hr_of_sig(signals[main_name], warmup_n)

        csv_path  = os.path.join(args.folder, out_name + ".csv")
        json_path = os.path.join(args.folder, out_name + ".json")

        with open(csv_path, "w") as f:
            f.write("time_s,datetime,green_raw," + ",".join(f"ppg_{k}" for k in names) + ",warmup\n")
            for i in range(N):
                abs_t = (rec_start + timedelta(seconds=float(ts_arr[i]))).isoformat(timespec="milliseconds") if rec_start else ""
                vals = ",".join(f"{signals[k][i]:.6f}" for k in names)
                f.write(f"{ts_arr[i]:.4f},{abs_t},{rgb_arr[i,1]:.4f},{vals},{warmup_mask[i]}\n")

        meta = {
            "mode": args.mode, "algorithm": algo_label, "valid_samples": int(valid_n),
            "roi": args.roi if args.mode == "face" else None,
            "active_rois": ROI_PRESETS[args.roi] if args.mode == "face" else None,
            "roi_shrink": args.roi_shrink if args.mode == "face" else None,
            "roi_landmark_indices": {k: ROI_POLYGONS[k] for k in ROI_PRESETS[args.roi]} if args.mode == "face" else None,
            "face_not_detected_frames": n_missed if args.mode == "face" else None,
            "frames_total": n_seen,
            "sampling_rate_hz": float(fs),
            "sampling_rate_meta_hz": float(fs_meta),
            "recording_start": rec_start.isoformat() if rec_start else None,
            "recording_start_source": rec_tag,
            "timestamp_source": "CAP_PROP_POS_MSEC" if using_file else "wall_clock",
            "duration_s": float(dur), "n_samples": N,
            "warmup_samples": int(warmup_n), "warmup_duration_s": wu_dur,
            "estimated_heart_rate_bpm": hr,
            "acquired_at": datetime.now().isoformat(),
        }
        if info:
            meta["fusion"] = {
                "mode": args.fusion, "window_s": info["window_s"], "n_windows": info["n_windows"],
                "per_algorithm": {k: {"hr_bpm": hr_of_sig(signals[k], warmup_n),
                                      "median_snr_db": info["median_snr_db"][k],
                                      "mean_weight": info["mean_weight"][k]} for k in info["names"]},
            }
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"\n[{algo_label}] " + (f"warm-up {warmup_n} campioni ({wu_dur:.2f}s, warmup=1 nel CSV)"
                                         if warmup_n > 0 else "nessun warm-up (filtro a fase zero)"))
        print(f"[{algo_label}] CSV : {csv_path}")
        print(f"[{algo_label}] JSON: {json_path}")
        if info:
            print(f"\nFusione ({args.fusion}, finestre da {info['window_s']:.1f}s, {info['n_windows']} finestre)")
            print(f"{'algoritmo':<14}{'SNR med [dB]':>14}{'peso medio':>12}{'HR [bpm]':>10}")
            for k in info["names"]:
                snr_k, h_k = info["median_snr_db"][k], hr_of_sig(signals[k], warmup_n)
                print(f"{ALGO_NAMES[k]:<14}{(f'{snr_k:.1f}' if snr_k is not None else '-'):>14}"
                      f"{info['mean_weight'][k]:>12.2f}{(f'{h_k:.1f}' if h_k else '-'):>10}")
        if hr: print(f"[{algo_label}] HR stimata: {hr:.1f} bpm  (escluso warm-up)")

        if args.plot:
            plot_stack(out_name + ".png", {k: signals[k] for k in names}, warmup_n,
                       title=algo_label + (f"  |  HR: {hr:.1f} bpm" if hr else ""))
        return {"hr": hr, "valid_n": valid_n}

    def plot_stack(png_name, sigs, warmup_n, title, labels=None):
        """Una riga per il verde grezzo + una per ogni segnale."""
        try:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib non installato.")
            return
        colors = {"pos": "steelblue", "chrom": "darkorange", "green": "seagreen", "fused": "crimson"}
        keys = list(sigs)
        fig, axes = plt.subplots(1 + len(keys), 1, figsize=(12, 2.2 * (1 + len(keys)) + 1.5), sharex=True)
        axes[0].plot(ts_arr, rgb_arr[:, 1], color="green", alpha=0.6, label="verde grezzo")
        axes[0].set_ylabel("Intensità verde"); axes[0].legend(loc="upper right")
        for ax, k in zip(axes[1:], keys):
            ax.plot(ts_arr, sigs[k], color=colors.get(k, "crimson"), lw=1.3,
                    label=(labels or {}).get(k, ALGO_NAMES[k]))
            if warmup_n > 0:
                ax.axvspan(ts_arr[0], ts_arr[min(warmup_n, N) - 1], alpha=0.15, color="orange")
            ax.set_ylabel("PPG (u.a.)"); ax.legend(loc="upper right")
        axes[-1].set_xlabel("Tempo (s)")
        axes[1].set_title(title + "   (zona arancione = warm-up)")
        plt.tight_layout()
        pp = os.path.join(args.folder, png_name)
        plt.savefig(pp, dpi=150)
        plt.close(fig)
        print(f"PNG  : {pp}")

    if args.algo == "all":
        # Stessi dati RGB per tutti: il video e' letto una volta sola.
        runs = {}
        for name in ("pos", "chrom", "green"):
            print(f"{ALGO_NAMES[name]}...")
            try:
                sig, wu, vn = run_algo(name, rgb_arr, fs)
            except ValueError as e:
                print(f"[WARN] {ALGO_NAMES[name]} saltato: {e}")
                continue
            sig = sig.astype(np.float64)
            sig[vn:] = np.nan
            res = write_outputs(f"{args.output}_{name}", {name: sig}, name, wu)
            runs[name] = {"signal": sig, "warmup_n": wu, "hr": res["hr"]}
        if len(runs) < 2:
            print("Meno di due algoritmi utilizzabili: nessun confronto.")
        else:
            start = max(r["warmup_n"] for r in runs.values())
            try:
                comp = compare_signals({k: r["signal"] for k, r in runs.items()}, fs, start)
            except ValueError as e:
                comp = None
                print(f"[WARN] confronto non eseguito: {e}")
            if comp:
                print(f"\nConfronto (porzione comune dopo warm-up: {comp['common_samples']} campioni, "
                      f"consenso {comp['consensus_hr_bpm']:.1f} bpm)")
                print(f"{'algoritmo':<14}{'HR [bpm]':>10}{'SNR [dB]':>10}")
                for k in runs:
                    h = runs[k]["hr"]
                    print(f"{ALGO_NAMES[k]:<14}{(f'{h:.1f}' if h else '-'):>10}{comp['snr_db_at_consensus'][k]:>10.1f}")
                print("|correlazione| a coppie: " +
                      "  ".join(f"{k}={v:.2f}" for k, v in comp["abs_correlation"].items()))
                comp["hr_bpm"] = {k: r["hr"] for k, r in runs.items()}
                cj = os.path.join(args.folder, args.output + "_compare.json")
                with open(cj, "w") as f:
                    json.dump(comp, f, indent=2)
                print(f"Confronto JSON: {cj}")
                if args.plot:
                    labels = {k: f"{ALGO_NAMES[k]}  HR {r['hr']:.1f}" if r["hr"] else ALGO_NAMES[k]
                              for k, r in runs.items()}
                    plot_stack(args.output + "_compare.png", {k: r["signal"] for k, r in runs.items()},
                               start, title="Confronto algoritmi (stesso video, stesse ROI)", labels=labels)
    elif args.algo == "multi":
        signals, wu_list = {}, []
        for name in ("pos", "chrom", "green"):
            try:
                sig, wu, vn = run_algo(name, rgb_arr, fs)
            except ValueError as e:
                print(f"[WARN] {ALGO_NAMES[name]} saltato: {e}")
                continue
            sig = sig.astype(np.float64)
            sig[vn:] = np.nan
            signals[name] = sig
            wu_list.append(wu)
        if len(signals) < 2:
            raise RuntimeError("--algo multi richiede almeno 2 algoritmi utilizzabili")
        fused, info = fuse_signals(signals, fs, win_sec=args.fusion_win, mode=args.fusion)
        signals["fused"] = fused
        write_outputs(args.output, signals, "fused", max(wu_list), info, ALGO_NAMES["multi"])
    else:
        print(f"{ALGO_NAMES[args.algo]}...")
        sig, wu, vn = run_algo(args.algo, rgb_arr, fs)
        sig = sig.astype(np.float64)
        sig[vn:] = np.nan
        write_outputs(args.output, {args.algo: sig}, args.algo, wu)
    # signals[nome][n] <-> ts_arr[n]  (allineamento 1:1)


if __name__ == "__main__":
    main()