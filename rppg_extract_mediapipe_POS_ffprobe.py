#!/usr/bin/env python3
"""rPPG Extractor — MediaPipe + POS_WANG (vedi docstring completo nel file)"""

import argparse, json, os, shutil, subprocess, time
from datetime import datetime, timedelta, timezone
import cv2, mediapipe as mp, numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy import sparse
from scipy.signal import butter, filtfilt

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
ROI_COLORS = {"forehead":(0,0,255), "cheek_l":(0,255,0), "cheek_r":(255,0,0)}
MIN_DIM_PX, MAX_DIM_REL = 40, 0.20

def build_face_detector(model_path):
    opts = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        output_face_blendshapes=False, output_facial_transformation_matrixes=False, num_faces=1)
    return vision.FaceLandmarker.create_from_options(opts)

def compute_rois(landmarks, active, shape):
    h, w, _ = shape
    result = []
    for name in active:
        if name not in ROI_LANDMARKS: continue
        xs = [int(landmarks[i].x * w) for i in ROI_LANDMARKS[name] if i < len(landmarks)]
        ys = [int(landmarks[i].y * h) for i in ROI_LANDMARKS[name] if i < len(landmarks)]
        if not xs: continue
        rx, ry = min(xs), min(ys)
        rw, rh = max(xs)-rx, max(ys)-ry
        if rw < MIN_DIM_PX or rh < MIN_DIM_PX:
            cx, cy = rx+rw//2, ry+rh//2
            rw, rh = max(rw,MIN_DIM_PX), max(rh,MIN_DIM_PX)
            rx, ry = cx-rw//2, cy-rh//2
        rw, rh = min(rw,int(w*MAX_DIM_REL)), min(rh,int(h*MAX_DIM_REL))
        rx, ry = max(0,min(rx,w-1)), max(0,min(ry,h-1))
        rw, rh = max(10,min(rw,w-rx)), max(10,min(rh,h-ry))
        result.append((name,rx,ry,rw,rh))
    return result

def extract_rgb_means(frame, rois):
    r,g,b = [],[],[]
    for (_,rx,ry,rw,rh) in rois:
        reg = frame[ry:ry+rh, rx:rx+rw]
        if reg.size==0: continue
        b.append(reg[:,:,0].ravel()); g.append(reg[:,:,1].ravel()); r.append(reg[:,:,2].ravel())
    if not r: return None
    return np.array([np.mean(np.concatenate(r)), np.mean(np.concatenate(g)), np.mean(np.concatenate(b))])

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

def estimate_hr(sig, fs):
    freqs = np.fft.rfftfreq(len(sig), 1./fs)
    fft   = np.abs(np.fft.rfft(sig - sig.mean()))
    mask  = (freqs>=0.7)&(freqs<=4.)
    return float(freqs[mask][np.argmax(fft[mask])]*60.) if np.any(mask) else None

def draw_rois(frame, face_box, rois, landmarks=None):
    fx,fy,fw,fh = face_box
    cv2.rectangle(frame,(fx,fy),(fx+fw,fy+fh),(200,200,200),1)
    if landmarks:
        h,w,_ = frame.shape
        for idx in [33,263]:
            if idx<len(landmarks):
                lm=landmarks[idx]
                cv2.circle(frame,(int(lm.x*w),int(lm.y*h)),4,(0,255,0),-1)
    for (name,rx,ry,rw,rh) in rois:
        col=ROI_COLORS.get(name,(255,255,0))
        cv2.rectangle(frame,(rx,ry),(rx+rw,ry+rh),col,2)
        cv2.putText(frame,name,(rx,max(ry-4,10)),cv2.FONT_HERSHEY_SIMPLEX,0.45,col,1)

def main():
    os.chdir(r'C:\Users\annad\PycharmProjects\rppg_custom_github\rPPG_custom')
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["face","finger"], required=True)
    p.add_argument("--roi", choices=list(ROI_PRESETS.keys()), default="forehead")
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
    last_fb = last_rois = last_lm = None
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
            if args.mode == "face":
                res = face_detector.detect(
                    mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                if res.face_landmarks:
                    last_lm = res.face_landmarks[0]
                    h,w,_ = frame.shape
                    xs=[int(l.x*w) for l in last_lm]; ys=[int(l.y*h) for l in last_lm]
                    last_fb = (min(xs),min(ys),max(xs)-min(xs),max(ys)-min(ys))
                    last_rois = compute_rois(last_lm, ROI_PRESETS[args.roi], frame.shape)
                elif last_lm:
                    last_rois = compute_rois(last_lm, ROI_PRESETS[args.roi], frame.shape)
                if last_rois:
                    rgb_val = extract_rgb_means(frame, last_rois)
            else:
                rgb_val = extract_rgb_finger(frame)

            if rgb_val is not None:
                rgb_buf.append(rgb_val); ts_buf.append(now)

            if not args.no_preview:
                if args.mode=="face" and last_fb and last_rois:
                    draw_rois(frame, last_fb, last_rois, last_lm)
                cv2.putText(frame,f"t={now:.1f}s n={len(ts_buf)}",
                            (10,25),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)
                cv2.imshow("rPPG preview (non salvata)", frame)
                if cv2.waitKey(1)&0xFF==ord('q'): break
    finally:
        if face_detector: face_detector.close()
        cap.release(); cv2.destroyAllWindows()

    if len(ts_buf) < 30:
        print("Troppi pochi campioni."); return

    rgb_arr = np.stack(rgb_buf)       # (N, 3)
    ts_arr  = np.array(ts_buf)        # (N,)
    N       = len(ts_arr)
    dur     = ts_arr[-1] - ts_arr[0]

    # fs: metadati per file (timing esatto), misurata per webcam (più fedele al reale)
    fs = fs_meta if using_file else (N-1)/dur
    print(f"fs usata: {fs:.3f} Hz")

    print("POS_WANG...")
    ppg, warmup_n = POS_WANG(rgb_arr, fs)
    # ppg[n] <-> ts_arr[n]  — allineamento 1:1
    # I primi warmup_n campioni hanno finestra POS incompleta -> meno affidabili

    warmup_mask = np.zeros(N, dtype=int)
    warmup_mask[:warmup_n] = 1
    wu_dur = ts_arr[min(warmup_n,N)-1] - ts_arr[0]
    print(f"Warm-up: {warmup_n} campioni ({wu_dur:.2f}s) — marcati warmup=1 nel CSV")

    hr = estimate_hr(ppg[warmup_n:], fs)

    csv_path  = os.path.join(args.folder, args.output + ".csv")
    json_path = os.path.join(args.folder, args.output + ".json")

    with open(csv_path,"w") as f:
        f.write("time_s,datetime,green_raw,ppg_pos,warmup\n")
        for t,g,pp,wu in zip(ts_arr, rgb_arr[:,1], ppg, warmup_mask):
            abs_t = (rec_start + timedelta(seconds=float(t))).isoformat(timespec="milliseconds") if rec_start else ""
            f.write(f"{t:.4f},{abs_t},{g:.4f},{pp:.6f},{wu}\n")

    with open(json_path,"w") as f:
        json.dump({
            "mode": args.mode, "algorithm": "POS_WANG",
            "roi": args.roi if args.mode=="face" else None,
            "active_rois": ROI_PRESETS[args.roi] if args.mode=="face" else None,
            "sampling_rate_hz": float(fs),
            "sampling_rate_meta_hz": float(fs_meta),
            "recording_start": rec_start.isoformat() if rec_start else None,
            "recording_start_source": rec_tag,
            "timestamp_source": "CAP_PROP_POS_MSEC" if using_file else "wall_clock",
            "duration_s": float(dur), "n_samples": N,
            "warmup_samples": int(warmup_n), "warmup_duration_s": float(wu_dur),
            "estimated_heart_rate_bpm": hr,
            "acquired_at": datetime.now().isoformat(),
        }, f, indent=2)

    print(f"\nCSV  : {csv_path}")
    print(f"JSON : {json_path}")
    if hr: print(f"HR stimata: {hr:.1f} bpm  (escluso warm-up)")

    if args.plot:
        try:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2,1,figsize=(12,7),sharex=True)

            axes[0].plot(ts_arr, rgb_arr[:,1], color="green", alpha=0.6, label="verde grezzo")
            axes[0].set_ylabel("Intensità verde"); axes[0].legend()

            axes[1].plot(ts_arr, ppg, color="crimson", lw=1.5, label="POS_WANG")
            axes[1].axvspan(ts_arr[0], ts_arr[min(warmup_n,N)-1],
                            alpha=0.15, color="orange", label=f"warm-up ({wu_dur:.1f}s)")
            axes[1].set_xlabel("Tempo (s)"); axes[1].set_ylabel("PPG (u.a.)")
            axes[1].set_title(f"POS_WANG" + (f"  |  HR: {hr:.1f} bpm" if hr else ""))
            axes[1].legend()

            plt.tight_layout()
            pp = os.path.join(args.folder, args.output+".png")
            plt.savefig(pp, dpi=150)
            print(f"PNG  : {pp}")
        except ImportError:
            print("matplotlib non installato.")

if __name__ == "__main__":
    main()