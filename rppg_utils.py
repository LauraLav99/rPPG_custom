import cv2, mediapipe as mp, numpy as np
from fontTools.cffLib import DictCompiler
from scipy import sparse
from scipy.signal import butter, filtfilt
from rppg_methods import POS_WANG,CHROM_DEHAAN,GREEN_SIGNAL
import argparse,sys,os,json
from datetime import datetime, timedelta, timezone

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

def extract_rgb_means(frame, rois,MIN_ROI_PIXELS):
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
    """Media RGB sull'intero frame (dito sull'obiettivo)."""
    b = float(np.mean(frame[:, :, 0]))
    g = float(np.mean(frame[:, :, 1]))
    r = float(np.mean(frame[:, :, 2]))
    return np.array([r, g, b], dtype=np.float64)

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

def run_algo(name, rgb, fs):
    """Ritorna (ppg[N], warmup_n, valid_n) per l'algoritmo scelto."""
    if name == "chrom":
        return CHROM_DEHAAN(rgb, fs)
    if name == "green":
        return GREEN_SIGNAL(rgb, fs)
    ppg, warmup_n = POS_WANG(rgb, fs)
    return ppg, warmup_n, len(ppg)

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

def trainlab_rppg_parser(p,ROI_PRESETS):
    p.add_argument("--mode", choices=["face", "finger"], required=True)
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
    p.add_argument("--input_folder", type=str, default=None)
    p.add_argument("--input_file",type=str, default=None)
    #p.add_argument("--model", type=str, default="face_landmarker.task")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--folder", type=str, default="OUTPUT")
    p.add_argument("--no-preview", action="store_true")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    return args

def return_camera_files(args):
    allFiles = os.listdir(path=args.input_folder)
    allFiles = [f for f in allFiles if os.path.isfile(args.input_folder + '/' + f)]
    #NOTE: here jsonfiles are used only as check -> I will look for the exact jsonfiles in for loop using
    # the camera filename
    # if is missing camera file os json file the program quit
    if args.input_file is None:
        #sort the json and camesta files
        jsonFiles=[f for f in allFiles if f.endswith('.json') ]
        camerafiles=[f for f in allFiles if f.endswith('.mp4') ]
        # check if there are the same number of json file and mp4.
        # Json files and mp4 have the same name-> compare json and mp4 filenames, if the number differs,
        # return the name is missing.
        if len(camerafiles)!=len(jsonFiles):
            jj=set([os.path.splitext(j)[0] for j in jsonFiles])
            cc = set([os.path.splitext(c)[0] for c in camerafiles])
            missing=list(sorted(cc-jj))
            print(f"Files missing\n: {missing}")
            sys.exit()
    else:
        camerafiles=args.input_file
        jsonFiles=os.path.splitext(camerafiles)[0]+'.json'
        # look if the json file exist in folder
        if not(jsonFiles in allFiles):
            sys.exit()
    return camerafiles

def estimate_hr(sig, fs):
    freqs = np.fft.rfftfreq(len(sig), 1./fs)
    fft   = np.abs(np.fft.rfft(sig - sig.mean()))
    mask  = (freqs>=0.7)&(freqs<=4.)
    return float(freqs[mask][np.argmax(fft[mask])]*60.) if np.any(mask) else None

def hr_of_sig(sig, fs,warmup_n):
    x = sig[warmup_n:]
    x = x[np.isfinite(x)]
    return estimate_hr(x, fs) if len(x) >= int(4 * fs) else None

def write_outputs(out_name, signals, main_name, rgb_arr,warmup_n,ALGO_NAMES,ROI,ts_arr,args,metainfo, info=None, algo_label=None):
        """Scrive CSV + JSON (+ PNG con --plot) per un insieme di segnali. Ritorna una sintesi."""
        ROI_PRESETS=ROI[0]
        ROI_POLYGONS=ROI[1]
        n_missed=metainfo[0]
        n_seen=metainfo[1]
        fs=metainfo[2]
        fs_meta=metainfo[3]
        rec_tag=metainfo[4]
        dur=metainfo[5]
        rec_start=metainfo[6]

        algo_label = algo_label or ALGO_NAMES[main_name]
        names = list(signals)
        finite_idx = np.flatnonzero(np.isfinite(signals[main_name]))
        valid_n = int(finite_idx[-1] + 1) if finite_idx.size else 0
        N=len(ts_arr)
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
            "timestamp_source": "CAP_PROP_POS_MSEC" ,
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

        return dict(hr=hr, valid_n=valid_n)


def plot_stack(png_name, sigs,ts_arr,rgb_arr,ALGO_NAMES,args, warmup_n, title, labels=None):
        """Una riga per il verde grezzo + una per ogni segnale."""
        try:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib non installato.")
            return
        N=len(ts_arr)
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