import numpy as np
from rppg_utils import detrend,_snr_at,_psd
from scipy.signal import butter, filtfilt, windows

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

