import numpy as np
from rppg_utils import detrend

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