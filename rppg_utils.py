import numpy as np
from scipy import sparse
from scipy.signal import butter, filtfilt

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

