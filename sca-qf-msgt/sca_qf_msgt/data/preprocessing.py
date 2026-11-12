"""Signal preprocessing and beat-level graph construction (Sections 3.5-3.6).

Pipeline for one ECG (leads x samples):
  resample to fs (510 Hz) -> zero-phase Butterworth band-pass 0.5-50 Hz (Eq. 1)
  -> per-lead z-score (Eq. 2) -> R-peaks on lead II (Pan-Tompkins, NeuroKit2)
  -> 0.6 s windows centred on each R-peak (Eq. 3) -> RR quality check
  -> node features [mean, std, max, min] (Eq. 4) -> Pearson edges, |rho| > tau (Eqs. 5-6)
"""
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy import signal

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


@dataclass
class BeatGraphs:
    X: np.ndarray            # (B, n, 4) node features
    A: np.ndarray            # (B, n, n) weighted adjacency, zero diagonal
    r_peaks: np.ndarray      # (B,) sample index of each kept beat
    windows: np.ndarray = None   # (B, n, W) beat windows, only if requested
    n_detected: int = 0
    n_discarded: int = 0


def resample(x, fs_in, fs_out):
    if fs_in == fs_out:
        return x
    fr = Fraction(fs_out, int(round(fs_in))).limit_denominator(1000)
    return signal.resample_poly(x, fr.numerator, fr.denominator, axis=-1)


def bandpass(x, fs, low=0.5, high=50.0, order=4):
    """Equation (1): zero-phase Butterworth band-pass."""
    high = min(high, 0.45 * fs)
    sos = signal.butter(order, [low, high], btype="band", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x, axis=-1)


def zscore(x, eps=1e-8):
    """Equation (2): per-lead z-score."""
    mu = x.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True)
    return (x - mu) / (sd + eps)


def detect_r_peaks(lead, fs):
    """Pan-Tompkins detector from NeuroKit2 on one lead; simple fallback if unavailable."""
    try:
        import neurokit2 as nk
        _, info = nk.ecg_peaks(lead, sampling_rate=fs, method="pantompkins1985")
        return np.asarray(info["ECG_R_Peaks"], dtype=int)
    except ImportError:
        d = np.abs(np.gradient(lead))
        peaks, _ = signal.find_peaks(d, distance=int(0.3 * fs), height=np.percentile(d, 98) * 0.5)
        # move to the local amplitude maximum
        w = int(0.05 * fs)
        return np.array([max(p - w, 0) + int(np.argmax(np.abs(lead[max(p - w, 0):p + w]))) for p in peaks], int)


def node_features(win):
    """Equation (4): [mean, std, max, min] per lead. win: (B, n, W) -> (B, n, 4)."""
    return np.stack([win.mean(-1), win.std(-1), win.max(-1), win.min(-1)], axis=-1).astype(np.float32)


def pearson_adjacency(win, tau=0.5, eps=1e-8):
    """Equations (5)-(6): Pearson correlation between lead windows, kept if |rho| > tau."""
    w = win - win.mean(-1, keepdims=True)
    w = w / (np.linalg.norm(w, axis=-1, keepdims=True) + eps)
    rho = np.einsum("bik,bjk->bij", w, w)
    A = np.where(np.abs(rho) > tau, rho, 0.0)
    idx = np.arange(A.shape[1])
    A[:, idx, idx] = 0.0
    return A.astype(np.float32)


def ecg_to_graphs(sig, fs_in, cfg, lead_names=None, keep_windows=False, r_lead="II"):
    """sig: (n_leads, T) raw ECG. Returns BeatGraphs, or None if the ECG fails quality control."""
    fs = cfg["fs"]
    x = resample(np.asarray(sig, float), fs_in, fs)
    x = zscore(bandpass(x, fs, *cfg["bandpass"], order=cfg.get("filter_order", 4)))
    lead_names = lead_names or LEADS[: x.shape[0]]
    li = lead_names.index(r_lead) if r_lead in lead_names else 0
    peaks = detect_r_peaks(x[li], fs)
    half = int(round(cfg["window_s"] * fs / 2))           # 153 samples at 510 Hz -> 306 per window
    n_det = len(peaks)
    if n_det < 2:
        return None
    rr = np.diff(peaks) / fs
    lo, hi = cfg["rr_range_s"]
    ok = np.ones(n_det, bool)
    ok[1:] &= (rr >= lo) & (rr <= hi)                     # RR to the previous beat
    ok[:-1] &= (rr >= lo) & (rr <= hi)                    # and to the next beat
    ok &= (peaks - half >= 0) & (peaks + half <= x.shape[1])
    n_bad = int((~ok).sum())
    if n_bad / n_det > cfg["max_bad_beat_frac"] or ok.sum() == 0:
        return None
    kept = peaks[ok]
    win = np.stack([x[:, p - half:p + half] for p in kept])   # (B, n, 2*half)
    return BeatGraphs(X=node_features(win), A=pearson_adjacency(win, cfg["tau"]), r_peaks=kept,
                      windows=win.astype(np.float16) if keep_windows else None,
                      n_detected=n_det, n_discarded=n_bad)
