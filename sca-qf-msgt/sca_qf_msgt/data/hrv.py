"""Features of the clinical baselines (Section 4.5.1): mean heart rate, SDNN, RMSSD and
pNN50 from the R-peaks, and QRS duration, QTc (Bazett) and PR interval from NeuroKit2
delineation of lead II. On 10-s ECGs these are ultra-short-term HRV values."""
import numpy as np

HRV_COLUMNS = ["mean_hr", "sdnn_ms", "rmssd_ms", "pnn50", "qrs_ms", "qtc_ms", "pr_ms"]


def hrv_features(lead_ii, r_peaks, fs):
    rr = np.diff(r_peaks) / fs * 1000.0
    out = dict.fromkeys(HRV_COLUMNS, np.nan)
    if len(rr) >= 2:
        d = np.diff(rr)
        out.update(mean_hr=60000.0 / rr.mean(), sdnn_ms=rr.std(ddof=1), rmssd_ms=np.sqrt((d ** 2).mean()),
                   pnn50=float((np.abs(d) > 50).mean()))
    try:
        import neurokit2 as nk
        _, w = nk.ecg_delineate(lead_ii, r_peaks, sampling_rate=fs, method="dwt")
        def med(a, b):
            a, b = np.asarray(w.get(a, []), float), np.asarray(w.get(b, []), float)
            k = min(len(a), len(b))
            v = (b[:k] - a[:k]) / fs * 1000.0
            v = v[np.isfinite(v) & (v > 0)]
            return float(np.median(v)) if len(v) else np.nan
        out["qrs_ms"] = med("ECG_R_Onsets", "ECG_R_Offsets")
        out["pr_ms"] = med("ECG_P_Onsets", "ECG_R_Onsets")
        qt = med("ECG_R_Onsets", "ECG_T_Offsets")
        if np.isfinite(qt) and len(rr):
            out["qtc_ms"] = qt / np.sqrt(rr.mean() / 1000.0)
    except Exception:
        pass
    return out
