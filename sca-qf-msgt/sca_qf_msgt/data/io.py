"""Reading ECG records named in a manifest: NumPy arrays (.npy, leads x samples) or
WFDB records (PhysioNet format, path without extension). Optional seg_start_s /
seg_end_s columns crop a segment from a continuous recording (SCD 5-min segments)."""
import os

import numpy as np

from .preprocessing import LEADS

_ALIASES = {"AVR": "aVR", "AVL": "aVL", "AVF": "aVF"}


def _norm_lead(name):
    n = name.strip().replace("lead", "").replace("Lead", "").strip()
    return _ALIASES.get(n.upper(), n if n in LEADS else n.upper())


def read_record(row, root=""):
    path = os.path.join(root, str(row["path"]))
    fs = float(row["fs"]) if "fs" in row and str(row["fs"]) not in ("", "nan") else None
    if path.endswith(".npy"):
        sig = np.load(path)
        leads = LEADS[: sig.shape[0]]
        if "leads" in row and isinstance(row["leads"], str) and row["leads"]:
            leads = [_norm_lead(l) for l in row["leads"].split("|")]
    else:
        import wfdb
        rec = wfdb.rdrecord(path)
        sig, fs = rec.p_signal.T, rec.fs
        leads = [_norm_lead(l) for l in rec.sig_name]
    if fs is None:
        raise ValueError("sampling rate missing for %s" % path)
    if "seg_start_s" in row and str(row.get("seg_start_s", "")) not in ("", "nan"):
        a, b = int(float(row["seg_start_s"]) * fs), int(float(row["seg_end_s"]) * fs)
        sig = sig[:, a:b]
    sig = np.nan_to_num(np.asarray(sig, dtype=float))
    # reorder to the standard 12-lead order, keeping only recorded leads (Section 3.3.7)
    order = [leads.index(l) for l in LEADS if l in leads]
    names = [leads[i] for i in order] if order else leads
    sig = sig[order] if order else sig
    return sig, fs, names
