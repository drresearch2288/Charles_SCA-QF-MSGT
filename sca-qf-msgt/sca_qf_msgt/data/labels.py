"""Labelling protocol (Section 3.3.8, Algorithm S1) and patient-level splits (Section 3.3.6).

Manifest columns (one row per ECG):
  source, patient_id, ecg_id, path, fs, ecg_time_h, arrest_time_h
ecg_time_h and arrest_time_h are hours on a common clock within the admission;
arrest_time_h is empty for patients without an arrest.
"""
import numpy as np
import pandas as pd


def label_manifest(df, horizon_h=12.0):
    """Adds hours_before_arrest and y; drops post-arrest ECGs and arrest ECGs earlier than H."""
    df = df.copy()
    df["arrest_time_h"] = pd.to_numeric(df["arrest_time_h"], errors="coerce")
    df["ecg_time_h"] = pd.to_numeric(df["ecg_time_h"], errors="coerce")
    arrest = df["arrest_time_h"].notna()
    df["is_arrest_patient"] = arrest.groupby(df["patient_id"]).transform("any").astype(int)
    d = df["arrest_time_h"] - df["ecg_time_h"]
    df["hours_before_arrest"] = np.where(arrest, d, np.nan)
    post = arrest & (d <= 0)                       # recorded at or after the arrest
    early = arrest & (d > horizon_h)               # uncertain state, excluded
    df = df[~(post | early)].copy()
    df["y"] = np.where(df["arrest_time_h"].notna(), 1, 0).astype(int)
    return df.reset_index(drop=True)


def patient_split(df, fractions=(0.70, 0.15, 0.15), seed=0):
    """Splits patients 70/15/15 within each source, stratified by outcome; adds column 'split'."""
    rng = np.random.default_rng(seed)
    out = {}
    pat = df.groupby(["source", "patient_id"])["is_arrest_patient"].max().reset_index()
    for (_, outcome), grp in pat.groupby(["source", "is_arrest_patient"]):
        ids = grp["patient_id"].to_numpy().copy()
        rng.shuffle(ids)
        n = len(ids)
        n_tr = int(round(fractions[0] * n))
        n_va = int(round(fractions[1] * n))
        for k, pid in enumerate(ids):
            out[(grp["source"].iloc[0], pid)] = "train" if k < n_tr else "val" if k < n_tr + n_va else "test"
    df = df.copy()
    df["split"] = [out[(s, p)] for s, p in zip(df["source"], df["patient_id"])]
    return df


def split_table(df):
    """Patients, arrest events, ECGs and prevalence per source and split (Table S1)."""
    rows = []
    for (src, sp), g in df.groupby(["source", "split"]):
        rows.append(dict(source=src, split=sp, patients=g["patient_id"].nunique(),
                         arrest_events=g.loc[g["is_arrest_patient"] == 1, "patient_id"].nunique(),
                         ecgs=len(g), prevalence=round(g["y"].mean(), 4)))
    return pd.DataFrame(rows)
