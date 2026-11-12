"""Builds the manifest for MIMIC-IV-ECG (PhysioNet, credentialed access).

Inputs
  --record-list   record_list.csv from MIMIC-IV-ECG (subject_id, study_id, ecg_time, path)
  --admissions    MIMIC-IV hosp/admissions.csv (subject_id, hadm_id, admittime, dischtime)
  --arrests       CSV you build from the linked clinical modules with one row per index
                  admission with a timestamped arrest: subject_id, hadm_id, arrest_time
                  (for example a charted cardiopulmonary resuscitation event, Section 3.3.3).
Controls are admissions with no arrest code and no such event; pass their hadm_id list
with --exclude-hadm for admissions that carry an arrest code but no timestamp.

ECGs are linked to the admission whose admittime-dischtime window contains ecg_time.
Times are converted to hours from admission.
"""
import argparse

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--record-list", required=True); ap.add_argument("--admissions", required=True)
ap.add_argument("--arrests", required=True); ap.add_argument("--exclude-hadm")
ap.add_argument("--out", default="data/manifests/mimic.csv")
a = ap.parse_args()
rec = pd.read_csv(a.record_list, parse_dates=["ecg_time"])
adm = pd.read_csv(a.admissions, parse_dates=["admittime", "dischtime"])[["subject_id", "hadm_id", "admittime", "dischtime"]]
arr = pd.read_csv(a.arrests, parse_dates=["arrest_time"]).sort_values("arrest_time").drop_duplicates("hadm_id")
m = rec.merge(adm, on="subject_id")
m = m[(m.ecg_time >= m.admittime) & (m.ecg_time <= m.dischtime)]
if a.exclude_hadm:
    m = m[~m.hadm_id.isin(pd.read_csv(a.exclude_hadm)["hadm_id"])]
m = m.merge(arr[["hadm_id", "arrest_time"]], on="hadm_id", how="left")
out = pd.DataFrame(dict(source="MIMIC-IV", patient_id=m.subject_id.astype(str) + "_" + m.hadm_id.astype(str),
                        ecg_id=m.study_id.astype(str), path=m.path, fs=500,
                        ecg_time_h=(m.ecg_time - m.admittime).dt.total_seconds() / 3600,
                        arrest_time_h=(m.arrest_time - m.admittime).dt.total_seconds() / 3600))
out.to_csv(a.out, index=False); print(len(out), "ECGs ->", a.out)
