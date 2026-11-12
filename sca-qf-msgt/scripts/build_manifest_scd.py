"""Builds the manifest for the SCD Holter Database (PhysioNet sddb): each continuous
two-channel record is cut into consecutive 5-min segments that play the role of ECGs
(Section 3.3.8). The paced record is excluded (Section 3.3.1).

  python scripts/build_manifest_scd.py --db data/raw/sddb --onsets scd_onsets.csv --exclude <paced record id>

--onsets: CSV with columns record, onset_s (arrest/VF onset in seconds from record start,
taken from the database documentation and annotations).
"""
import argparse
import os

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--db", required=True); ap.add_argument("--onsets", required=True)
ap.add_argument("--exclude", nargs="*", default=[]); ap.add_argument("--segment-min", type=float, default=5)
ap.add_argument("--out", default="data/manifests/scd.csv")
a = ap.parse_args()
import wfdb
on = pd.read_csv(a.onsets, dtype={"record": str})
rows = []
seg = a.segment_min * 60
for _, r in on.iterrows():
    if r.record in a.exclude:
        continue
    h = wfdb.rdheader(os.path.join(a.db, r.record))
    dur = h.sig_len / h.fs
    end = min(dur, float(r.onset_s))
    k = 0; t = 0.0
    while t + seg <= end:
        rows.append(dict(source="SCD", patient_id=r.record, ecg_id="%s_%04d" % (r.record, k), path=r.record, fs=h.fs,
                         seg_start_s=t, seg_end_s=t + seg, ecg_time_h=(t + seg) / 3600, arrest_time_h=float(r.onset_s) / 3600))
        t += seg; k += 1
pd.DataFrame(rows).to_csv(a.out, index=False); print(len(rows), "segments ->", a.out)
