"""Labels, splits and preprocesses every ECG in one or more manifests into beat graphs.

  python scripts/prepare_data.py --manifest data/manifests/scaos.csv data/manifests/mimic.csv \
      --root data/raw --out data/processed [--keep-windows] [--hrv]

SCD (external two-channel test) is prepared the same way into its own bundle with
--external, which skips the split and marks every ECG as 'test'.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sca_qf_msgt.config import load_config
from sca_qf_msgt.data.hrv import hrv_features
from sca_qf_msgt.data.io import read_record
from sca_qf_msgt.data.labels import label_manifest, patient_split, split_table
from sca_qf_msgt.data.preprocessing import ecg_to_graphs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--root", default="")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--name", default="bundle")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--keep-windows", action="store_true", help="store beat windows (needed by ResNet/CNN-LSTM)")
    ap.add_argument("--hrv", action="store_true", help="compute HRV/interval features for the clinical baselines")
    ap.add_argument("--external", action="store_true", help="no split; every ECG is a test ECG (SCD)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    dc = dict(cfg["data"])
    man = pd.concat([pd.read_csv(m) for m in a.manifest], ignore_index=True)
    man = label_manifest(man, dc["horizon_h"])
    man = man.assign(split="test") if a.external else patient_split(man, dc["split"], cfg["seed"])
    ecgs, qc, hrv_rows = [], [], []
    for _, r in tqdm(man.iterrows(), total=len(man)):
        try:
            sig, fs, leads = read_record(r, a.root)
            g = ecg_to_graphs(sig, fs, dc, leads, keep_windows=a.keep_windows)
        except Exception as ex:                       # unreadable record
            g = None; print("skip", r["ecg_id"], ex)
        qc.append(dict(ecg_id=r["ecg_id"], kept=g is not None))
        if g is None:
            continue
        e = dict(source=r["source"], patient_id=str(r["patient_id"]), ecg_id=str(r["ecg_id"]), split=r["split"],
                 ecg_time_h=float(r["ecg_time_h"]), hours_before_arrest=float(r["hours_before_arrest"]),
                 is_arrest_patient=int(r["is_arrest_patient"]), y=int(r["y"]), X=g.X, A=g.A, leads=leads)
        if a.keep_windows:
            e["windows"] = g.windows
        ecgs.append(e)
        if a.hrv:
            from sca_qf_msgt.data.preprocessing import bandpass, resample
            x = resample(sig, fs, dc["fs"])
            li = leads.index("II") if "II" in leads else 0
            hrv_rows.append(dict(ecg_id=e["ecg_id"], **hrv_features(bandpass(x, dc["fs"])[li], g.r_peaks, dc["fs"])))
    os.makedirs(a.out, exist_ok=True)
    torch.save({"ecgs": ecgs, "meta": {"config": dc, "manifests": a.manifest}}, os.path.join(a.out, a.name + ".pt"))
    kept = pd.DataFrame(qc)
    man = man.merge(kept, on="ecg_id", how="left")
    man.to_csv(os.path.join(a.out, a.name + "_labelled.csv"), index=False)
    split_table(man[man["kept"] == True]).to_csv(os.path.join(a.out, a.name + "_table_S1.csv"), index=False)
    if hrv_rows:
        pd.DataFrame(hrv_rows).to_csv(os.path.join(a.out, a.name + "_hrv.csv"), index=False)
    print("kept %d of %d ECGs; %d excluded by quality control" % (len(ecgs), len(man), len(man) - len(ecgs)))


if __name__ == "__main__":
    main()
