"""Per-beat inference latency decomposition (Section 4.8, Table S10): mean +- SD over
test beats, batch size 1, after 100 warm-up beats. Run on the machine you report.

  python scripts/latency.py --model runs/sca_qf_msgt/seed0/model.pt --manifest data/manifests/mimic.csv --root data/raw
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sca_qf_msgt.data import preprocessing as P
from sca_qf_msgt.data.io import read_record
from sca_qf_msgt.models.sca_qf_msgt import build_model


def tick(dev):
    if dev == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", default=""); ap.add_argument("--beats", type=int, default=10000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    ck = torch.load(a.model, weights_only=False); dc = ck["config"]["data"]
    model = build_model(ck["config"]["model"]); model.load_state_dict(ck["state_dict"]); model.eval().to(a.device)
    man = pd.read_csv(a.manifest)
    T = {k: [] for k in ["R-peak detection", "Beat segmentation and node features", "Graph construction",
                         "Graph attention encoder", "Circuit simulation", "Fusion gate and head"]}
    n = 0
    with torch.no_grad():
        for _, r in man.iterrows():
            sig, fs, leads = read_record(r, a.root)
            x = P.zscore(P.bandpass(P.resample(sig, fs, dc["fs"]), dc["fs"]))
            t0 = time.perf_counter(); pk = P.detect_r_peaks(x[leads.index("II") if "II" in leads else 0], dc["fs"])
            t_r = (time.perf_counter() - t0) / max(len(pk), 1)
            half = int(dc["window_s"] * dc["fs"] / 2)
            for p in pk:
                if p - half < 0 or p + half > x.shape[1]:
                    continue
                t0 = time.perf_counter(); w = x[None, :, p - half:p + half]; X = P.node_features(w); t1 = time.perf_counter()
                A = P.pearson_adjacency(w, dc["tau"]); t2 = time.perf_counter()
                Xt, At = torch.as_tensor(X, device=a.device), torch.as_tensor(A, device=a.device)
                s0 = tick(a.device); z = model.encoder(Xt, At); s1 = tick(a.device)
                th = np.pi * torch.tanh(model.to_angles(z)); q = model.block(th); s2 = tick(a.device)
                zq = model.from_q(q); g = torch.sigmoid(model.gate(torch.cat([z, zq], -1)))
                model.head(g * zq + (1 - g) * z); s3 = tick(a.device)
                n += 1
                if n > 100:
                    for k, v in zip(T, [t_r, t1 - t0, t2 - t1, s1 - s0, s2 - s1, s3 - s2]):
                        T[k].append(v * 1000)
                if n >= a.beats + 100:
                    break
            if n >= a.beats + 100:
                break
    tot = np.sum([np.array(v) for v in T.values()], axis=0)
    for k, v in T.items():
        print("%-40s %.1f +- %.1f ms" % (k, np.mean(v), np.std(v)))
    print("%-40s %.1f +- %.1f ms (%d beats, %s)" % ("Total", tot.mean(), tot.std(), len(tot), a.device))


if __name__ == "__main__":
    main()
