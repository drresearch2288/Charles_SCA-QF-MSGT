"""Applies trained models to another bundle without retraining (external SCD test, Table S8).

  python scripts/predict.py --models "runs/sca_qf_msgt/seed*/model.pt" --data data/processed/scd.pt --out runs/scd_external
"""
import argparse
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sca_qf_msgt.data.dataset import PatientSequences
from sca_qf_msgt.models.sca_qf_msgt import build_model
from sca_qf_msgt.training.trainer import device_of, predict

ap = argparse.ArgumentParser()
ap.add_argument("--models", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
data = PatientSequences(torch.load(a.data, weights_only=False))
for k, path in enumerate(sorted(glob.glob(a.models))):
    ck = torch.load(path, weights_only=False)
    model = build_model(ck["config"]["model"], ck.get("seed", 0)); model.load_state_dict(ck["state_dict"])
    dev = device_of(ck["config"]["train"].get("device", "auto")); model.to(dev)
    p = predict(model, data, dev)
    p["model"], p["delta"] = "SCA-QF-MSGT", ck["delta"]
    d = os.path.join(a.out, "seed%d" % ck.get("seed", k)); os.makedirs(d, exist_ok=True)
    p.to_csv(os.path.join(d, "test_predictions.csv"), index=False)
    print(path, "->", d, "(%d ECGs)" % len(p))
