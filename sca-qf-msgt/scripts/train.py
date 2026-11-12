"""Trains SCA-QF-MSGT, an ablation variant, a classical control or a deep baseline.

  python scripts/train.py --data data/processed/bundle.pt --out runs/sca_qf_msgt --seeds 0 1 2 3 4
  python scripts/train.py ... --set model.quantum_block=bottleneck_mlp      # Section 4.6.1 control
  python scripts/train.py ... --set model.quantum_block=none                 # ablation row
  python scripts/train.py ... --baseline multiscale_gcn                      # Table 5 baseline
  python scripts/train.py ... --train-sources MIMIC-IV                        # LODO (Table S6)
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sca_qf_msgt.config import load_config
from sca_qf_msgt.data.dataset import PatientSequences
from sca_qf_msgt.models.baselines import BASELINES
from sca_qf_msgt.models.sca_qf_msgt import build_model
from sca_qf_msgt.training.trainer import predict, set_seed, train, device_of


NAMES = {"resnet_attention": "ResNet-Attention", "multiscale_gcn": "Multi-Scale GCN", "cnn_lstm": "CNN-LSTM"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. train.epochs=5")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--baseline", choices=sorted(BASELINES))
    ap.add_argument("--train-sources", nargs="*", help="default: config data.sources")
    ap.add_argument("--name", help="model name written to the predictions file")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    bundle = torch.load(a.data, weights_only=False)
    src = a.train_sources or cfg["data"]["sources"]
    tr = PatientSequences(bundle, "train", src)
    va = PatientSequences(bundle, "val", src)
    te = PatientSequences(bundle, "test")
    print("patients: train %d, val %d, test %d" % (len(tr), len(va), len(te)))
    for seed in a.seeds or cfg["seeds"]:
        set_seed(seed)
        model = BASELINES[a.baseline]() if a.baseline else build_model(cfg["model"], seed)
        out = os.path.join(a.out, "seed%d" % seed)
        model, delta = train(model, tr, va, cfg, out, seed)
        pred = predict(model, te, device_of(cfg["train"]["device"]), with_windows=getattr(model, "needs_windows", False))
        pred["model"] = a.name or NAMES.get(a.baseline, "SCA-QF-MSGT")
        pred["delta"] = delta
        pred.to_csv(os.path.join(out, "test_predictions.csv"), index=False)
        print("seed %d done: delta %.3f, predictions in %s" % (seed, delta, out))


if __name__ == "__main__":
    main()
