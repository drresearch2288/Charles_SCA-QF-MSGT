"""Domain-shift check of Section 4.3 / Table S6: logistic regression predicting the source
from the learned graph embedding z_G; AUC near 0.5 means the sources are indistinguishable.

  python scripts/domain_classifier.py --model runs/sca_qf_msgt/seed0/model.pt \
      --data data/processed/bundle.pt --external data/processed/scd.pt --pairs MIMIC-IV:SCD MIMIC-IV:SCAOS
"""
import argparse
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sca_qf_msgt.models.sca_qf_msgt import build_model


@torch.no_grad()
def embeddings(model, ecgs, max_per_source=5000, seed=0):
    rng = np.random.default_rng(seed); by = {}
    for e in ecgs:
        by.setdefault(e["source"], []).append(e)
    out = {}
    for s, lst in by.items():
        pick = rng.permutation(len(lst))[:max_per_source]
        z = [model.encoder(torch.as_tensor(lst[i]["X"]), torch.as_tensor(lst[i]["A"])).mean(0).numpy() for i in pick]
        out[s] = np.stack(z)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--external"); ap.add_argument("--pairs", nargs="+", default=["MIMIC-IV:SCAOS"])
    a = ap.parse_args()
    ck = torch.load(a.model, weights_only=False)
    model = build_model(ck["config"]["model"]); model.load_state_dict(ck["state_dict"]); model.eval()
    ecgs = [e for e in torch.load(a.data, weights_only=False)["ecgs"] if e["split"] == "test"]
    if a.external:
        ecgs += torch.load(a.external, weights_only=False)["ecgs"]
    Z = embeddings(model, ecgs)
    for pair in a.pairs:
        s1, s2 = pair.split(":")
        X = np.concatenate([Z[s1], Z[s2]]); y = np.r_[np.zeros(len(Z[s1])), np.ones(len(Z[s2]))]
        p = cross_val_predict(LogisticRegression(max_iter=2000), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                              method="predict_proba")[:, 1]
        rng = np.random.default_rng(0); aucs = []
        for _ in range(2000):
            i = rng.integers(0, len(y), len(y))
            if len(np.unique(y[i])) == 2:
                aucs.append(roc_auc_score(y[i], p[i]))
        print("%s versus %s: AUC %.3f (%.3f-%.3f)" % (s1, s2, roc_auc_score(y, p), *np.quantile(aucs, [0.025, 0.975])))


if __name__ == "__main__":
    main()
