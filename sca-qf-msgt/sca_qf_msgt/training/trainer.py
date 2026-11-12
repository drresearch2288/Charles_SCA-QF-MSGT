"""Training loop (Section 3.10, Table 4): Adam, 100 epochs, batches of 5 complete patient
sequences, delta re-selected on the validation set after every epoch, early stopping
with patience 15 on validation AUPRC."""
import copy
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from ..data.dataset import collate
from ..evaluation.metrics import best_f1_threshold, ecg_metrics
from .losses import focal_loss, l2_penalty, time_loss


def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def device_of(name):
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def forward_batch(model, batch):
    """Beat logits -> ECG scores (mean of beat scores, Section 3.12). Returns
    (beat_logits, beat_ecg_index, ecg_scores)."""
    logits, idx = [], []
    for n, g in batch["groups"].items():
        logits.append(model(g["X"], g["A"], g.get("W")))
        idx.append(g["ecg"])
    logits, idx = torch.cat(logits), torch.cat(idx)
    p = torch.sigmoid(logits)
    n_ecg = batch["n_ecg"]
    s = torch.zeros(n_ecg, device=p.device).index_add_(0, idx, p)
    c = torch.zeros(n_ecg, device=p.device).index_add_(0, idx, torch.ones_like(p))
    return logits, idx, s / c.clamp_min(1)


@torch.no_grad()
def predict(model, data, device, n_seq=16, with_windows=False):
    """Returns one row per ECG with its score, for evaluation and for figures."""
    model.eval()
    rows = []
    for seqs in data.batches(n_seq, shuffle=False):
        b = collate(seqs, device, with_windows)
        _, _, s = forward_batch(model, b)
        s = s.cpu().numpy(); k = 0
        for seq in seqs:
            for e in seq:
                rows.append(dict(source=e["source"], patient_id=e["patient_id"], ecg_id=e["ecg_id"],
                                 ecg_time_h=e["ecg_time_h"], hours_before_arrest=e.get("hours_before_arrest", np.nan),
                                 is_arrest_patient=e["is_arrest_patient"], y=e["y"], score=float(s[k])))
                k += 1
    return pd.DataFrame(rows)


def train(model, train_data, val_data, cfg, out_dir, seed=0, log=print):
    tc, lc = cfg["train"], cfg["loss"]
    dev = device_of(tc.get("device", "auto"))
    model.to(dev)
    windows = bool(getattr(model, "needs_windows", False))
    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"])               # Eq. 27
    delta = tc.get("delta_init", 0.5)
    best, best_state, bad, hist = -1.0, None, 0, []
    os.makedirs(out_dir, exist_ok=True)
    for ep in range(tc["epochs"]):
        model.train(); t0 = time.time(); tot = 0.0; nb = 0
        for seqs in train_data.batches(tc["batch_sequences"], shuffle=True, seed=seed * 1000 + ep):
            b = collate(seqs, dev, windows)
            logits, idx, ecg_s = forward_batch(model, b)
            y_beat = b["ecg_y"][idx]
            loss = focal_loss(logits, y_beat, lc["focal_alpha"], lc["focal_gamma"])           # Eq. 24
            if lc.get("use_time_loss", True):
                loss = loss + lc["lambda_t"] * time_loss(ecg_s, b["ecg_seq"], b["ecg_y"], b["ecg_dt"],
                                                         delta, lc["target_h"], lc["temp"])     # Eq. 25
            loss = loss + lc["lambda_r"] * l2_penalty(model)                                    # Eq. 26
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        vp = predict(model, val_data, dev, with_windows=windows)
        y, s = vp["y"].to_numpy(), vp["score"].to_numpy()
        delta = best_f1_threshold(y, s)                                  # fixed for the next epoch
        m = ecg_metrics(y, s, delta)
        hist.append(dict(epoch=ep + 1, loss=tot / max(nb, 1), delta=delta, **m, sec=time.time() - t0))
        log("epoch %3d  loss %.4f  val AUPRC %.4f  F1 %.4f  delta %.3f" % (ep + 1, hist[-1]["loss"], m["AUPRC"], m["F1"], delta))
        if m["AUPRC"] > best:
            best, bad = m["AUPRC"], 0
            best_state = copy.deepcopy(model.state_dict()); best_delta = delta
        else:
            bad += 1
            if bad >= tc["patience"]:
                log("early stopping at epoch %d" % (ep + 1)); break
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "delta": best_delta, "config": dict(cfg), "seed": seed},
               os.path.join(out_dir, "model.pt"))
    pd.DataFrame(hist).to_csv(os.path.join(out_dir, "history.csv"), index=False)
    json.dump({"best_val_auprc": best, "delta": best_delta}, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    return model, best_delta
