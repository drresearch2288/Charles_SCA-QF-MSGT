"""Patient-sequence dataset. Each item is one patient: the time-ordered ECGs of the
admission, each holding its beat graphs. Training batches hold complete sequences
so that the lead-time loss (Eqs. 25a-25c) can be computed (Section 3.10)."""
import random

import numpy as np
import torch


class PatientSequences:
    def __init__(self, bundle, split=None, sources=None):
        """bundle: dict saved by scripts/prepare_data.py with keys 'ecgs' (list of dicts) and 'meta'."""
        ecgs = [e for e in bundle["ecgs"]
                if (split is None or e["split"] == split) and (sources is None or e["source"] in sources)]
        self.patients = {}
        for e in ecgs:
            self.patients.setdefault((e["source"], e["patient_id"]), []).append(e)
        for k in self.patients:
            self.patients[k].sort(key=lambda e: e["ecg_time_h"])
        self.keys = sorted(self.patients)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        return self.patients[self.keys[i]]

    def n_ecgs(self):
        return sum(len(v) for v in self.patients.values())

    def batches(self, n_seq, shuffle=True, seed=None):
        idx = list(range(len(self)))
        if shuffle:
            random.Random(seed).shuffle(idx)
        for s in range(0, len(idx), n_seq):
            yield [self[i] for i in idx[s:s + n_seq]]


def collate(sequences, device="cpu", with_windows=False):
    """Stacks all beats of all ECGs in the batch, grouped by node count (12 or fewer leads).

    Returns a dict with tensors per node-count group plus the index maps needed to go
    from beats back to ECGs and from ECGs back to patient sequences."""
    groups = {}
    ecg_rows = []                          # per ECG: (seq_idx, y, hours_before_arrest)
    beat_ecg = {}
    for si, seq in enumerate(sequences):
        for e in seq:
            ei = len(ecg_rows)
            ecg_rows.append((si, e["y"], e.get("hours_before_arrest", np.nan)))
            n = e["X"].shape[1]
            g = groups.setdefault(n, {"X": [], "A": [], "W": []})
            g["X"].append(e["X"]); g["A"].append(e["A"])
            if with_windows:
                g["W"].append(e["windows"])
            beat_ecg.setdefault(n, []).extend([ei] * len(e["X"]))
    out = {"groups": {}, "n_ecg": len(ecg_rows)}
    for n, g in groups.items():
        d = {"X": torch.as_tensor(np.concatenate(g["X"]), dtype=torch.float32, device=device),
             "A": torch.as_tensor(np.concatenate(g["A"]), dtype=torch.float32, device=device),
             "ecg": torch.as_tensor(beat_ecg[n], dtype=torch.long, device=device)}
        if with_windows:
            d["W"] = torch.as_tensor(np.concatenate(g["W"]).astype(np.float32), device=device)
        out["groups"][n] = d
    er = np.array(ecg_rows, dtype=float)
    out["ecg_seq"] = torch.as_tensor(er[:, 0], dtype=torch.long, device=device)
    out["ecg_y"] = torch.as_tensor(er[:, 1], dtype=torch.float32, device=device)
    out["ecg_dt"] = torch.as_tensor(np.nan_to_num(er[:, 2], nan=0.0), dtype=torch.float32, device=device)
    return out
