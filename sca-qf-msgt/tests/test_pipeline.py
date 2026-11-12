"""End-to-end check on a synthetic signal: preprocessing shapes and one training step."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sca_qf_msgt.config import load_config
from sca_qf_msgt.data.preprocessing import ecg_to_graphs
from sca_qf_msgt.models.baselines import BASELINES
from sca_qf_msgt.models.sca_qf_msgt import build_model
from sca_qf_msgt.training.losses import focal_loss

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = load_config(os.path.join(ROOT, "configs", "default.yaml"))


def _fake_ecg(fs=500, sec=10, leads=12, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(fs * sec) / fs
    beat = np.zeros_like(t)
    for c in np.arange(0.4, sec, 0.8):
        beat += np.exp(-((t - c) / 0.012) ** 2) - 0.2 * np.exp(-((t - c - 0.25) / 0.05) ** 2)
    return np.stack([beat * rng.uniform(0.5, 1.5) + 0.02 * rng.normal(size=t.size) for _ in range(leads)])


def test_beat_graph_shapes():
    g = ecg_to_graphs(_fake_ecg(), 500, dict(CFG["data"]), keep_windows=True)
    assert g is not None
    assert g.X.shape[1:] == (12, 4) and g.A.shape[1:] == (12, 12)
    assert g.windows.shape[-1] == 306                     # 0.6 s at 510 Hz
    assert np.allclose(np.diagonal(g.A, axis1=1, axis2=2), 0)
    assert (np.abs(g.A[g.A != 0]) > 0.5).all()


def test_two_lead_graph():
    g = ecg_to_graphs(_fake_ecg(leads=2), 250, dict(CFG["data"]), lead_names=["V1", "II"])
    assert g.X.shape[1:] == (2, 4)


def test_model_forward_and_step():
    g = ecg_to_graphs(_fake_ecg(), 500, dict(CFG["data"]), keep_windows=True)
    X, A = torch.tensor(g.X), torch.tensor(g.A)
    W = torch.tensor(g.windows.astype(np.float32))
    for name in ("circuit", "bottleneck_mlp", "random_features", "wide_mlp", "identity", "none"):
        m = build_model({**CFG["model"], "quantum_block": name})
        opt = torch.optim.Adam(m.parameters(), 1e-3)
        loss = focal_loss(m(X, A), torch.ones(len(X)))
        loss.backward(); opt.step()
        assert torch.isfinite(loss)
    for b in BASELINES.values():
        assert b()(X, A, W).shape == (len(X),)
