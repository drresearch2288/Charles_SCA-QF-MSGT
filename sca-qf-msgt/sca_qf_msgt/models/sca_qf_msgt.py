"""The full SCA-QF-MSGT model (Sections 3.7-3.9) and its ablation variants (Section 4.6)."""
import math

import torch
import torch.nn as nn

from .gat import MultiScaleGATEncoder
from .quantum import make_block


class SCAQFMSGT(nn.Module):
    """Beat graph -> z_G (Eq. 12) -> theta (Eqs. 13-14) -> q (Eqs. 15-17) -> z_Q (Eq. 18)
    -> gated fusion (Eqs. 19-21) -> logit u (Eq. 22). Risk y_hat = sigmoid(u) (Eq. 23).

    Ablation switches (Table 6):
      quantum_block="none"   Without quantum refinement (z_F = z_G)
      fusion="concat"        Without adaptive fusion gate (concatenation of z_G and z_Q)
      edge_prior=False       Without edge-weighted graph attention (standard GAT)
      (Without earliness-aware loss is a loss setting: loss.use_time_loss=false)
    The 'Standard GAT + MLP' row uses edge_prior=False, quantum_block="none".
    """

    def __init__(self, in_dim=4, hidden=64, layers=3, heads=5, qubits=4, q_layers=2,
                 quantum_block="circuit", quantum_backend="torch", fusion="gate",
                 edge_prior=True, dropout=0.1, seed=0):
        super().__init__()
        self.encoder = MultiScaleGATEncoder(in_dim, hidden, layers, heads, dropout, edge_prior)
        self.use_q = quantum_block != "none"
        self.fusion = fusion
        if self.use_q:
            self.to_angles = nn.Linear(hidden, qubits)                      # W_q, b_q (Eq. 13)
            self.block = make_block(quantum_block, qubits, q_layers, quantum_backend, seed)
            self.from_q = nn.Linear(qubits, hidden)                         # W_o, b_o (Eq. 18)
            if fusion == "gate":
                self.gate = nn.Linear(2 * hidden, hidden)                   # W_g, b_g (Eq. 19)
                head_in = hidden
            else:
                head_in = 2 * hidden
        else:
            head_in = hidden
        self.head = nn.Linear(head_in, 1)                                   # W_c, b_c (Eq. 22)

    def embed(self, X, A):
        z_g = self.encoder(X, A)
        if not self.use_q:
            return z_g, {}
        theta = math.pi * torch.tanh(self.to_angles(z_g))                   # Eq. 14
        q = self.block(theta)                                               # Eqs. 15-17
        z_q = self.from_q(q)                                                # Eq. 18
        if self.fusion == "gate":
            g = torch.sigmoid(self.gate(torch.cat([z_g, z_q], -1)))         # Eqs. 19-20
            z_f = g * z_q + (1 - g) * z_g                                   # Eq. 21
        else:
            g, z_f = None, torch.cat([z_g, z_q], -1)
        return z_f, {"z_G": z_g, "theta": theta, "q": q, "z_Q": z_q, "gate": g}

    def forward(self, X, A, W=None):
        z_f, _ = self.embed(X, A)
        return self.head(z_f).squeeze(-1)                                   # logit u


def build_model(mcfg, seed=0):
    return SCAQFMSGT(in_dim=mcfg.get("in_dim", 4), hidden=mcfg.get("hidden", 64), layers=mcfg.get("layers", 3),
                     heads=mcfg.get("heads", 5), qubits=mcfg.get("qubits", 4), q_layers=mcfg.get("q_layers", 2),
                     quantum_block=mcfg.get("quantum_block", "circuit"),
                     quantum_backend=mcfg.get("quantum_backend", "torch"), fusion=mcfg.get("fusion", "gate"),
                     edge_prior=mcfg.get("edge_prior", True), dropout=mcfg.get("dropout", 0.1), seed=seed)
