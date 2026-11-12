"""Edge-weighted multi-head graph attention and multi-scale readout (Section 3.7, Eqs. 7-12).

Graphs are dense and batched: X (B, n, F), A (B, n, n) with zero diagonal. The
neighbourhood N(i) is {j : A_ij != 0} plus i itself (self-loop with prior 0), so a
lead with no retained edge still keeps its own features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeWeightedGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=5, dropout=0.0, negative_slope=0.2):
        super().__init__()
        self.heads, self.out_dim = heads, out_dim
        self.W = nn.Linear(in_dim, heads * out_dim, bias=False)          # W^m, all heads
        self.a_src = nn.Parameter(torch.empty(heads, out_dim))           # a^m, first half
        self.a_dst = nn.Parameter(torch.empty(heads, out_dim))           # a^m, second half
        self.lam = nn.Parameter(torch.tensor(1.0))                       # lambda, Eq. 8
        # Eq. 11 concatenates M heads; with M = 5 and width 64 the concatenation (5 x 64)
        # is projected back to the working width so every layer outputs 64 features.
        self.proj = nn.Linear(heads * out_dim, out_dim)
        self.drop = nn.Dropout(dropout)
        self.slope = negative_slope
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src); nn.init.xavier_uniform_(self.a_dst)

    def forward(self, h, A, edge_prior=True):
        B, n, _ = h.shape
        Wh = self.W(h).view(B, n, self.heads, self.out_dim)                       # (B, n, M, d)
        e = (Wh * self.a_src).sum(-1).unsqueeze(2) + (Wh * self.a_dst).sum(-1).unsqueeze(1)  # (B, i, j, M)
        e = F.leaky_relu(e, self.slope)                                           # Eq. 7
        if edge_prior:
            e = e + self.lam * A.unsqueeze(-1)                                    # Eq. 8
        eye = torch.eye(n, dtype=torch.bool, device=h.device)
        mask = (A != 0) | eye
        e = e.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        alpha = self.drop(torch.softmax(e, dim=2))                                # Eq. 9, over j
        out = torch.einsum("bijm,bjmd->bimd", alpha, Wh)                          # Eq. 10 per head
        out = F.elu(out).reshape(B, n, self.heads * self.out_dim)                 # Eq. 11, concat
        return self.proj(out), alpha


class MultiScaleGATEncoder(nn.Module):
    """L stacked layers; z_G = sum_l beta_l * mean_i h_i^(l) (Eq. 12)."""

    def __init__(self, in_dim=4, hidden=64, layers=3, heads=5, dropout=0.1, edge_prior=True):
        super().__init__()
        self.layers = nn.ModuleList(
            [EdgeWeightedGATLayer(in_dim if l == 0 else hidden, hidden, heads, dropout) for l in range(layers)])
        self.beta = nn.Parameter(torch.full((layers,), 1.0 / layers))
        self.edge_prior = edge_prior

    def forward(self, X, A, return_attention=False):
        """edge_prior=False gives the standard GAT of the ablation (Eq. 7 only, same graph)."""
        h, z, att = X, 0.0, []
        for l, layer in enumerate(self.layers):
            h, a = layer(h, A, self.edge_prior)
            z = z + self.beta[l] * h.mean(dim=1)               # READOUT averages over present nodes
            att.append(a)
        return (z, att) if return_attention else z
