"""Deep baselines of Table 5. They see the same beats as SCA-QF-MSGT and are trained
with the same loss, splits and protocol. All return one logit per beat.

* ResNetAttention : 1-D residual CNN over the 12 x 306 beat window with attention pooling.
* MultiScaleGCN   : GCN on the same beat graphs (|A| as edge weights) with multi-scale readout.
* CNNLSTM         : per-lead 1-D CNN features read by an LSTM across the 12 leads.
ResNetAttention and CNNLSTM need beat windows (prepare_data.py --keep-windows).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResBlock(nn.Module):
    def __init__(self, c_in, c_out, stride):
        super().__init__()
        self.c1 = nn.Conv1d(c_in, c_out, 7, stride, 3, bias=False); self.b1 = nn.BatchNorm1d(c_out)
        self.c2 = nn.Conv1d(c_out, c_out, 7, 1, 3, bias=False); self.b2 = nn.BatchNorm1d(c_out)
        self.skip = nn.Sequential() if (c_in == c_out and stride == 1) else nn.Sequential(
            nn.Conv1d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm1d(c_out))

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        return F.relu(self.b2(self.c2(y)) + self.skip(x))


class ResNetAttention(nn.Module):
    needs_windows = True

    def __init__(self, leads=12, width=64):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(leads, width // 2, 15, 2, 7, bias=False),
                                  nn.BatchNorm1d(width // 2), nn.ReLU())
        self.blocks = nn.Sequential(_ResBlock(width // 2, width, 2), _ResBlock(width, width, 2),
                                    _ResBlock(width, width, 1))
        self.att = nn.Linear(width, 1)
        self.head = nn.Linear(width, 1)

    def forward(self, X, A, W):
        h = self.blocks(self.stem(W)).transpose(1, 2)            # (B, T', C)
        a = torch.softmax(self.att(h), dim=1)
        return self.head((a * h).sum(1)).squeeze(-1)


class MultiScaleGCN(nn.Module):
    needs_windows = False

    def __init__(self, in_dim=4, hidden=64, layers=3):
        super().__init__()
        self.lins = nn.ModuleList([nn.Linear(in_dim if l == 0 else hidden, hidden) for l in range(layers)])
        self.beta = nn.Parameter(torch.full((layers,), 1.0 / layers))
        self.head = nn.Linear(hidden, 1)

    def forward(self, X, A, W=None):
        n = A.shape[1]
        Ah = A.abs() + torch.eye(n, device=A.device)
        d = Ah.sum(-1).clamp_min(1e-6).rsqrt()
        An = d.unsqueeze(-1) * Ah * d.unsqueeze(-2)              # symmetric normalisation
        h, z = X, 0.0
        for l, lin in enumerate(self.lins):
            h = F.relu(An @ lin(h))
            z = z + self.beta[l] * h.mean(1)
        return self.head(z).squeeze(-1)


class CNNLSTM(nn.Module):
    needs_windows = True

    def __init__(self, hidden=64):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv1d(1, 16, 9, 2, 4), nn.ReLU(), nn.Conv1d(16, 32, 9, 2, 4), nn.ReLU(),
                                 nn.AdaptiveAvgPool1d(8))
        self.lstm = nn.LSTM(32 * 8, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, X, A, W):
        B, L, T = W.shape
        f = self.cnn(W.reshape(B * L, 1, T)).reshape(B, L, -1)  # per-lead features
        _, (h, _) = self.lstm(f)
        return self.head(h[-1]).squeeze(-1)


BASELINES = {"resnet_attention": ResNetAttention, "multiscale_gcn": MultiScaleGCN, "cnn_lstm": CNNLSTM}
