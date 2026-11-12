"""Training objective (Section 3.10): focal loss (Eq. 24), soft first-crossing lead-time
loss (Eqs. 25, 25a-25c) and L2 regulariser (Eq. 26). Matches Listing S1."""
import torch
import torch.nn.functional as F


def focal_loss(logits, y, alpha=0.25, gamma=2.0):
    """Equation (24), with alpha weighting the positive class and 1 - alpha the negative."""
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")      # -log(p_t)
    p_t = p * y + (1 - p) * (1 - y)
    a_t = alpha * y + (1 - alpha) * (1 - y)
    return (a_t * (1 - p_t) ** gamma * ce).mean()


def soft_lead_time(scores, dt_hours, delta, temp=0.05):
    """scores, dt_hours: (K,) for one positive sequence in time order (earliest first)."""
    c = torch.sigmoid((scores - delta) / temp)                                # Eq. 25a
    survive = torch.cumprod(torch.cat([c.new_ones(1), 1 - c[:-1]]), dim=0)
    omega = c * survive                                                        # Eq. 25b
    return (omega * dt_hours).sum()                                            # Eq. 25c; no alarm -> 0 h


def hard_lead_time(scores, dt_hours, delta):
    idx = torch.nonzero(scores > delta)
    return dt_hours[idx[0, 0]] if len(idx) else scores.new_zeros(())


def time_loss(ecg_scores, ecg_seq, ecg_y, ecg_dt, delta, target=6.0, temp=0.05):
    """Equation (25) averaged over the positive sequences of the batch.
    ECGs within each sequence are already in time order (collate keeps the order)."""
    losses = []
    for s in torch.unique(ecg_seq):
        m = ecg_seq == s
        if ecg_y[m].max() < 1:
            continue
        t_lead = soft_lead_time(ecg_scores[m], ecg_dt[m], delta, temp)
        losses.append(torch.clamp(target - t_lead, min=0) ** 2)
    return torch.stack(losses).mean() if losses else ecg_scores.new_zeros(())


def l2_penalty(model):
    return sum((p ** 2).sum() for n, p in model.named_parameters() if p.requires_grad and "bias" not in n)
