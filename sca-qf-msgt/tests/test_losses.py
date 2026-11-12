"""Checks of the soft first-crossing lead time (Eqs. 25a-25c)."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sca_qf_msgt.training.losses import focal_loss, hard_lead_time, soft_lead_time, time_loss


def test_soft_tends_to_hard_as_temperature_falls():
    s = torch.tensor([0.1, 0.3, 0.7, 0.9]); dt = torch.tensor([10.0, 7.0, 4.0, 1.0])
    assert abs(float(soft_lead_time(s, dt, 0.5, 1e-4)) - float(hard_lead_time(s, dt, 0.5))) < 1e-3


def test_no_alarm_gives_zero_lead_time():
    s = torch.tensor([0.1, 0.2]); dt = torch.tensor([5.0, 2.0])
    assert float(soft_lead_time(s, dt, 0.5, 1e-4)) < 1e-6


def test_time_loss_has_gradient_everywhere():
    s = torch.tensor([0.40, 0.45, 0.48], requires_grad=True)
    loss = time_loss(s, torch.zeros(3, dtype=torch.long), torch.ones(3), torch.tensor([9.0, 5.0, 2.0]), 0.42)
    loss.backward()
    assert s.grad.abs().sum() > 0


def test_focal_loss_positive():
    assert float(focal_loss(torch.randn(10), (torch.rand(10) > 0.5).float())) > 0
