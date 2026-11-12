"""Checks of the circuit (Section 3.8): exact simulation, R_Z untrainability, gradients."""
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sca_qf_msgt.models.quantum import (BottleneckMLP, QuantumCircuit, RandomFeatures, WideMLP, simulate)


def test_outputs_in_range_and_shape():
    q = simulate(torch.randn(7, 4), torch.randn(2, 4))
    assert q.shape == (7, 4) and q.abs().max() <= 1 + 1e-6


def test_all_zero_angles_leave_ground_state():
    # |0000> is unchanged by R_Y(0) and by the CNOT ring, so every <Z_i> = +1
    q = simulate(torch.zeros(1, 4), torch.zeros(2, 4))
    assert torch.allclose(q, torch.ones(1, 4))


def test_rz_variational_angles_get_no_gradient():
    th = torch.randn(5, 4, dtype=torch.float64)
    phi = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    simulate(th, phi, rot="Z").sum().backward()
    assert phi.grad.abs().max() < 1e-5          # R_Z before a Z-basis readout is invisible


def test_ry_gradient_matches_parameter_shift():
    th = torch.randn(3, 4, dtype=torch.float64)
    phi = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    simulate(th, phi).sum().backward()
    ps = torch.zeros(2, 4, dtype=torch.float64)
    for d in range(2):
        for i in range(4):
            e = torch.zeros(2, 4, dtype=torch.float64); e[d, i] = math.pi / 2
            ps[d, i] = 0.5 * (simulate(th, phi.detach() + e).sum() - simulate(th, phi.detach() - e).sum())
    assert torch.allclose(phi.grad, ps, atol=1e-8)


def test_parameter_counts():
    assert QuantumCircuit().n_params == 8
    assert sum(p.numel() for p in BottleneckMLP().parameters()) == 8
    assert sum(p.numel() for p in RandomFeatures().parameters()) == 8
    assert sum(p.numel() for p in WideMLP().parameters()) == 40


def test_qiskit_backend_agrees():
    pytest.importorskip("qiskit_machine_learning")
    th = torch.randn(3, 4)
    qk = QuantumCircuit(backend="qiskit")
    phi = qk.qnn.weight.detach().double().reshape(2, 4)
    t = th.clone().requires_grad_(True)
    out = qk(t); out.sum().backward()
    t2 = th.double().requires_grad_(True)
    ref = simulate(t2, phi); ref.sum().backward()
    assert torch.allclose(out.double(), ref, atol=1e-5)
    assert torch.allclose(t.grad.double(), t2.grad, atol=1e-5)
