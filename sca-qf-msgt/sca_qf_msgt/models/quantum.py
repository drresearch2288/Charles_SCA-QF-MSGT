"""Four-qubit feature transformation (Section 3.8, Eqs. 15-17) and the parameter-matched
classical controls of Section 4.6.1.

Every block maps the four angles theta (Eq. 14) to four outputs q.

* QuantumCircuit, backend "torch": exact state-vector simulation in PyTorch. The state
  is real because R_Y and CNOT are real, so the 16 amplitudes are simulated directly;
  autograd gives the exact gradient, identical to the parameter-shift rule.
* QuantumCircuit, backend "qiskit": the same circuit built in Qiskit, evaluated with an
  exact estimator and wrapped by Qiskit Machine Learning's TorchConnector (parameter-shift
  gradients). Slower; tests/test_quantum.py checks that both backends agree.
"""
import math

import torch
import torch.nn as nn

RING = lambda n: [(i, (i + 1) % n) for i in range(n)]        # CNOT ring q1->q2->...->qn->q1


def _ry(state, theta, qubit, n):
    """Apply R_Y(theta) on `qubit`. state: (B, 2, ..., 2); theta: (B,) or scalar."""
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    st = state.movedim(qubit + 1, 1)
    a0, a1 = st[:, 0], st[:, 1]
    shape = (-1,) + (1,) * (n - 1)
    c = c.reshape(shape) if c.dim() else c
    s = s.reshape(shape) if s.dim() else s
    new = torch.stack([c * a0 - s * a1, s * a0 + c * a1], dim=1)
    return new.movedim(1, qubit + 1)


def _cnot(state, ctrl, tgt):
    """CNOT: flip the `tgt` axis where `ctrl` = 1 (a permutation of amplitudes)."""
    s0 = state.select(ctrl + 1, 0)
    s1 = state.select(ctrl + 1, 1)
    t_axis = tgt + 1 if tgt < ctrl else tgt      # the ctrl axis was removed by select
    return torch.stack([s0, s1.flip(t_axis)], dim=ctrl + 1)


def simulate(theta, phi, rot="Y"):
    """theta: (B, n) encoding angles; phi: (D, n) variational angles. Returns <Z_i>, (B, n)."""
    B, n = theta.shape
    state = torch.zeros((B,) + (2,) * n, dtype=theta.dtype, device=theta.device)
    state[(slice(None),) + (0,) * n] = 1.0
    for i in range(n):                                      # Eq. 15
        state = _ry(state, theta[:, i], i, n)
    for d in range(phi.shape[0]):                           # Eq. 16
        for i in range(n):
            if rot == "Y":
                state = _ry(state, phi[d, i], i, n)
            else:                                           # R_Z, kept only for the check in tests
                state = _rz_real_check(state, phi[d, i], i, n)
        for c, t in RING(n):
            state = _cnot(state, c, t)
    p = state.reshape(B, -1) ** 2 if rot == "Y" else state.reshape(B, -1).abs() ** 2
    z = []
    for i in range(n):                                      # Eq. 17
        bits = ((torch.arange(2 ** n, device=theta.device) >> (n - 1 - i)) & 1).to(theta.dtype)
        z.append((p * (1 - 2 * bits)).sum(-1))
    return torch.stack(z, -1)


def _rz_real_check(state, phi, qubit, n):
    """R_Z(phi) = diag(e^{-i phi/2}, e^{i phi/2}); applied on a complex copy."""
    st = state.to(torch.complex64).movedim(qubit + 1, 1)
    ph = torch.exp(-0.5j * phi.to(torch.complex64))
    new = torch.stack([ph * st[:, 0], ph.conj() * st[:, 1]], 1)
    return new.movedim(1, qubit + 1)


class QuantumCircuit(nn.Module):
    def __init__(self, n_qubits=4, n_layers=2, backend="torch"):
        super().__init__()
        self.n, self.D, self.backend = n_qubits, n_layers, backend
        if backend == "torch":
            self.phi = nn.Parameter(torch.empty(n_layers, n_qubits).uniform_(-math.pi, math.pi))
        elif backend == "qiskit":
            self.qnn = _qiskit_layer(n_qubits, n_layers)
        else:
            raise ValueError(backend)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, theta):
        if self.backend == "torch":
            return simulate(theta, self.phi)
        return self.qnn(theta)


def build_qiskit_circuit(n_qubits=4, n_layers=2):
    from qiskit import QuantumCircuit as QC
    from qiskit.circuit import ParameterVector
    x = ParameterVector("theta", n_qubits)
    w = ParameterVector("phi", n_qubits * n_layers)
    qc = QC(n_qubits)
    for i in range(n_qubits):
        qc.ry(x[i], i)
    for d in range(n_layers):
        for i in range(n_qubits):
            qc.ry(w[d * n_qubits + i], i)
        for c, t in RING(n_qubits):
            qc.cx(c, t)
    return qc, x, w


def _qiskit_layer(n_qubits, n_layers):
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_machine_learning.connectors import TorchConnector
    from qiskit_machine_learning.neural_networks import EstimatorQNN
    from qiskit_machine_learning.gradients import ParamShiftEstimatorGradient
    from qiskit.primitives import StatevectorEstimator
    qc, x, w = build_qiskit_circuit(n_qubits, n_layers)
    # Qiskit labels are little-endian: the rightmost character is qubit 0
    obs = [SparsePauliOp("I" * (n_qubits - 1 - i) + "Z" + "I" * i) for i in range(n_qubits)]
    est = StatevectorEstimator()
    qnn = EstimatorQNN(circuit=qc, observables=obs, input_params=list(x), weight_params=list(w),
                       estimator=est, gradient=ParamShiftEstimatorGradient(est), input_gradients=True,
                       default_precision=0.0)                     # exact expectation values
    init = torch.empty(n_qubits * n_layers).uniform_(-math.pi, math.pi).numpy()
    return TorchConnector(qnn, initial_weights=init)


class BottleneckMLP(nn.Module):
    """(a) 4-1-4 MLP without biases, q = tanh(W2 tanh(W1 theta)); 8 parameters."""
    def __init__(self, n=4):
        super().__init__()
        self.l1 = nn.Linear(n, 1, bias=False); self.l2 = nn.Linear(1, n, bias=False)

    def forward(self, t):
        return torch.tanh(self.l2(torch.tanh(self.l1(t))))


class RandomFeatures(nn.Module):
    """(b) q = cos(s * (Omega theta) + beta), Omega fixed Gaussian, s and beta trainable; 8 parameters."""
    def __init__(self, n=4, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("omega", torch.randn(n, n, generator=g))
        self.s = nn.Parameter(torch.ones(n)); self.beta = nn.Parameter(torch.zeros(n))

    def forward(self, t):
        return torch.cos(self.s * (t @ self.omega.T) + self.beta)


class WideMLP(nn.Module):
    """Generous control: 4-4-4 MLP with biases, 40 parameters."""
    def __init__(self, n=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n, n), nn.Tanh(), nn.Linear(n, n), nn.Tanh())

    def forward(self, t):
        return self.net(t)


class Identity(nn.Module):
    """No module: theta is passed directly to Eq. 18."""
    def forward(self, t):
        return t


def make_block(name, n_qubits=4, n_layers=2, backend="torch", seed=0):
    return {"circuit": lambda: QuantumCircuit(n_qubits, n_layers, backend),
            "bottleneck_mlp": lambda: BottleneckMLP(n_qubits),
            "random_features": lambda: RandomFeatures(n_qubits, seed),
            "wide_mlp": lambda: WideMLP(n_qubits),
            "identity": Identity}[name]()
