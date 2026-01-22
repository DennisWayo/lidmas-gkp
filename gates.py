import numpy as np
from typing import Tuple, Optional
# -----------------------------
# Basic gates / states
# -----------------------------
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
Sd = S.conj().T
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

# 2-qubit CNOT (control is qubit 0, target is qubit 1) for ordering (q0 ⊗ q1)
CNOT = np.array(
    [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 1],
     [0, 0, 1, 0]], dtype=complex
)

# Projectors (X basis)
PROJ_XP = 0.5 * np.array([[1, 1], [1, 1]], dtype=complex)   # |+><+|
PROJ_XM = 0.5 * np.array([[1, -1], [-1, 1]], dtype=complex) # |-><-|


def kron(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.kron(a, b)


def apply_unitary(rho: np.ndarray, U: np.ndarray) -> np.ndarray:
    return U @ rho @ U.conj().T


def partial_trace_keep_first(rho2: np.ndarray) -> np.ndarray:
    """Trace out second qubit, keep first; assumes ordering (q0 ⊗ q1)."""
    out = np.zeros((2, 2), dtype=complex)
    for q1 in (0, 1):
        idx = [0 * 2 + q1, 1 * 2 + q1]
        out += rho2[np.ix_(idx, idx)]
    return out


def measure_projective_twoqubit(
    rho2: np.ndarray,
    which_qubit: int,
    proj: np.ndarray
) -> Tuple[float, Optional[np.ndarray]]:
    """
    Projective measurement on one qubit (0 or 1) with projector 'proj' (2x2).
    Returns probability and post-measurement 2-qubit state (normalized), or None if prob ~ 0.
    """
    P = kron(proj, I) if which_qubit == 0 else kron(I, proj)
    p = float(np.real(np.trace(P @ rho2)))
    if p < 1e-15:
        return 0.0, None
    rho_post = (P @ rho2 @ P.conj().T) / p
    return p, rho_post


def pure_state_density(psi: np.ndarray) -> np.ndarray:
    psi = psi.reshape(2, 1)
    return psi @ psi.conj().T


def fidelity_to_pure(rho: np.ndarray, phi: np.ndarray) -> float:
    """F(ρ, |phi>) = <phi|ρ|phi>."""
    phi = phi.reshape(2, 1)
    return float(np.real((phi.conj().T @ rho @ phi)[0, 0]))

