
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List

from gates import X, Y, Z
# -----------------------------
# Noise models (logical approx)
# -----------------------------
@dataclass
class LogicalNoise:
    """
    Simple logical-noise knobs:
    - p_dephase: Z errors
    - p_depol: depolarizing (X,Y,Z)
    - p_erasure: erasure (flagged loss) -> we treat as 'attempt fails' (like heralded loss)
    """
    p_dephase: float = 0.0
    p_depol: float = 0.0
    p_erasure: float = 0.0

    def validate(self) -> None:
        for name, p in [
            ("p_dephase", self.p_dephase),
            ("p_depol", self.p_depol),
            ("p_erasure", self.p_erasure),
        ]:
            if not (0.0 <= p <= 1.0):
                raise ValueError(f"{name} must be in [0,1], got {p}.")


def apply_dephasing(rho: np.ndarray, p: float) -> np.ndarray:
    return (1 - p) * rho + p * (Z @ rho @ Z)


def apply_depolarizing(rho: np.ndarray, p: float) -> np.ndarray:
    if p == 0.0:
        return rho
    return (1 - p) * rho + (p / 3.0) * (X @ rho @ X + Y @ rho @ Y + Z @ rho @ Z)


def maybe_erasure(rng: np.random.Generator, p_erasure: float) -> bool:
    return rng.random() < p_erasure


def apply_logical_noise_1q(rng: np.random.Generator, rho: np.ndarray, noise: LogicalNoise) -> Tuple[np.ndarray, bool]:
    """
    Applies 1-qubit noise. If erasure occurs, returns (rho, True) meaning 'flagged failure'.
    Otherwise returns noisy rho and False.
    """
    noise.validate()
    if maybe_erasure(rng, noise.p_erasure):
        return rho, True
    rho = apply_dephasing(rho, noise.p_dephase)
    rho = apply_depolarizing(rho, noise.p_depol)
    return rho, False
