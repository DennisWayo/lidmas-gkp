
# -----------------------------
# Outer code abstraction (surface code)
# -----------------------------
def surface_code_logical_error(p_phys: float, d: int, p_th: float = 0.01, A: float = 0.5) -> float:
    """
    Surface-code logical error scaling-law abstraction.

    p_L ~ A * (p_phys / p_th)^((d+1)/2), for p_phys < p_th

    - p_phys: effective physical Pauli error probability per logical operation
    - d: code distance (odd integer typical)
    - p_th: threshold (order ~1% is a common ballpark)
    - A: prefactor (order ~0.1–1)

    NOTE:
    This is a standard architecture-level abstraction; it avoids explicit syndrome/decoder simulation.
    """
    if d < 1:
        raise ValueError("Surface code distance d must be >= 1.")
    if p_phys <= 0.0:
        return 0.0
    if p_phys >= p_th:
        return 0.5  # above threshold -> essentially random logical output (maximally pessimistic cap)
    exponent = (d + 1) / 2.0
    return float(A * (p_phys / p_th) ** exponent)


def apply_outer_code_to_fidelity(
    f_in: float,
    p_phys: float,
    d: int,
    p_th: float = 0.01,
    A: float = 0.5
) -> float:
    """
    Convert injected fidelity to an outer-code-adjusted logical fidelity by mixing with a random state
    according to p_L (a conservative approximation).

    If a logical failure occurs, we approximate output fidelity -> 1/2 (random qubit wrt target).
    """
    pL = surface_code_logical_error(p_phys=p_phys, d=d, p_th=p_th, A=A)
    return float((1.0 - pL) * f_in + pL * 0.5)

