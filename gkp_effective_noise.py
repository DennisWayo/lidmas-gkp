import math
# -----------------------------
# GKP proxy maps + coupling
# -----------------------------
def squeezing_db_to_p_dephase(s_db: float, scale: float = 0.06) -> float:
    """
    Placeholder map: higher squeezing (dB) -> lower effective logical phase error.
    Tunable proxy for architecture sweeps.
    """
    return float(min(0.5, max(0.0, scale * math.exp(-0.35 * s_db))))


def effective_erasure(p_base: float, pz: float, alpha: float = 6.0) -> float:
    """
    Loss–squeezing coupling proxy:
    finite squeezing (modeled by pz) increases effective loss/erasure sensitivity.

    - p_base: baseline loss-like erasure probability (dominant photonics impairment)
    - pz: effective dephasing proxy from squeezing map
    - alpha: coupling strength (tune; start modest)
    """
    return float(min(1.0, max(0.0, p_base * (1.0 + alpha * pz))))

