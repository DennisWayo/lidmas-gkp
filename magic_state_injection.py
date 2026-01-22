import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List

from gates import (
    I, X, Y, Z,
    H, S, Sd, T,
    CNOT,
    PROJ_XP, PROJ_XM,
    kron,
    apply_unitary,
    partial_trace_keep_first,
    measure_projective_twoqubit,
    pure_state_density,
    fidelity_to_pure,
)

from logical_noise import LogicalNoise, apply_logical_noise_1q
from outer_code import apply_outer_code_to_fidelity
# -----------------------------
# Magic state injection (T gadget)
# -----------------------------
def magic_state_A() -> np.ndarray:
    """|A> = T|+>"""
    plus = (1 / np.sqrt(2)) * np.array([1, 1], dtype=complex)
    return T @ plus


@dataclass
class InjectionResult:
    success: bool
    erased: bool
    outcome: int
    rho_out: Optional[np.ndarray]
    fidelity: Optional[float]


def injection_attempt_T(
    rng: np.random.Generator,
    psi_in: np.ndarray,
    noise_data: LogicalNoise,
    noise_anc: LogicalNoise,
    noise_after: LogicalNoise,
) -> InjectionResult:
    """
    One attempt at T injection using |A>=T|+> ancilla.
    Ordering: data qubit is q0, ancilla is q1.

    Circuit (logical):
      1) Prepare data in |psi>, anc in |A>
      2) Apply CNOT (data control -> anc target)
      3) Measure anc in X basis
      4) Apply Clifford feedforward (we use S or Sd depending on branch)

    Architectural note:
      - Depending on outcome, you may effectively get T or T† up to Cliffords.
      - We label "success" as one branch; the other is treated as requiring an additional corrective round,
        hence repeat-until-success (RUS) abstraction.
    """
    rho_data = pure_state_density(psi_in)
    rho_anc = pure_state_density(magic_state_A())

    # Pre-noise
    rho_data, erased_d = apply_logical_noise_1q(rng, rho_data, noise_data)
    rho_anc, erased_a = apply_logical_noise_1q(rng, rho_anc, noise_anc)
    if erased_d or erased_a:
        return InjectionResult(success=False, erased=True, outcome=-1, rho_out=None, fidelity=None)

    rho2 = kron(rho_data, rho_anc)

    # Entangle
    rho2 = apply_unitary(rho2, CNOT)

    # Measure ancilla (q1) in X basis
    p_plus, rho_plus = measure_projective_twoqubit(rho2, which_qubit=1, proj=PROJ_XP)
    p_minus, rho_minus = measure_projective_twoqubit(rho2, which_qubit=1, proj=PROJ_XM)

    p = np.array([p_plus, p_minus], dtype=float)
    if p.sum() <= 0:
        return InjectionResult(success=False, erased=True, outcome=-1, rho_out=None, fidelity=None)
    p = p / p.sum()
    outcome = int(rng.choice([0, 1], p=p))

    rho_post = rho_plus if outcome == 0 else rho_minus
    if rho_post is None:
        return InjectionResult(success=False, erased=True, outcome=outcome, rho_out=None, fidelity=None)

    # Reduce to data (q0)
    rho_out = partial_trace_keep_first(rho_post)

    # Clifford feedforward (architecture-style abstraction)
    if outcome == 0:
        rho_out = apply_unitary(rho_out, Sd)
        success = True
    else:
        rho_out = apply_unitary(rho_out, S)
        success = False

    # Post-noise on output
    rho_out, erased_out = apply_logical_noise_1q(rng, rho_out, noise_after)
    if erased_out:
        return InjectionResult(success=False, erased=True, outcome=outcome, rho_out=None, fidelity=None)

    # Fidelity vs ideal T|psi>
    phi = T @ psi_in
    fid = fidelity_to_pure(rho_out, phi)

    return InjectionResult(success=success, erased=False, outcome=outcome, rho_out=rho_out, fidelity=fid)


@dataclass
class SweepConfig:
    squeezing_db_values: List[float]
    loss_base_values: List[float]
    distances: List[int]
    max_rounds: int = 10
    trials: int = 5000  # increase for smoother curves; keep reasonable for laptop runs
    seed: int = 42

    # Noise floor not tied to squeezing
    p_depol_data: float = 0.01
    p_depol_anc: float = 0.02
    p_depol_after: float = 0.01

    # Coupling strength
    alpha_loss_squeezing: float = 6.0

    # Outer code scaling params
    p_th: float = 0.01
    A_prefactor: float = 0.5


def run_repeat_until_success_with_outer_code(
    rng: np.random.Generator,
    psi_in: np.ndarray,
    max_rounds: int,
    trials: int,
    noise_data: LogicalNoise,
    noise_anc: LogicalNoise,
    noise_after: LogicalNoise,
    surface_distance: int,
    p_th: float,
    A_prefactor: float,
) -> Dict[str, float]:
    """
    RUS loop + injected fidelity + outer code adjusted logical fidelity.
    """
    erased = 0
    succeeded = 0
    total_rounds = 0

    fidelities_injected = []
    fidelities_logical = []

    # Define an effective "physical Pauli" rate that the outer code suppresses.
    # (Conservative: include both dephasing and depolarizing contributions.)
    def p_phys_eff(n: LogicalNoise) -> float:
        # Effective Pauli error seen by the outer code after GKP + Clifford protection
        w_z = 0.3  # residual Z-type errors after GKP EC
        w_p = 0.1  # residual depolarizing errors after stabilizer cycles
        return float(min(1.0, max(0.0, w_z * n.p_dephase + w_p * n.p_depol)))

    p_eff = p_phys_eff(noise_data)  # you can also combine data+after if desired

    for _ in range(trials):
        r = 0
        while r < max_rounds:
            r += 1
            res = injection_attempt_T(rng, psi_in, noise_data, noise_anc, noise_after)
            if res.erased:
                erased += 1
                break
            if res.success:
                succeeded += 1
                total_rounds += r
                f_inj = float(res.fidelity if res.fidelity is not None else 0.0)
                f_log = apply_outer_code_to_fidelity(
                    f_in=f_inj,
                    p_phys=p_eff,
                    d=surface_distance,
                    p_th=p_th,
                    A=A_prefactor
                )
                fidelities_injected.append(f_inj)
                fidelities_logical.append(f_log)
                break
        # if capped out without success: treat as failure (no fidelity recorded)

    success_prob = succeeded / trials
    erasure_rate = erased / trials
    avg_rounds = (total_rounds / succeeded) if succeeded > 0 else float("inf")
    avg_f_inj = float(np.mean(fidelities_injected)) if fidelities_injected else 0.0
    avg_f_log = float(np.mean(fidelities_logical)) if fidelities_logical else 0.0

    return {
        "trials": float(trials),
        "success_prob_within_cap": float(success_prob),
        "erasure_rate": float(erasure_rate),
        "avg_rounds_given_success": float(avg_rounds),
        "avg_fidelity_injected_given_success": float(avg_f_inj),
        "avg_fidelity_logical_given_success": float(avg_f_log),
        "p_phys_eff": float(p_eff),
    }
