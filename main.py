#!/usr/bin/env python3
"""
Investigating Fault-Tolerant Logical Magic-State Preparation in GKP-Based Photonic Architectures
------------------------------------------------------------------------------------------------
Milestone 1→4: Logical-layer simulator for T-gate magic-state injection with noise + repeat-until-success,
plus (i) loss–squeezing coupling, (ii) a real outer-code abstraction (surface code scaling law),
(iii) parameter sweeps.

Modeling philosophy (architecture-first, honest abstractions):
- Inner (bosonic) layer: GKP-like behavior is approximated by an effective Pauli channel (dephasing + depolarizing),
  and a flagged erasure channel (loss).
- Crucially, loss is coupled to finite squeezing via a simple, tunable effective_erasure() map to reflect
  the fact that finite squeezing can amplify sensitivity to loss in practice.
- Outer (qubit) layer: we use a standard surface-code logical error scaling law (not a toy invented code),
  without implementing explicit stabilizers/decoders. This keeps the study in “architecture tradeoffs” territory.

"""

from __future__ import annotations
from typing import Tuple, Dict, Optional, List
from gates import *
from logical_noise import LogicalNoise
from gkp_effective_noise import (
    squeezing_db_to_p_dephase,
    effective_erasure,
)
from outer_code import apply_outer_code_to_fidelity
from magic_state_injection import (
    SweepConfig,
    run_repeat_until_success_with_outer_code,
)

from analysis_plots import (
    write_csv,
    plot_metric_vs_squeezing_by_loss,
    plot_sensitivity_heatmaps,
    plot_phase_boundary,
)

def main():
    # Deterministic input state for comparability across sweeps
    rng0 = np.random.default_rng(123)
    v = rng0.normal(size=2) + 1j * rng0.normal(size=2)
    psi = v / np.linalg.norm(v)

    cfg = SweepConfig(
        squeezing_db_values=list(np.arange(8.0, 16.5, 0.5)),
        loss_base_values=[0.005, 0.01, 0.015, 0.02, 0.025, 0.03],
        distances=[1, 3, 5, 7],
        max_rounds=10,
        trials=5000,
        seed=42,
    )

    rng = np.random.default_rng(cfg.seed)

    rows: List[Dict[str, float]] = []

    for loss_base in cfg.loss_base_values:
        for s_db in cfg.squeezing_db_values:
            pz = squeezing_db_to_p_dephase(float(s_db))
            # Coupled erasure probabilities
            pE_data = effective_erasure(loss_base, pz, alpha=cfg.alpha_loss_squeezing)
            pE_anc  = effective_erasure(min(1.0, 1.5 * loss_base), pz, alpha=cfg.alpha_loss_squeezing)  # anc slightly worse
            pE_after = 0.0  # post-stage erasure is often effectively heralded earlier; keep 0 for now

            noise_data = LogicalNoise(p_dephase=pz, p_depol=cfg.p_depol_data, p_erasure=pE_data)
            noise_anc  = LogicalNoise(p_dephase=pz, p_depol=cfg.p_depol_anc,  p_erasure=pE_anc)
            noise_after= LogicalNoise(p_dephase=pz, p_depol=cfg.p_depol_after, p_erasure=pE_after)

            for d in cfg.distances:
                stats = run_repeat_until_success_with_outer_code(
                    rng=rng,
                    psi_in=psi,
                    max_rounds=cfg.max_rounds,
                    trials=cfg.trials,
                    noise_data=noise_data,
                    noise_anc=noise_anc,
                    noise_after=noise_after,
                    surface_distance=int(d),
                    p_th=cfg.p_th,
                    A_prefactor=cfg.A_prefactor,
                )

                row = {
                    "squeezing_db": float(s_db),
                    "pZ_proxy": float(pz),
                    "loss_base": float(loss_base),
                    "pE_data": float(pE_data),
                    "pE_anc": float(pE_anc),
                    "distance": float(d),
                    **stats,
                }
                rows.append(row)

                print(
                    f"\n=== loss_base={loss_base:.3f} | squeezing={s_db:>2} dB | d={d} "
                    f"| pZ≈{pz:.5f} | pE_data≈{pE_data:.5f} ==="
                )
                for k in [
                    "success_prob_within_cap",
                    "erasure_rate",
                    "avg_rounds_given_success",
                    "avg_fidelity_injected_given_success",
                    "avg_fidelity_logical_given_success",
                    "p_phys_eff",
                ]:
                    print(f"{k:>34}: {row[k]:.6f}")

    # Save CSV
    csv_path = "results_magic_state_sweep.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    write_csv(csv_path, rows, fieldnames)
    print(f"\n[Saved] {csv_path}")

    # Plots
    plot_metric_vs_squeezing_by_loss(
        rows,
        metric="avg_fidelity_logical_given_success",
        ylabel="Average logical fidelity (given success)",
        title="Logical magic-state fidelity vs squeezing",
        outfile="fig_fidelity_vs_squeezing.png",
    )

    plot_metric_vs_squeezing_by_loss(
        rows,
        metric="avg_rounds_given_success",
        ylabel="Average RUS rounds (given success)",
        title="RUS overhead vs squeezing",
        outfile="fig_overhead_vs_squeezing.png",
    )

    plot_metric_vs_squeezing_by_loss(
        rows,
        metric="avg_fidelity_logical_given_success",
        ylabel="Average logical fidelity (given success)",
        title="Logical magic-state fidelity vs squeezing",
        outfile="fig_fidelity_vs_squeezing.png",
    )

    plot_sensitivity_heatmaps(
        rows,
        metric="avg_fidelity_logical_given_success",
        outfile_prefix="fig_sensitivity"
    )

    plot_phase_boundary(
        rows,
        metric_success="success_prob_within_cap",
        metric_fidelity="avg_fidelity_logical_given_success",
        success_thresh=0.95,
        fidelity_thresh=0.79,
        outfile="fig_phase_boundary.png"
    )

    print("[Saved] sensitivity heatmaps")
    print("[Saved] phase boundary plot")
    print("[Saved] fig_success_vs_squeezing.png")
    print("[Saved] fig_overhead_vs_squeezing.png")
    print("[Saved] fig_fidelity_vs_squeezing.png")


if __name__ == "__main__":
    main()