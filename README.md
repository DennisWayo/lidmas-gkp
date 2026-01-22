## LiDMaS

LiDMaS is a lightweight, architecture-level simulator for investigating logical magic-state injection in Gottesman–Kitaev–Preskill (GKP)–encoded photonic qubits.

Rather than performing full continuous-variable wavefunction simulations or decoder-level syndrome tracking, LiDMaS adopts a density-matrix abstraction in which encoded logical qubits are represented as **2x2 density matrices** and dominant error mechanisms are modeled as effective logical channels. This enables rapid and transparent exploration of fault-tolerant design trade-offs in photonic quantum architectures.

The simulator is specifically designed to study:
- Repeat-until-success (RUS) logical T-gate magic-state injection
- Finite-squeezing-induced logical noise in GKP encodings
- Heralded photon-loss-induced erasure
- Outer-code protection via surface-code–inspired scaling laws

### Model Architect

LiDMaS follows an architecture-first abstraction, intentionally avoiding microscopic simulation details while preserving the structure of logical error propagation.

What is modeled
- Finite GKP squeezing → effective logical dephasing
- Residual Clifford imperfections → logical depolarizing noise
- Photon loss → heralded erasure (abort and restart, no unheralded corruption)
- Outer code protection → surface-code logical error suppression (scaling law)

What is not modeled
- Continuous-variable wavefunctions
- GKP stabilizer decoding or syndrome extraction
- Explicit surface-code stabilizer circuits
- Quantum software frameworks (PennyLane)

This deliberate abstraction allows efficient sweeps over squeezing, loss, and code distance while retaining physical interpretability.

### Structure 
.
├── main.py                     # Parameter sweeps and experiment driver
├── magic_state_injection.py    # Repeat-until-success (RUS) T-gate injection logic
├── logical_noise.py             # Logical noise channels (dephasing, depolarizing, erasure)
├── gkp_effective_noise.py       # Finite squeezing → effective GKP noise mappings
├── outer_code.py                # Surface-code-inspired logical error scaling
├── analysis_plots.py            # Plotting utilities and sensitivity analysis
├── results_magic_state_sweep.csv# Raw simulation output (architecture-level metrics)
└── figures/
├── fig_success_vs_squeezing.png
├── fig_overhead_vs_squeezing.png
├── fig_fidelity_vs_squeezing.png
├── fig_sensitivity_*.png
└── fig_phase_boundary.png


### Respresentative Results 

| Metric (given success)            | Typical range        | Key observation                                                                 |
|----------------------------------|----------------------|----------------------------------------------------------------------------------|
| RUS success probability          | 0.90 – 0.98          | Increases monotonically with squeezing; weak dependence on code distance         |
| Average RUS rounds               | 1.15 – 1.20          | Overhead remains close to unity due to efficient heralding                        |
| Logical fidelity (d = 3–7)       | 0.77 – 0.80          | Strongly improved by squeezing and outer-code distance                            |
| Sensitivity to loss              | ≈ 0                  | Loss primarily affects success probability, not logical fidelity                 |
| Sensitivity to squeezing         | Non-zero at low values | Finite-energy GKP noise is the dominant continuous error mechanism              |


### Intended Use

LiDMaS is intended for:
- Architecture-level exploration of photonic fault tolerance
- Hardware–software co-design studies
- Rapid evaluation of squeezing vs error-correction trade-offs
- Complementing (not replacing) decoder-level or CV-level simulations

### Citation



