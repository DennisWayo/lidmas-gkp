![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/status-research--grade-brightgreen)
![Scope](https://img.shields.io/badge/scope-architecture--level-lightgrey)
![Quantum](https://img.shields.io/badge/domain-quantum%20error%20correction-purple)
![Simulation](https://img.shields.io/badge/simulation-density--matrix-informational)

# LiDMaS

LiDMaS is a lightweight, architecture-level simulator for investigating logical magic-state injection in Gottesman–Kitaev–Preskill (GKP)–encoded photonic qubits.

Rather than performing full continuous-variable wavefunction simulations or decoder-level syndrome tracking, LiDMaS adopts a density-matrix abstraction in which encoded logical qubits are represented as **2x2 density matrices** and dominant error mechanisms are modeled as effective logical channels. This enables rapid and transparent exploration of fault-tolerant design trade-offs in photonic quantum architectures.

The simulator is designed to study:
- Repeat-until-success (RUS) logical T-gate magic-state injection
- Finite-squeezing-induced logical noise in GKP encodings
- Heralded photon-loss-induced erasure
- Outer-code protection via surface-code–inspired scaling laws

### Model Architecture

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

Core architecture-level noise model:

$$
\mathcal{E}(\rho)=
\begin{cases}
\text{erasure}, & \text{with probability } p_E \\
\mathcal{D}_{\text{depol}} \circ \mathcal{D}_Z(\rho), & \text{otherwise}
\end{cases}
$$

where:

$$
\mathcal{D}_Z(\rho)=(1-p_Z)\rho + p_Z Z\rho Z
$$

and

$$
\mathcal{D}_{\text{depol}}
$$

is a standard depolarizing channel. The RUS injection loop repeats until success or a round cap is hit, enabling direct estimates of success probability, overhead, and logical fidelity.

### Install Dependencies and Run

```bash
pip install numpy matplotlib scipy
python main.py
```
No quantum SDKs such as PennyLane, Qiskit, or Strawberry Fields are required.

### Configuring the parameter sweep

```python
cfg = SweepConfig(
    squeezing_db_values=list(np.arange(8.0, 16.5, 0.5)),
    loss_base_values=[0.005, 0.01, 0.015, 0.02, 0.025, 0.03],
    distances=[1, 3, 5, 7],
    max_rounds=10,
    trials=5000,
    seed=42,
)
```



### Structure 

```text
.
├── main.py                     # Parameter sweeps and experiment driver
├── magic_state_injection.py    # Repeat-until-success (RUS) T-gate injection logic
├── logical_noise.py            # Logical noise channels (dephasing, depolarizing, erasure)
├── gkp_effective_noise.py      # Finite squeezing → effective GKP noise mappings
├── outer_code.py               # Surface-code-inspired logical error scaling
├── analysis_plots.py           # Plotting utilities and sensitivity analysis
├── results_magic_state_sweep.csv  # Raw simulation output (architecture-level metrics)
```

### Representative Results 

| Metric (given success)            | Typical range        | Key observation                                                                 |
|----------------------------------|----------------------|----------------------------------------------------------------------------------|
| RUS success probability          | 0.904 – 0.989        | Increases monotonically with squeezing; weak dependence on code distance         |
| Average RUS rounds               | 1.151 – 1.203        | Overhead remains close to unity due to efficient heralding                        |
| Logical fidelity (d = 3–7)       | 0.765 – 0.796        | Strongly improved by squeezing and outer-code distance                            |
| Sensitivity to loss              | ≈ 0                  | Loss primarily affects success probability, not logical fidelity                 |
| Sensitivity to squeezing         | Non-zero at low values | Finite-energy GKP noise is the dominant continuous error mechanism              |


### Intended Use

LiDMaS is intended for:
- Architecture-level exploration of photonic fault tolerance
- Hardware–software co-design studies
- Rapid evaluation of squeezing vs error-correction trade-offs
- Complementing (not replacing) decoder-level or CV-level simulations

### Citation

If you use this work, please cite:

```bibtex
@misc{wayo2026lidmas,
      title={LiDMaS: Architecture-Level Modeling of Fault-Tolerant Magic-State Injection in GKP Photonic Qubits}, 
      author={Dennis Delali Kwesi Wayo},
      year={2026},
      eprint={2601.16244},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2601.16244}, 
}
```
