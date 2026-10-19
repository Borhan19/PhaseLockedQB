# Phase-Locked Quantum Battery Charging via Dissipative Mode Selection

Numerical and plotting source accompanying the manuscript **Phase-Locked Quantum Battery Charging via Dissipative Mode Selection**.

This repository contains the four Python scripts used in the numerical study. `phase_locked_3.py` includes the corrected coherent-intensity plot in the finite-gain comparison; see `SHA256SUMS.txt` for source checksums. Physical parameters for the principal linear and nonlinear reference cases are defined directly in the scripts, and a convenient read-only snapshot is supplied in `benchmark_parameters.json`.

## Contents

| File | Purpose |
| --- | --- |
| `phase_locked_1.py` | Linear Gaussian two-mode calculations: stability landscapes, gain thresholds, charging dynamics, first-passage performance, work-like SNR, phase-locking diagnostics, collective-reservoir and mediator checks, and gross-input accounting. Exports figure PDFs, CSV data, and JSON summaries. |
| `phase_locked_2.py` | Exact single-emitter saturable-gain dynamics, linear-gain comparison, exact battery ergotropy, and Fock-space cutoff-convergence checks. |
| `phase_locked_3.py` | Extended saturable-gain calculations, including the exact single-emitter and many-emitter mean-field workflows and additional End-Matter figures. |
| `phase_locked_4.py` | Finite-capacity reference comparison and independent coupling-resource robustness landscapes (fixed-pump and matched-growth protocols), with CSV/JSON outputs. |
| `benchmark_parameters.json` | Snapshot of reference benchmark parameters extracted from the supplied scripts. The scripts themselves define their runtime parameters; this JSON is documentation, not an input dependency. |
| `requirements.txt` | Python numerical and plotting dependencies. |
| `SHA256SUMS.txt` | SHA-256 checksums of the four current scripts and selected auxiliary files. |

**Figure numbering note:** Filenames inside the scripts retain some historical naming conventions. Match generated figures by their scientific content and plotted quantities, rather than assuming internal output filenames match the final manuscript's figure numbers.

## Installation

Use Python **3.10 or newer**, preferably in a fresh environment.

```bash
python -m venv .venv
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

QuTiP 5 is required for the exact pumped-emitter master-equation calculations. Linear Gaussian simulations, validation, and the mean-field emitter-ensemble calculation do not require QuTiP at runtime.

## Reproduction examples

### Linear Gaussian analysis and plots

```bash
python phase_locked_1.py validate --output results/linear
python phase_locked_1.py all-linear --output results/linear
```

Alternatively, run individual components, such as:

```bash
python phase_locked_1.py stability --output results/linear
python phase_locked_1.py dynamics --output results/linear
python phase_locked_1.py finite-capacity --output results/linear
python phase_locked_1.py gross-input --output results/linear
python phase_locked_1.py theta-tradeoff --output results/linear
python phase_locked_1.py mediator --output results/linear
```

`finite-capacity` also calls the gross-input plotting routine. `all-linear` includes linear Gaussian analyses, including `finite-capacity`; nonlinear emitter calculations are separate. For the full list, run `python phase_locked_1.py --help`.

### Exact pumped-emitter saturation

Start by checking the small-signal gain matching without performing the expensive density-matrix evolution:

```bash
python phase_locked_2.py derive --profile low_energy
python phase_locked_3.py derive --profile low_energy
```

To perform exact convergence and final-time simulations with QuTiP, for example:

```bash
python phase_locked_3.py all --profile low_energy --cutoffs 6x6,8x8,10x10,12x10,14x10,16x10,18x10 --output results/saturation
```

For the related original single-emitter implementation, use `phase_locked_2.py` with its `derive`, `convergence`, `final`, or `all` commands. Full density-matrix and cutoff-convergence calculations may take substantial memory and runtime. See `python phase_locked_2.py --help` and `python phase_locked_3.py --help` for options.

### Coupling-resource landscapes

```bash
python phase_locked_4.py validate
python phase_locked_4.py all
```

The new landscape calculations vary coherent coupling and collective damping independently. The fixed-pump map does **not** match growth rates throughout the scan; the alternative map recalibrates pumps at each point. See `python phase_locked_4.py --help` for runtime options.

### Many-emitter mean-field results

```bash
python phase_locked_3.py ensemble --profile paper --emitter-counts 1,10,30,100 --output results/ensemble
```

The many-emitter calculation is a **mean-field approximation**, not an exact many-emitter master-equation simulation.

## Outputs and scientific conventions

The scripts write PDF figures, CSV tables, and/or JSON parameter and summary files into their selected output folder, commonly using subdirectories `Figs_New/`, `data/`, `summaries/`, and `diagnostics/`.

- Reference rate: `kappa_0 = 0.01` in code units.
- Time and rates are reported in units of `kappa_0 t` and `rate / kappa_0`.
- Linear-state ergotropy follows the Gaussian battery definition; the exact nonlinear calculation obtains ergotropy from the reduced battery density matrix.
- Frequency optimization is applied to the coherent drive while the bare mode frequencies remain fixed.
- Gross input and signed active-energy accounting are **model conventions**, not microscopic wall-plug power costs.

## Validation status

During repository preparation, the initial three scripts passed Python syntax compilation, and the subsequently added `phase_locked_4.py` also passed compilation; `phase_locked_1.py validate` passed its built-in checks; `phase_locked_1.py stability` generated its figure and numerical datasets; and `phase_locked_2.py derive` and `phase_locked_3.py derive` completed successfully. These are **smoke tests, not a full rerun of every manuscript figure**. The exact nonlinear QuTiP evolution was not executed during this packaging check.

## License

No license has been assigned to this repository bundle. The repository owner should add a license if a specific reuse permission is intended.

## Repository revisions

The `main` branch contains the corrected coherent-intensity plotting lines in `phase_locked_3.py` and the additional `phase_locked_4.py` resource-landscape script. A previously published release tag may point to an earlier immutable snapshot; check the tag's commit before citing it as the complete current code.
