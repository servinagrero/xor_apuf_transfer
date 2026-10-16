# Correlation and transferability in XOR Arbiter PUF populations

This repository studies whether XOR composition can make conventional PUF
quality metrics appear favorable while systematic inter-device and intra-device
structure remains exploitable by transfer learning. The implementation supports
both controlled PyPUF populations and measurements from multi-device FPGA
populations.

## Study pipeline

| Study | Data computation | Analysis and figures |
| --- | --- | --- |
| Population metrics over $\alpha$, $\beta$, and $k$ | `scripts/00-study_alpha_beta.py` | `scripts/00-figures_alpha_beta.py` |
| Target-only modeling for $k=1,\ldots,9$ | `scripts/01-study_baseline_modeling.py` | `scripts/01-figures_baseline_modeling.py` |
| Simulated population transfer | `scripts/02-study_theoretical_transfer.py` | `scripts/02-figures_theoretical_transfer.py` |
| FPGA population transfer | `scripts/03-study_empirical_transfer.py` | `scripts/03-figures_empirical_transfer.py` |

## Installation

Python 3.11 or newer and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync --locked --dev
```

The lock file installs PyTorch and the scientific Python stack used for model
training and analysis.

## Simulation smoke runs

The first three studies generate their own data and can be checked without the
FPGA dataset:

```bash
bash run_publication_study.sh alpha-beta smoke
bash run_publication_study.sh baseline smoke
bash run_publication_study.sh theory smoke
```

The `all` phase also executes empirical preparation, transfer, controls, and
figures. It therefore requires a compatible CRP CSV and challenge array. See
[Empirical data and transfer](empirical_transfer.md) for the input contract and
adaptation guidance.

## Outputs

Study tables are written under `data/`; figures and analysis tables are written
under `figures/`. Both directories are ignored because full publication runs
produce large, environment-specific outputs. Random seeds, study grids, and
empirical run metadata are recorded with the results.

## Verification

```bash
uv run python -m unittest discover -s tests -v
uv run ruff check scripts src/puf_transfer tests
```

The tests cover the correlated population endpoints, output-noise calibration,
empirical alignment, split isolation, transfer freezing, and fixed-cohort
controls. They do not run the expensive publication grids.
