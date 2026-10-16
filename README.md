# Correlation and transferability in XOR Arbiter PUF populations

This repository studies whether XOR composition can make conventional PUF
quality metrics appear favorable while systematic inter-device and intra-device
structure remains exploitable by transfer learning.

The active publication pipeline contains four paired studies:

| Study | Data computation | Analysis and figures |
| --- | --- | --- |
| Population metrics over \((\alpha,\beta,k)\) | `scripts/00-study_alpha_beta.py` | `scripts/00-figures_alpha_beta.py` |
| Target-only modeling for \(k=1,\ldots,9\) | `scripts/01-study_baseline_modeling.py` | `scripts/01-figures_baseline_modeling.py` |
| Simulated population transfer | `scripts/02-study_theoretical_transfer.py` | `scripts/02-figures_theoretical_transfer.py` |
| FPGA population transfer | `scripts/03-study_empirical_transfer.py` | `scripts/03-figures_empirical_transfer.py` |

## Reproduce the studies

Install the locked environment:

```bash
uv sync --locked --dev
```

The simulation studies generate their own data and can be checked independently:

```bash
bash run_publication_study.sh alpha-beta smoke
bash run_publication_study.sh baseline smoke
bash run_publication_study.sh theory smoke
```

The `all` phase additionally runs the empirical study and therefore requires a
compatible CRP CSV and challenge array:

```bash
EMPIRICAL_CSV=/path/to/crps.csv \
CHALLENGES=/path/to/all_challenges.npy \
PUF_JOBS=8 \
bash run_publication_study.sh all publication
```

Individual phases are `alpha-beta`, `baseline`, `theory`, `theory-calibrated`,
`prepare`, `empirical`, `pooled`, `fixed-cohort`, `validation`, `control`, and
`figures`. Empirical outputs include the revision suffix `ber_cost_v2`; override
it with `PUF_STUDY_REVISION`. Figure-only reruns do not recompute study data:

```bash
bash run_publication_study.sh figures publication
```

Corrected empirical figures are written to a revisioned directory ending in `analysis_v2`; override that suffix with `PUF_FIGURE_REVISION`. The empirical analysis also emits transfer-frontier, crossover, and break-even CSV tables, plus a selected LaTeX break-even table.

The empirical CSV loader streams only `challenge_id`, `sample_id`, `device_id`, `slice`, and `response`. See [the empirical data contract](docs/empirical_transfer.md#input-data-contract) for adaptation guidance, cache semantics, data splits, metric definitions, and source-similarity controls.

## Evaluation conventions

- Target and source budgets include validation labels; validation is not a
  hidden acquisition cost.
- Simulated studies calibrate PyPUF noise to requested output BERs of 0%, 3%
  and 5%, while retaining calibrated raw noisiness and achieved BER. Raw
  `--noisiness` remains available only as a diagnostic interface.
- Source pretraining, total attack and amortized per-target costs are reported
  separately; target-only baselines carry no source-pretraining cost.
- Transfer and target-only models use paired target CRPs and test challenges.
- Simulation confidence intervals use independent population trials.
- Empirical confidence intervals resample physical target devices, not CRP rows
  or dependent device pairs.
- `Min_H_UPD` and `Min_H_UPC` are empirical descriptive scores, not certified
  cryptographic entropy bounds.
- Individual-chain FPGA responses are used for population characterization;
  the transfer attack receives only the selected XOR response.
- The challenge-permuted source-label control preserves budgets and response
  balance while destroying reusable challenge-aligned source structure.

## Tests

```bash
uv run python -m unittest discover -s tests -v
ruff check scripts src/puf_transfer tests
```
