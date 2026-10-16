# Empirical FPGA transfer study

This pipeline uses the same FiLM NN (`PUFModel`) as
`scripts/02-study_theoretical_transfer.py`.
Every source and target model sees only the XOR response for its selected slice
combination. Individual-chain labels are used separately for descriptive
population estimation, never to initialize or select the transfer model.

## Input data contract

Preparation requires a CRP CSV and a NumPy challenge array. The CSV may contain
additional metadata columns, but these five columns are required:

| Column | Required representation |
| --- | --- |
| `challenge_id` | Zero-based integer row index into the challenge array |
| `sample_id` | Integer equal to `0`; repeated measurements require an explicit preprocessing policy |
| `device_id` | Stable string identifier for one physical device |
| `slice` | Stable string identifier for one individual APUF chain |
| `response` | Binary chain response in `{0,1}` |

For example, the same challenge may appear once for each measured chain and
device:

```csv
challenge_id,sample_id,device_id,slice,response
0,0,D01,S0,0
0,0,D01,S1,1
0,0,D02,S0,1
0,0,D02,S1,0
```

The challenge file must be a two-dimensional `.npy` array with shape
`(n_challenges, n_stages)` and binary `{0,1}` entries. Row `i` must contain the
challenge referenced by `challenge_id == i`. Columns must follow physical stage
order from the chain input to the arbiter unless preparation uses
`--bit-order reverse`.

Each `(device_id, slice, challenge_id)` key must be unique. Measurements may be
sparse: a k-XOR response is available only on the intersection of challenge IDs
present for all selected chains on that device. Individual-chain responses are
required to construct arbitrary slice combinations and estimate the
inter-device and intra-device population moments. A dataset containing only
pre-XORed outputs can support a separately adapted prediction experiment, but
not the current alpha/beta estimation or combination pipeline.

### Adapting another dataset

Normalize source-specific formats before invoking the study. The narrow import
boundary is `prepare_empirical_dataset()` in `src/puf_transfer/empirical.py`:

1. Rename source columns to the five names above, or change the Arrow `types`
   and normalization immediately after each input batch.
2. Remap non-contiguous challenge identifiers to `0,...,n_challenges-1` and
   reorder the challenge-array rows with the same mapping.
3. Convert polar responses using the convention expected downstream:
   `+1 -> 0` and `-1 -> 1`.
4. For repeated measurements, choose a policy such as a fixed acquisition or
   majority response before preparation, then emit one row with `sample_id=0`.
   Keep repeated measurements separately when estimating reliability.
5. If challenges are embedded in the source table, deduplicate them into one
   challenge array and replace each vector with its row index.
6. Use `--slices S0,S1,...` when only a subset of chain identifiers should be
   included. The identifiers themselves need not match the names used in the
   FPGA study.

The loader deliberately does not guess any of these policies: incorrect
challenge alignment or response encoding can create artificial transfer.

## Prepare once

```bash
uv run python scripts/03-study_empirical_transfer.py --prepare \
    --csv /path/to/full/crps.csv \
    --challenges data/raw/all_challenges.npy \
    --cache-dir data/empirical_cache
```

The reader streams `challenge_id,sample_id,device_id,slice,response`, ignoring
metadata. Each chain becomes a one-byte-per-challenge, memory-mapped NPY array.
For example, 29 devices, four slices and 1,005,000 IDs occupy about 117 MB in
the response cache. XOR datasets are computed on demand; the source CSV is never
loaded whole.

Challenge IDs must be zero-based indices into the NPY rows. Duplicate
`(device_id,slice,challenge_id)` keys, invalid IDs, non-binary responses and
sample IDs other than zero fail explicitly. Missing measurements have a sentinel;
they are never treated as a zero response. XOR combinations use the intersection
of available challenge IDs across their constituent slices on each device.

The manifest stores coverage, input paths and a challenge-array checksum. An
incomplete preparation does not publish a manifest. Caches are not overwritten;
use a new directory after an interrupted or invalid preparation. A copied cache
can use `--challenges` to point to a relocated, checksum-identical NPY file.

Bits are mapped as `0 -> +1, 1 -> -1`, then suffix parity is applied. By default,
the columns are assumed to follow stage order from chain input to arbiter.
If hardware serialization reverses this, prepare with `--bit-order reverse`.
Choose this from the acquisition design, not held-out test accuracy.

## Fixed global partitions

| Challenge IDs | Role |
| --- | --- |
| 0 through 999,999 | Source training and target adaptation |
| 1,000,000 through 1,000,999 | Validation and chain-fit diagnostics |
| 1,001,000 through 1,004,999 | Final target test |

No source FPGA trains on any validation/test challenge ID. Defaults use 200
validation labels per source/target; the test uses every available complete CRP
in the final partition. `--train-stop` and `--validation-size` configure a new
cache; subsequent runs use the cached partitions.

The adaptation budget `m` is the total attacker-visible target budget. Training
and validation labels are both drawn from this budget and are recorded
separately; `target_total_labeled_crps = m` for both transfer and target-only
models.
These are XOR CRP counts; acquisition of a k-XOR CRP entails k chain readouts.
There are no repeated measurements, so BER/reliability cannot be measured directly.

## Estimate alpha and beta

```bash
uv run python scripts/03-study_empirical_transfer.py --estimate \
    --cache-dir data/empirical_cache \
    --output-dir data/empirical_transfer \
    --estimate-budget 20000
```

The primary estimator works directly from polar responses on the same sampled
training challenges. With `r = 1 - 2 * response`, the Gaussian sign identity is
`E[sign(X)sign(Y)] = 2/pi * asin(rho)`. Under the repository's equal-variance
latent model:

```text
m_alpha = mean E[r(device,s) r(other_device,s)]
alpha^2 = sin(pi/2 * m_alpha)

m_beta = mean E[r(device,s) r(device,t)]
alpha^2 beta^2 = sin(pi/2 * m_beta)
beta^2 = (alpha^2 beta^2) / alpha^2
```

For a multi-slice combination, the estimator averages the same-slice
inter-device pair moments for its alpha and all within-device slice-pair moments
for its common beta. Pair-equal and observation-weighted moments are both
exported; the equations use the pair-equal expectation. `response_moments.csv`
retains each pair's raw polar product, agreement, centered Pearson correlation,
uniformities and number of common challenges. The arcsine equations use the raw
polar product, not centered Pearson correlation.

For k=1, beta is undefined; when alpha-squared is near zero, beta is also
undefined. Negative transformed moments or beta-squared outside `[0,1]` make
the estimate undefined and set `moment_model_consistent=false`. Values are not
clipped into a plausible range.

These are approximate method-of-moments estimates, not measurements of physical
Gaussian mixing coefficients. Nonzero chain means, unequal delay variances,
noise and a finite number of stages can violate the zero-mean Gaussian identity.
Responses do not identify absolute delay scale. Raw moments, transformed
correlations, the raw beta-squared ratio, pair counts and
`moment_model_consistent` are exported alongside estimates.
Pairs share devices/features and are not independent replicates; no spurious
pair-count-based confidence interval is reported.

Logistic single-chain fits remain as diagnostics. Their direction cosines are
stored under `lr_*` columns and in `chain_pairs.csv`; they no longer define
alpha or beta. `chain_fits.csv` reports held-out model accuracy, while
`response_moments.csv` contains the primary sufficient statistics and
`population_estimates.csv` contains one response-based estimate per slice
combination. These diagnostic budgets are separate from attack budgets and are
never passed to the attack.

`population_parameter_pairs.csv` additionally retains pairwise transformed
estimates. Inter-device pairs identify alpha; within-device chain pairs identify
gamma = alpha beta directly and beta only after division by the combination's
global alpha estimate. Invalid transformations remain explicit rather than
being clipped. `population_model_fit.csv` reports pairwise residual dispersion
around the global first-order model.

## Publication cost and control artifacts

The figure stage writes a long-form `theory_empirical_budget_accuracy.csv` with
source-pretraining, total and amortized costs. For S source devices, N labels
per source, M target labels and T subsequent targets, the reported amortized
cost is `M + S*N/T`; the marginal target cost remains M. Pareto plots and the
minimum costs reaching fixed accuracy thresholds are derived from this table.

The `control` runner phase permutes source labels independently over challenges
before source training. It leaves target adaptation unchanged and tests whether
the transfer gain requires challenge-aligned population structure. This is a
negative control, not an alternative attack.

## Population quality metrics

`--estimate` also computes empirical uniformity, bitaliasing and uniqueness for
all requested slice combinations. Run only this inexpensive stage with:

```bash
uv run python scripts/03-study_empirical_transfer.py --metrics \
    --cache-dir data/empirical_cache \
    --output-dir data/empirical_transfer \
    --metric-budget 20000
```

One seeded training-challenge sample is shared by every combination. A device
is included for a combination only when all selected slices are measured on the
complete sample. This produces a rectangular response matrix and makes the
three distributions directly consistent:

```text
uniformity[d] = mean response of device d over challenges
bitaliasing[c] = mean response to challenge c over devices
uniqueness[d1,d2] = Hamming distance between two devices over challenges
```

`population_metric_distributions.npy` stores a Python dictionary containing
the raw distributions, device IDs, excluded device IDs, challenge IDs and
uniqueness device pairs for each of the 15 combinations. It also stores the
publication-facing transformed distributions:

```text
UPD[c] = H2(bitaliasing[c])
UPC[d] = H2(uniformity[d])
H_uniqueness[d1,d2] = H2(uniqueness[d1,d2])
Min_H_UPD[c] = H_inf(bitaliasing[c])
Min_H_UPC[d] = H_inf(uniformity[d])
```

UPD therefore uses the challenge axis after averaging across devices; UPC uses
the device axis after averaging across challenges. Both min-entropies are
reported to avoid selecting one after inspecting the results. `Min_H_UPD` is
the primary population-predictability metric. Load the dictionary only from a
trusted study output because NumPy dictionaries use pickle:

```python
metrics = np.load(
    "data/empirical_transfer/population_metric_distributions.npy",
    allow_pickle=True,
).item()
```

`population_metrics_summary.csv` is a 90-row long-form Polars-compatible table:
UPD, UPC, Uniqueness, H(Uniqueness), Min_H_UPD and Min_H_UPC for each
combination. Every row reports its source metric, aggregation axis, observation
count, mean, standard deviation and range. Entropy uses the same definitions as
`scripts/00-study_alpha_beta.py`. `EMPIRICAL_METRIC_BUDGET` configures the
sample when using `run_publication_study.sh`.

## Source-similarity validation

The `validation` pipeline phase tests whether transfer depends on near-clone
source devices:

```bash
bash run_publication_study.sh validation publication
```

It reruns the attack after excluding the three source devices with the smallest
pairwise uniqueness to each target. Results are written under the revisioned
`data/empirical_transfer_exclude_nearest_*` directory and never overwrite the
random-source experiment. The publication profile uses 10 remaining sources;
the smoke profile excludes one nearest source and uses two sources. Change
`NEAREST_EXCLUSIONS` and `VALIDATION_SOURCE_COUNT` in the runner, or invoke
`03-study_empirical_transfer.py` directly, to study another policy.

`scripts/03-figures_empirical_transfer.py` writes pairwise uniqueness,
cluster membership, per-cohort target/source distances, Pearson/Spearman
correlations and matched random-versus-excluded comparisons. Unmatched cells
remain absent from the comparison rather than being silently averaged.

The validation run also reports several controls with different threat models:

- `source_embedding_ensemble_acc` averages predictions from learned source
  embeddings and uses no target labels.
- `pooled_source_acc` trains a device-agnostic model on exactly the source CRP
  budget and uses no target labels.
- `diagnostic_closest_source_embedding_acc` selects a source using empirical
  target/source uniqueness and is diagnostic-only.
- `oracle_best_source_embedding_acc` selects using target test accuracy and is
  an oracle diagnostic, never an attack result.
- `source_consensus_acc` queries source devices on target test challenges. Its
  additional acquisition cost is recorded in `source_consensus_extra_crps`.

Target-label accounting is explicit: zero-shot controls use zero target labels,
embedding adaptation uses `m`, and the from-scratch NN uses `m` training plus
the validation labels. Oracle and diagnostic values cannot select transfer
parameters, checkpoints, source polarity or predictions.

## Train and evaluate

The complete theoretical-to-empirical workflow is available through one
phase-based runner once both empirical input files satisfy the contract above:

```bash
bash run_publication_study.sh all publication
```

The publication profile uses `k=1,2,3,4`, source counts `1,5,10,20`, source
budgets `500,1000,5000`, target budgets from 100 to 20,000 CRPs, and five
trials. The smoke profile uses smaller grids but still requires empirical data.
Configure input paths, cache location, parallelism, and output revision names
through the runner environment:

```bash
EMPIRICAL_CSV=/path/to/crps.csv \
CHALLENGES=/path/to/all_challenges.npy \
EMPIRICAL_CACHE=/path/to/cache \
PUF_JOBS=8 \
PUF_STUDY_REVISION=my_run \
    bash run_publication_study.sh all publication
```

Study grids are declared near the top of `run_publication_study.sh`. For a
one-off custom grid, invoke the numbered study script directly rather than
adding dataset-specific behavior to the runner. Resume metadata rejects mixing
trial counts or incompatible configurations in one output directory.

Preparation reuses an existing cache manifest. The empirical phase resumes a
matching transfer run but always recomputes the population estimates, which is
required when the estimator changes.

Start with a smaller source configuration, then expand:

```bash
uv run python scripts/03-study_empirical_transfer.py --run \
    --cache-dir data/empirical_cache --output-dir data/empirical_transfer \
    --k-values 1,2,3,4 --k-pretrain 1,2,10,20,28 \
    --n-base-budget 1000,5000 --m-target-budget 100,500,1000,5000 \
    --n-trials 1 --n-jobs 4
```

Four slices give 4 + 6 + 4 + 1 = 15 combinations. Slice names are discovered
from the data; `--slices` can choose an explicit comma-separated subset.
Every device is a held-out target by default. `--targets UID1,UID2` limits the
experiment. A target can use at most one fewer source than the number of
eligible devices.

Some slices may have fewer usable source devices because a device is missing
enough complete training or validation CRPs. The default
`--insufficient-sources skip` keeps every feasible requested source count and
writes `skipped_source_configurations.csv` with the target, combination, budget,
eligible count and device lists for every omission. Use
`--insufficient-sources error` for strict behavior. The plotting summary reports
`n_target_devices` for each curve; compare source counts only after checking that
coverage. For a completely balanced study, choose a `--k-pretrain` no larger
than the minimum eligible count in the skip report.

For each target and trial, source cohorts are nested as source count grows.
One backbone is trained per combination/cohort/source budget and restored before
each target budget. Target CRPs are nested across budgets and independent of
source-cohort choice. The new target embedding starts at the mean source
embedding, then only embeddings are adapted using a 20-step L-BFGS routine.
The best target embedding is selected by validation loss, using labels included
in the declared target budget.
Backbone pretraining and the from-scratch baseline use validation early stopping.

The baseline is the same NN trained from scratch on the identical target CRPs
and validation set. Zero-shot prediction and a training-majority predictor are
also reported. Predictions retain their output orientation; test labels never
select a polarity, checkpoint, hyperparameter, or source cohort.

The number of fits grows across slice combinations, source cohorts, budgets,
targets, and trials. Use a small subset first. Results are flushed after each
configuration. `--resume` skips completed configurations only when run metadata
match (parallel worker count may change). A new configuration requires a new
output directory. Source UIDs, seeds, budgets and architecture are recorded.

`PUFModel.fit` now trains incomplete batches, including datasets smaller than
the batch size, and restores its best validation checkpoint after a normal
epoch-limit exit. These fixes also affect future theoretical/baseline runs;
old curves should be rerun for a tightly controlled comparison.

## Plot and compare

```bash
uv run python scripts/03-figures_empirical_transfer.py \
    --input data/empirical_transfer/transfer_results.csv
```

This writes detailed figures for each XOR degree, the aggregate
`empirical_transfer.{png,pdf}`, publication summaries such as
`transfer_advantage_map.{png,pdf}`, and their CSV tables. It averages trials
within each target first and bootstraps physical target devices for confidence
intervals; source cohorts overlap. Plot defaults select the largest source
count and source budget. Explicit flags choose another configuration.

Passing theoretical results adds them to the long-form cost table after checking
that the stage count and selected BER agree. It does not interpolate or rescale
simulation accuracy to match empirical data:

```bash
uv run python scripts/03-figures_empirical_transfer.py \
    --k-pretrain 20 --n-base-budget 5000 \
    --estimates data/empirical_transfer/population_estimates.csv \
    --theory data/theoretical_transfer.csv
```

`--theory-n-bits` declares the stage count for legacy simulation tables without
that column (default 64). The simulation uses different population generation,
noise, and challenge sampling, so the combined table is an exploratory cost
comparison rather than evidence that both domains are identical. The plotting
script rejects population estimates made by the former LR-direction estimator.

Run focused tests with:

```bash
uv run python -m unittest discover -s tests -p 'test_empirical_transfer.py'
```
