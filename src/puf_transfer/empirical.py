"""ID-aligned, disk-backed empirical chains and descriptive population estimates."""

from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.csv as arrow_csv
from sklearn.linear_model import LogisticRegression

from puf_transfer import suffix_parity_transform


def prepare_empirical_dataset(
    csv_path,
    challenges_path,
    cache_dir,
    *,
    train_stop=1_000_000,
    validation_size=1000,
    block_size=16 * 1024 * 1024,
    bit_order="forward",
):
    """Stream only five CSV columns; missing responses use sentinel 255, never zero."""
    csv_path, challenges_path, cache_dir = map(
        Path, (csv_path, challenges_path, cache_dir)
    )
    if bit_order not in ("forward", "reverse"):
        raise ValueError("bit_order must be forward or reverse.")
    if cache_dir.exists() and any(cache_dir.iterdir()):
        raise ValueError(f"Cache is not empty: {cache_dir}. Use a new directory.")
    challenges = np.load(challenges_path, mmap_mode="r")
    if challenges.ndim != 2 or not 0 < train_stop < train_stop + validation_size < len(
        challenges
    ):
        raise ValueError(
            "Require a 2-D challenge array and nonempty train, validation and test ID ranges."
        )
    for start in range(0, len(challenges), 100_000):
        block = challenges[start : start + 100_000]
        if not np.all((block == 0) | (block == 1)):
            raise ValueError(
                "Challenge file must contain binary 0/1 bits in stage order."
            )
    cache_dir.mkdir(parents=True, exist_ok=True)
    types = {
        "challenge_id": pa.int64(),
        "sample_id": pa.int64(),
        "device_id": pa.string(),
        "slice": pa.string(),
        "response": pa.int64(),
    }
    reader = arrow_csv.open_csv(
        csv_path,
        read_options=arrow_csv.ReadOptions(block_size=block_size),
        convert_options=arrow_csv.ConvertOptions(
            include_columns=list(types), column_types=types
        ),
    )
    arrays, entries = {}, {}
    for batch in reader:
        frame = pl.from_arrow(batch)
        if any(frame.null_count().row(0)):
            raise ValueError("Required CRP fields contain null values.")
        if frame.filter(
            (pl.col("sample_id") != 0) | ~pl.col("response").is_in([0, 1])
        ).height:
            raise ValueError(
                "Require sample_id=0 and binary responses; repeated samples need an explicit policy."
            )
        for key, group in frame.partition_by(
            ["device_id", "slice"], as_dict=True
        ).items():
            ids = group["challenge_id"].to_numpy()
            if np.any(ids < 0) or np.any(ids >= len(challenges)):
                raise ValueError(f"Challenge ID outside the array for {key}.")
            if len(np.unique(ids)) != len(ids):
                raise ValueError(
                    f"Duplicate (device, slice, challenge_id) within CSV batch for {key}."
                )
            if key not in arrays:
                filename = hashlib.sha256(json.dumps(key).encode()).hexdigest() + ".npy"
                array = np.lib.format.open_memmap(
                    cache_dir / filename,
                    mode="w+",
                    dtype=np.uint8,
                    shape=(len(challenges),),
                )
                array[:] = 255
                arrays[key] = array
                entries[key] = dict(
                    device_id=key[0], slice=key[1], file=filename, rows=0
                )
            array = arrays[key]
            if np.any(array[ids] != 255):
                raise ValueError(
                    f"Duplicate (device, slice, challenge_id) across CSV batches for {key}."
                )
            array[ids] = group["response"].to_numpy().astype(np.uint8)
            entries[key]["rows"] += len(ids)
    if not arrays:
        raise ValueError("The CSV contains no CRPs.")
    for array in arrays.values():
        array.flush()
    stat = csv_path.stat()
    manifest = dict(
        version=1,
        csv_path=str(csv_path.resolve()),
        csv_size=stat.st_size,
        csv_mtime_ns=stat.st_mtime_ns,
        challenges_path=str(challenges_path.resolve()),
        challenge_sha256=hashlib.sha256(challenges_path.read_bytes()).hexdigest(),
        n_challenges=len(challenges),
        n_bits=challenges.shape[1],
        bit_order=bit_order,
        train_stop=train_stop,
        validation_stop=train_stop + validation_size,
        chains=sorted(entries.values(), key=lambda r: (r["device_id"], r["slice"])),
    )
    # Publish only after validation completes; incomplete caches cannot be loaded.
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


@dataclass
class EmpiricalDataset:
    cache_dir: Path
    challenges_path: Path | None = None

    def __post_init__(self):
        self.cache_dir = Path(self.cache_dir)
        self.manifest = json.loads((self.cache_dir / "manifest.json").read_text())
        path = Path(self.challenges_path or self.manifest["challenges_path"])
        self.challenges = np.load(path, mmap_mode="r")
        if list(self.challenges.shape) != [
            self.manifest["n_challenges"],
            self.manifest["n_bits"],
        ]:
            raise ValueError("Challenge array shape differs from cache manifest.")
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != self.manifest["challenge_sha256"]
        ):
            raise ValueError(
                "Challenge array contents differ from the cached ID mapping."
            )
        self.chains = {
            (r["device_id"], r["slice"]): np.load(
                self.cache_dir / r["file"], mmap_mode="r"
            )
            for r in self.manifest["chains"]
        }

    @property
    def devices(self):
        return sorted({d for d, _ in self.chains})

    @property
    def slices(self):
        return sorted({s for _, s in self.chains})

    def combinations(self, ks=(1, 2, 3, 4), slices=None):
        slices = sorted(slices or self.slices)
        if not set(slices) <= set(self.slices) or any(
            k < 1 or k > len(slices) for k in ks
        ):
            raise ValueError(
                f"Requested k/slices incompatible with available slices: {self.slices}"
            )
        return [c for k in ks for c in itertools.combinations(slices, k)]

    def ids(self, device, slices, split):
        boundaries = {
            "train": (0, self.manifest["train_stop"]),
            "validation": (
                self.manifest["train_stop"],
                self.manifest["validation_stop"],
            ),
            "test": (self.manifest["validation_stop"], self.manifest["n_challenges"]),
        }
        start, stop = boundaries[split]
        present = np.ones(stop - start, dtype=bool)
        for s in slices:
            if (device, s) not in self.chains:
                return np.array([], dtype=np.int64)
            present &= self.chains[device, s][start:stop] != 255
        return np.flatnonzero(present) + start

    def responses(self, device, slices, ids):
        values = np.stack([self.chains[device, s][ids] for s in slices])
        if np.any(values == 255):
            raise ValueError("Cannot XOR missing chain responses.")
        return np.bitwise_xor.reduce(values, axis=0)

    def features(self, ids):
        # Match the original empirical runner's convention: bit 0 -> +1.
        polar = 1 - 2 * np.asarray(self.challenges[ids], dtype=np.int8)
        if self.manifest.get("bit_order", "forward") == "reverse":
            polar = polar[:, ::-1]
        return np.array(suffix_parity_transform(polar), dtype=np.float32, copy=True)


def binary_shannon_entropy(probabilities):
    """Binary Shannon entropy for scalar or array-like probabilities."""
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-9, 1 - 1e-9)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def binary_min_entropy(probabilities):
    """Binary min-entropy for scalar or array-like probabilities."""
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-9, 1 - 1e-9)
    return -np.log2(np.maximum(p, 1 - p))


def summarize_metric_combination(combination):
    """Return publication-facing entropy and uniqueness metrics in long form."""
    definitions = (
        ("UPD", "bitaliasing", "challenges", combination["UPD"]),
        ("UPC", "uniformity", "devices", combination["UPC"]),
        ("Uniqueness", "uniqueness", "device_pairs", combination["uniqueness"]),
        (
            "H(Uniqueness)",
            "uniqueness",
            "device_pairs",
            combination["H_uniqueness"],
        ),
        ("Min_H_UPD", "bitaliasing", "challenges", combination["Min_H_UPD"]),
        ("Min_H_UPC", "uniformity", "devices", combination["Min_H_UPC"]),
    )
    rows = []
    for metric, source_metric, axis, distribution in definitions:
        values = np.asarray(distribution, dtype=float)
        if values.size == 0:
            raise ValueError(f"Cannot summarize an empty {metric} distribution.")
        rows.append(
            {
                "k_xor": int(combination["k_xor"]),
                "slices": "+".join(combination["slices"]),
                "metric": metric,
                "source_metric": source_metric,
                "axis": axis,
                "n_observations": len(values),
                "mean_metric": float(np.mean(values)),
                "std_metric": float(np.std(values)),
                "p05_metric": float(np.quantile(values, 0.05)),
                "median_metric": float(np.median(values)),
                "p95_metric": float(np.quantile(values, 0.95)),
                "min_metric": float(np.min(values)),
                "max_metric": float(np.max(values)),
            }
        )
    return pl.DataFrame(rows)


def compute_population_metrics(
    dataset,
    output_dir,
    *,
    budget=20_000,
    seed=42,
    ks=(1, 2, 3, 4),
    slices=None,
    null_trials=100,
):
    """Store empirical metric distributions for every requested XOR combination.

    Each combination uses the same seeded challenge sample. A device is included
    only when every selected slice is present for that complete sample, giving a
    rectangular device-by-challenge response matrix for all three metrics.
    """
    if budget < 2 or null_trials < 0:
        raise ValueError("Metric budget must be at least two and null_trials nonnegative.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_slices = slices or dataset.slices
    available = dataset.manifest["train_stop"]
    challenge_ids = np.random.default_rng(seed).permutation(available)[:budget]
    combinations = {}
    summaries = []
    null_summaries = []
    null_rng = np.random.default_rng(seed + 1)

    for combo in dataset.combinations(ks, selected_slices):
        device_ids = []
        excluded_devices = []
        response_rows = []
        for device in dataset.devices:
            chains = [dataset.chains.get((device, slice_name)) for slice_name in combo]
            if any(chain is None for chain in chains) or any(
                np.any(chain[challenge_ids] == 255) for chain in chains
            ):
                excluded_devices.append(device)
                continue
            values = np.stack([chain[challenge_ids] for chain in chains])
            response_rows.append(np.bitwise_xor.reduce(values, axis=0))
            device_ids.append(device)

        if len(device_ids) < 2:
            raise ValueError(
                f"{combo} has only {len(device_ids)} devices complete on the "
                f"{len(challenge_ids)} sampled challenges."
            )
        responses = np.stack(response_rows).astype(np.uint8, copy=False)
        pair_indices = np.asarray(
            list(itertools.combinations(range(len(device_ids)), 2)), dtype=np.int64
        )
        uniqueness = np.mean(
            responses[pair_indices[:, 0]] != responses[pair_indices[:, 1]], axis=1
        )
        key = "+".join(combo)
        entry = {
            "k_xor": len(combo),
            "slices": tuple(combo),
            "device_ids": np.asarray(device_ids),
            "excluded_device_ids": np.asarray(excluded_devices),
            "challenge_ids": challenge_ids.copy(),
            "uniformity": responses.mean(axis=1),
            "bitaliasing": responses.mean(axis=0),
            "uniqueness": uniqueness,
            "uniqueness_device_pairs": np.asarray(
                [(device_ids[i], device_ids[j]) for i, j in pair_indices]
            ),
        }
        entry.update(
            {
                "UPD": binary_shannon_entropy(entry["bitaliasing"]),
                "UPC": binary_shannon_entropy(entry["uniformity"]),
                "H_uniqueness": binary_shannon_entropy(entry["uniqueness"]),
                "Min_H_UPD": binary_min_entropy(entry["bitaliasing"]),
                "Min_H_UPC": binary_min_entropy(entry["uniformity"]),
            }
        )
        null_metrics = {
            name: []
            for name in (
                "UPD",
                "UPC",
                "Uniqueness",
                "H(Uniqueness)",
                "Min_H_UPD",
                "Min_H_UPC",
            )
        }
        for _ in range(null_trials):
            permuted = np.stack(
                [null_rng.permutation(device_row) for device_row in responses]
            )
            null_uniformity = permuted.mean(axis=1)
            null_bitaliasing = permuted.mean(axis=0)
            null_uniqueness = np.mean(
                permuted[pair_indices[:, 0]] != permuted[pair_indices[:, 1]],
                axis=1,
            )
            null_metrics["UPD"].append(
                float(np.mean(binary_shannon_entropy(null_bitaliasing)))
            )
            null_metrics["UPC"].append(
                float(np.mean(binary_shannon_entropy(null_uniformity)))
            )
            null_metrics["Uniqueness"].append(float(np.mean(null_uniqueness)))
            null_metrics["H(Uniqueness)"].append(
                float(np.mean(binary_shannon_entropy(null_uniqueness)))
            )
            null_metrics["Min_H_UPD"].append(
                float(np.mean(binary_min_entropy(null_bitaliasing)))
            )
            null_metrics["Min_H_UPC"].append(
                float(np.mean(binary_min_entropy(null_uniformity)))
            )
        entry["permutation_null"] = {
            name: np.asarray(values, dtype=float) for name, values in null_metrics.items()
        }
        combinations[key] = entry
        observed_summary = summarize_metric_combination(entry)
        summaries.append(observed_summary)
        for row in observed_summary.iter_rows(named=True):
            null_values = entry["permutation_null"][row["metric"]]
            ideal = 0.5 if row["metric"] == "Uniqueness" else 1.0
            observed_distance = abs(row["mean_metric"] - ideal)
            p_value = (
                1
                + np.count_nonzero(np.abs(null_values - ideal) >= observed_distance)
            ) / (len(null_values) + 1) if len(null_values) else np.nan
            null_summaries.append(
                {
                    "k_xor": len(combo),
                    "slices": key,
                    "metric": row["metric"],
                    "null_model": "independent_challenge_permutation",
                    "n_null_trials": len(null_values),
                    "observed_mean": row["mean_metric"],
                    "null_mean": float(np.mean(null_values)) if len(null_values) else np.nan,
                    "null_p05": float(np.quantile(null_values, 0.05)) if len(null_values) else np.nan,
                    "null_p95": float(np.quantile(null_values, 0.95)) if len(null_values) else np.nan,
                    "null_p_value": float(p_value),
                }
            )

    result = {
        "metadata": {
            "version": 1,
            "split": "train",
            "budget": len(challenge_ids),
            "seed": seed,
            "response_encoding": "binary_0_1",
            "uniformity": "mean response over challenges for each device",
            "bitaliasing": "mean response over devices for each challenge",
            "uniqueness": "pairwise device Hamming distance over challenges",
            "UPD": "H2(mean response across devices), one value per challenge",
            "UPC": "H2(mean response across challenges), one value per device",
            "H_uniqueness": "H2(pairwise device Hamming distance)",
            "Min_H_UPD": "H_inf(mean response across devices), one value per challenge",
            "Min_H_UPC": "H_inf(mean response across challenges), one value per device",
            "device_policy": "complete on every sampled challenge and selected slice",
            "null_model": "independent per-device challenge permutation",
            "null_trials": null_trials,
        },
        "combinations": combinations,
    }
    np.save(output_dir / "population_metric_distributions.npy", result, allow_pickle=True)
    summary = pl.concat(summaries).sort(["k_xor", "slices", "metric"])
    summary.write_csv(output_dir / "population_metrics_summary.csv")
    pl.DataFrame(null_summaries).sort(["k_xor", "slices", "metric"]).write_csv(
        output_dir / "population_metrics_null_summary.csv"
    )
    return result, summary


def estimate_population(
    dataset,
    output_dir,
    *,
    budget=20_000,
    seed=42,
    min_accuracy=0.9,
    ks=(1, 2, 3, 4),
    slices=None,
):
    """Estimate response moments and retain LR-direction diagnostics.

    For zero-mean jointly Gaussian delays, E[sign(X)sign(Y)] = 2/pi asin(rho).
    The repository population model has rho=alpha^2 across devices for one slice
    and rho=alpha^2 beta^2 across slices within one device.
    """
    output_dir = Path(output_dir)
    if budget < 2 or not 0 <= min_accuracy <= 1:
        raise ValueError("Require estimation budget >=2 and minimum accuracy in [0,1].")
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_slices = slices or dataset.slices
    fits, directions = [], {}
    for device in dataset.devices:
        for s in selected_slices:
            ids = dataset.ids(device, (s,), "train")
            val = dataset.ids(device, (s,), "validation")
            rng = np.random.default_rng(seed)
            ids = rng.permutation(ids)[:budget]
            row = dict(
                device_id=device,
                slice=s,
                fit_crps=len(ids),
                validation_crps=len(val),
                validation_acc=np.nan,
                bias_over_weight_norm=np.nan,
                status="insufficient_data",
            )
            if len(ids) >= 2 and len(val):
                y = dataset.responses(device, (s,), ids)
                if len(np.unique(y)) < 2:
                    row["status"] = "constant_training_response"
                else:
                    model = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
                    model.fit(dataset.features(ids), y)
                    acc = float(
                        model.score(
                            dataset.features(val), dataset.responses(device, (s,), val)
                        )
                    )
                    norm = np.linalg.norm(model.coef_[0])
                    row.update(
                        validation_acc=acc,
                        bias_over_weight_norm=float(
                            model.intercept_[0] / max(norm, 1e-12)
                        ),
                        status="ok"
                        if acc >= min_accuracy and norm > 1e-12
                        else "poor_chain_fit",
                    )
                    if row["status"] == "ok":
                        directions[device, s] = model.coef_[0] / norm
            fits.append(row)
    pair_rows = []
    for (d1, s1), (d2, s2) in itertools.combinations(directions, 2):
        kind = (
            "intra_device"
            if d1 == d2
            else ("inter_device_same_slice" if s1 == s2 else "inter_device_cross_slice")
        )
        pair_rows.append(
            dict(
                device_1=d1,
                slice_1=s1,
                device_2=d2,
                slice_2=s2,
                kind=kind,
                cosine=float(directions[d1, s1] @ directions[d2, s2]),
            )
        )
    # Direct response moments use one common challenge sample for all pairs.
    sampled_ids = np.random.default_rng(seed).permutation(
        dataset.manifest["train_stop"]
    )[:budget]
    response_rows = []

    def append_response_pair(kind, d1, s1, d2, s2):
        first = dataset.chains.get((d1, s1))
        second = dataset.chains.get((d2, s2))
        if first is None or second is None:
            return
        present = (first[sampled_ids] != 255) & (second[sampled_ids] != 255)
        ids = sampled_ids[present]
        if len(ids) < 2:
            return
        x = 1 - 2 * first[ids].astype(np.int8)
        y = 1 - 2 * second[ids].astype(np.int8)
        x_std, y_std = float(np.std(x)), float(np.std(y))
        pearson = float(np.corrcoef(x, y)[0, 1]) if x_std > 0 and y_std > 0 else np.nan
        response_rows.append(
            dict(
                kind=kind,
                device_1=d1,
                slice_1=s1,
                device_2=d2,
                slice_2=s2,
                challenge_count=len(ids),
                polar_product_mean=float(np.mean(x * y)),
                centered_pearson=pearson,
                agreement=float(np.mean(x == y)),
                uniformity_1=float(np.mean((1 - x) / 2)),
                uniformity_2=float(np.mean((1 - y) / 2)),
            )
        )

    for s in selected_slices:
        devices = [d for d in dataset.devices if (d, s) in dataset.chains]
        for d1, d2 in itertools.combinations(devices, 2):
            append_response_pair("inter_device_same_slice", d1, s, d2, s)
    for device in dataset.devices:
        available = [s for s in selected_slices if (device, s) in dataset.chains]
        for s1, s2 in itertools.combinations(available, 2):
            append_response_pair("intra_device_cross_slice", device, s1, device, s2)

    estimates = []
    for combo in dataset.combinations(ks, selected_slices):
        pairs = [
            r for r in pair_rows if r["slice_1"] in combo and r["slice_2"] in combo
        ]

        def moment(kind):
            values = [r["cosine"] for r in pairs if r["kind"] == kind]
            return float(np.mean(values)) if values else np.nan, len(values)

        lr_inter, lr_n_inter = moment("inter_device_same_slice")
        lr_intra, lr_n_intra = moment("intra_device")
        lr_cross, lr_n_cross = moment("inter_device_cross_slice")
        response = [
            r for r in response_rows if r["slice_1"] in combo and r["slice_2"] in combo
        ]

        def response_moment(kind):
            rows = [r for r in response if r["kind"] == kind]
            pair_mean = (
                float(np.mean([r["polar_product_mean"] for r in rows]))
                if rows
                else np.nan
            )
            observations = sum(r["challenge_count"] for r in rows)
            pooled = (
                float(
                    sum(r["polar_product_mean"] * r["challenge_count"] for r in rows)
                    / observations
                )
                if observations
                else np.nan
            )
            return pair_mean, pooled, len(rows), observations

        inter, inter_pooled, n_inter, obs_inter = response_moment(
            "inter_device_same_slice"
        )
        intra, intra_pooled, n_intra, obs_intra = response_moment(
            "intra_device_cross_slice"
        )
        alpha_squared = (
            float(np.sin(np.pi * inter / 2)) if np.isfinite(inter) else np.nan
        )
        intra_rho = float(np.sin(np.pi * intra / 2)) if np.isfinite(intra) else np.nan
        alpha_valid = bool(np.isfinite(alpha_squared) and 0 <= alpha_squared <= 1)
        alpha = float(np.sqrt(alpha_squared)) if alpha_valid else np.nan
        ratio = (
            intra_rho / alpha_squared
            if np.isfinite(intra_rho) and alpha_squared > 1e-12
            else np.nan
        )
        beta_valid = bool(np.isfinite(ratio) and 0 <= ratio <= 1)
        beta = float(np.sqrt(ratio)) if len(combo) > 1 and beta_valid else np.nan
        consistent = alpha_valid and (len(combo) == 1 or beta_valid)

        devices_in_moments = sorted(
            {
                row[device_key]
                for row in response
                for device_key in ("device_1", "device_2")
            }
        )
        jackknife_alpha, jackknife_beta = [], []
        for omitted in devices_in_moments:
            retained = [
                row
                for row in response
                if row["device_1"] != omitted and row["device_2"] != omitted
            ]
            inter_values = [
                row["polar_product_mean"]
                for row in retained
                if row["kind"] == "inter_device_same_slice"
            ]
            intra_values = [
                row["polar_product_mean"]
                for row in retained
                if row["kind"] == "intra_device_cross_slice"
            ]
            if inter_values:
                alpha_squared_j = float(
                    np.sin(np.pi * np.mean(inter_values) / 2)
                )
                if 0 <= alpha_squared_j <= 1:
                    jackknife_alpha.append(float(np.sqrt(alpha_squared_j)))
                    if len(combo) > 1 and intra_values and alpha_squared_j > 1e-12:
                        intra_rho_j = float(
                            np.sin(np.pi * np.mean(intra_values) / 2)
                        )
                        ratio_j = intra_rho_j / alpha_squared_j
                        if 0 <= ratio_j <= 1:
                            jackknife_beta.append(float(np.sqrt(ratio_j)))

        def jackknife_interval(estimate, values):
            values = np.asarray(values, dtype=float)
            if not np.isfinite(estimate) or len(values) < 2:
                return np.nan, np.nan, np.nan
            standard_error = np.sqrt(
                (len(values) - 1) / len(values)
                * np.sum((values - values.mean()) ** 2)
            )
            return (
                float(standard_error),
                float(np.clip(estimate - 1.96 * standard_error, 0, 1)),
                float(np.clip(estimate + 1.96 * standard_error, 0, 1)),
            )

        alpha_se, alpha_ci_low, alpha_ci_high = jackknife_interval(
            alpha, jackknife_alpha
        )
        beta_se, beta_ci_low, beta_ci_high = jackknife_interval(
            beta, jackknife_beta
        )
        estimates.append(
            dict(
                k_xor=len(combo),
                slices="+".join(combo),
                alpha_hat=alpha,
                beta_hat=beta,
                inter_response_moment=inter,
                intra_response_moment=intra,
                inter_response_moment_pooled=inter_pooled,
                intra_response_moment_pooled=intra_pooled,
                alpha_squared_raw=alpha_squared,
                alpha_squared_beta_squared_raw=intra_rho,
                beta_squared_raw=ratio,
                inter_pairs=n_inter,
                intra_pairs=n_intra,
                inter_observations=obs_inter,
                intra_observations=obs_intra,
                moment_model_consistent=consistent,
                jackknife_devices=len(devices_in_moments),
                alpha_jackknife_se=alpha_se,
                alpha_ci95_low=alpha_ci_low,
                alpha_ci95_high=alpha_ci_high,
                beta_jackknife_se=beta_se,
                beta_ci95_low=beta_ci_low,
                beta_ci95_high=beta_ci_high,
                estimator="polar_response_arcsine_moments",
                estimation_budget=budget,
                lr_rho_inter_raw=lr_inter,
                lr_rho_intra_raw=lr_intra,
                lr_rho_cross_raw=lr_cross,
                lr_inter_pairs=lr_n_inter,
                lr_intra_pairs=lr_n_intra,
                lr_cross_pairs=lr_n_cross,
                lr_min_fit_accuracy=min_accuracy,
            )
        )
    pl.DataFrame(fits).write_csv(output_dir / "chain_fits.csv")
    pl.DataFrame(
        pair_rows,
        schema={
            "device_1": pl.String,
            "slice_1": pl.String,
            "device_2": pl.String,
            "slice_2": pl.String,
            "kind": pl.String,
            "cosine": pl.Float64,
        },
    ).write_csv(output_dir / "chain_pairs.csv")
    response_schema = {
        "kind": pl.String,
        "device_1": pl.String,
        "slice_1": pl.String,
        "device_2": pl.String,
        "slice_2": pl.String,
        "challenge_count": pl.Int64,
        "polar_product_mean": pl.Float64,
        "centered_pearson": pl.Float64,
        "agreement": pl.Float64,
        "uniformity_1": pl.Float64,
        "uniformity_2": pl.Float64,
    }
    pl.DataFrame(response_rows, schema=response_schema).write_csv(
        output_dir / "response_moments.csv"
    )
    frame = pl.DataFrame(estimates)
    frame.write_csv(output_dir / "population_estimates.csv")

    estimate_lookup = {
        frozenset(row["slices"].split("+")): row
        for row in frame.iter_rows(named=True)
        if row["k_xor"] == 2
    }
    parameter_pairs = []
    for row in response_rows:
        rho = float(np.sin(np.pi * row["polar_product_mean"] / 2))
        result = {
            **row,
            "latent_rho_hat": rho,
            "alpha_pair_hat": np.nan,
            "gamma_pair_hat": np.nan,
            "beta_pair_hat": np.nan,
            "global_alpha_hat": np.nan,
            "valid_transformation": False,
            "invalid_reason": "",
        }
        if row["kind"] == "inter_device_same_slice":
            result["alpha_pair_hat"] = np.sqrt(rho) if 0 <= rho <= 1 else np.nan
            result["valid_transformation"] = bool(0 <= rho <= 1)
            if not result["valid_transformation"]:
                result["invalid_reason"] = "inter_rho_outside_[0,1]"
        elif row["kind"] == "intra_device_cross_slice":
            combo = frozenset((row["slice_1"], row["slice_2"]))
            estimate = estimate_lookup.get(combo)
            alpha_hat = estimate["alpha_hat"] if estimate else np.nan
            ratio = rho / alpha_hat**2 if np.isfinite(alpha_hat) and alpha_hat > 0 else np.nan
            result["global_alpha_hat"] = alpha_hat
            result["gamma_pair_hat"] = np.sqrt(rho) if 0 <= rho <= 1 else np.nan
            result["beta_pair_hat"] = np.sqrt(ratio) if 0 <= ratio <= 1 else np.nan
            result["valid_transformation"] = bool(0 <= ratio <= 1)
            if not 0 <= rho <= 1:
                result["invalid_reason"] = "intra_rho_outside_[0,1]"
            elif not np.isfinite(ratio):
                result["invalid_reason"] = "missing_global_alpha"
            elif not 0 <= ratio <= 1:
                result["invalid_reason"] = "beta_squared_outside_[0,1]"
        parameter_pairs.append(result)
    pl.DataFrame(parameter_pairs).write_csv(
        output_dir / "population_parameter_pairs.csv"
    )

    fit_rows = []
    for estimate in estimates:
        combo = set(estimate["slices"].split("+"))
        for kind, observed_name, predicted_rho in (
            (
                "inter_device_same_slice",
                "inter_response_moment",
                estimate["alpha_hat"] ** 2,
            ),
            (
                "intra_device_cross_slice",
                "intra_response_moment",
                estimate["alpha_hat"] ** 2 * estimate["beta_hat"] ** 2,
            ),
        ):
            values = np.asarray(
                [
                    row["polar_product_mean"]
                    for row in response_rows
                    if row["kind"] == kind
                    and row["slice_1"] in combo
                    and row["slice_2"] in combo
                ],
                dtype=float,
            )
            predicted = (
                float(2 / np.pi * np.arcsin(predicted_rho))
                if np.isfinite(predicted_rho) and 0 <= predicted_rho <= 1
                else np.nan
            )
            residuals = values - predicted
            fit_rows.append(
                {
                    "k_xor": estimate["k_xor"],
                    "slices": estimate["slices"],
                    "moment_kind": kind,
                    "n_pairs": len(values),
                    "observed_global_moment": estimate[observed_name],
                    "predicted_global_moment": predicted,
                    "pair_residual_mean": float(np.mean(residuals)) if len(values) else np.nan,
                    "pair_residual_std": float(np.std(residuals)) if len(values) else np.nan,
                    "pair_residual_mae": float(np.mean(np.abs(residuals))) if len(values) else np.nan,
                    "pair_residual_rmse": float(np.sqrt(np.mean(residuals**2))) if len(values) else np.nan,
                    "pair_residual_p05": float(np.quantile(residuals, 0.05)) if len(values) else np.nan,
                    "pair_residual_p95": float(np.quantile(residuals, 0.95)) if len(values) else np.nan,
                }
            )
    pl.DataFrame(fit_rows).write_csv(output_dir / "population_model_fit.csv")
    return frame
