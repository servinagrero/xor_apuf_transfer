"""Prepare streamed FPGA data, estimate population moments, and run NN transfer."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl

from puf_transfer.empirical import (
    compute_population_metrics,
    EmpiricalDataset,
    estimate_population,
    prepare_empirical_dataset,
)
from puf_transfer.empirical_transfer import transfer_trial
from puf_transfer.empirical_validation import (
    filter_source_pool,
    load_metric_distributions,
    source_distance_statistics,
    uniqueness_geometry,
)
from study_utils import (
    parse_int_list,
    parse_str_list,
    split_labeled_budget,
)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--estimate", action="store_true")
    p.add_argument(
        "--metrics",
        action="store_true",
        help="Compute empirical quality-metric distributions without fitting chain models.",
    )
    p.add_argument("--run", action="store_true")
    p.add_argument("--csv", type=Path, default=Path("data/raw/crps_short.csv"))
    p.add_argument(
        "--challenges", type=Path, default=Path("data/raw/all_challenges.npy")
    )
    p.add_argument("--cache-dir", type=Path, default=Path("data/empirical_cache"))
    p.add_argument("--output-dir", type=Path, default=Path("data/empirical_transfer"))
    p.add_argument("--train-stop", type=int, default=1_000_000)
    p.add_argument("--validation-size", type=int, default=1000)
    p.add_argument(
        "--bit-order",
        choices=["forward", "reverse"],
        default="forward",
        help="Challenge-column order for a new cache; choose using the hardware mapping, not test accuracy.",
    )
    p.add_argument("--k-values", default="1,2,3,4")
    p.add_argument("--slices", default=None)
    p.add_argument(
        "--targets",
        default=None,
        help="Comma-separated UIDs; default: all eligible devices.",
    )
    p.add_argument("--k-pretrain", default="1,2,10,20,28")
    p.add_argument("--n-base-budget", default="1000,5000")
    p.add_argument("--m-target-budget", default="100,500,1000,5000")
    p.add_argument("--validation-fraction", type=float, default=0.2)
    p.add_argument("--estimate-budget", type=int, default=20000)
    p.add_argument("--metric-budget", type=int, default=20000)
    p.add_argument("--metric-null-trials", type=int, default=100)
    p.add_argument(
        "--source-policy",
        choices=["random", "leave-cluster-out", "exclude-nearest"],
        default="random",
    )
    p.add_argument(
        "--metric-distributions",
        type=Path,
        default=None,
        help="Population metric NPY required by uniqueness-aware source policies.",
    )
    p.add_argument("--cluster-threshold", type=float, default=0.20)
    p.add_argument(
        "--exclude-nearest-count",
        type=int,
        default=3,
        help=(
            "For --source-policy exclude-nearest, remove this many sources "
            "with the smallest target-source uniqueness before source sampling."
        ),
    )
    p.add_argument("--diagnostic-baselines", action="store_true")
    p.add_argument(
        "--source-label-control",
        choices=["none", "challenge-permuted"],
        default="none",
        help="Negative control that destroys challenge-aligned source structure.",
    )
    p.add_argument("--min-chain-accuracy", type=float, default=0.9)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--embedding-dim", type=int, default=16)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--n-trials", type=int, default=1)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--insufficient-sources",
        choices=["skip", "error"],
        default="skip",
        help="Skip infeasible target/combination/source-count cells, or stop at the first one.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume completed configurations from matching run metadata.",
    )
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not (args.prepare or args.estimate or args.metrics or args.run):
        parser.print_help()
        return
    if (
        min(
            args.n_trials,
            args.epochs,
            args.batch_size,
            args.estimate_budget,
            args.metric_budget,
        )
        < 1
    ):
        parser.error("Counts and budgets must be positive.")
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be in (0, 1).")
    if args.exclude_nearest_count < 1:
        parser.error("--exclude-nearest-count must be positive.")
    if args.prepare:
        prepare_empirical_dataset(
            args.csv,
            args.challenges,
            args.cache_dir,
            train_stop=args.train_stop,
            validation_size=args.validation_size,
            bit_order=args.bit_order,
        )
    data = EmpiricalDataset(args.cache_dir, args.challenges)
    print(
        f"Cache: {len(data.devices)} devices, {len(data.slices)} slices, "
        f"{sum(r['rows'] for r in data.manifest['chains'])} measured chain CRPs."
    )
    if not (args.estimate or args.metrics or args.run):
        return
    ks = parse_int_list(args.k_values)
    slices = parse_str_list(args.slices) if args.slices else data.slices
    combos = data.combinations(ks, slices)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.estimate or args.metrics:
        _, metric_summary = compute_population_metrics(
            data,
            args.output_dir,
            budget=args.metric_budget,
            seed=args.seed,
            ks=ks,
            slices=slices,
            null_trials=args.metric_null_trials,
        )
        with pl.Config(tbl_rows=-1, tbl_cols=-1):
            print(metric_summary)
    metric_distributions = None
    if args.source_policy != "random" or args.diagnostic_baselines:
        metric_path = args.metric_distributions or (
            args.output_dir / "population_metric_distributions.npy"
        )
        if not metric_path.exists():
            parser.error(f"Uniqueness-aware validation requires {metric_path}.")
        metric_distributions = load_metric_distributions(metric_path)
    if args.estimate:
        estimate_population(
            data,
            args.output_dir,
            budget=args.estimate_budget,
            seed=args.seed,
            min_accuracy=args.min_chain_accuracy,
            ks=ks,
            slices=slices,
        )
    if not args.run:
        return
    if len(data.devices) < 2:
        parser.error(
            "Transfer needs at least two devices. The short fixture cannot run this study."
        )
    source_counts = sorted(set(parse_int_list(args.k_pretrain)))
    source_budgets = sorted(set(parse_int_list(args.n_base_budget)))
    target_budgets = sorted(set(parse_int_list(args.m_target_budget)))
    if (
        not all((source_counts, source_budgets, target_budgets))
        or min(source_counts) < 1
        or min(source_budgets + target_budgets) < 2
    ):
        parser.error("Source counts must be positive and labeled budgets at least two.")
    target_splits = {
        budget: split_labeled_budget(
            budget,
            validation_fraction=args.validation_fraction,
            min_validation=20,
            max_validation=(
                data.manifest["validation_stop"] - data.manifest["train_stop"]
            ),
        )
        for budget in target_budgets
    }
    max_target_train = max(split.train for split in target_splits.values())
    max_target_validation = max(
        split.validation for split in target_splits.values()
    )
    targets = parse_str_list(args.targets) if args.targets else data.devices
    if not set(targets) <= set(data.devices):
        parser.error("Unknown target UID.")
    jobs, coverage, skipped = [], [], []
    for combo in combos:
        combo_name = "+".join(combo)
        geometry = (
            uniqueness_geometry(metric_distributions, combo_name, args.cluster_threshold)
            if metric_distributions is not None
            else None
        )
        counts = {
            d: {s: len(data.ids(d, combo, s)) for s in ("train", "validation", "test")}
            for d in data.devices
        }
        for d, c in counts.items():
            coverage.append(
                dict(k_xor=len(combo), slices="+".join(combo), device_id=d, **c)
            )
        for target in targets:
            c = counts[target]
            if (
                c["train"] < max_target_train
                or c["validation"] < max_target_validation
                or not c["test"]
            ):
                if args.insufficient_sources == "error":
                    parser.error(
                        f"Incomplete target {target}, slices {combo}: {c}. "
                        "Choose smaller budgets or explicit targets/slices."
                    )
                for n in source_budgets:
                    for trial in range(args.n_trials):
                        for count in source_counts:
                            skipped.append(
                                dict(
                                    k_xor=len(combo),
                                    slices="+".join(combo),
                                    target_device=target,
                                    n_base_budget=n,
                                    trial=trial,
                                    requested_k_pretrain=count,
                                    eligible_source_count=0,
                                    eligible_source_devices="[]",
                                    ineligible_source_devices="[]",
                                    target_train_count=c["train"],
                                    target_validation_count=c["validation"],
                                    target_test_count=c["test"],
                                    reason="insufficient_complete_target_data",
                                )
                            )
                continue
            for n in source_budgets:
                source_split = split_labeled_budget(
                    n,
                    validation_fraction=args.validation_fraction,
                    min_validation=20,
                    max_validation=(
                        data.manifest["validation_stop"]
                        - data.manifest["train_stop"]
                    ),
                )
                eligible = [
                    d
                    for d in data.devices
                    if d != target
                    and counts[d]["train"] >= source_split.train
                    and counts[d]["validation"] >= source_split.validation
                ]
                target_cluster = -1
                excluded_cluster = []
                if geometry is not None:
                    eligible, target_cluster, excluded_cluster = filter_source_pool(
                        geometry,
                        target,
                        eligible,
                        args.source_policy,
                        exclude_nearest_count=args.exclude_nearest_count,
                    )
                for trial in range(args.n_trials):
                    rng = np.random.default_rng(
                        args.seed + trial * 100_000 + data.devices.index(target)
                    )
                    ordered = rng.permutation(eligible).tolist()
                    for count in source_counts:
                        if count > len(eligible):
                            skipped.append(
                                dict(
                                    k_xor=len(combo),
                                    slices="+".join(combo),
                                    target_device=target,
                                    n_base_budget=n,
                                    trial=trial,
                                    requested_k_pretrain=count,
                                    eligible_source_count=len(eligible),
                                    eligible_source_devices=json.dumps(
                                        sorted(eligible)
                                    ),
                                    ineligible_source_devices=json.dumps(
                                        sorted(
                                            d
                                            for d in data.devices
                                            if d != target and d not in eligible
                                        )
                                    ),
                                    target_train_count=c["train"],
                                    target_validation_count=c["validation"],
                                    target_test_count=c["test"],
                                    reason="insufficient_complete_source_devices",
                                )
                            )
                            if args.insufficient_sources == "error":
                                parser.error(
                                    f"Target {target}, slices {combo}: only {len(eligible)} "
                                    f"eligible sources at budget {n}; requested {count}."
                                )
                            continue
                        selected_sources = ordered[:count]
                        source_stats = (
                            source_distance_statistics(geometry, target, selected_sources)
                            if geometry is not None
                            else None
                        )
                        jobs.append(
                            dict(
                                cache_dir=str(args.cache_dir.resolve()),
                                challenges_path=str(args.challenges.resolve()),
                                slices=combo,
                                target=target,
                                source_devices=selected_sources,
                                n_base_budget=n,
                                m_budgets=target_budgets,
                                trial=trial,
                                seed=args.seed,
                                validation_fraction=args.validation_fraction,
                                epochs=args.epochs,
                                batch_size=args.batch_size,
                                hidden_dim=args.hidden_dim,
                                embedding_dim=args.embedding_dim,
                                patience=args.patience,
                                source_policy=args.source_policy,
                                cluster_threshold=args.cluster_threshold
                                if geometry is not None
                                else np.nan,
                                target_cluster=target_cluster,
                                excluded_cluster_devices=excluded_cluster,
                                source_uniqueness_values=(
                                    source_stats["source_uniqueness_values"]
                                    if source_stats is not None
                                    else None
                                ),
                                diagnostic_baselines=args.diagnostic_baselines,
                                source_label_control=args.source_label_control,
                            )
                        )
    output = args.output_dir / "transfer_results.csv"
    metadata_path = args.output_dir / "transfer_run.json"
    metadata = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()
        if k
        not in (
            "prepare",
            "estimate",
            "metrics",
            "metric_budget",
            "source_policy",
            "metric_distributions",
            "cluster_threshold",
            "diagnostic_baselines",
            "run",
            "resume",
            "n_jobs",
            "csv",
            "insufficient_sources",
        )
    }
    if args.source_policy != "random":
        metadata["source_policy"] = args.source_policy
        metadata["cluster_threshold"] = args.cluster_threshold
        metadata["exclude_nearest_count"] = args.exclude_nearest_count
        metadata["metric_distributions"] = str(args.metric_distributions)
    if args.diagnostic_baselines:
        metadata["diagnostic_baselines"] = True
    metadata["cache_sha256"] = hashlib.sha256(
        (args.cache_dir / "manifest.json").read_bytes()
    ).hexdigest()
    metadata["train_stop"] = data.manifest["train_stop"]
    metadata["validation_size"] = (
        data.manifest["validation_stop"] - data.manifest["train_stop"]
    )
    metadata["bit_order"] = data.manifest.get("bit_order", "forward")

    def key(row):
        return (
            row["slices"],
            row["target_device"],
            int(row["k_pretrain"]),
            int(row["n_base_budget"]),
            int(row["trial"]),
        )

    completed = {}
    if output.exists():
        if (
            not args.resume
            or not metadata_path.exists()
            or json.loads(metadata_path.read_text()) != metadata
        ):
            parser.error(
                "Output exists. Use a new output directory or --resume with the identical configuration."
            )
        for row in pl.read_csv(output).iter_rows(named=True):
            completed.setdefault(key(row), set()).add(row["m_target_budget"])
    pl.DataFrame(coverage).write_csv(args.output_dir / "coverage.csv")
    skipped_schema = {
        "k_xor": pl.Int64,
        "slices": pl.String,
        "target_device": pl.String,
        "n_base_budget": pl.Int64,
        "trial": pl.Int64,
        "requested_k_pretrain": pl.Int64,
        "eligible_source_count": pl.Int64,
        "eligible_source_devices": pl.String,
        "ineligible_source_devices": pl.String,
        "target_train_count": pl.Int64,
        "target_validation_count": pl.Int64,
        "target_test_count": pl.Int64,
        "reason": pl.String,
    }
    pl.DataFrame(skipped, schema=skipped_schema).write_csv(
        args.output_dir / "skipped_source_configurations.csv"
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    if not jobs:
        parser.error(
            "No feasible transfer configurations remain. See coverage.csv and "
            "skipped_source_configurations.csv."
        )
    pending = []
    for job in jobs:
        job_key = (
            "+".join(job["slices"]),
            job["target"],
            len(job["source_devices"]),
            job["n_base_budget"],
            job["trial"],
        )
        if not set(target_budgets) <= completed.get(job_key, set()):
            pending.append(job)
    print(
        f"{len(combos)} slice combinations; {len(jobs)} feasible source fits; "
        f"{len(pending)} pending; {len(skipped)} infeasible configurations skipped."
    )
    if skipped:
        minimum = min(row["eligible_source_count"] for row in skipped)
        print(
            "See skipped_source_configurations.csv for omissions. "
            f"The smallest source pool among skipped cells is {minimum}."
        )
    results = joblib.Parallel(n_jobs=args.n_jobs, return_as="generator", verbose=10)(
        joblib.delayed(transfer_trial)(**job) for job in pending
    )
    with output.open("a", newline="") as stream:
        writer = None
        for rows in results:
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                if stream.tell() == 0:
                    writer.writeheader()
            for row in rows:
                if row["m_target_budget"] not in completed.get(key(row), set()):
                    writer.writerow(row)
            stream.flush()
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
