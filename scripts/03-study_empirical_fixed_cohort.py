"""Reuse one empirical population model across a fixed held-out target cohort."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl

from puf_transfer.empirical import EmpiricalDataset
from puf_transfer.empirical_transfer import fixed_cohort_trial
from study_utils import parse_int_list, split_labeled_budget


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenges", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k-values", default="1,2,3,4")
    parser.add_argument("--k-pretrain", type=int, default=10)
    parser.add_argument("--n-base-budget", type=int, default=1000)
    parser.add_argument("--m-target-budget", type=int, default=100)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if min(
        args.k_pretrain,
        args.n_base_budget,
        args.m_target_budget,
        args.n_trials,
        args.epochs,
    ) < 1:
        parser.error("Counts and budgets must be positive.")
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be in (0, 1).")

    data = EmpiricalDataset(args.cache_dir, args.challenges)
    combos = data.combinations(parse_int_list(args.k_values), data.slices)
    source_split = split_labeled_budget(
        args.n_base_budget,
        validation_fraction=args.validation_fraction,
        min_validation=20,
        max_validation=data.manifest["validation_stop"]
        - data.manifest["train_stop"],
    )
    target_split = split_labeled_budget(
        args.m_target_budget,
        validation_fraction=args.validation_fraction,
        min_validation=20,
        max_validation=data.manifest["validation_stop"]
        - data.manifest["train_stop"],
    )

    jobs = []
    coverage = []
    for combo in combos:
        combo_name = "+".join(combo)
        eligible = []
        for device in data.devices:
            counts = {
                split: len(data.ids(device, combo, split))
                for split in ("train", "validation", "test")
            }
            complete = (
                counts["train"] >= max(source_split.train, target_split.train)
                and counts["validation"]
                >= max(source_split.validation, target_split.validation)
                and counts["test"] > 0
            )
            coverage.append(
                dict(
                    k_xor=len(combo),
                    slices=combo_name,
                    device_id=device,
                    eligible=complete,
                    **counts,
                )
            )
            if complete:
                eligible.append(device)
        if len(eligible) <= args.k_pretrain:
            parser.error(
                f"{combo_name} has {len(eligible)} eligible devices; fixed-cohort "
                f"evaluation needs more than {args.k_pretrain}."
            )
        combo_offset = int.from_bytes(
            hashlib.sha256(combo_name.encode()).digest()[:4], "little"
        )
        for trial in range(args.n_trials):
            rng = np.random.default_rng(args.seed + combo_offset + trial * 100_000)
            ordered = rng.permutation(eligible).tolist()
            sources = ordered[: args.k_pretrain]
            targets = ordered[args.k_pretrain :]
            jobs.append(
                dict(
                    cache_dir=str(args.cache_dir.resolve()),
                    challenges_path=str(args.challenges.resolve()),
                    slices=combo,
                    source_devices=sources,
                    target_devices=targets,
                    n_base_budget=args.n_base_budget,
                    m_target_budget=args.m_target_budget,
                    trial=trial,
                    seed=args.seed,
                    validation_fraction=args.validation_fraction,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    hidden_dim=args.hidden_dim,
                    embedding_dim=args.embedding_dim,
                    patience=args.patience,
                )
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "fixed_cohort_results.csv"
    metadata_path = args.output_dir / "fixed_cohort_run.json"
    metadata = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in {"resume", "n_jobs"}
    }
    metadata["cache_sha256"] = hashlib.sha256(
        (args.cache_dir / "manifest.json").read_bytes()
    ).hexdigest()
    completed = set()
    if output.exists():
        if (
            not args.resume
            or not metadata_path.exists()
            or json.loads(metadata_path.read_text()) != metadata
        ):
            parser.error(
                "Output exists. Use --resume with the identical configuration or "
                "choose a new output directory."
            )
        previous = pl.read_csv(output)
        complete = (
            previous.group_by(["slices", "cohort_trial"])
            .agg(
                pl.len().alias("rows"),
                pl.col("held_out_target_count").first().alias("expected"),
            )
            .filter(pl.col("rows") == pl.col("expected"))
        )
        completed = set(complete.select("slices", "cohort_trial").iter_rows())

    pending = [
        job
        for job in jobs
        if ("+".join(job["slices"]), job["trial"]) not in completed
    ]
    pl.DataFrame(coverage).write_csv(args.output_dir / "fixed_cohort_coverage.csv")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"{len(combos)} slice combinations; {len(jobs)} fixed source models; "
        f"{len(pending)} pending."
    )
    results = joblib.Parallel(n_jobs=args.n_jobs, return_as="generator", verbose=10)(
        joblib.delayed(fixed_cohort_trial)(**job) for job in pending
    )
    with output.open("a", newline="") as stream:
        writer = None
        for rows in results:
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                if stream.tell() == 0:
                    writer.writeheader()
            writer.writerows(rows)
            stream.flush()
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
