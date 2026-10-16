import argparse
from typing import Literal

import numpy as np
import polars as pl
from scipy.spatial.distance import pdist

from puf_transfer import (
    CorrelatedXORArbiterPUF,
    generate_weights_bias,
    logger,
    min_entropy,
    random_crps,
    shannon_entropy,
)
from study_utils import (
    ParallelConfig,
    StudyPaths,
    flatten_rows,
    parse_float_list,
    parse_int_list,
    parse_linspace,
    run_grid,
    write_csv,
)

def _nearest_point(df: pl.DataFrame, alpha: float, beta: float) -> dict:
    ranked = df.with_columns(
        ((pl.col("alpha") - alpha) ** 2 + (pl.col("beta") - beta) ** 2).alias(
            "_distance"
        )
    ).sort("_distance")
    row = ranked.row(0, named=True)
    row.pop("_distance", None)
    return row


def _metric_mask(df: pl.DataFrame, mode: str, tolerance: float) -> pl.Expr:
    if mode == "raw":
        return ((pl.col("uniformity_mean") - 0.5).abs() <= tolerance) & (
            (pl.col("uniqueness_mean") - 0.5).abs() <= tolerance
        )
    if mode == "entropy":
        return (pl.col("uniformity_h_mean") >= 1 - tolerance) & (
            pl.col("uniqueness_h_mean") >= 1 - tolerance
        )
    if mode == "min-entropy":
        return (pl.col("H_min_unif_mean") >= 1 - tolerance) & (
            pl.col("H_min_ba_mean") >= 1 - tolerance
        )
    if mode == "all":
        return (
            ((pl.col("uniformity_mean") - 0.5).abs() <= tolerance)
            & ((pl.col("uniqueness_mean") - 0.5).abs() <= tolerance)
            & (pl.col("uniformity_h_mean") >= 1 - tolerance)
            & (pl.col("uniqueness_h_mean") >= 1 - tolerance)
            & (pl.col("H_min_unif_mean") >= 1 - tolerance)
            & (pl.col("H_min_ba_mean") >= 1 - tolerance)
        )
    if mode == "baseline-relative":
        baseline = _nearest_point(df, 0.0, 0.0)
        return (
            (
                (pl.col("uniformity_mean") - baseline["uniformity_mean"]).abs()
                <= tolerance
            )
            & (
                (pl.col("uniqueness_mean") - baseline["uniqueness_mean"]).abs()
                <= tolerance
            )
            & (pl.col("uniformity_h_mean") >= baseline["uniformity_h_mean"] - tolerance)
            & (pl.col("uniqueness_h_mean") >= baseline["uniqueness_h_mean"] - tolerance)
            & (pl.col("H_min_unif_mean") >= baseline["H_min_unif_mean"] - tolerance)
            & (pl.col("H_min_ba_mean") >= baseline["H_min_ba_mean"] - tolerance)
        )
    raise ValueError(f"Unknown stealth metric mode: {mode}")


def compute_stealth_representatives(
    df: pl.DataFrame,
    *,
    mode: Literal["raw", "entropy", "min-entropy", "all", "baseline-relative"] = "raw",
    tolerances: list[float] | None = None,
) -> pl.DataFrame:
    tolerances = tolerances or [0.01, 0.03, 0.05, 0.10]
    if "uniformity_h_mean" not in df.columns and "uniformity_mean" in df.columns:
        df = df.with_columns(
            pl.col("uniformity_mean")
            .map_elements(
                lambda value: float(shannon_entropy(value)), return_dtype=pl.Float64
            )
            .alias("uniformity_h_mean")
        )
    if "uniqueness_h_mean" not in df.columns and "uniqueness_mean" in df.columns:
        df = df.with_columns(
            pl.col("uniqueness_mean")
            .map_elements(
                lambda value: float(shannon_entropy(value)), return_dtype=pl.Float64
            )
            .alias("uniqueness_h_mean")
        )
    rows = []
    required = {
        "k",
        "alpha",
        "beta",
        "uniformity_mean",
        "uniqueness_mean",
    }
    if mode in {"entropy", "all", "baseline-relative"}:
        required |= {"uniformity_h_mean", "uniqueness_h_mean"}
    if mode in {"min-entropy", "all", "baseline-relative"}:
        required |= {"H_min_unif_mean", "H_min_ba_mean"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns for stealth extraction: {sorted(missing)}"
        )

    for group, df_k in df.group_by("k", maintain_order=True):
        # k = int(group[0])
        baseline = _nearest_point(df_k, 0.0, 0.0)
        rows.append(
            {"selection": "baseline_uncorrelated", "tolerance": np.nan, **baseline}
        )

        for tolerance in tolerances:
            acceptable = df_k.filter(_metric_mask(df_k, mode, tolerance))
            if acceptable.is_empty():
                continue
            selected = acceptable.sort(
                ["expected_chain_corr", "alpha", "beta"], descending=[True, True, True]
            ).row(0, named=True)
            selected["selection"] = f"stealth_{mode}_tol_{tolerance:g}"
            selected["tolerance"] = tolerance
            rows.append(selected)

        health = df_k.sort(
            ["expected_chain_corr", "alpha", "beta"], descending=[True, True, True]
        ).row(0, named=True)
        health["selection"] = "high_correlation_health_check"
        health["tolerance"] = np.nan
        rows.append(health)

    return pl.DataFrame(rows).sort(["k", "selection"])


def make_study_grid(
    stealth_df: pl.DataFrame,
    *,
    mode: str,
    low_tolerance: float = 0.01,
    mid_tolerance: float = 0.03,
    high_tolerance: float = 0.05,
    stress_tolerance: float | None = 0.10,
) -> pl.DataFrame:
    targets = [
        ("baseline", "baseline_uncorrelated"),
        ("stealth_low", f"stealth_{mode}_tol_{low_tolerance:g}"),
        ("stealth_mid", f"stealth_{mode}_tol_{mid_tolerance:g}"),
        ("stealth_high", f"stealth_{mode}_tol_{high_tolerance:g}"),
    ]
    if stress_tolerance is not None:
        targets.append(("stress", f"stealth_{mode}_tol_{stress_tolerance:g}"))
    targets.append(("health", "high_correlation_health_check"))

    rows = []
    for label, source_selection in targets:
        selected = stealth_df.filter(pl.col("selection") == source_selection)
        for row in selected.iter_rows(named=True):
            row = dict(row)
            row["study_region"] = label
            row["source_selection"] = row.pop("selection")
            row["selection"] = label
            rows.append(row)

    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).sort(["k", "study_region"])


def compute_alpha_beta_metrics(
    alpha: float,
    beta: float,
    k: int = 1,
    n_bits: int = 64,
    n_devices: int = 100,
    budget: int = 5000,
    n_trials: int = 20,
    noisiness: float = 0.0,
    seed: int = 42,
):
    rows = []
    for trial in range(n_trials):
        trial_seed = seed + trial * 10_000

        # Ideal PUF parameters: mean 0, variance 1
        shared_weights, shared_bias = generate_weights_bias(
            n_bits, 0, 1, 0, 0.5, k=k, seed=trial_seed, beta=beta
        )

        x_raw = random_crps(budget, n_bits, seed=trial_seed)
        responses = np.empty((n_devices, budget), dtype=np.int8)
        latent_responses = np.empty_like(responses)

        for dev_id in range(n_devices):
            puf = CorrelatedXORArbiterPUF(
                n_bits,
                k,
                shared_weights,
                shared_bias,
                alpha,
                seed=trial_seed + dev_id + 1,
                noisiness=noisiness,
            )
            responses[dev_id, :] = np.where(puf.eval(x_raw) < 0, 1, 0)
            latent_responses[dev_id, :] = np.where(puf.val(x_raw) < 0, 1, 0)

        bitaliasing = responses.mean(axis=0)
        uniformity = responses.mean(axis=1)
        uniqueness = pdist(responses, "hamming")
        distributions = {
            "uniformity": uniformity,
            "bitaliasing": bitaliasing,
            "UPC": shannon_entropy(uniformity),
            "UPD": shannon_entropy(bitaliasing),
            "uniqueness": uniqueness,
            "uniqueness_h": shannon_entropy(uniqueness),
            "H_min_unif": min_entropy(uniformity),
            "H_min_ba": min_entropy(bitaliasing),
        }
        row = {
            "k": k,
            "alpha": alpha,
            "beta": beta,
            "inter_die_weight_corr": alpha**2,
            "intra_die_weight_corr": (alpha * beta) ** 2,
            "expected_chain_corr": (alpha * beta) ** 2,
            "noisiness": noisiness,
            "n_bits": n_bits,
            "n_devices": n_devices,
            "budget": budget,
            "trial": trial,
            "seed": trial_seed,
            "observed_ber": float(np.mean(responses != latent_responses)),
        }
        for name, values in distributions.items():
            row.update(
                {
                    f"{name}_mean": float(np.mean(values)),
                    f"{name}_std": float(np.std(values)),
                    f"{name}_p05": float(np.quantile(values, 0.05)),
                    f"{name}_median": float(np.median(values)),
                    f"{name}_p95": float(np.quantile(values, 0.95)),
                }
            )
        # Compatibility aliases used by the stealth-region code and old figures.
        row["uniformity_h_mean"] = row["UPC_mean"]
        row["uniformity_h_std"] = row["UPC_std"]
        rows.append(row)
    return rows


def aggregate_trials(df: pl.DataFrame) -> pl.DataFrame:
    keys = [
        "k",
        "alpha",
        "beta",
        "inter_die_weight_corr",
        "intra_die_weight_corr",
        "expected_chain_corr",
        "noisiness",
        "n_bits",
        "n_devices",
        "budget",
    ]
    values = [column for column in df.columns if column not in keys + ["trial", "seed"]]
    return (
        df.group_by(keys)
        .agg(
            pl.len().alias("n_trials"),
            *[pl.col(column).mean().alias(column) for column in values],
            *[
                pl.col(column).std().fill_null(0).alias(f"{column}_trial_std")
                for column in values
                if column.endswith("_mean") or column == "observed_ber"
            ],
        )
        .sort(["k", "alpha", "beta", "noisiness"])
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Map alpha and beta to k-XOR APUF population metrics."
    )
    parser.add_argument(
        "--run", action="store_true", help="Recompute the CSV before plotting."
    )
    parser.add_argument(
        "--k-values", default="1,2,3,4,5,6,7,8,9", help="Comma-separated k values."
    )
    parser.add_argument(
        "--alpha", default="0:1:21", help="Linspace as start:stop:count."
    )
    parser.add_argument(
        "--beta", default="0:1:21", help="Linspace as start:stop:count."
    )
    parser.add_argument("--n-bits", type=int, default=64)
    parser.add_argument("--n-devices", type=int, default=100)
    parser.add_argument("--budget", type=int, default=5000)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--noisiness", type=float, default=0.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=6)

    parser.add_argument("--output", default="alpha_beta_metrics_trials.csv")
    parser.add_argument("--summary-output", default="alpha_beta_metrics_summary.csv")

    parser.add_argument("--stealth-output", default="alpha_beta_stealth_regions.csv")
    parser.add_argument(
        "--study-grid-output",
        default="alpha_beta_study_grid.csv",
        help=(
            "Compact per-k alpha/beta grid for attack scripts. Includes baseline, "
            "stealth_low, stealth_mid, stealth_high, stress, and health when present."
        ),
    )
    parser.add_argument(
        "--stealth-mode",
        choices=["raw", "entropy", "min-entropy", "all", "baseline-relative"],
        default="raw",
        help="Metrics used for stealth tolerances.",
    )
    parser.add_argument(
        "--stealth-tolerances",
        default="0.01,0.03,0.05,0.10",
        help=(
            "Comma-separated tolerances. For raw mode this is distance from 0.5; "
            "for entropy mode this is distance from 1.0."
        ),
    )
    parser.add_argument(
        "--no-stress-region",
        action="store_true",
        help="Exclude the 0.10-tolerance stress point from the compact study grid.",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    paths = StudyPaths()
    paths.ensure()
    output_path = paths.data_dir / args.output

    if args.run:
        points = []
        beta_values = parse_linspace(args.beta)
        for k in parse_int_list(args.k_values):
            for alpha in parse_linspace(args.alpha):
                effective_betas = (
                    [0.0] if k == 1 or np.isclose(alpha, 0) else beta_values
                )
                for beta in effective_betas:
                    points.append({"k": k, "alpha": alpha, "beta": beta})
        logger.info("Grid size after removing equivalent points: %d", len(points))

        results = run_grid(
            (
                {
                    **params,
                    "n_bits": args.n_bits,
                    "n_devices": args.n_devices,
                    "budget": args.budget,
                    "n_trials": args.n_trials,
                    "noisiness": args.noisiness,
                    "seed": args.seed,
                }
                for params in points
            ),
            compute_alpha_beta_metrics,
            ParallelConfig(n_jobs=args.n_jobs, verbose=20),
        )
        df = write_csv(flatten_rows(results), output_path)
        summary_df = aggregate_trials(df)
        summary_df.write_csv(paths.data_dir / args.summary_output)
    else:
        if not output_path.exists():
            parser.error(f"Output does not exist: {output_path}. Use --run first.")
        df = pl.read_csv(output_path)
        summary_path = paths.data_dir / args.summary_output
        summary_df = (
            pl.read_csv(summary_path) if summary_path.exists() else aggregate_trials(df)
        )

    # Compute "stealth regions" and write them to a CSV
    tolerances = parse_float_list(args.stealth_tolerances)
    stealth_df = compute_stealth_representatives(
        summary_df,
        mode=args.stealth_mode,
        tolerances=tolerances,
    )
    stealth_df.write_csv(paths.data_dir / args.stealth_output)
    study_grid = make_study_grid(
        stealth_df,
        mode=args.stealth_mode,
        stress_tolerance=None if args.no_stress_region else 0.10,
    )
    if not study_grid.is_empty():
        study_grid.write_csv(paths.data_dir / args.study_grid_output)
    report_columns = [
        "selection",
        "k",
        "alpha",
        "beta",
        "inter_die_weight_corr",
        "intra_die_weight_corr",
        "uniformity_mean",
        "uniqueness_mean",
        "H_min_ba_mean",
    ]
    print(stealth_df.select(report_columns))
    if not study_grid.is_empty():
        print(
            study_grid.select(
                ["selection", "k", "alpha", "beta", "expected_chain_corr"]
            )
        )
