"""Create empirical transfer, population-diagnostic, and cost artifacts."""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import t as student_t

from puf_transfer.empirical_validation import (
    annotate_transfer_results,
    clustering_tables,
    distance_correlations,
    load_metric_distributions,
)

METHODS = (
    ("transfer_acc", "Transfer (embedding only)", "#0072B2"),
    ("baseline_acc", "From-scratch NN", "#D55E00"),
    ("zero_shot_acc", "Zero-shot (mean source embedding)", "#009E73"),
)

SLICE_LABELS = {
    "LEFT-LOWER-0": "LL-0",
    "LEFT-LOWER-1": "LL-1",
    "LEFT-UPPER-0": "LU-0",
    "LEFT-UPPER-1": "LU-1",
}


def save_figure(fig, output, formats, *, dpi=300):
    """Save a figure in every requested format."""
    for extension in formats:
        fig.savefig(output.with_suffix(f".{extension}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def short_slice_label(value):
    """Shorten slice names for presentation without changing stored identifiers."""
    return "+".join(SLICE_LABELS.get(part, part) for part in str(value).split("+"))


def summarize_device_results(devices, keys, metrics, *, resamples=2000, seed=42):
    """Compute uncertainty with the physical target device as sampling unit."""
    rows = []
    rng = np.random.default_rng(seed)
    for group_values, group in devices.group_by(keys, maintain_order=True):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        row = dict(zip(keys, group_values))
        row["n_target_devices"] = group["target_device"].n_unique()
        for metric in metrics:
            values = group[metric].to_numpy()
            row[f"mean_{metric}"] = float(np.mean(values))
            row[f"std_{metric}"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
            if len(values) > 1 and resamples:
                means = np.mean(
                    rng.choice(values, size=(resamples, len(values)), replace=True),
                    axis=1,
                )
                row[f"ci95_low_{metric}"] = float(np.quantile(means, 0.025))
                row[f"ci95_high_{metric}"] = float(np.quantile(means, 0.975))
            else:
                row[f"ci95_low_{metric}"] = row[f"mean_{metric}"]
                row[f"ci95_high_{metric}"] = row[f"mean_{metric}"]
        rows.append(row)
    return pl.DataFrame(rows).sort(keys)


def analyze_similarity_control(
    random_path, excluded_path, metrics_path, output_dir, threshold
):
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = load_metric_distributions(metrics_path)
    random_frame = pl.read_csv(random_path)
    excluded = pl.read_csv(excluded_path)
    annotated = annotate_transfer_results(random_frame, metrics, threshold)
    annotated.write_csv(output_dir / "transfer_distance_rows.csv")
    distance_correlations(annotated).write_csv(
        output_dir / "transfer_distance_correlations.csv"
    )
    pairs, clusters = clustering_tables(metrics, (0.15, 0.20, 0.25))
    pairs.write_csv(output_dir / "uniqueness_pairs.csv")
    clusters.write_csv(output_dir / "device_clusters.csv")

    keys = [
        "k_xor",
        "slices",
        "target_device",
        "k_pretrain",
        "n_base_budget",
        "m_target_budget",
        "trial",
    ]
    columns = ["transfer_acc", "zero_shot_acc", "baseline_acc"]
    matched = (
        random_frame.select(keys + columns)
        .unique(subset=keys)
        .join(
            excluded.select(keys + columns).unique(subset=keys),
            on=keys,
            suffix="_similarity_control",
        )
        .with_columns(
            (pl.col("transfer_acc_similarity_control") - pl.col("transfer_acc")).alias(
                "delta_transfer_acc"
            )
        )
    )
    matched.write_csv(output_dir / "source_policy_comparison.csv")
    summary = (
        matched.group_by(["k_xor", "k_pretrain", "m_target_budget"])
        .agg(
            pl.len().alias("n_matched_cells"),
            pl.col("transfer_acc").mean().alias("random_transfer_acc"),
            pl.col("transfer_acc_similarity_control")
            .mean()
            .alias("similarity_control_transfer_acc"),
            pl.col("delta_transfer_acc").mean().alias("mean_delta_transfer_acc"),
        )
        .sort(["k_xor", "k_pretrain", "m_target_budget"])
    )
    summary.write_csv(output_dir / "source_policy_summary.csv")


def plot_population_metrics(summary_path, null_path, output):
    observed = pl.read_csv(summary_path)
    null = pl.read_csv(null_path)
    metrics = ["UPD", "UPC", "Uniqueness", "H(Uniqueness)", "Min_H_UPD"]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6), squeeze=False)
    for ax, metric in zip(axes.flat, metrics):
        obs = observed.filter(pl.col("metric") == metric)
        nul = null.filter(pl.col("metric") == metric)
        for k in sorted(obs["k_xor"].unique().to_list()):
            values = obs.filter(pl.col("k_xor") == k)["mean_metric"].to_numpy()
            null_values = nul.filter(pl.col("k_xor") == k)["null_mean"].to_numpy()
            x = np.full(len(values), k, dtype=float)
            ax.scatter(x - 0.08, values, s=12, alpha=0.7, color="#0072B2")
            ax.scatter(
                x[: len(null_values)] + 0.08,
                null_values,
                s=12,
                alpha=0.7,
                color="#D55E00",
            )
        ax.set_title(metric)
        ax.set_xlabel("XOR degree")
        ax.set_ylim(0, 1.01)
        ax.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Metric value")
    axes[1, 0].set_ylabel("Metric value")
    axes.flat[-1].set_axis_off()
    axes[0, 0].scatter([], [], color="#0072B2", label="observed")
    axes[0, 0].scatter([], [], color="#D55E00", label="permuted null")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metric_distributions(metrics_path, output, formats):
    metrics = load_metric_distributions(metrics_path)
    definitions = (
        ("uniformity", "Uniformity", 0.5),
        ("bitaliasing", "Bit-aliasing", 0.5),
        ("uniqueness", "Uniqueness", 0.5),
        ("UPD", "UPD", 1.0),
        ("UPC", "UPC", 1.0),
        ("H_uniqueness", r"$H(\mathrm{Uniqueness})$", 1.0),
    )
    entries = list(metrics["combinations"].values())
    ks = sorted({entry["k_xor"] for entry in entries})
    fig, axes = plt.subplots(2, 3, figsize=(12, 5), sharex=True, squeeze=False)
    for ax, (key, title, ideal) in zip(axes.flat, definitions):
        grouped = []
        for k in ks:
            values = [
                np.asarray(entry[key], dtype=float)
                for entry in entries
                if entry["k_xor"] == k
            ]
            grouped.append(np.concatenate(values))
        boxes = ax.boxplot(
            grouped, positions=ks, widths=0.55, showfliers=False, patch_artist=True
        )
        for box in boxes["boxes"]:
            box.set(facecolor="#D9EAF4", edgecolor="#0072B2")
        ax.axhline(ideal, color="black", linestyle=":", linewidth=1)
        ax.set_title(title)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis="y", alpha=0.2)
    for ax in axes[-1]:
        ax.set_xlabel("XOR degree")
        ax.set_xticks(ks)
    for ax in axes[:, 0]:
        ax.set_ylabel("Metric value")
    fig.tight_layout()
    save_figure(fig, output, formats)


def plot_parameter_diagnostics(parameter_path, fit_path, output_dir, formats):
    pairs = pl.read_csv(parameter_path)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), squeeze=False)
    specs = [
        (
            "alpha_pair_hat",
            r"Effective inter-die $\hat{\alpha}$",
            "inter_device_same_slice",
        ),
        (
            "gamma_pair_hat",
            r"Effective $\widehat{\alpha\beta}$",
            "intra_device_cross_slice",
        ),
        (
            "beta_pair_hat",
            r"Derived effective $\hat{\beta}$",
            "intra_device_cross_slice",
        ),
    ]
    for ax, (column, title, kind) in zip(axes.flat, specs):
        panel = pairs.filter(pl.col("kind") == kind)
        group_columns = (
            ["slice_1"] if kind == "inter_device_same_slice" else ["slice_1", "slice_2"]
        )
        for group_values, group in panel.group_by(group_columns, maintain_order=True):
            if not isinstance(group_values, tuple):
                group_values = (group_values,)
            values = np.sort(group[column].drop_nans().to_numpy())
            if not len(values):
                continue
            label = "+".join(short_slice_label(value) for value in group_values)
            ax.step(
                values,
                np.arange(1, len(values) + 1) / len(values),
                where="post",
                label=label,
            )
        valid = len(panel[column].drop_nans())
        invalid = len(panel) - valid
        ax.set(
            title=title,
            xlabel="Effective coefficient",
            ylabel="ECDF",
            xlim=(0, 1),
            ylim=(0, 1.01),
        )
        ax.text(
            0.02,
            0.96,
            f"invalid={invalid}/{len(panel)}",
            fontsize=10,
            transform=ax.transAxes,
            va="top",
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=6, loc="lower left")
    fig.tight_layout()
    save_figure(fig, output_dir / "population_parameters", formats)

    fit = pl.read_csv(fit_path).filter(pl.col("n_pairs") > 0)
    labels = [
        f"{row['k_xor']}:{short_slice_label(row['slices'])}"
        for row in fit.iter_rows(named=True)
    ]
    colors = [
        "#0072B2" if value == "inter_device_same_slice" else "#D55E00"
        for value in fit["moment_kind"]
    ]
    fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(fit)), 3.5))
    ax.bar(np.arange(len(fit)), fit["pair_residual_rmse"], color=colors)
    ax.set_xticks(np.arange(len(fit)), labels, rotation=90, fontsize=6)
    ax.set_ylabel("Pairwise moment residual RMSE")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(
        output_dir / "population_model_residuals.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)


def build_cost_table(empirical, theory, amortization_targets):
    rows = []

    def append_row(domain, row, method, accuracy, low, high):
        uses_sources = method != "target_only"
        source_count = row["k_pretrain"] if uses_sources else 0
        source_budget = row["n_base_budget"] if uses_sources else 0
        source_cost = source_count * source_budget
        target_budget = 0 if method == "zero_shot" else row["m_target_budget"]
        for targets in amortization_targets:
            rows.append(
                {
                    "domain": domain,
                    "k_xor": row["k_xor"],
                    "slices_or_region": row.get("slices", row.get("selection", "")),
                    "alpha": row.get("alpha", np.nan),
                    "beta": row.get("beta", np.nan),
                    "target_ber": row.get("target_ber", np.nan),
                    "k_pretrain": source_count,
                    "n_base_budget": source_budget,
                    "source_pretraining_cost": source_cost,
                    "m_target_budget": target_budget,
                    "total_attack_cost": source_cost + target_budget,
                    "amortization_targets": targets,
                    "amortized_cost_per_target": target_budget + source_cost / targets,
                    "method": method,
                    "mean_accuracy": accuracy,
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_target_devices": row.get("n_target_devices", np.nan),
                    "n_trials": row.get("n_trials", np.nan),
                }
            )

    for row in empirical.iter_rows(named=True):
        for metric, method in (
            ("transfer_acc", "transfer"),
            ("baseline_acc", "target_only"),
            ("zero_shot_acc", "zero_shot"),
        ):
            append_row(
                "empirical",
                row,
                method,
                row[f"mean_{metric}"],
                row[f"ci95_low_{metric}"],
                row[f"ci95_high_{metric}"],
            )
    if theory is not None and len(theory):
        keys = [
            "selection",
            "k_xor",
            "alpha",
            "beta",
            "target_ber",
            "k_pretrain",
            "n_base_budget",
            "m_target_budget",
        ]
        aggregate = theory.group_by(keys).agg(
            pl.len().alias("n_trials"),
            *[
                pl.col(metric).mean().alias(f"mean_{metric}")
                for metric in ("transfer_acc", "baseline_acc", "zero_shot_acc")
            ],
            *[
                (1.96 * pl.col(metric).std().fill_null(0) / pl.len().sqrt()).alias(
                    f"ci_{metric}"
                )
                for metric in ("transfer_acc", "baseline_acc", "zero_shot_acc")
            ],
        )
        for row in aggregate.iter_rows(named=True):
            for metric, method in (
                ("transfer_acc", "transfer"),
                ("baseline_acc", "target_only"),
                ("zero_shot_acc", "zero_shot"),
            ):
                mean, ci = row[f"mean_{metric}"], row[f"ci_{metric}"]
                append_row("theoretical", row, method, mean, mean - ci, mean + ci)
    return pl.DataFrame(rows, infer_schema_length=None).unique(maintain_order=True)


def summarize_costs(costs, thresholds=(0.75, 0.90, 0.95)):
    rows = []
    keys = ["domain", "k_xor", "slices_or_region", "method", "amortization_targets"]
    for values, group in costs.group_by(keys):
        identity = dict(zip(keys, values))
        for threshold in thresholds:
            eligible = group.filter(pl.col("mean_accuracy") >= threshold).sort(
                ["amortized_cost_per_target", "mean_accuracy"], descending=[False, True]
            )
            rows.append(
                {
                    **identity,
                    "accuracy_threshold": threshold,
                    "minimum_amortized_cost": (
                        eligible["amortized_cost_per_target"][0]
                        if len(eligible)
                        else np.nan
                    ),
                    "achieved_accuracy": eligible["mean_accuracy"][0]
                    if len(eligible)
                    else np.nan,
                    "k_pretrain": eligible["k_pretrain"][0] if len(eligible) else None,
                    "n_base_budget": eligible["n_base_budget"][0]
                    if len(eligible)
                    else None,
                    "m_target_budget": eligible["m_target_budget"][0]
                    if len(eligible)
                    else None,
                }
            )
    return pl.DataFrame(rows).sort(keys + ["accuracy_threshold"])


def plot_cost_frontiers(costs, output):
    targets = int(costs["amortization_targets"].max())
    panel = costs.filter(
        (pl.col("amortization_targets") == targets) & (pl.col("domain") == "empirical")
    )
    ks = sorted(panel["k_xor"].unique().to_list())
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7), squeeze=False, sharey=True)
    colors = {"transfer": "#0072B2", "target_only": "#D55E00", "zero_shot": "#009E73"}
    for ax, k in zip(axes.flat, ks):
        subset = panel.filter(pl.col("k_xor") == k)
        for method in subset["method"].unique().sort().to_list():
            points = (
                subset.filter(pl.col("method") == method)
                .group_by("amortized_cost_per_target")
                .agg(pl.col("mean_accuracy").mean().alias("mean_accuracy"))
                .sort("amortized_cost_per_target")
            )
            x = points["amortized_cost_per_target"].to_numpy()
            y = points["mean_accuracy"].to_numpy()
            keep = np.maximum.accumulate(y) == y
            ax.scatter(x, y, color=colors[method], alpha=0.18, s=10)
            ax.plot(
                x[keep],
                y[keep],
                color=colors[method],
                marker="o",
                markersize=3,
                label=method.replace("_", " "),
            )
        ax.set_title(f"{k}-XOR")
        ax.set_xscale("log")
        ax.set_ylim(0.45, 1.01)
        ax.grid(alpha=0.2)
    for ax in axes[-1]:
        ax.set_xlabel(f"Amortized CRPs/target (T={targets})")
    for ax in axes[:, 0]:
        ax.set_ylabel("Best achieved accuracy")
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(
        "Empirical accuracy-cost Pareto frontiers; faint points are dominated configurations"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_cost_latex(summary, output):
    selected = summary.filter(
        pl.col("amortization_targets") == pl.col("amortization_targets").max()
    )
    lines = [
        r"\begin{tabular}{lllrlrrrr}",
        r"\toprule",
        r"Domain & Region & Method & $k$ & Accuracy & Sources & CRPs/source & Target CRPs & Amortized CRPs \\",
        r"\midrule",
    ]
    for row in selected.iter_rows(named=True):
        cost = row["minimum_amortized_cost"]
        region = short_slice_label(row["slices_or_region"]).replace("_", r"\_")
        if np.isfinite(cost):
            line = (
                f"{row['domain']} & {region} & "
                f"{row['method'].replace('_', ' ')} & {row['k_xor']} & "
                f"{row['accuracy_threshold']:.2f} & {row['k_pretrain'] or '--'} & "
                f"{row['n_base_budget'] or '--'} & "
                f"{row['m_target_budget'] or '--'} & {cost:.1f}"
            )
        else:
            line = (
                f"{row['domain']} & {region} & "
                f"{row['method'].replace('_', ' ')} & {row['k_xor']} & "
                f"{row['accuracy_threshold']:.2f} & -- & -- & -- & --"
            )
        lines.append(line + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    output.write_text("\n".join(lines) + "\n")


def analyze_negative_control(reference_path, control_path, output_dir):
    reference = pl.read_csv(reference_path)
    control = pl.read_csv(control_path)
    keys = [
        "k_xor",
        "slices",
        "target_device",
        "k_pretrain",
        "n_base_budget",
        "m_target_budget",
        "trial",
    ]
    matched = (
        reference.select(keys + ["transfer_acc"])
        .join(control.select(keys + ["transfer_acc"]), on=keys, suffix="_control")
        .with_columns(
            (pl.col("transfer_acc_control") - pl.col("transfer_acc")).alias(
                "control_delta"
            )
        )
    )
    matched.write_csv(output_dir / "negative_control_rows.csv")
    summary = (
        matched.group_by(["k_xor", "k_pretrain", "m_target_budget"])
        .agg(
            pl.len().alias("n_matched"),
            pl.col("transfer_acc").mean().alias("reference_accuracy"),
            pl.col("transfer_acc_control").mean().alias("control_accuracy"),
            pl.col("control_delta").mean().alias("mean_control_delta"),
        )
        .sort(["k_xor", "k_pretrain", "m_target_budget"])
    )
    summary.write_csv(output_dir / "negative_control_summary.csv")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    groups = [
        matched.filter(pl.col("k_xor") == k)["control_delta"].to_numpy()
        for k in sorted(matched["k_xor"].unique().to_list())
    ]
    ax.boxplot(groups, showfliers=False)
    ax.axhline(0, color="black", linestyle=":")
    ax.set(
        xticks=range(1, len(groups) + 1),
        xticklabels=sorted(matched["k_xor"].unique().to_list()),
        xlabel="XOR degree",
        ylabel="Permuted-source minus reference accuracy",
    )
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "negative_control.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def analyze_pooled_control(results_path, output_dir, *, resamples=2000, seed=42):
    """Compare device-conditioned transfer with a source-pooled predictor."""
    frame = pl.read_csv(results_path)
    if "pooled_source_acc" not in frame.columns:
        raise ValueError("Pooled-control results lack pooled_source_acc.")
    frame = frame.filter(pl.col("pooled_source_acc").is_finite())
    if frame.is_empty():
        raise ValueError(
            "Pooled-control results are empty; rerun with --diagnostic-baselines."
        )

    configuration = frame.select(
        "k_pretrain", "n_base_budget", "m_target_budget"
    ).unique()
    if len(configuration) != 1:
        raise ValueError(
            "Select one source/target configuration for the pooled control."
        )

    methods = ["transfer_acc", "pooled_source_acc", "zero_shot_acc", "baseline_acc"]
    devices = (
        frame.group_by(["k_xor", "target_device"])
        .agg([pl.col(metric).mean().alias(metric) for metric in methods])
        .with_columns(
            (pl.col("pooled_source_acc") - pl.col("zero_shot_acc")).alias(
                "pooled_minus_zero_shot"
            ),
            (pl.col("transfer_acc") - pl.col("pooled_source_acc")).alias(
                "transfer_minus_pooled"
            ),
            (pl.col("pooled_source_acc") - pl.col("baseline_acc")).alias(
                "pooled_minus_target_only"
            ),
        )
        .sort(["k_xor", "target_device"])
    )
    devices.write_csv(output_dir / "pooled_control_device_rows.csv")
    metrics = methods + [
        "pooled_minus_zero_shot",
        "transfer_minus_pooled",
        "pooled_minus_target_only",
    ]
    summary = summarize_device_results(
        devices, ["k_xor"], metrics, resamples=resamples, seed=seed
    )
    config = configuration.row(0, named=True)
    summary = summary.with_columns(
        pl.lit(config["k_pretrain"]).alias("k_pretrain"),
        pl.lit(config["n_base_budget"]).alias("n_base_budget"),
        pl.lit(config["m_target_budget"]).alias("m_target_budget"),
    )
    summary.write_csv(output_dir / "pooled_control_summary.csv")

    ks = summary["k_xor"].to_numpy()
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    styles = (
        ("transfer_acc", "FiLM transfer", "#0072B2", "o"),
        ("pooled_source_acc", "Pooled sources", "#CC79A7", "s"),
        ("zero_shot_acc", "Mean embedding", "#009E73", "^"),
        ("baseline_acc", "Target only", "#D55E00", "D"),
    )
    for metric, label, color, marker in styles:
        mean = summary[f"mean_{metric}"].to_numpy()
        low = summary[f"ci95_low_{metric}"].to_numpy()
        high = summary[f"ci95_high_{metric}"].to_numpy()
        axes[0].errorbar(
            ks,
            mean,
            yerr=np.vstack((mean - low, high - mean)),
            color=color,
            marker=marker,
            capsize=3,
            label=label,
        )
    axes[0].axhline(0.5, color="grey", linestyle=":", linewidth=1)
    axes[0].set(
        xlabel="XOR degree",
        ylabel="Accuracy",
        xticks=ks,
        ylim=(0.45, 1.01),
    )
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.2)

    delta_styles = (
        ("pooled_minus_zero_shot", "Pooled - mean embedding", "#CC79A7"),
        ("transfer_minus_pooled", "FiLM transfer - pooled", "#0072B2"),
        ("pooled_minus_target_only", "Pooled - target only", "#E69F00"),
    )
    offsets = np.linspace(-0.22, 0.22, len(delta_styles))
    for offset, (metric, label, color) in zip(offsets, delta_styles):
        groups = [
            devices.filter(pl.col("k_xor") == k)[metric].to_numpy() for k in ks
        ]
        plot = axes[1].boxplot(
            groups,
            positions=ks + offset,
            widths=0.18,
            showfliers=False,
            patch_artist=True,
            manage_ticks=False,
        )
        for box in plot["boxes"]:
            box.set(facecolor=color, alpha=0.35, edgecolor=color)
        for element in ("whiskers", "caps", "medians"):
            for artist in plot[element]:
                artist.set(color=color)
        axes[1].plot([], [], color=color, linewidth=6, alpha=0.35, label=label)
    axes[1].axhline(0, color="black", linestyle=":", linewidth=1)
    axes[1].set(
        xlabel="XOR degree",
        ylabel="Paired accuracy difference",
        xticks=ks,
    )
    axes[1].legend(fontsize=7)
    axes[1].grid(axis="y", alpha=0.2)
    figure.suptitle(
        f"Pooled-source control ({config['k_pretrain']} sources, "
        f"{config['n_base_budget']} CRPs/source, {config['m_target_budget']} target CRPs)"
    )
    figure.tight_layout()
    figure.savefig(
        output_dir / "pooled_source_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close(figure)


def analyze_fixed_cohort(fixed_path, baseline_path, output_dir):
    """Summarize models that are each reused across a fixed target cohort."""
    fixed = pl.read_csv(fixed_path).rename({"cohort_trial": "trial"})
    baseline = pl.read_csv(baseline_path).filter(
        (pl.col("k_pretrain") == fixed["k_pretrain"][0])
        & (pl.col("n_base_budget") == fixed["n_base_budget"][0])
        & (pl.col("m_target_budget") == fixed["m_target_budget"][0])
    )
    keys = ["k_xor", "slices", "target_device", "trial", "m_target_budget"]
    baseline = baseline.select(keys + ["baseline_acc"]).unique(subset=keys)
    targets = (
        fixed.join(baseline, on=keys, how="inner", validate="1:1")
        .with_columns(
            (pl.col("transfer_acc") - pl.col("baseline_acc")).alias(
                "acc_gain_delta"
            )
        )
        .sort(keys)
    )
    if targets.height != fixed.height:
        raise ValueError(
            "Fixed-cohort rows do not have a one-to-one matched target-only result."
        )
    targets.write_csv(output_dir / "fixed_cohort_target_rows.csv")

    metrics = ["transfer_acc", "zero_shot_acc", "baseline_acc", "acc_gain_delta"]
    model_keys = ["k_xor", "slices", "trial", "source_devices"]
    models = (
        targets.group_by(model_keys)
        .agg(
            pl.col("target_device").n_unique().alias("held_out_target_count"),
            *[pl.col(metric).mean().alias(metric) for metric in metrics],
        )
        .sort(model_keys)
    )
    models.write_csv(output_dir / "fixed_cohort_model_rows.csv")

    # Average slice combinations before treating the five cohort trials as
    # replicates, so XOR degrees with more combinations receive no extra weight.
    trials = (
        models.group_by(["k_xor", "trial"])
        .agg(*[pl.col(metric).mean().alias(metric) for metric in metrics])
        .sort(["k_xor", "trial"])
    )
    rows = []
    for (k_xor,), group in trials.group_by("k_xor", maintain_order=True):
        model_group = models.filter(pl.col("k_xor") == k_xor)
        row = {
            "k_xor": k_xor,
            "n_cohort_trials": group.height,
            "held_out_targets_min": int(model_group["held_out_target_count"].min()),
            "held_out_targets_max": int(model_group["held_out_target_count"].max()),
        }
        critical = (
            float(student_t.ppf(0.975, group.height - 1))
            if group.height > 1
            else 0.0
        )
        for metric in metrics:
            values = group[metric].to_numpy()
            mean = float(np.mean(values))
            half_width = (
                critical * float(np.std(values, ddof=1)) / np.sqrt(len(values))
                if len(values) > 1
                else 0.0
            )
            row[f"mean_{metric}"] = mean
            row[f"ci95_low_{metric}"] = mean - half_width
            row[f"ci95_high_{metric}"] = mean + half_width
        rows.append(row)
    pl.DataFrame(rows).sort("k_xor").write_csv(
        output_dir / "fixed_cohort_summary.csv"
    )


def write_coverage_table(coverage_path, skipped_path, output):
    coverage = pl.read_csv(coverage_path)
    summary = coverage.group_by(["k_xor", "slices"]).agg(
        pl.len().alias("n_devices_in_cache"),
        (pl.col("train") > 0).sum().alias("devices_with_training_data"),
        (pl.col("test") > 0).sum().alias("devices_with_test_data"),
        pl.col("train").min().alias("minimum_training_crps"),
        pl.col("test").min().alias("minimum_test_crps"),
    )
    if skipped_path and skipped_path.exists():
        skipped = pl.read_csv(skipped_path)
        skipped_summary = skipped.group_by(["k_xor", "slices"]).agg(
            pl.len().alias("skipped_configurations"),
            pl.col("target_device").n_unique().alias("skipped_target_devices"),
        )
        summary = summary.join(skipped_summary, on=["k_xor", "slices"], how="left")
    else:
        summary = summary.with_columns(
            pl.lit(0).alias("skipped_configurations"),
            pl.lit(0).alias("skipped_target_devices"),
        )
    summary = summary.fill_null(0).with_columns(
        (pl.col("n_devices_in_cache") - pl.col("skipped_target_devices")).alias(
            "eligible_target_devices"
        ),
        pl.col("slices")
        .map_elements(short_slice_label, return_dtype=pl.String)
        .alias("display_slices"),
    )
    summary.sort(["k_xor", "slices"]).write_csv(output)


def build_transfer_frontier(devices, combination_summary, *, resamples, seed):
    """Aggregate paired gain by physical target and derive crossover cells."""
    keys = ["k_xor", "k_pretrain", "n_base_budget", "m_target_budget"]
    per_target = devices.group_by(keys + ["target_device"]).agg(
        pl.col("transfer_acc").mean(),
        pl.col("baseline_acc").mean(),
        pl.col("zero_shot_acc").mean(),
        pl.col("acc_gain_delta").mean(),
    )
    aggregate = summarize_device_results(
        per_target,
        keys,
        ["transfer_acc", "baseline_acc", "zero_shot_acc", "acc_gain_delta"],
        resamples=resamples,
        seed=seed,
    )
    combo_counts = combination_summary.group_by(keys).agg(
        pl.len().alias("n_slice_combinations"),
        (pl.col("ci95_low_acc_gain_delta") > 0)
        .sum()
        .alias("significant_positive_combinations"),
        (pl.col("mean_acc_gain_delta") > 0).sum().alias("positive_mean_combinations"),
    )
    aggregate = aggregate.join(combo_counts, on=keys).with_columns(
        (pl.col("ci95_low_acc_gain_delta") > 0).alias(
            "aggregate_significant_transfer_gain"
        )
    )

    crossover_rows = []
    group_keys = ["k_xor", "k_pretrain", "n_base_budget"]
    for values, group in aggregate.group_by(group_keys, maintain_order=True):
        group = group.sort("m_target_budget")
        significant = group.filter(pl.col("aggregate_significant_transfer_gain"))
        nonpositive = group.filter(pl.col("mean_acc_gain_delta") <= 0)
        last = significant.tail(1).row(0, named=True) if len(significant) else None
        crossover_rows.append(
            {
                **dict(zip(group_keys, values)),
                "source_pretraining_cost": values[1] * values[2],
                "largest_significantly_beneficial_target_budget": (
                    last["m_target_budget"] if last else None
                ),
                "gain_at_largest_beneficial_budget": (
                    last["mean_acc_gain_delta"] if last else np.nan
                ),
                "first_nonpositive_target_budget": (
                    nonpositive["m_target_budget"][0] if len(nonpositive) else None
                ),
                "maximum_mean_gain": group["mean_acc_gain_delta"].max(),
            }
        )
    return aggregate, pl.DataFrame(crossover_rows).sort(group_keys)


def build_break_even_table(frontier):
    """Compare transfer accuracy with the smallest measured target-only budget."""
    baseline = (
        frontier.group_by(["k_xor", "m_target_budget"])
        .agg(pl.col("mean_baseline_acc").mean().alias("baseline_accuracy"))
        .sort(["k_xor", "m_target_budget"])
    )
    rows = []
    for row in frontier.iter_rows(named=True):
        candidates = baseline.filter(
            (pl.col("k_xor") == row["k_xor"])
            & (pl.col("baseline_accuracy") >= row["mean_transfer_acc"])
        ).sort("m_target_budget")
        equivalent = candidates["m_target_budget"][0] if len(candidates) else None
        savings = (
            equivalent - row["m_target_budget"] if equivalent is not None else None
        )
        source_cost = row["k_pretrain"] * row["n_base_budget"]
        rows.append(
            {
                "k_xor": row["k_xor"],
                "k_pretrain": row["k_pretrain"],
                "n_base_budget": row["n_base_budget"],
                "source_pretraining_cost": source_cost,
                "m_target_budget": row["m_target_budget"],
                "transfer_accuracy": row["mean_transfer_acc"],
                "target_only_accuracy_same_budget": row["mean_baseline_acc"],
                "paired_gain": row["mean_acc_gain_delta"],
                "gain_ci95_low": row["ci95_low_acc_gain_delta"],
                "gain_ci95_high": row["ci95_high_acc_gain_delta"],
                "baseline_equivalent_budget": equivalent,
                "target_crp_savings": savings,
                "break_even_targets": (
                    math.ceil(source_cost / savings)
                    if savings is not None and savings > 0
                    else None
                ),
                "baseline_equivalent_right_censored": equivalent is None,
            }
        )
    return pl.DataFrame(rows).sort(
        ["k_xor", "k_pretrain", "n_base_budget", "m_target_budget"]
    )


def plot_transfer_gain_frontier(frontier, output, formats):
    """Plot paired transfer gain and mark cells without significant gain."""
    ks = sorted(frontier["k_xor"].unique().to_list())
    budgets = sorted(frontier["m_target_budget"].unique().to_list())
    configs = (
        frontier.select("k_pretrain", "n_base_budget")
        .unique()
        .with_columns(
            (pl.col("k_pretrain") * pl.col("n_base_budget")).alias("source_cost")
        )
        .sort(["source_cost", "k_pretrain", "n_base_budget"])
    )
    labels = [
        f"{row['k_pretrain']}x{row['n_base_budget']}"
        for row in configs.iter_rows(named=True)
    ]
    limit = 100 * max(
        abs(frontier["mean_acc_gain_delta"].min()),
        abs(frontier["mean_acc_gain_delta"].max()),
    )
    if limit == 0:
        limit = 1
    norm = mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    fig, axes = plt.subplots(1, len(ks), figsize=(4 * len(ks), 4), sharey=True)
    axes = np.atleast_1d(axes)
    image = None
    for ax, k in zip(axes.flat, ks):
        values = np.full((len(configs), len(budgets)), np.nan)
        significant = np.zeros_like(values, dtype=bool)
        panel = frontier.filter(pl.col("k_xor") == k)
        for row_index, config in enumerate(configs.iter_rows(named=True)):
            for column_index, budget in enumerate(budgets):
                cell = panel.filter(
                    (pl.col("k_pretrain") == config["k_pretrain"])
                    & (pl.col("n_base_budget") == config["n_base_budget"])
                    & (pl.col("m_target_budget") == budget)
                )
                if cell.height != 1:
                    continue
                values[row_index, column_index] = 100 * cell[
                    "mean_acc_gain_delta"
                ][0]
                significant[row_index, column_index] = (
                    cell["ci95_low_acc_gain_delta"][0] > 0
                    or cell["ci95_high_acc_gain_delta"][0] < 0
                )
        image = ax.imshow(
            values, aspect="auto", origin="lower", cmap="RdYlGn", norm=norm
        )
        for row_index in range(values.shape[0]):
            for column_index in range(values.shape[1]):
                if not np.isfinite(values[row_index, column_index]):
                    continue
                if not significant[row_index, column_index]:
                    ax.add_patch(
                        Rectangle(
                            (column_index - 0.5, row_index - 0.5),
                            1,
                            1,
                            fill=False,
                            hatch="////",
                            edgecolor="0.35",
                            linewidth=0,
                        )
                    )
                else:
                    ax.text(
                        column_index,
                        row_index,
                        f"{values[row_index, column_index]:.0f}",
                        color=(
                            "white"
                            if np.abs(values[row_index, column_index]) >= 10
                            else "black"
                        ),
                        weight="bold",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )
        ax.set_title(rf"$k={k}$")
        ax.set_xticks(
            range(len(budgets)), [str(value) for value in budgets], rotation=45
        )
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
        ax.set_xlabel("Target CRPs ($M$)")
    axes[0].set_ylabel("Sources x CRPs/source ($S$ x $N$)")
    fig.subplots_adjust(
        left=0.1, right=0.9, bottom=0.1, top=0.95, wspace=0.12, hspace=0.20
    )
    divider = make_axes_locatable(axes[-1])
    colorbar_axis = divider.append_axes("right", size="7%", pad=0.15)
    fig.colorbar(
        image, cax=colorbar_axis, label="Paired transfer gain (percentage points)"
    )
    save_figure(fig, output, formats)


def write_break_even_latex(table, source_count, source_budget, output):
    selected = table.filter(
        (pl.col("k_pretrain") == source_count)
        & (pl.col("n_base_budget") == source_budget)
        & (pl.col("m_target_budget") == pl.col("m_target_budget").min())
    ).sort("k_xor")
    lines = [
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"$k$ & Transfer acc. & Target-only acc. & Equivalent target CRPs & Saved CRPs & Break-even targets \\",
        r"\midrule",
    ]
    for row in selected.iter_rows(named=True):
        equivalent = row["baseline_equivalent_budget"]
        savings = row["target_crp_savings"]
        break_even = row["break_even_targets"]
        equivalent_text = (
            str(equivalent)
            if equivalent is not None
            else f">{table['m_target_budget'].max()}"
        )
        line = (
            f"{row['k_xor']} & {100 * row['transfer_accuracy']:.1f} & "
            f"{100 * row['target_only_accuracy_same_budget']:.1f} & "
            f"{equivalent_text} & "
            f"{savings if savings is not None else '--'} & "
            f"{break_even if break_even is not None else '--'}"
        )
        lines.append(line + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    output.write_text("\n".join(lines) + "\n")


def _per_device(frame, value):
    return (
        frame.group_by(["k_xor", "target_device"])
        .agg(pl.col(value).mean().alias(value))
        .sort(["k_xor", "target_device"])
    )


def plot_validation_controls(output_dir, output, formats):
    """Compare permutation and nearest-source-exclusion controls by target."""
    permutation = _per_device(
        pl.read_csv(output_dir / "negative_control_rows.csv"), "control_delta"
    )
    exclusion = _per_device(
        pl.read_csv(output_dir / "validation/source_policy_comparison.csv"),
        "delta_transfer_acc",
    )
    specs = (
        (permutation, "control_delta", "Challenge permutation", "Permuted - reference"),
        (
            exclusion,
            "delta_transfer_acc",
            "Exclude nearest sources",
            "Excluded-nearest - random",
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(9, 3), sharex=True)
    for ax, (frame, value, title, ylabel) in zip(axes, specs):
        ks = sorted(frame["k_xor"].unique().to_list())
        groups = [
            100 * frame.filter(pl.col("k_xor") == k)[value].to_numpy() for k in ks
        ]
        boxes = ax.boxplot(
            groups, positions=ks, widths=0.55, showfliers=False, patch_artist=True
        )
        for box in boxes["boxes"]:
            box.set(facecolor="#E6E6E6", edgecolor="#333333")
        ax.axhline(0, color="black", linestyle=":", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("XOR degree")
        ax.set_ylabel(f"{ylabel} accuracy (pp)")
        ax.set_xticks(ks)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, output, formats)


def bootstrap_mean(values, rng, resamples):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if len(values) < 2 or resamples == 0:
        return mean, mean, mean
    samples = rng.choice(values, size=(resamples, len(values)), replace=True).mean(
        axis=1
    )
    return mean, float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def plot_source_scaling(
    results_path, source_budgets, target_budget, output, formats, seed, resamples
):
    """Plot source-count scaling by XOR degree and per-source budget."""
    if not source_budgets or len(source_budgets) != len(set(source_budgets)):
        raise ValueError("Source scaling requires distinct source budgets")
    frame = pl.read_csv(results_path).filter(
        pl.col("n_base_budget").is_in(source_budgets)
        & (pl.col("m_target_budget") == target_budget)
    )
    available = sorted(frame["n_base_budget"].unique().to_list())
    if available != sorted(source_budgets):
        raise ValueError(
            f"Requested source budgets {sorted(source_budgets)} but found {available}"
        )
    per_device = frame.group_by(
        ["k_xor", "n_base_budget", "k_pretrain", "target_device"]
    ).agg(pl.col("transfer_acc").mean().alias("transfer_acc"))
    rng = np.random.default_rng(seed)
    rows = []
    for values, group in per_device.group_by(
        ["k_xor", "n_base_budget", "k_pretrain"], maintain_order=True
    ):
        mean, low, high = bootstrap_mean(
            group["transfer_acc"].to_numpy(), rng, resamples
        )
        rows.append(
            {
                "k_xor": values[0],
                "n_base_budget": values[1],
                "k_pretrain": values[2],
                "mean": mean,
                "low": low,
                "high": high,
            }
        )
    summary = pl.DataFrame(rows).sort(["k_xor", "n_base_budget", "k_pretrain"])
    ks = sorted(summary["k_xor"].unique().to_list())
    source_counts = sorted(summary["k_pretrain"].unique().to_list())
    fig, axes = plt.subplots(1, len(ks), figsize=(3 * len(ks), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    palette = plt.get_cmap("tab10")
    for ax, k in zip(axes, ks):
        k_panel = summary.filter(pl.col("k_xor") == k)
        for index, source_budget in enumerate(sorted(source_budgets)):
            panel = k_panel.filter(pl.col("n_base_budget") == source_budget)
            x = panel["k_pretrain"].to_numpy().astype(int)
            color = palette(index)
            ax.plot(
                x,
                panel["mean"],
                color=color,
                marker="o",
                linewidth=1.2,
                markersize=3.5,
                label=rf"$N={source_budget}$",
            )
            ax.fill_between(
                x,
                panel["low"],
                panel["high"],
                color=color,
                alpha=0.14,
                linewidth=0,
            )
        ax.axhline(0.5, color="0.5", linestyle=":", linewidth=1)
        ax.set_title(rf"$k={k}$")
        ax.grid(alpha=0.2)
        ax.set_xlabel("Source devices ($S$)")
        ax.set_ylim(0.45, 1.03)
        ax.set_xticks(source_counts)
    axes[0].set_ylabel("Transfer accuracy")
    axes[0].legend(title="CRPs/source", fontsize=7, title_fontsize=7)
    fig.tight_layout()
    save_figure(fig, output, formats)


def plot_overview(
    devices, source_count, base_budget, output_dir, resamples, seed, formats
):
    """Summarize all target/combination cells without hiding coverage."""
    selected = devices.filter(
        (pl.col("k_pretrain") == source_count)
        & (pl.col("n_base_budget") == base_budget)
    )
    per_target = selected.group_by(["k_xor", "m_target_budget", "target_device"]).agg(
        *[pl.col(metric).mean().alias(metric) for metric, _, _ in METHODS]
    )
    aggregate = summarize_device_results(
        per_target,
        ["k_xor", "m_target_budget"],
        [metric for metric, _, _ in METHODS],
        resamples=resamples,
        seed=seed,
    )
    aggregate.write_csv(output_dir / "overview_summary.csv")

    ks = aggregate["k_xor"].unique().sort().to_list()
    fig, axes = plt.subplots(
        2, 2, figsize=(11, 7.5), squeeze=False, sharex=True, sharey=True
    )
    for ax, k in zip(axes.flat, ks):
        panel = aggregate.filter(pl.col("k_xor") == k).sort("m_target_budget")
        x = panel["m_target_budget"].to_numpy()
        for metric, label, color in METHODS:
            mean = panel[f"mean_{metric}"].to_numpy()
            low = panel[f"ci95_low_{metric}"].to_numpy()
            high = panel[f"ci95_high_{metric}"].to_numpy()
            ax.plot(x, mean, marker="o", label=label, color=color)
            ax.fill_between(
                x,
                np.maximum(0, low),
                np.minimum(1, high),
                color=color,
                alpha=0.1,
            )
        coverage = panel["n_target_devices"].min()
        ax.set_title(f"{k}-XOR ({coverage} target devices)")
        ax.set_xscale("log")
        ax.set_ylim(0.45, 1.01)
        ax.axhline(0.5, color="grey", linestyle=":")
        ax.grid(alpha=0.2)
    for ax in axes[-1]:
        ax.set_xlabel("Total target labeled CRPs")
    for ax in axes[:, 0]:
        ax.set_ylabel("Accuracy")
    axes.flat[0].legend(fontsize=11)
    # fig.suptitle(
    #     f"Empirical transfer: {source_count} sources, {base_budget} total CRPs/source; bands = device bootstrap 95% CI"
    # )
    fig.tight_layout()
    save_figure(fig, output_dir / "empirical_transfer", formats, dpi=200)

    scaling = (
        devices.filter(pl.col("n_base_budget") == base_budget)
        .group_by(["k_xor", "k_pretrain", "m_target_budget"])
        .agg(
            pl.col("transfer_acc").mean(), pl.len().alias("n_target_combination_cells")
        )
        .sort(["k_xor", "m_target_budget", "k_pretrain"])
    )
    scaling.write_csv(output_dir / "source_scaling_summary.csv")
    fig, axes = plt.subplots(
        2, 2, figsize=(11, 7.5), squeeze=False, sharex=True, sharey=True
    )
    colors = plt.get_cmap("viridis")(
        np.linspace(0.15, 0.85, scaling["m_target_budget"].n_unique())
    )
    for ax, k in zip(axes.flat, ks):
        panel = scaling.filter(pl.col("k_xor") == k)
        for color, budget in zip(
            colors, sorted(panel["m_target_budget"].unique().to_list())
        ):
            curve = panel.filter(pl.col("m_target_budget") == budget).sort("k_pretrain")
            ax.plot(
                curve["k_pretrain"],
                curve["transfer_acc"],
                marker="o",
                color=color,
                label=f"m={budget:g}",
            )
        ax.set_title(f"{k}-XOR")
        ax.set_ylim(0.45, 1.01)
        ax.axhline(0.5, color="grey", linestyle=":")
        ax.grid(alpha=0.2)
    for ax in axes[-1]:
        ax.set_xlabel("Number of source devices")
    for ax in axes[:, 0]:
        ax.set_ylabel("Transfer accuracy")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"Source-population scaling ({base_budget} CRPs/source)")
    fig.tight_layout()
    fig.savefig(output_dir / "source_scaling.png", dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        type=Path,
        default=Path("data/empirical_transfer/transfer_results.csv"),
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("figures/empirical_transfer")
    )
    p.add_argument("--k-pretrain", type=int, default=None)
    p.add_argument("--n-base-budget", type=int, default=None)
    p.add_argument("--estimates", type=Path, default=None)
    p.add_argument(
        "--theory",
        type=Path,
        default=None,
        help="Optional theoretical results for the long-form cost table; not overlaid on empirical curves.",
    )
    p.add_argument(
        "--theory-n-bits",
        type=int,
        default=64,
        help="Stage count of legacy theoretical CSVs that lack an n_bits column.",
    )
    p.add_argument("--theory-target-ber", type=float, default=0.0)
    p.add_argument("--amortization-targets", default="1,10,100")
    p.add_argument("--bootstrap-resamples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--formats", default="png,pdf")
    p.add_argument("--source-scaling-source-budgets", default=None)
    p.add_argument("--source-scaling-target-budget", type=int, default=100)
    p.add_argument(
        "--similarity-control-results",
        "--leave-cluster-out-results",
        dest="similarity_control_results",
        type=Path,
        default=None,
    )
    p.add_argument("--metric-distributions", type=Path, default=None)
    p.add_argument("--metrics-summary", type=Path, default=None)
    p.add_argument("--metrics-null-summary", type=Path, default=None)
    p.add_argument("--population-parameter-pairs", type=Path, default=None)
    p.add_argument("--population-model-fit", type=Path, default=None)
    p.add_argument("--negative-control-results", type=Path, default=None)
    p.add_argument(
        "--pooled-control-results",
        type=Path,
        default=None,
        help="Targeted run produced with --diagnostic-baselines.",
    )
    p.add_argument(
        "--fixed-cohort-results",
        type=Path,
        default=None,
        help="Results from one pretrained model reused across held-out targets.",
    )
    p.add_argument("--coverage", type=Path, default=None)
    p.add_argument("--skipped-configurations", type=Path, default=None)
    p.add_argument("--cluster-threshold", type=float, default=0.15)
    args = p.parse_args()
    formats = tuple(value.strip() for value in args.formats.split(",") if value.strip())
    invalid_formats = set(formats) - {"png", "pdf"}
    if not formats or invalid_formats:
        p.error(
            "--formats must contain png and/or pdf"
            + (
                f"; unsupported: {', '.join(sorted(invalid_formats))}"
                if invalid_formats
                else ""
            )
        )
    df = pl.read_csv(args.input).with_columns(
        (pl.col("transfer_acc") - pl.col("zero_shot_acc")).alias("adaptation_gain")
    )
    # Average repeated trials within a device before reporting spread across devices.
    keys = ["k_xor", "slices", "k_pretrain", "n_base_budget", "m_target_budget"]
    metrics = [
        "transfer_acc",
        "baseline_acc",
        "acc_gain_delta",
        "zero_shot_acc",
        "adaptation_gain",
        "majority_acc",
    ]
    devices = df.group_by(keys + ["target_device"]).agg(
        [pl.col(m).mean().alias(m) for m in metrics]
    )
    summary = summarize_device_results(
        devices,
        keys,
        metrics,
        resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.write_csv(args.output_dir / "transfer_summary.csv")
    (
        summary.select(
            "k_xor",
            "slices",
            "k_pretrain",
            "n_base_budget",
            "m_target_budget",
            "n_target_devices",
            "mean_zero_shot_acc",
            "mean_adaptation_gain",
            "mean_transfer_acc",
            "mean_baseline_acc",
            "mean_acc_gain_delta",
        )
        .sort(
            [
                "k_xor",
                "slices",
                "k_pretrain",
                "n_base_budget",
                "m_target_budget",
            ]
        )
        .write_csv(args.output_dir / "transfer_decomposition.csv")
    )
    source_count = args.k_pretrain or int(summary["k_pretrain"].max())
    base_budget = args.n_base_budget or int(summary["n_base_budget"].max())
    selected = summary.filter(
        (pl.col("k_pretrain") == source_count)
        & (pl.col("n_base_budget") == base_budget)
    )
    if selected.is_empty():
        p.error("No rows match the requested source configuration.")
    estimates = pl.read_csv(args.estimates) if args.estimates else None
    theory = pl.read_csv(args.theory) if args.theory else None
    if estimates is not None:
        expected = "polar_response_arcsine_moments"
        if "estimator" not in estimates.columns or set(
            estimates["estimator"].unique()
        ) != {expected}:
            p.error(
                "Population estimates use the obsolete LR-direction estimator. "
                "Rerun scripts/03-study_empirical_transfer.py --estimate before plotting."
            )
    if theory is not None:
        n_bits = df["n_bits"].unique().to_list()
        if len(n_bits) != 1:
            p.error("Select a single empirical stage count before comparing theory.")
        if "n_bits" in theory.columns:
            theory = theory.filter(pl.col("n_bits") == n_bits[0])
        elif args.theory_n_bits != n_bits[0]:
            p.error("Empirical and theoretical stage counts differ.")
        if "target_ber" in theory.columns:
            theory = theory.filter(
                (pl.col("target_ber") - args.theory_target_ber).abs() < 1e-9
            )
    plot_overview(
        devices,
        source_count,
        base_budget,
        args.output_dir,
        args.bootstrap_resamples,
        args.seed,
        formats,
    )
    frontier, crossover = build_transfer_frontier(
        devices,
        summary,
        resamples=args.bootstrap_resamples,
        seed=args.seed + 1,
    )
    frontier.write_csv(args.output_dir / "transfer_frontier.csv")
    crossover.write_csv(args.output_dir / "transfer_crossover_summary.csv")
    break_even = build_break_even_table(frontier)
    break_even.write_csv(args.output_dir / "transfer_break_even.csv")
    plot_transfer_gain_frontier(
        frontier, args.output_dir / "transfer_advantage_map", formats
    )
    for k in sorted(selected["k_xor"].unique().to_list()):
        rows = selected.filter(pl.col("k_xor") == k)
        combos = sorted(rows["slices"].unique().to_list())
        fig, axes = plt.subplots(
            int(np.ceil(len(combos) / 2)),
            min(2, len(combos)),
            figsize=(12, 3.5 * int(np.ceil(len(combos) / 2))),
            squeeze=False,
        )
        for ax, combo in zip(axes.flat, combos):
            panel = rows.filter(pl.col("slices") == combo).sort("m_target_budget")
            x = panel["m_target_budget"].to_numpy()
            for metric, label, color in METHODS:
                mean = panel[f"mean_{metric}"].to_numpy()
                low = panel[f"ci95_low_{metric}"].to_numpy()
                high = panel[f"ci95_high_{metric}"].to_numpy()
                ax.plot(x, mean, marker="o", label=label, color=color)
                ax.fill_between(
                    x,
                    np.maximum(0, low),
                    np.minimum(1, high),
                    color=color,
                    alpha=0.1,
                )
            ax.set(
                title=short_slice_label(combo),
                xlabel="Total target labeled CRPs",
                ylabel="Accuracy",
                xscale="log",
                ylim=(0, 1.01),
            )
            ax.axhline(0.5, color="grey", linestyle=":")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7)
        for ax in axes.flat[len(combos) :]:
            ax.set_axis_off()
        fig.suptitle(
            f"{k}-XOR: {source_count} sources, {base_budget} total CRPs/source; bands = device bootstrap 95% CI"
        )
        fig.tight_layout()
        save_figure(fig, args.output_dir / f"transfer_k{k}", formats, dpi=180)
    if args.metrics_summary and args.metrics_null_summary:
        plot_population_metrics(
            args.metrics_summary,
            args.metrics_null_summary,
            args.output_dir / "population_metrics.png",
        )
    if args.metric_distributions:
        plot_metric_distributions(
            args.metric_distributions,
            args.output_dir / "empirical_metric_distributions",
            formats,
        )
    if args.population_parameter_pairs and args.population_model_fit:
        plot_parameter_diagnostics(
            args.population_parameter_pairs,
            args.population_model_fit,
            args.output_dir,
            formats,
        )
    if args.source_scaling_source_budgets:
        source_budgets = [
            int(value)
            for value in args.source_scaling_source_budgets.split(",")
            if value
        ]
        plot_source_scaling(
            args.input,
            source_budgets,
            args.source_scaling_target_budget,
            args.output_dir / "source_scaling_low_budget",
            formats,
            args.seed,
            args.bootstrap_resamples,
        )
    costs = build_cost_table(
        summary,
        theory,
        [int(value) for value in args.amortization_targets.split(",") if value],
    )
    costs.write_csv(args.output_dir / "theory_empirical_budget_accuracy.csv")
    cost_summary = summarize_costs(costs)
    cost_summary.write_csv(args.output_dir / "minimum_cost_by_accuracy_threshold.csv")
    write_cost_latex(cost_summary, args.output_dir / "attack_cost_summary.tex")
    plot_cost_frontiers(costs, args.output_dir / "accuracy_vs_amortized_cost.png")
    if args.negative_control_results:
        analyze_negative_control(
            args.input, args.negative_control_results, args.output_dir
        )
    if args.pooled_control_results:
        analyze_pooled_control(
            args.pooled_control_results,
            args.output_dir,
            resamples=args.bootstrap_resamples,
            seed=args.seed + 2,
        )
    if args.fixed_cohort_results:
        analyze_fixed_cohort(
            args.fixed_cohort_results,
            args.input,
            args.output_dir,
        )
    if args.coverage:
        write_coverage_table(
            args.coverage,
            args.skipped_configurations,
            args.output_dir / "experiment_coverage.csv",
        )
    if args.similarity_control_results:
        if args.metric_distributions is None:
            p.error("--similarity-control-results requires --metric-distributions")
        analyze_similarity_control(
            args.input,
            args.similarity_control_results,
            args.metric_distributions,
            args.output_dir / "validation",
            args.cluster_threshold,
        )
    if args.negative_control_results and args.similarity_control_results:
        plot_validation_controls(
            args.output_dir,
            args.output_dir / "validation_controls",
            formats,
        )
    print(f"Wrote summary and figures to {args.output_dir}")


if __name__ == "__main__":
    main()
