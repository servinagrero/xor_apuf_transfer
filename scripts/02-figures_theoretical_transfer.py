"""Summarize and plot paired theoretical transfer experiments."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import t as student_t

GROUPS = [
    "selection",
    "k_xor",
    "alpha",
    "beta",
    "noise_mode",
    "target_ber",
    "requested_noisiness",
    "k_pretrain",
    "n_base_budget",
    "m_target_budget",
]
METRICS = [
    "transfer_acc",
    "transfer_latent_acc",
    "baseline_acc",
    "baseline_latent_acc",
    "zero_shot_acc",
    "majority_acc",
    "acc_gain_delta",
    "test_ber",
    "accuracy_ceiling",
    "calibration_ber",
    "noisiness",
    "classification_noise_penalty",
]


def save_figure(fig, output: Path, formats: tuple[str, ...]) -> None:
    for extension in formats:
        fig.savefig(
            output.with_suffix(f".{extension}"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def summarize(frame: pl.DataFrame) -> pl.DataFrame:
    expressions = [pl.len().alias("n_trials")]
    for metric in METRICS:
        expressions.extend(
            [
                pl.col(metric).mean().alias(f"mean_{metric}"),
                pl.col(metric).std().fill_null(0).alias(f"std_{metric}"),
            ]
        )
    summary = frame.group_by(GROUPS).agg(expressions).sort(GROUPS)
    critical_value = pl.col("n_trials").map_elements(
        lambda count: float(student_t.ppf(0.975, count - 1)) if count > 1 else 0.0,
        return_dtype=pl.Float64,
    )
    return summary.with_columns(
        *[
            (
                critical_value * pl.col(f"std_{metric}") / pl.col("n_trials").sqrt()
            ).alias(f"ci95_{metric}")
            for metric in METRICS
        ]
    )


def plot_budget_curves(
    summary,
    source_count,
    source_budget,
    output,
    formats=("png",),
):
    selected = summary.filter(
        (pl.col("k_pretrain") == source_count)
        & (pl.col("n_base_budget") == source_budget)
    )
    ks = sorted(selected["k_xor"].unique().to_list())
    fig, axes = plt.subplots(
        int(np.ceil(len(ks) / 3)),
        3,
        figsize=(10.5, 2.9 * int(np.ceil(len(ks) / 3))),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    region_colors = dict(
        zip(
            sorted(selected["selection"].unique().to_list()),
            plt.get_cmap("viridis")(
                np.linspace(0.1, 0.9, selected["selection"].n_unique())
            ),
        )
    )
    for ax, k in zip(axes.flat, ks):
        panel = selected.filter(pl.col("k_xor") == k)
        for region in sorted(panel["selection"].unique().to_list()):
            curve = panel.filter(pl.col("selection") == region).sort("m_target_budget")
            x = curve["m_target_budget"].to_numpy()
            mean = curve["mean_transfer_acc"].to_numpy()
            ci = curve["ci95_transfer_acc"].to_numpy()
            ax.plot(
                x,
                mean,
                marker="o",
                linewidth=1,
                color=region_colors[region],
                label=region,
            )
            ax.fill_between(
                x, mean - ci, mean + ci, color=region_colors[region], alpha=0.12
            )
        baseline = (
            panel.group_by("m_target_budget")
            .agg(pl.col("mean_baseline_acc").mean())
            .sort("m_target_budget")
        )
        ax.plot(
            baseline["m_target_budget"],
            baseline["mean_baseline_acc"],
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="target only",
        )
        ax.set_title(f"{k}-XOR")
        ax.set_xscale("log")
        ax.set_ylim(0.45, 1.01)
        ax.axhline(0.5, color="0.6", linestyle=":", linewidth=1)
        ax.grid(alpha=0.2)
    for ax in axes.flat[len(ks) :]:
        ax.set_axis_off()
    for ax in axes[-1]:
        ax.set_xlabel("Total target CRPs")
    for ax in axes[:, 0]:
        ax.set_ylabel("Accuracy")
    axes.flat[0].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    save_figure(fig, output, formats)


def _format_budget(value: int) -> str:
    return f"{value // 1000}k" if value >= 1000 and value % 1000 == 0 else str(value)


def plot_gain(
    summary,
    source_count,
    source_budget,
    output,
    table_output=None,
    formats=("png",),
):
    """Plot paired gain without mixing controls and masking regions in each panel.

    The metric-selected ``stealth_high`` point is fixed independently of attack
    accuracy. Baseline and health panels provide the two causal controls. Colors
    are clipped so that publication-scale gains remain visible; cell annotations
    retain the exact values.
    """
    selected = summary.filter(
        (pl.col("k_pretrain") == source_count)
        & (pl.col("n_base_budget") == source_budget)
    )
    panel_specs = (
        ("baseline", "Uncorrelated control"),
        ("stealth_high", "Metric-compatible boundary"),
        ("health", "Correlation health check"),
    )
    panel_names = [name for name, _ in panel_specs]
    plot_data = (
        selected.filter(pl.col("selection").is_in(panel_names))
        .select(
            "selection",
            "k_xor",
            "alpha",
            "beta",
            "k_pretrain",
            "n_base_budget",
            "m_target_budget",
            "n_trials",
            "mean_transfer_acc",
            "mean_baseline_acc",
            "mean_acc_gain_delta",
            "ci95_acc_gain_delta",
        )
        .sort(["selection", "k_xor", "m_target_budget"])
    )
    if table_output is not None:
        plot_data.write_csv(table_output)

    duplicate_cells = (
        plot_data.group_by(["selection", "k_xor", "m_target_budget"])
        .len()
        .filter(pl.col("len") != 1)
    )
    if duplicate_cells.height:
        raise ValueError(
            "Gain plot requires one row per selection, k, and target budget; "
            f"found duplicates:\n{duplicate_cells}"
        )

    ks = sorted(plot_data["k_xor"].unique().to_list())
    budgets = sorted(plot_data["m_target_budget"].unique().to_list())
    fig, axes = plt.subplots(
        1,
        len(panel_specs),
        figsize=(11.8, 4.35),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    norm = TwoSlopeNorm(vmin=-10, vcenter=0, vmax=10)
    image = None
    for ax, (selection, title) in zip(axes, panel_specs):
        panel = plot_data.filter(pl.col("selection") == selection)
        gains = np.full((len(ks), len(budgets)), np.nan)
        intervals = np.full_like(gains, np.nan)
        for row in panel.iter_rows(named=True):
            i = ks.index(row["k_xor"])
            j = budgets.index(row["m_target_budget"])
            gains[i, j] = 100 * row["mean_acc_gain_delta"]
            intervals[i, j] = 100 * row["ci95_acc_gain_delta"]

        image = ax.imshow(
            gains,
            cmap="RdYlGn",
            norm=norm,
            aspect="auto",
            interpolation="nearest",
        )
        for i, j in np.ndindex(gains.shape):
            if np.isnan(gains[i, j]):
                continue
            excludes_zero = abs(gains[i, j]) > intervals[i, j]
            ax.text(
                j,
                i,
                f"{gains[i, j]:.1f}",
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold" if excludes_zero else "normal",
                color="white" if abs(gains[i, j]) >= 5 else "black",
            )
        ax.set_title(title, fontsize=9)
        ax.set_xticks(range(len(budgets)), [_format_budget(v) for v in budgets])
        ax.set_yticks(range(len(ks)), [str(k) for k in ks])
        ax.set_xlabel("Target CRPs")
        ax.tick_params(axis="both", labelsize=7, length=0)
        ax.set_xticks(np.arange(-0.5, len(budgets), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(ks), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)

    axes[0].set_ylabel("XOR degree $k$")
    # colorbar = fig.colorbar(image, ax=axes, extend="both", shrink=0.9, pad=0.02)
    colorbar = fig.colorbar(image, ax=axes, shrink=0.9, pad=0.02)
    colorbar.set_label("Transfer gain over target-only (percentage points)", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    # fig.suptitle(
    #     f"Paired theoretical transfer gain ($S={source_count}$, "
    #     f"$N={source_budget:,}$ CRPs/source)",
    #     fontsize=10,
    # )
    # fig.text(
    #     0.5,
    #     -0.015,
    #     "Bold values: paired 95% CI excludes zero. Color is clipped at +/-10 pp; annotations are exact.",
    #     ha="center",
    #     fontsize=7,
    # )
    save_figure(fig, output, formats)


def plot_noise_sensitivity(
    summary,
    source_count,
    source_budget,
    output,
    formats=("png",),
):
    selected = summary.filter(
        (pl.col("noise_mode") == "target_ber")
        & (pl.col("k_pretrain") == source_count)
        & (pl.col("n_base_budget") == source_budget)
        & (pl.col("m_target_budget") == pl.col("m_target_budget").max())
    )
    preferred = "stealth_mid"
    regions = selected["selection"].unique().to_list()
    region = preferred if preferred in regions else sorted(regions)[0]
    selected = selected.filter(pl.col("selection") == region)
    ks = sorted(selected["k_xor"].unique().to_list())
    fig, axes = plt.subplots(
        int(np.ceil(len(ks) / 3)),
        3,
        figsize=(10.5, 2.8 * int(np.ceil(len(ks) / 3))),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for ax, k in zip(axes.flat, ks):
        curve = selected.filter(pl.col("k_xor") == k).sort("target_ber")
        ax.plot(
            curve["target_ber"],
            curve["mean_transfer_acc"],
            marker="o",
            label="observed",
        )
        ax.plot(
            curve["target_ber"],
            curve["mean_transfer_latent_acc"],
            marker="s",
            label="latent",
        )
        ax.plot(
            curve["target_ber"],
            1 - curve["target_ber"],
            color="black",
            linestyle=":",
            label=r"$1-\mathrm{BER}$",
        )
        ax.set_title(f"{k}-XOR")
        ax.grid(alpha=0.2)
    for ax in axes.flat[len(ks) :]:
        ax.set_axis_off()
    for ax in axes[-1]:
        ax.set_xlabel("Requested output BER")
    for ax in axes[:, 0]:
        ax.set_ylabel("Accuracy")
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(f"Reliability sensitivity ({region}, maximum target budget)")
    fig.tight_layout()
    save_figure(fig, output, formats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("data/theoretical_transfer.csv")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("figures/theoretical_transfer")
    )
    parser.add_argument("--k-pretrain", type=int, default=None)
    parser.add_argument("--n-base-budget", type=int, default=None)
    parser.add_argument("--target-ber", type=float, default=0.0)
    parser.add_argument("--formats", default="png,pdf")
    args = parser.parse_args()
    formats = tuple(item.strip() for item in args.formats.split(",") if item.strip())
    invalid = set(formats) - {"png", "pdf"}
    if not formats or invalid:
        parser.error(f"Unsupported output formats: {sorted(invalid)}")

    frame = pl.read_csv(args.input)
    summary = summarize(frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.write_csv(args.output_dir / "theoretical_transfer_summary.csv")
    (
        summary.filter(pl.col("noise_mode") == "target_ber")
        .group_by(["k_xor", "target_ber"])
        .agg(
            pl.col("mean_calibration_ber").mean().alias("mean_calibration_ber"),
            pl.col("mean_test_ber").mean().alias("mean_achieved_test_ber"),
            pl.col("mean_noisiness").mean().alias("mean_calibrated_noisiness"),
            pl.col("mean_accuracy_ceiling").mean().alias("mean_accuracy_ceiling"),
            pl.col("mean_classification_noise_penalty")
            .mean()
            .alias("symmetric_noise_sample_penalty"),
        )
        .sort(["k_xor", "target_ber"])
        .write_csv(args.output_dir / "ber_calibration_pac_reference.csv")
    )
    source_count = args.k_pretrain or int(summary["k_pretrain"].max())
    source_budget = args.n_base_budget or int(summary["n_base_budget"].max())
    selected = summary.filter(
        (pl.col("noise_mode") != "target_ber")
        | ((pl.col("target_ber") - args.target_ber).abs() < 1e-9)
    )
    plot_budget_curves(
        selected,
        source_count,
        source_budget,
        args.output_dir / "theoretical_transfer",
        formats,
    )
    plot_gain(
        selected,
        source_count,
        source_budget,
        args.output_dir / "theoretical_transfer_gain",
        args.output_dir / "theoretical_transfer_gain.csv",
        formats,
    )
    if (
        summary.filter(pl.col("noise_mode") == "target_ber")["target_ber"].n_unique()
        > 1
    ):
        plot_noise_sensitivity(
            summary,
            source_count,
            source_budget,
            args.output_dir / "theoretical_transfer_noise_sensitivity",
            formats,
        )


if __name__ == "__main__":
    main()
