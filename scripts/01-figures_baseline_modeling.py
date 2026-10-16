"""Plot measured baseline-modeling curves without interpolating missing budgets."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def save_figure(fig, output: Path, formats: tuple[str, ...]) -> None:
    for extension in formats:
        fig.savefig(
            output.with_suffix(f".{extension}"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def _summary(frame: pl.DataFrame) -> pl.DataFrame:
    keys = [
        "k",
        "model",
        "alpha",
        "beta",
        "noise_mode",
        "target_ber",
        "requested_noisiness",
        "budget",
    ]
    return (
        frame.group_by(keys)
        .agg(
            pl.len().alias("n_trials"),
            pl.col("observed_acc").mean().alias("mean_observed_acc"),
            pl.col("observed_acc").std().fill_null(0).alias("std_observed_acc"),
            pl.col("latent_acc").mean().alias("mean_latent_acc"),
            pl.col("test_ber").mean().alias("mean_test_ber"),
            pl.col("test_ber").std().fill_null(0).alias("std_test_ber"),
            pl.col("calibration_ber").mean().alias("mean_calibration_ber"),
            pl.col("noisiness").mean().alias("mean_calibrated_noisiness"),
            pl.col("accuracy_ceiling").mean().alias("mean_accuracy_ceiling"),
            pl.col("classification_noise_penalty")
            .mean()
            .alias("mean_classification_noise_penalty"),
            pl.col("majority_acc").mean().alias("mean_majority_acc"),
        )
        .sort(keys)
    )


def plot_baselines(
    summary: pl.DataFrame,
    output: Path,
    formats: tuple[str, ...] = ("png",),
) -> None:
    ks = sorted(summary["k"].unique().to_list())
    fig, ax = plt.subplots(figsize=(6, 3))
    palette = plt.get_cmap("tab10")

    for index, k in enumerate(ks):
        curve = summary.filter(
            (pl.col("k") == k) & (pl.col("model") == "target_nn")
        ).sort("budget")
        x = curve["budget"].to_numpy()
        mean = curve["mean_observed_acc"].to_numpy()
        std = curve["std_observed_acc"].to_numpy()

        color = palette(index)
        ax.plot(
            x,
            mean,
            marker="o",
            linewidth=1.2,
            markersize=3.5,
            color=color,
            label=rf"$k={k}$",
        )
        ax.fill_between(
            x,
            np.clip(mean - std, 0, 1),
            np.clip(mean + std, 0, 1),
            color=color,
            alpha=0.14,
            linewidth=0,
        )

    ax.set_xscale("log")
    ax.set_ylim(0.45, 1.02)
    ax.axhline(0.5, color="0.5", linestyle=":", linewidth=1)
    ax.grid(alpha=0.2)
    ax.set_xlabel("Total labeled CRPs")
    ax.set_ylabel("Observed-response accuracy")
    ax.legend(
        title="XOR degree",
        fontsize=7,
        title_fontsize=7,
        ncol=3,
        loc="upper left",
    )
    fig.tight_layout()
    save_figure(fig, output, formats)


def plot_noise_sensitivity(
    summary: pl.DataFrame,
    output: Path,
    formats: tuple[str, ...] = ("png",),
) -> None:
    panel = summary.filter(
        (pl.col("noise_mode") == "target_ber")
        & (pl.col("model") == "target_nn")
        & (pl.col("budget") == pl.col("budget").max())
    )
    ks = sorted(panel["k"].unique().to_list())
    fig, axes = plt.subplots(
        int(np.ceil(len(ks) / 3)),
        3,
        figsize=(10.5, 2.8 * int(np.ceil(len(ks) / 3))),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for ax, k in zip(axes.flat, ks):
        curve = panel.filter(pl.col("k") == k).sort("target_ber")
        ax.plot(
            curve["target_ber"],
            curve["mean_observed_acc"],
            marker="o",
            label="observed",
        )
        ax.plot(
            curve["target_ber"], curve["mean_latent_acc"], marker="s", label="latent"
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
    fig.tight_layout()
    save_figure(fig, output, formats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("data/baseline_modeling.csv")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("figures/baseline_modeling")
    )
    parser.add_argument("--target-ber", type=float, default=0.0)
    parser.add_argument("--formats", default="png,pdf")
    args = parser.parse_args()
    formats = tuple(item.strip() for item in args.formats.split(",") if item.strip())
    invalid = set(formats) - {"png", "pdf"}
    if not formats or invalid:
        parser.error(f"Unsupported output formats: {sorted(invalid)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = _summary(pl.read_csv(args.input))
    summary.write_csv(args.output_dir / "baseline_modeling_summary.csv")
    (
        summary.filter(pl.col("noise_mode") == "target_ber")
        .group_by(["k", "target_ber"])
        .agg(
            pl.col("mean_calibration_ber").mean(),
            pl.col("mean_test_ber").mean().alias("mean_achieved_test_ber"),
            pl.col("mean_calibrated_noisiness").mean(),
            pl.col("mean_accuracy_ceiling").mean(),
            pl.col("mean_classification_noise_penalty")
            .mean()
            .alias("symmetric_noise_sample_penalty"),
        )
        .sort(["k", "target_ber"])
        .write_csv(args.output_dir / "ber_calibration_pac_reference.csv")
    )
    selected = summary.filter(
        (pl.col("noise_mode") != "target_ber")
        | ((pl.col("target_ber") - args.target_ber).abs() < 1e-9)
    )
    plot_baselines(
        selected,
        args.output_dir / "baseline_modeling",
        formats,
    )
    if (
        summary.filter(pl.col("noise_mode") == "target_ber")["target_ber"].n_unique()
        > 1
    ):
        plot_noise_sensitivity(
            summary,
            args.output_dir / "baseline_noise_sensitivity",
            formats,
        )


if __name__ == "__main__":
    main()
