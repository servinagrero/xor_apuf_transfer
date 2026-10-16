"""Create selective publication figures for the alpha/beta metric sweep."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from study_utils import parse_int_list

METRICS = (
    ("uniformity_mean", "Uniformity", True),
    ("UPC_mean", "UPC", False),
    ("uniqueness_mean", "Uniqueness", True),
    ("uniqueness_h_mean", r"$H(\mathrm{Uniqueness})$", False),
    ("H_min_ba_mean", r"Mean plug-in $H_{min}(\mathrm{BA})$", False),
)

CONTOUR_LEVELS = np.linspace(0, 1, 41)


def save_figure(fig, output, formats):
    for extension in formats:
        fig.savefig(
            output.with_suffix(f".{extension}"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def metric_grid(frame, column, alpha, beta):
    """Project a metric onto the complete sweep grid.

    Some configurations are stored sparsely when a parameter is irrelevant.
    In particular, k=1 has no intra-die parameter and is evaluated at one beta
    value only. A row with one observation is therefore constant over beta.
    """
    grid = np.full((len(alpha), len(beta)), np.nan)
    alpha_index = {value: index for index, value in enumerate(alpha)}
    beta_index = {value: index for index, value in enumerate(beta)}
    for row in frame.select("alpha", "beta", column).iter_rows(named=True):
        grid[alpha_index[row["alpha"]], beta_index[row["beta"]]] = row[column]
    for index in range(len(alpha)):
        present = np.flatnonzero(np.isfinite(grid[index]))
        if len(present) == 1:
            grid[index, :] = grid[index, present[0]]
    return alpha, beta, grid


def plot_heatmaps(frame, selected_k, output, formats=("png",)):
    selected = frame.filter(pl.col("k").is_in(selected_k))
    ks = sorted(selected["k"].unique().to_list())
    missing = sorted(set(selected_k) - set(ks))
    if missing:
        raise ValueError(f"Requested XOR degrees missing from input: {missing}")

    alpha = np.sort(frame["alpha"].unique().to_numpy())
    beta = np.sort(frame["beta"].unique().to_numpy())
    fig, axes = plt.subplots(
        len(METRICS),
        len(ks),
        figsize=(2.0 * len(ks), 1.8 * len(METRICS)),
        squeeze=False,
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    raw_norm = colors.TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)
    entropy_norm = colors.Normalize(0, 1)
    row_images = []
    for column_index, k in enumerate(ks):
        panel = selected.filter(pl.col("k") == k)
        for row_index, (metric, label, raw) in enumerate(METRICS):
            _, _, grid = metric_grid(panel, metric, alpha, beta)
            if not np.isfinite(grid).all():
                raise ValueError(f"Incomplete {metric} grid for k={k}")
            image = axes[row_index, column_index].contourf(
                beta,
                alpha,
                grid,
                levels=CONTOUR_LEVELS,
                cmap="Spectral" if raw else "magma",
                norm=raw_norm if raw else entropy_norm,
                antialiased=True,
            )
            if column_index == 0:
                row_images.append(image)
            if row_index == 0:
                axes[row_index, column_index].set_title(rf"$k={k}$")
            if column_index == 0:
                axes[row_index, column_index].set_ylabel(f"{label}\n$\\alpha$")
            if row_index == len(METRICS) - 1:
                axes[row_index, column_index].set_xlabel(r"$\beta$")
            axes[row_index, column_index].set_xlim(0, 1)
            axes[row_index, column_index].set_ylim(0, 1)
            axes[row_index, column_index].set_xticks((0, 0.5, 1))
            axes[row_index, column_index].set_yticks((0, 0.5, 1))
    for row_index, image in enumerate(row_images):
        fig.colorbar(
            image,
            ax=axes[row_index, :],
            ticks=(0, 0.5, 1),
            fraction=0.012,
            pad=0.01,
            aspect=10,
        )
    save_figure(fig, output, formats)


def plot_parity_sections(frame, output, alpha_value=0.9, formats=("png",)):
    available = frame["alpha"].unique().to_numpy()
    alpha_value = float(available[np.argmin(np.abs(available - alpha_value))])
    selected = frame.filter(pl.col("alpha") == alpha_value)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True)
    columns = (
        ("uniformity_mean", "Uniformity"),
        ("uniqueness_mean", "Uniqueness"),
        ("H_min_ba_mean", r"Mean plug-in $H_{min}(\mathrm{BA})$"),
    )
    palette = plt.get_cmap("tab10")
    for index, (metric, label) in enumerate(columns):
        for k in sorted(selected["k"].unique().to_list()):
            curve = selected.filter(pl.col("k") == k).sort("beta")
            if curve.height < 2:
                continue
            axes[index].plot(
                curve["beta"],
                curve[metric],
                marker="o",
                markersize=2,
                linewidth=1,
                color=palette((k - 1) % 10),
                linestyle="-" if k % 2 else "--",
                label=f"k={k}",
            )
        axes[index].set_title(label)
        axes[index].set_xlabel(r"$\beta$")
        axes[index].set_ylim(0, 1.01)
        axes[index].grid(alpha=0.2)
    axes[0].set_ylabel(f"Metric value ($\\alpha={alpha_value:g}$)")
    axes[-1].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    save_figure(fig, output, formats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("data/alpha_beta_metrics_summary.csv")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figures/alpha_beta"))
    parser.add_argument("--k-values", default="1,2,3,4,5,6,7,8,9")
    parser.add_argument("--section-alpha", type=float, default=0.9)
    parser.add_argument("--formats", default="png,pdf")
    args = parser.parse_args()
    formats = tuple(item.strip() for item in args.formats.split(",") if item.strip())
    invalid = set(formats) - {"png", "pdf"}
    if not formats or invalid:
        parser.error(f"Unsupported output formats: {sorted(invalid)}")

    frame = pl.read_csv(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_heatmaps(
        frame,
        parse_int_list(args.k_values),
        args.output_dir / "metric_masking",
        formats,
    )
    plot_parity_sections(
        frame,
        args.output_dir / "parity_effect",
        args.section_alpha,
        formats,
    )


if __name__ == "__main__":
    main()
