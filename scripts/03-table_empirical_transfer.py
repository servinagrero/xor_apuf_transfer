"""Generate the publication empirical transfer-cost table."""

import argparse
from pathlib import Path

import polars as pl

KEYS = ["k_xor", "k_pretrain", "n_base_budget", "m_target_budget"]


def require_columns(frame, columns, source):
    """Raise a useful error when an input is stale or incomplete."""
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {', '.join(missing)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--break-even", type=Path, required=True)
    parser.add_argument("--frontier", type=Path, required=True)
    parser.add_argument("--source-count", type=int, required=True)
    parser.add_argument("--source-budget", type=int, required=True)
    parser.add_argument("--target-budget", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, default=None)
    args = parser.parse_args()

    break_even = pl.read_csv(args.break_even)
    frontier = pl.read_csv(args.frontier)
    require_columns(
        break_even,
        KEYS
        + [
            "source_pretraining_cost",
            "transfer_accuracy",
            "target_only_accuracy_same_budget",
            "baseline_equivalent_budget",
            "break_even_targets",
        ],
        args.break_even,
    )
    frontier_columns = KEYS + [
        "ci95_low_transfer_acc",
        "ci95_high_transfer_acc",
        "ci95_low_baseline_acc",
        "ci95_high_baseline_acc",
        "mean_acc_gain_delta",
        "ci95_low_acc_gain_delta",
        "ci95_high_acc_gain_delta",
    ]
    require_columns(frontier, frontier_columns, args.frontier)

    selected = (
        break_even.filter(
            (pl.col("k_pretrain") == args.source_count)
            & (pl.col("n_base_budget") == args.source_budget)
            & (pl.col("m_target_budget") == args.target_budget)
        )
        .join(
            frontier.select(frontier_columns),
            on=KEYS,
            how="inner",
            validate="1:1",
        )
        .with_columns(
            pl.col("mean_acc_gain_delta").alias("paired_gain"),
            pl.col("ci95_low_acc_gain_delta").alias("gain_ci95_low"),
            pl.col("ci95_high_acc_gain_delta").alias("gain_ci95_high"),
        )
        .sort("k_xor")
    )

    expected_ks = break_even["k_xor"].n_unique()
    if len(selected) != expected_ks:
        raise ValueError(
            f"Expected one selected row for each of {expected_ks} XOR degrees; "
            f"found {len(selected)}"
        )

    source_costs = selected["source_pretraining_cost"].unique().to_list()
    equivalent_budgets = (
        selected["baseline_equivalent_budget"].drop_nulls().unique().to_list()
    )
    break_even_values = (
        selected["break_even_targets"].drop_nulls().unique().to_list()
    )
    if len(source_costs) != 1:
        raise ValueError("Selected rows do not have a common source cost")
    if len(equivalent_budgets) != 1 or len(break_even_values) != 1:
        raise ValueError(
            "Compact Table 4 requires a common B_tau and break-even value"
        )

    selected = selected.with_columns(
        (
            1
            - pl.col("m_target_budget")
            / pl.col("baseline_equivalent_budget")
        ).alias("target_crp_reduction")
    )

    csv_output = args.csv_output or args.output.with_suffix(".csv")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    selected.write_csv(csv_output)

    lines = [
        r"\begin{table}[t]",
        r"    \centering",
        r"    \small",
        r"    \begin{tabular}{@{}rccccc@{}}",
        r"        \toprule",
        (
            r"        $k$ & Transfer acc. & Target-only acc. "
            r"& Paired $\Delta$ & 95\% CI & Target reduction \\"
        ),
        r"        \midrule",
    ]
    for row in selected.iter_rows(named=True):
        significant = row["gain_ci95_low"] > 0 or row["gain_ci95_high"] < 0
        gain = f"{100 * row['paired_gain']:+.1f}"
        interval = (
            f"[{100 * row['gain_ci95_low']:.1f},"
            f"{100 * row['gain_ci95_high']:.1f}]"
        )
        if significant:
            gain = rf"\textbf{{{gain}}}"
            interval = rf"\textbf{{{interval}}}"
        lines.append(
            f"        {row['k_xor']} & "
            f"{100 * row['transfer_accuracy']:.1f}\\% & "
            f"{100 * row['target_only_accuracy_same_budget']:.1f}\\% & "
            f"{gain} & {interval} & "
            f"{100 * row['target_crp_reduction']:.1f}\\% \\\\"
        )
    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}",
            (
                rf"    \caption{{Matched-cost empirical results with "
                rf"$S={args.source_count}$ sources, "
                rf"$N={args.source_budget}$ CRPs per source "
                rf"($C_{{\mathrm{{src}}}}={source_costs[0]}$), and "
                rf"$M={args.target_budget}$ target CRPs. Target-only accuracy "
                rf"uses the same $M$ CRPs. For every row, the smallest tested "
                rf"target-only budget matching the transfer accuracy is "
                rf"$B_\tau={equivalent_budgets[0]}$, giving a break-even "
                rf"population of $T={break_even_values[0]}$ targets. "
                rf"$B_\tau$ is resolved only on the measured budget grid. "
                rf"Target reduction is the marginal reduction "
                rf"$1-M/B_\tau$ after source pretraining; it does not include "
                rf"the one-time source cost. Bold paired gains and intervals "
                rf"exclude zero.}}"
            ),
            r"    \label{tab:empirical-transfer}",
            r"\end{table}",
        ]
    )
    args.output.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
