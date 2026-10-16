from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
import numpy.typing as npt
import polars as pl

from puf_transfer.budgets import split_labeled_budget as split_labeled_budget


@dataclass(frozen=True)
class StudyPaths:
    data_dir: Path = Path("data")
    figures_dir: Path = Path("figures")

    def ensure(self) -> None:
        self.data_dir.mkdir(exist_ok=True)
        self.figures_dir.mkdir(exist_ok=True)


@dataclass(frozen=True)
class ParallelConfig:
    n_jobs: int = 1
    verbose: int = 10


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_linspace(value: str) -> npt.NDArray:
    start, stop, count = value.split(":")
    return np.linspace(float(start), float(stop), int(count))


def run_grid(grid: Iterable[dict], compute_fn, parallel: ParallelConfig):
    return joblib.Parallel(n_jobs=parallel.n_jobs, verbose=parallel.verbose)(
        joblib.delayed(compute_fn)(**params) for params in grid
    )


def flatten_rows(results: Sequence):
    return list(itertools.chain.from_iterable(results))


def write_csv(rows, output_path: Path) -> pl.DataFrame:
    df = pl.DataFrame(rows)
    df.write_csv(output_path)
    return df


def resolve_data_path(path: str | Path, data_dir: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return data_dir / path


def load_study_points(
    path: str | Path,
    *,
    data_dir: Path,
    selections: Sequence[str] | None = None,
) -> list[dict]:
    df = pl.read_csv(resolve_data_path(path, data_dir))
    required = {"k", "alpha", "beta"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Study grid is missing required columns: {sorted(missing)}")

    if selections is not None and "selection" in df.columns:
        df = df.filter(pl.col("selection").is_in(list(selections)))

    columns = ["k", "alpha", "beta"]
    if "selection" in df.columns:
        columns.append("selection")

    points = []
    for row in df.select(columns).iter_rows(named=True):
        row["k"] = int(row["k"])
        row["alpha"] = float(row["alpha"])
        row["beta"] = float(row["beta"])
        points.append(row)
    return points
