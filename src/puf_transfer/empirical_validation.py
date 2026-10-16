"""Uniqueness-aware controls for empirical transfer experiments."""

import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr


def load_metric_distributions(path):
    return np.load(Path(path), allow_pickle=True).item()


def uniqueness_geometry(metrics, combination, threshold=0.20):
    entry = metrics["combinations"][combination]
    devices = entry["device_ids"].astype(str).tolist()
    index = {device: i for i, device in enumerate(devices)}
    distances = np.zeros((len(devices), len(devices)), dtype=float)
    for pair, value in zip(entry["uniqueness_device_pairs"], entry["uniqueness"]):
        first, second = map(str, pair)
        i, j = index[first], index[second]
        distances[i, j] = distances[j, i] = float(value)
    if len(devices) < 2:
        labels = np.ones(len(devices), dtype=int)
    else:
        labels = fcluster(
            linkage(squareform(distances), method="complete"),
            t=threshold,
            criterion="distance",
        )
    return {
        "devices": devices,
        "index": index,
        "distances": distances,
        "clusters": {device: int(labels[i]) for i, device in enumerate(devices)},
        "threshold": float(threshold),
    }


def source_distance_statistics(geometry, target, sources):
    sources = list(sources)
    if target not in geometry["index"] or not sources:
        return None
    known = [source for source in sources if source in geometry["index"]]
    if len(known) != len(sources):
        return None
    target_index = geometry["index"][target]
    values = np.asarray(
        [geometry["distances"][target_index, geometry["index"][source]] for source in sources]
    )
    cluster = geometry["clusters"][target]
    same_cluster = [source for source in sources if geometry["clusters"][source] == cluster]
    nearest_index = int(np.argmin(values))
    return {
        "target_source_min_uniqueness": float(values.min()),
        "target_source_mean_uniqueness": float(values.mean()),
        "target_source_max_uniqueness": float(values.max()),
        "diagnostic_closest_source_device": sources[nearest_index],
        "target_cluster": cluster,
        "selected_same_cluster_sources": len(same_cluster),
        "selected_same_cluster_device_ids": json.dumps(same_cluster),
        "source_uniqueness_values": values.tolist(),
    }


def filter_source_pool(
    geometry,
    target,
    eligible,
    policy,
    *,
    exclude_nearest_count=0,
):
    """Apply a uniqueness-aware evaluation split to an eligible source pool."""
    eligible = list(eligible)
    if policy == "random":
        return eligible, -1, []
    if target not in geometry["clusters"]:
        return [], -1, eligible
    target_cluster = geometry["clusters"][target]
    if policy == "exclude-nearest":
        if exclude_nearest_count < 1:
            raise ValueError(
                "exclude-nearest requires a positive exclude_nearest_count."
            )
        target_index = geometry["index"][target]
        ordered = sorted(
            eligible,
            key=lambda device: geometry["distances"][
                target_index, geometry["index"][device]
            ],
        )
        excluded = ordered[:exclude_nearest_count]
        excluded_set = set(excluded)
        return (
            [device for device in eligible if device not in excluded_set],
            target_cluster,
            excluded,
        )

    if policy != "leave-cluster-out":
        raise ValueError(f"Unknown source policy: {policy}")
    excluded = [
        device
        for device in eligible
        if geometry["clusters"].get(device) == target_cluster
    ]
    return [device for device in eligible if device not in excluded], target_cluster, excluded


def annotate_transfer_results(frame, metrics, threshold=0.20):
    """Attach uniqueness geometry without changing any attack result."""
    geometries = {
        name: uniqueness_geometry(metrics, name, threshold)
        for name in metrics["combinations"]
    }
    rows = []
    for row in frame.iter_rows(named=True):
        sources = json.loads(row["source_devices"])
        stats = source_distance_statistics(geometries[row["slices"]], row["target_device"], sources)
        if stats is None:
            continue
        stats.pop("source_uniqueness_values")
        rows.append({**row, **stats, "cluster_threshold": threshold})
    return pl.DataFrame(rows)


def clustering_tables(metrics, thresholds=(0.15, 0.20, 0.25)):
    pairs, clusters = [], []
    for name, entry in metrics["combinations"].items():
        for pair, value in zip(entry["uniqueness_device_pairs"], entry["uniqueness"]):
            pairs.append(
                {
                    "k_xor": entry["k_xor"],
                    "slices": name,
                    "device_1": str(pair[0]),
                    "device_2": str(pair[1]),
                    "uniqueness": float(value),
                }
            )
        for threshold in thresholds:
            geometry = uniqueness_geometry(metrics, name, threshold)
            counts = {}
            for cluster in geometry["clusters"].values():
                counts[cluster] = counts.get(cluster, 0) + 1
            for device, cluster in geometry["clusters"].items():
                clusters.append(
                    {
                        "k_xor": entry["k_xor"],
                        "slices": name,
                        "threshold": threshold,
                        "device_id": device,
                        "cluster": cluster,
                        "cluster_size": counts[cluster],
                    }
                )
    return pl.DataFrame(pairs), pl.DataFrame(clusters)


def distance_correlations(frame):
    rows = []
    for keys, group in frame.group_by(["k_xor", "k_pretrain", "m_target_budget"]):
        k_xor, k_pretrain, budget = keys
        for distance in ("target_source_min_uniqueness", "target_source_mean_uniqueness"):
            x = group[distance].to_numpy()
            for outcome in ("zero_shot_acc", "transfer_acc", "acc_gain_delta"):
                y = group[outcome].to_numpy()
                valid = np.isfinite(x) & np.isfinite(y)
                pearson = float(np.corrcoef(x[valid], y[valid])[0, 1]) if valid.sum() > 1 else np.nan
                spearman = float(spearmanr(x[valid], y[valid]).statistic) if valid.sum() > 1 else np.nan
                rows.append(
                    {
                        "k_xor": k_xor,
                        "k_pretrain": k_pretrain,
                        "m_target_budget": budget,
                        "distance": distance,
                        "outcome": outcome,
                        "n_cells": int(valid.sum()),
                        "pearson": pearson,
                        "spearman": spearman,
                    }
                )
    return pl.DataFrame(rows).sort(["k_xor", "k_pretrain", "m_target_budget", "distance", "outcome"])
