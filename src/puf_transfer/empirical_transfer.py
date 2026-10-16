"""Leave-one-device-out FiLM transfer using only XOR labels for the chosen slices."""

import copy
from functools import lru_cache
import json
from time import perf_counter

import numpy as np
import torch

from puf_transfer import PUFModel
from puf_transfer.empirical import EmpiricalDataset
from puf_transfer.budgets import split_labeled_budget


@lru_cache(maxsize=2)
def load_dataset(cache_dir, challenges_path=None):
    return EmpiricalDataset(cache_dir, challenges_path)


def transfer_trial(
    *,
    cache_dir,
    challenges_path,
    slices,
    target,
    source_devices,
    n_base_budget,
    m_budgets,
    trial=0,
    seed=42,
    validation_fraction=0.2,
    epochs=100,
    batch_size=512,
    hidden_dim=64,
    embedding_dim=16,
    patience=15,
    source_policy="random",
    cluster_threshold=np.nan,
    target_cluster=-1,
    excluded_cluster_devices=None,
    source_uniqueness_values=None,
    diagnostic_baselines=False,
    source_label_control="none",
):
    torch.set_num_threads(1)
    data = load_dataset(cache_dir, challenges_path)
    sources = list(source_devices)
    excluded_cluster_devices = list(excluded_cluster_devices or [])
    if target in sources or len(sources) != len(set(sources)) or not sources:
        raise ValueError("Source devices must be distinct and exclude the target.")
    if min(m_budgets) < 2 or n_base_budget < 2 or epochs < 1:
        raise ValueError("Labeled budgets must be at least two and epochs positive.")
    seed = seed + trial * 100_000
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    def examples(device, split, budget=None, shuffle=False):
        ids = data.ids(device, slices, split)
        if budget is not None and len(ids) < budget:
            raise ValueError(
                f"{device}, {slices}: {split} has {len(ids)} CRPs, needs {budget}."
            )
        if shuffle:
            ids = rng.permutation(ids)
        if budget is not None:
            ids = ids[:budget]
        return data.features(ids), data.responses(device, slices, ids), ids

    source_split = split_labeled_budget(
        n_base_budget,
        validation_fraction=validation_fraction,
        min_validation=20,
        max_validation=min(
            len(data.ids(device, slices, "validation")) for device in sources
        ),
    )
    target_splits = {
        m: split_labeled_budget(
            m,
            validation_fraction=validation_fraction,
            min_validation=20,
            max_validation=len(data.ids(target, slices, "validation")),
        )
        for m in sorted(set(m_budgets))
    }
    pre_x, pre_y, pre_d, val_x, val_y, val_d = [], [], [], [], [], []
    for index, device in enumerate(sources):
        x, y, _ = examples(device, "train", source_split.train, shuffle=True)
        vx, vy, _ = examples(device, "validation", source_split.validation)
        pre_x.append(x)
        pre_y.append(y)
        pre_d.append(np.full(len(y), index))
        val_x.append(vx)
        val_y.append(vy)
        val_d.append(np.full(len(vy), index))
    if source_label_control == "challenge-permuted":
        pre_y = [rng.permutation(labels) for labels in pre_y]
        val_y = [rng.permutation(labels) for labels in val_y]
    elif source_label_control != "none":
        raise ValueError(f"Unknown source-label control: {source_label_control}")
    target_id = len(sources)

    def new_model(devices):
        return PUFModel(
            n_inputs=data.manifest["n_bits"],
            num_devices=devices,
            device_emb_dim=embedding_dim,
            hidden_dim=hidden_dim,
            k_xor=len(slices),
            epochs=epochs,
            batch_size=batch_size,
        )

    model = new_model(target_id + 1)
    start = perf_counter()
    model.fit(
        np.vstack(pre_x),
        np.concatenate(pre_d),
        np.concatenate(pre_y),
        np.vstack(val_x),
        np.concatenate(val_d),
        np.concatenate(val_y),
        patience=patience,
    )
    pretrain_seconds = perf_counter() - start
    with torch.no_grad():
        model.model.embedding.weight[target_id].copy_(
            model.model.embedding.weight[:target_id].mean(dim=0)
        )
    base_state = copy.deepcopy(model.model.state_dict())
    # Independent target ordering keeps the same nested budgets across source configurations.
    target_rng = np.random.default_rng(seed + 1)
    max_target_train = max(split.train for split in target_splits.values())
    max_target_validation = max(split.validation for split in target_splits.values())
    target_ids = target_rng.permutation(data.ids(target, slices, "train"))
    if len(target_ids) < max_target_train:
        raise ValueError(f"Target {target} has too few complete training CRPs.")
    target_ids = target_ids[:max_target_train]
    tx, ty = data.features(target_ids), data.responses(target, slices, target_ids)
    vx, vy, _ = examples(target, "validation", max_target_validation)
    ex, ey, test_ids = examples(target, "test")
    if not len(ey):
        raise ValueError(f"Target {target} has no complete test CRPs.")
    test_d = np.full(len(ey), target_id)
    zero_shot = float(np.mean((model.predict(ex, test_d).ravel() >= 0.5) == ey))
    source_uniqueness_values = list(source_uniqueness_values or [])
    if source_uniqueness_values and len(source_uniqueness_values) != len(sources):
        raise ValueError("Source uniqueness values must align with source devices.")
    source_distance = {
        "target_source_min_uniqueness": float(np.min(source_uniqueness_values))
        if source_uniqueness_values
        else np.nan,
        "target_source_mean_uniqueness": float(np.mean(source_uniqueness_values))
        if source_uniqueness_values
        else np.nan,
        "target_source_max_uniqueness": float(np.max(source_uniqueness_values))
        if source_uniqueness_values
        else np.nan,
    }
    diagnostic = {
        "source_embedding_ensemble_acc": np.nan,
        "oracle_best_source_embedding_acc": np.nan,
        "diagnostic_closest_source_embedding_acc": np.nan,
        "diagnostic_closest_source_device": "",
        "source_consensus_acc": np.nan,
        "source_consensus_extra_crps": 0,
        "pooled_source_acc": np.nan,
        "diagnostic_baselines_seconds": 0.0,
    }
    if diagnostic_baselines:
        start = perf_counter()
        source_probabilities = np.stack(
            [
                model.predict(ex, np.full(len(ey), index, dtype=int)).ravel()
                for index in range(len(sources))
            ]
        )
        source_accuracies = np.mean((source_probabilities >= 0.5) == ey, axis=1)
        diagnostic["source_embedding_ensemble_acc"] = float(
            np.mean((source_probabilities.mean(axis=0) >= 0.5) == ey)
        )
        diagnostic["oracle_best_source_embedding_acc"] = float(source_accuracies.max())
        if source_uniqueness_values:
            closest = int(np.argmin(source_uniqueness_values))
            diagnostic["diagnostic_closest_source_embedding_acc"] = float(
                source_accuracies[closest]
            )
            diagnostic["diagnostic_closest_source_device"] = sources[closest]

        consensus_rows = []
        for source in sources:
            if len(data.ids(source, slices, "test")) and all(
                data.chains[source, slice_name][test_ids].min() != 255
                for slice_name in slices
            ):
                consensus_rows.append(data.responses(source, slices, test_ids))
        if consensus_rows:
            consensus = np.mean(np.stack(consensus_rows), axis=0) >= 0.5
            diagnostic["source_consensus_acc"] = float(np.mean(consensus == ey))
            diagnostic["source_consensus_extra_crps"] = len(consensus_rows) * len(test_ids)

        torch.manual_seed(seed + 3)
        pooled = new_model(1)
        pooled.fit(
            np.vstack(pre_x),
            np.zeros(sum(map(len, pre_y)), dtype=int),
            np.concatenate(pre_y),
            np.vstack(val_x),
            np.zeros(sum(map(len, val_y)), dtype=int),
            np.concatenate(val_y),
            patience=patience,
        )
        diagnostic["pooled_source_acc"] = float(
            np.mean(
                (pooled.predict(ex, np.zeros(len(ey), dtype=int)).ravel() >= 0.5)
                == ey
            )
        )
        diagnostic["diagnostic_baselines_seconds"] = perf_counter() - start
    rows = []
    for m in sorted(set(m_budgets)):
        target_split = target_splits[m]
        train_count = target_split.train
        validation_count = target_split.validation
        model.model.load_state_dict(base_state)
        start = perf_counter()
        model.fit_transfer(
            tx[:train_count],
            target_id,
            ty[:train_count],
            vx[:validation_count],
            np.full(validation_count, target_id),
            vy[:validation_count],
            patience=patience,
        )
        adapt_seconds = perf_counter() - start
        pred = model.predict(ex, test_d).ravel() >= 0.5
        acc = float(np.mean(pred == ey))
        torch.manual_seed(seed + 2)
        baseline = new_model(1)
        start = perf_counter()
        baseline.fit(
            tx[:train_count],
            np.zeros(train_count, dtype=int),
            ty[:train_count],
            vx[:validation_count],
            np.zeros(validation_count, dtype=int),
            vy[:validation_count],
            patience=patience,
        )
        baseline_seconds = perf_counter() - start
        baseline_acc = float(
            np.mean(
                (baseline.predict(ex, np.zeros(len(ey), dtype=int)).ravel() >= 0.5)
                == ey
            )
        )
        majority = int(np.mean(ty[:train_count]) >= 0.5)
        rows.append(
            dict(
                k_xor=len(slices),
                slices="+".join(slices),
                target_device=target,
                source_devices=json.dumps(sources),
                source_policy=source_policy,
                source_label_control=source_label_control,
                cluster_threshold=cluster_threshold,
                target_cluster=target_cluster,
                excluded_cluster_devices=json.dumps(excluded_cluster_devices),
                k_pretrain=target_id,
                n_base_budget=n_base_budget,
                m_target_budget=m,
                trial=trial,
                seed=seed,
                n_bits=data.manifest["n_bits"],
                source_training_crps=target_id * source_split.train,
                source_validation_crps=target_id * source_split.validation,
                source_total_labeled_crps=target_id * n_base_budget,
                target_training_crps=train_count,
                target_validation_crps=validation_count,
                target_total_labeled_crps=m,
                transfer_target_labeled_crps=m,
                baseline_target_labeled_crps=m,
                test_crps=len(ey),
                test_id_min=int(test_ids.min()),
                test_id_max=int(test_ids.max()),
                transfer_acc=acc,
                baseline_acc=baseline_acc,
                acc_gain_delta=acc - baseline_acc,
                zero_shot_acc=zero_shot,
                majority_acc=float(np.mean(ey == majority)),
                test_uniformity=float(np.mean(ey)),
                pretrain_seconds=pretrain_seconds,
                adaptation_seconds=adapt_seconds,
                baseline_seconds=baseline_seconds,
                hidden_dim=hidden_dim,
                embedding_dim=embedding_dim,
                epochs=epochs,
                batch_size=batch_size,
                patience=patience,
                target_initialization="source_embedding_mean",
                **source_distance,
                **diagnostic,
            )
        )
    return rows


def fixed_cohort_trial(
    *,
    cache_dir,
    challenges_path,
    slices,
    source_devices,
    target_devices,
    n_base_budget,
    m_target_budget,
    trial=0,
    seed=42,
    validation_fraction=0.2,
    epochs=100,
    batch_size=512,
    hidden_dim=64,
    embedding_dim=16,
    patience=15,
):
    """Pretrain once on a fixed cohort and reuse that model for every target."""
    torch.set_num_threads(1)
    data = load_dataset(cache_dir, challenges_path)
    sources = list(source_devices)
    targets = list(target_devices)
    if not sources or not targets:
        raise ValueError("Fixed-cohort evaluation needs sources and held-out targets.")
    if len(sources) != len(set(sources)) or len(targets) != len(set(targets)):
        raise ValueError("Source and target device lists must not contain duplicates.")
    if set(sources) & set(targets):
        raise ValueError("Fixed source and target cohorts must be disjoint.")
    if min(n_base_budget, m_target_budget) < 2 or epochs < 1:
        raise ValueError("Labeled budgets must be at least two and epochs positive.")

    run_seed = seed + trial * 100_000
    torch.manual_seed(run_seed)
    rng = np.random.default_rng(run_seed)

    def examples(device, split, budget=None, shuffle=False):
        ids = data.ids(device, slices, split)
        if budget is not None and len(ids) < budget:
            raise ValueError(
                f"{device}, {slices}: {split} has {len(ids)} CRPs, needs {budget}."
            )
        if shuffle:
            ids = rng.permutation(ids)
        if budget is not None:
            ids = ids[:budget]
        return data.features(ids), data.responses(device, slices, ids), ids

    source_split = split_labeled_budget(
        n_base_budget,
        validation_fraction=validation_fraction,
        min_validation=20,
        max_validation=min(
            len(data.ids(device, slices, "validation")) for device in sources
        ),
    )
    target_split = split_labeled_budget(
        m_target_budget,
        validation_fraction=validation_fraction,
        min_validation=20,
        max_validation=min(
            len(data.ids(device, slices, "validation")) for device in targets
        ),
    )

    pre_x, pre_y, pre_d, val_x, val_y, val_d = [], [], [], [], [], []
    for index, device in enumerate(sources):
        x, y, _ = examples(device, "train", source_split.train, shuffle=True)
        vx, vy, _ = examples(device, "validation", source_split.validation)
        pre_x.append(x)
        pre_y.append(y)
        pre_d.append(np.full(len(y), index))
        val_x.append(vx)
        val_y.append(vy)
        val_d.append(np.full(len(vy), index))

    target_id = len(sources)
    model = PUFModel(
        n_inputs=data.manifest["n_bits"],
        num_devices=target_id + 1,
        device_emb_dim=embedding_dim,
        hidden_dim=hidden_dim,
        k_xor=len(slices),
        epochs=epochs,
        batch_size=batch_size,
    )
    start = perf_counter()
    model.fit(
        np.vstack(pre_x),
        np.concatenate(pre_d),
        np.concatenate(pre_y),
        np.vstack(val_x),
        np.concatenate(val_d),
        np.concatenate(val_y),
        patience=patience,
    )
    pretrain_seconds = perf_counter() - start
    with torch.no_grad():
        model.model.embedding.weight[target_id].copy_(
            model.model.embedding.weight[:target_id].mean(dim=0)
        )
    pretrained_state = copy.deepcopy(model.model.state_dict())

    rows = []
    for target in targets:
        # Match the nested target-budget ordering used by the main empirical study.
        target_rng = np.random.default_rng(run_seed + 1)
        train_ids = target_rng.permutation(data.ids(target, slices, "train"))
        train_ids = train_ids[: target_split.train]
        tx = data.features(train_ids)
        ty = data.responses(target, slices, train_ids)
        vx, vy, _ = examples(target, "validation", target_split.validation)
        ex, ey, test_ids = examples(target, "test")
        if len(train_ids) != target_split.train or not len(ey):
            raise ValueError(f"Target {target} lacks complete fixed-cohort data.")

        model.model.load_state_dict(pretrained_state)
        test_d = np.full(len(ey), target_id)
        zero_shot = float(
            np.mean((model.predict(ex, test_d).ravel() >= 0.5) == ey)
        )
        start = perf_counter()
        model.fit_transfer(
            tx,
            target_id,
            ty,
            vx,
            np.full(len(vy), target_id),
            vy,
            patience=patience,
        )
        adaptation_seconds = perf_counter() - start
        transfer_acc = float(
            np.mean((model.predict(ex, test_d).ravel() >= 0.5) == ey)
        )
        rows.append(
            dict(
                k_xor=len(slices),
                slices="+".join(slices),
                cohort_trial=trial,
                seed=run_seed,
                source_devices=json.dumps(sources),
                target_device=target,
                k_pretrain=len(sources),
                held_out_target_count=len(targets),
                n_base_budget=n_base_budget,
                m_target_budget=m_target_budget,
                source_training_crps=len(sources) * source_split.train,
                source_validation_crps=len(sources) * source_split.validation,
                source_total_labeled_crps=len(sources) * n_base_budget,
                target_training_crps=target_split.train,
                target_validation_crps=target_split.validation,
                target_total_labeled_crps=m_target_budget,
                test_crps=len(ey),
                test_id_min=int(test_ids.min()),
                test_id_max=int(test_ids.max()),
                zero_shot_acc=zero_shot,
                transfer_acc=transfer_acc,
                adaptation_gain=transfer_acc - zero_shot,
                pretrain_seconds=pretrain_seconds,
                adaptation_seconds=adaptation_seconds,
                hidden_dim=hidden_dim,
                embedding_dim=embedding_dim,
                epochs=epochs,
                batch_size=batch_size,
                patience=patience,
            )
        )
    return rows
