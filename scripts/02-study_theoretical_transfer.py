"""Run paired transfer and target-only attacks on simulated k-XOR populations."""

import argparse
import copy
import itertools
from time import perf_counter

import numpy as np
import polars as pl
import torch

from puf_transfer import (
    CorrelatedXORArbiterPUF,
    PUFModel,
    generate_weights_bias,
    logger,
    random_crps,
    suffix_parity_transform,
)
from study_utils import (
    ParallelConfig,
    StudyPaths,
    flatten_rows,
    load_study_points,
    parse_float_list,
    parse_int_list,
    parse_str_list,
    run_grid,
    split_labeled_budget,
    write_csv,
)
from puf_transfer.noise import (
    calibrate_noisiness,
    classification_noise_penalty,
    latent_response,
)

torch.set_num_threads(1)


def _binary(response):
    return np.asarray(response < 0, dtype=np.uint8)


def _accuracy(labels, probabilities):
    return float(np.mean((np.asarray(probabilities).ravel() >= 0.5) == labels))


def compute_theoretical_transfer(
    alpha: float,
    beta: float,
    k_xor: int,
    k_pretrain: int,
    n_base_budget: int,
    m_budgets: list[int],
    *,
    selection: str = "grid",
    n_bits: int = 64,
    n_trials: int = 5,
    seed: int = 42,
    noisiness: float | None = None,
    target_ber: float | None = 0.0,
    calibration_size: int = 20_000,
    validation_fraction: float = 0.2,
    test_size: int = 5000,
    epochs: int = 100,
    batch_size: int = 512,
    hidden_dim: int = 64,
    embedding_dim: int = 16,
    patience: int = 15,
):
    rows = []
    source_split = split_labeled_budget(
        n_base_budget, validation_fraction=validation_fraction, min_validation=20
    )
    target_splits = {
        m: split_labeled_budget(
            m, validation_fraction=validation_fraction, min_validation=20
        )
        for m in sorted(set(m_budgets))
    }

    for trial in range(n_trials):
        trial_seed = seed + trial * 1_000_000 + k_xor * 10_000
        torch.manual_seed(trial_seed)
        shared_weights, shared_bias = generate_weights_bias(
            n_bits, k=k_xor, beta=beta, seed=trial_seed
        )
        target_seed = trial_seed + 900_000

        def target_factory(noise):
            return CorrelatedXORArbiterPUF(
                n_bits,
                k_xor,
                shared_weights,
                shared_bias,
                alpha,
                seed=target_seed,
                noisiness=noise,
            )

        if target_ber is not None:
            calibration_x = random_crps(
                calibration_size, n_bits, seed=trial_seed + 800_000
            )
            calibrated_noisiness, calibration_ber = calibrate_noisiness(
                target_factory, calibration_x, target_ber
            )
        else:
            calibrated_noisiness = float(noisiness or 0.0)
            calibration_ber = np.nan

        pre_x, pre_y, pre_d = [], [], []
        val_x, val_y, val_d = [], [], []
        for device_id in range(k_pretrain):
            device_seed = trial_seed + 100 + device_id
            puf = CorrelatedXORArbiterPUF(
                n_bits,
                k_xor,
                shared_weights,
                shared_bias,
                alpha,
                seed=device_seed,
                noisiness=calibrated_noisiness,
            )
            x_train = random_crps(
                source_split.train, n_bits, seed=device_seed + 10_000
            )
            x_val = random_crps(
                source_split.validation, n_bits, seed=device_seed + 20_000
            )
            pre_x.append(suffix_parity_transform(x_train))
            pre_y.append(_binary(puf.eval(x_train)))
            pre_d.append(np.full(source_split.train, device_id, dtype=int))
            val_x.append(suffix_parity_transform(x_val))
            val_y.append(_binary(puf.eval(x_val)))
            val_d.append(np.full(source_split.validation, device_id, dtype=int))

        target_id = k_pretrain

        def new_model(num_devices):
            return PUFModel(
                n_inputs=n_bits,
                num_devices=num_devices,
                device_emb_dim=embedding_dim,
                hidden_dim=hidden_dim,
                k_xor=k_xor,
                epochs=epochs,
                batch_size=batch_size,
            )

        model = new_model(k_pretrain + 1)
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

        target = target_factory(calibrated_noisiness)
        max_train = max(split.train for split in target_splits.values())
        max_validation = max(split.validation for split in target_splits.values())
        x_train = random_crps(max_train, n_bits, seed=target_seed + 1)
        x_val = random_crps(max_validation, n_bits, seed=target_seed + 2)
        x_test = random_crps(test_size, n_bits, seed=target_seed + 3)
        phi_train = suffix_parity_transform(x_train)
        phi_val = suffix_parity_transform(x_val)
        phi_test = suffix_parity_transform(x_test)
        y_train = _binary(target.eval(x_train))
        y_val = _binary(target.eval(x_val))
        y_test_observed = _binary(target.eval(x_test))
        y_test_latent = _binary(latent_response(target, x_test))
        test_devices = np.full(test_size, target_id, dtype=int)
        zero_prob = model.predict(phi_test, test_devices)
        test_ber = float(np.mean(y_test_observed != y_test_latent))

        for m, target_split in target_splits.items():
            train_count = target_split.train
            validation_count = target_split.validation
            tx, ty = phi_train[:train_count], y_train[:train_count]
            vx, vy = phi_val[:validation_count], y_val[:validation_count]

            model.model.load_state_dict(copy.deepcopy(base_state))
            start = perf_counter()
            model.fit_transfer(
                tx,
                target_id,
                ty,
                vx,
                np.full(validation_count, target_id, dtype=int),
                vy,
                patience=patience,
            )
            adaptation_seconds = perf_counter() - start
            transfer_prob = model.predict(phi_test, test_devices)

            torch.manual_seed(trial_seed + m)
            baseline = new_model(1)
            start = perf_counter()
            baseline.fit(
                tx,
                np.zeros(train_count, dtype=int),
                ty,
                vx,
                np.zeros(validation_count, dtype=int),
                vy,
                patience=patience,
            )
            baseline_seconds = perf_counter() - start
            baseline_prob = baseline.predict(
                phi_test, np.zeros(test_size, dtype=int)
            )
            majority = int(ty.mean() >= 0.5)

            transfer_observed = _accuracy(y_test_observed, transfer_prob)
            baseline_observed = _accuracy(y_test_observed, baseline_prob)
            rows.append(
                {
                    "selection": selection,
                    "alpha": alpha,
                    "beta": beta,
                    "k_xor": k_xor,
                    "k_pretrain": k_pretrain,
                    "n_base_budget": n_base_budget,
                    "source_train_crps_per_device": source_split.train,
                    "source_validation_crps_per_device": source_split.validation,
                    "source_total_labeled_crps": k_pretrain * n_base_budget,
                    "m_target_budget": m,
                    "target_train_crps": train_count,
                    "target_validation_crps": validation_count,
                    "target_total_labeled_crps": m,
                    "test_crps": test_size,
                    "trial": trial,
                    "seed": trial_seed,
                    "n_bits": n_bits,
                    "noise_mode": "target_ber" if target_ber is not None else "raw_noisiness",
                    "target_ber": target_ber if target_ber is not None else np.nan,
                    "requested_noisiness": (
                        float(noisiness or 0.0) if target_ber is None else np.nan
                    ),
                    "noisiness": calibrated_noisiness,
                    "calibration_ber": calibration_ber,
                    "test_ber": test_ber,
                    "accuracy_ceiling": 1 - test_ber,
                    "classification_noise_penalty": classification_noise_penalty(
                        test_ber
                    ),
                    "transfer_acc": transfer_observed,
                    "transfer_latent_acc": _accuracy(y_test_latent, transfer_prob),
                    "baseline_acc": baseline_observed,
                    "baseline_latent_acc": _accuracy(y_test_latent, baseline_prob),
                    "acc_gain_delta": transfer_observed - baseline_observed,
                    "zero_shot_acc": _accuracy(y_test_observed, zero_prob),
                    "zero_shot_latent_acc": _accuracy(y_test_latent, zero_prob),
                    "majority_acc": float(np.mean(y_test_observed == majority)),
                    "test_uniformity": float(y_test_observed.mean()),
                    "pretrain_seconds": pretrain_seconds,
                    "adaptation_seconds": adaptation_seconds,
                    "baseline_seconds": baseline_seconds,
                    "hidden_dim": hidden_dim,
                    "embedding_dim": embedding_dim,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "patience": patience,
                    "target_initialization": "source_embedding_mean",
                }
            )
    return rows


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--grid-csv", default=None)
    parser.add_argument(
        "--grid-selections",
        default="baseline,stealth_low,stealth_mid,stealth_high,stress,health",
    )
    parser.add_argument("--alpha", default="0.0,0.75,0.9,0.95")
    parser.add_argument("--beta", default="0.0,0.5,0.8,0.95")
    parser.add_argument(
        "--calibrated-alpha",
        type=float,
        default=None,
        help=(
            "Run only an empirical-calibration sweep at this alpha instead of "
            "loading the study grid or constructing the Cartesian grid."
        ),
    )
    parser.add_argument(
        "--calibrated-beta",
        default="0.5,0.75,0.95",
        help="Comma-separated beta values used with --calibrated-alpha.",
    )
    parser.add_argument("--k-xor", default="1,2,3,4,5,6,7,8,9")
    parser.add_argument("--k-pretrain", default="1,5,10,20")
    parser.add_argument("--n-base-budget", default="1000")
    parser.add_argument("--m-target-budget", default="100,500,1000,5000,10000")
    parser.add_argument("--n-bits", type=int, default=64)
    noise = parser.add_mutually_exclusive_group()
    noise.add_argument(
        "--target-ber",
        default=None,
        help="Comma-separated output BERs; default: 0,0.03,0.05.",
    )
    noise.add_argument(
        "--noisiness",
        default=None,
        help="Raw PyPUF noisiness values for diagnostic use.",
    )
    parser.add_argument("--calibration-size", type=int, default=20_000)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--test-size", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--output", default="theoretical_transfer.csv")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Keep existing output rows and replace only matching jobs.",
    )
    return parser


def write_results(rows, output_path, *, append):
    new = pl.DataFrame(rows)
    if not append or not output_path.exists():
        return write_csv(rows, output_path)

    existing = pl.read_csv(output_path)
    combined = pl.concat([existing, new], how="diagonal_relaxed").with_columns(
        pl.col("target_ber").fill_nan(-1.0).alias("_target_ber_key"),
        pl.col("requested_noisiness")
        .fill_nan(-1.0)
        .alias("_requested_noisiness_key"),
    )
    identity = [
        "selection",
        "alpha",
        "beta",
        "k_xor",
        "k_pretrain",
        "n_base_budget",
        "m_target_budget",
        "trial",
        "noise_mode",
        "_target_ber_key",
        "_requested_noisiness_key",
    ]
    combined = (
        combined.unique(subset=identity, keep="last", maintain_order=True)
        .drop("_target_ber_key", "_requested_noisiness_key")
        .sort(
            [
                "selection",
                "k_xor",
                "alpha",
                "beta",
                "k_pretrain",
                "n_base_budget",
                "m_target_budget",
                "trial",
                "target_ber",
            ]
        )
    )
    combined.write_csv(output_path)
    return combined


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return
    paths = StudyPaths()
    paths.ensure()
    requested_k = set(parse_int_list(args.k_xor))
    if args.calibrated_alpha is not None:
        if args.grid_csv:
            parser.error("--calibrated-alpha and --grid-csv are mutually exclusive")
        points = [
            {
                "k_xor": k,
                "alpha": args.calibrated_alpha,
                "beta": 0.0 if k == 1 else beta,
                "selection": "empirical_calibrated",
            }
            for k, beta in itertools.product(
                sorted(requested_k), parse_float_list(args.calibrated_beta)
            )
        ]
        points = list(
            {
                (point["k_xor"], point["alpha"], point["beta"]): point
                for point in points
            }.values()
        )
    elif args.grid_csv:
        points = load_study_points(
            args.grid_csv,
            data_dir=paths.data_dir,
            selections=parse_str_list(args.grid_selections),
        )
        points = [point for point in points if point["k"] in requested_k]
        points = [
            {
                "k_xor": point.pop("k"),
                **point,
            }
            for point in points
        ]
    else:
        points = [
            {
                "k_xor": k,
                "alpha": alpha,
                "beta": 0.0 if k == 1 else beta,
                "selection": "cartesian",
            }
            for k, alpha, beta in itertools.product(
                sorted(requested_k),
                parse_float_list(args.alpha),
                parse_float_list(args.beta),
            )
        ]
        points = list({(p["k_xor"], p["alpha"], p["beta"]): p for p in points}.values())

    noise_points = (
        [
            {"target_ber": None, "noisiness": value}
            for value in parse_float_list(args.noisiness)
        ]
        if args.noisiness is not None
        else [
            {"target_ber": value, "noisiness": None}
            for value in parse_float_list(args.target_ber or "0,0.03,0.05")
        ]
    )
    jobs = [
        {
            **point,
            **noise_point,
            "k_pretrain": source_count,
            "n_base_budget": source_budget,
            "m_budgets": parse_int_list(args.m_target_budget),
            "n_bits": args.n_bits,
            "n_trials": args.n_trials,
            "seed": args.seed,
            "calibration_size": args.calibration_size,
            "validation_fraction": args.validation_fraction,
            "test_size": args.test_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "embedding_dim": args.embedding_dim,
            "patience": args.patience,
        }
        for point, source_count, source_budget, noise_point in itertools.product(
            points,
            parse_int_list(args.k_pretrain),
            parse_int_list(args.n_base_budget),
            noise_points,
        )
    ]
    logger.info("Theoretical transfer jobs: %d", len(jobs))
    results = run_grid(
        jobs,
        compute_theoretical_transfer,
        ParallelConfig(n_jobs=args.n_jobs, verbose=20),
    )
    output_path = paths.data_dir / args.output
    frame = write_results(flatten_rows(results), output_path, append=args.append)
    logger.info("Wrote %d rows to %s", frame.height, output_path)


if __name__ == "__main__":
    main()
