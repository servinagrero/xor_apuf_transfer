"""Compute target-only modeling curves for correlated k-XOR APUFs."""

import argparse
import itertools

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score

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


def _binary(response):
    return np.asarray(response < 0, dtype=np.uint8)


def compute_baseline(
    k: int,
    budget: int,
    beta: float,
    *,
    alpha: float = 0.0,
    n_bits: int = 64,
    n_trials: int = 5,
    seed: int = 42,
    noisiness: float | None = None,
    target_ber: float | None = 0.0,
    calibration_size: int = 20_000,
    test_size: int = 5000,
    validation_fraction: float = 0.2,
    models: tuple[str, ...] = ("logreg", "target_nn"),
    epochs: int = 100,
    batch_size: int = 512,
    patience: int = 15,
):
    rows = []
    split = split_labeled_budget(
        budget, validation_fraction=validation_fraction, min_validation=20
    )
    for trial in range(n_trials):
        trial_seed = seed + trial * 100_000 + k * 1_000
        shared_weights, shared_bias = generate_weights_bias(
            n_bits, k=k, beta=beta, seed=trial_seed
        )
        puf_seed = trial_seed + 1

        def puf_factory(noise):
            return CorrelatedXORArbiterPUF(
                n_bits,
                k,
                shared_weights,
                shared_bias,
                alpha=alpha,
                seed=puf_seed,
                noisiness=noise,
            )

        if target_ber is not None:
            calibration_x = random_crps(
                calibration_size, n_bits, seed=trial_seed + 50_000
            )
            calibrated_noisiness, calibration_ber = calibrate_noisiness(
                puf_factory, calibration_x, target_ber
            )
        else:
            calibrated_noisiness = float(noisiness or 0.0)
            calibration_ber = np.nan
        puf = puf_factory(calibrated_noisiness)

        x_train = random_crps(split.train, n_bits, seed=trial_seed + 2)
        x_val = random_crps(split.validation, n_bits, seed=trial_seed + 3)
        x_test = random_crps(test_size, n_bits, seed=trial_seed + 4)
        y_train = _binary(puf.eval(x_train))
        y_train_latent = _binary(latent_response(puf, x_train))
        y_val = _binary(puf.eval(x_val))
        y_test_observed = _binary(puf.eval(x_test))
        y_test_latent = _binary(latent_response(puf, x_test))
        phi_train = suffix_parity_transform(x_train)
        phi_val = suffix_parity_transform(x_val)
        phi_test = suffix_parity_transform(x_test)

        majority = int(y_train.mean() >= 0.5)
        common = {
            "k": k,
            "alpha": alpha,
            "beta": beta,
            "noise_mode": "target_ber" if target_ber is not None else "raw_noisiness",
            "target_ber": target_ber if target_ber is not None else np.nan,
            "requested_noisiness": (
                float(noisiness or 0.0) if target_ber is None else np.nan
            ),
            "noisiness": calibrated_noisiness,
            "calibration_ber": calibration_ber,
            "budget": budget,
            "train_crps": split.train,
            "validation_crps": split.validation,
            "test_crps": test_size,
            "trial": trial,
            "seed": trial_seed,
            "majority_acc": float(np.mean(y_test_observed == majority)),
            "train_ber": float(np.mean(y_train != y_train_latent)),
            "test_ber": float(np.mean(y_test_observed != y_test_latent)),
            "accuracy_ceiling": 1 - float(np.mean(y_test_observed != y_test_latent)),
            "classification_noise_penalty": classification_noise_penalty(
                float(np.mean(y_test_observed != y_test_latent))
            ),
            "epochs": epochs,
            "batch_size": batch_size,
            "patience": patience,
        }

        for model_name in models:
            if model_name == "logreg" and k != 1:
                continue
            if len(np.unique(y_train)) < 2:
                prediction = np.full(test_size, majority, dtype=np.uint8)
            elif model_name == "logreg":
                model = LogisticRegression(max_iter=2000, C=1.0, random_state=trial_seed)
                model.fit(phi_train, y_train)
                prediction = model.predict(phi_test)
            elif model_name == "target_nn":
                model = PUFModel(
                    n_inputs=n_bits,
                    num_devices=1,
                    device_emb_dim=16,
                    hidden_dim=64,
                    k_xor=k,
                    epochs=epochs,
                    batch_size=batch_size,
                )
                model.fit(
                    phi_train,
                    np.zeros(split.train, dtype=int),
                    y_train,
                    phi_val,
                    np.zeros(split.validation, dtype=int),
                    y_val,
                    patience=patience,
                )
                prediction = (
                    model.predict(phi_test, np.zeros(test_size, dtype=int)).ravel()
                    >= 0.5
                )
            else:
                raise ValueError(f"Unknown baseline model: {model_name}")

            rows.append(
                {
                    **common,
                    "model": model_name,
                    "observed_acc": accuracy_score(y_test_observed, prediction),
                    "latent_acc": accuracy_score(y_test_latent, prediction),
                    "balanced_acc": balanced_accuracy_score(
                        y_test_observed, prediction
                    ),
                }
            )
    return rows


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--k-values", default="1,2,3,4,5,6,7,8,9")
    parser.add_argument(
        "--budgets", default="100,500,1000,5000,10000,20000"
    )
    parser.add_argument("--models", default="logreg,target_nn")
    parser.add_argument("--n-bits", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--beta", default="0.0")
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
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=6)
    parser.add_argument("--output", default="baseline_modeling.csv")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return
    models = tuple(parse_str_list(args.models))
    unknown = set(models) - {"logreg", "target_nn"}
    if unknown:
        parser.error(f"Unknown models: {sorted(unknown)}")

    paths = StudyPaths()
    paths.ensure()
    target_bers = parse_float_list(args.target_ber or "0,0.03,0.05")
    noise_points = (
        [{"target_ber": None, "noisiness": value} for value in parse_float_list(args.noisiness)]
        if args.noisiness is not None
        else [{"target_ber": value, "noisiness": None} for value in target_bers]
    )
    grid = [
        {"k": k, "budget": budget, "beta": beta, **noise_point}
        for k, budget, beta, noise_point in itertools.product(
            parse_int_list(args.k_values),
            parse_int_list(args.budgets),
            parse_float_list(args.beta),
            noise_points,
        )
    ]
    logger.info("Grid size: %d", len(grid))
    results = run_grid(
        (
            {
                **params,
                "alpha": args.alpha,
                "n_bits": args.n_bits,
                "n_trials": args.n_trials,
                "seed": args.seed,
                "test_size": args.test_size,
                "calibration_size": args.calibration_size,
                "validation_fraction": args.validation_fraction,
                "models": models,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "patience": args.patience,
            }
            for params in grid
        ),
        compute_baseline,
        ParallelConfig(n_jobs=args.n_jobs, verbose=20),
    )
    write_csv(flatten_rows(results), paths.data_dir / args.output)


if __name__ == "__main__":
    main()
