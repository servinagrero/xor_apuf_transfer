import csv
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import polars as pl
import torch

from puf_transfer import PUFModel, generate_weights_bias, suffix_parity_transform
from puf_transfer.empirical import (
    binary_min_entropy,
    binary_shannon_entropy,
    compute_population_metrics,
    EmpiricalDataset,
    estimate_population,
    prepare_empirical_dataset,
    summarize_metric_combination,
)
from puf_transfer.empirical_transfer import (
    fixed_cohort_trial,
    load_dataset,
    transfer_trial,
)
from puf_transfer.empirical_validation import (
    annotate_transfer_results,
    filter_source_pool,
    source_distance_statistics,
    uniqueness_geometry,
)


def fixture(root, duplicate=False):
    rng = np.random.default_rng(4)
    challenges = rng.integers(0, 2, size=(160, 8), dtype=np.uint8)
    np.save(root / "challenges.npy", challenges)
    phi = suffix_parity_transform(1 - 2 * challenges.astype(np.int8))
    rows = []
    for device in ("A", "B", "C"):
        for s in ("S0", "S1", "S2", "S3"):
            labels = (phi @ rng.normal(size=8) > 0).astype(int)
            for cid, y in enumerate(labels):
                if device == "A" and s == "S1" and cid == 3:
                    continue
                rows.append((cid, 0, device, s, int(y), "ignored"))
    rng.shuffle(rows)
    if duplicate:
        rows.append(rows[0])
    with (root / "crps.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["challenge_id", "sample_id", "device_id", "slice", "response", "metadata"]
        )
        writer.writerows(rows)
    prepare_empirical_dataset(
        root / "crps.csv",
        root / "challenges.npy",
        root / "cache",
        train_stop=120,
        validation_size=20,
        block_size=4096,
    )
    return EmpiricalDataset(root / "cache")


class EmpiricalTests(unittest.TestCase):
    def test_generated_bias_is_the_evaluated_final_coefficient(self):
        weights, bias = generate_weights_bias(8, k=3, beta=0.7, seed=9)
        self.assertEqual(weights.shape, (3, 9))
        self.assertEqual(bias.shape, (3, 1))
        np.testing.assert_allclose(weights[:, -1:], bias)

    def test_nn_restores_best_checkpoint_at_epoch_limit(self):
        torch.manual_seed(5)
        short = PUFModel(4, 1, 4, 8, epochs=1, batch_size=512)
        long = copy.deepcopy(short)
        long.epochs = 4
        X = np.zeros((8, 4), dtype=np.float32)
        D = np.zeros(8, dtype=int)
        # Every training update helps the positive class and harms validation.
        for model in (short, long):
            model.fit(X, D, np.ones(8), X, D, np.zeros(8), patience=10)
        for name, value in short.model.state_dict().items():
            torch.testing.assert_close(value, long.model.state_dict()[name])

    def test_alignment_xors_and_global_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = fixture(Path(tmp))
            self.assertEqual(len(data.combinations()), 15)
            for combo in data.combinations():
                ids = data.ids("A", combo, "train")
                expected = np.zeros(len(ids), dtype=np.uint8)
                for s in combo:
                    expected ^= data.chains["A", s][ids]
                np.testing.assert_array_equal(data.responses("A", combo, ids), expected)
                self.assertTrue(np.all(ids < 120))
            self.assertNotIn(3, data.ids("A", ("S0", "S1"), "train"))
            self.assertIn(3, data.ids("A", ("S0",), "train"))
            self.assertTrue(np.all(data.ids("B", ("S0",), "validation") >= 120))
            self.assertTrue(np.all(data.ids("B", ("S0",), "test") >= 140))
            with self.assertRaises(ValueError):
                data.responses("A", ("S1",), [3])

    def test_duplicate_identity_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                fixture(Path(tmp), duplicate=True)
            self.assertFalse((Path(tmp) / "cache/manifest.json").exists())

    def test_population_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = fixture(Path(tmp))
            frame = estimate_population(
                data, Path(tmp) / "estimates", budget=80, min_accuracy=0
            )
            self.assertEqual(frame.height, 15)
            self.assertTrue(
                np.isnan(frame.filter(frame["k_xor"] == 1)["beta_hat"].to_numpy()).all()
            )
            self.assertTrue((Path(tmp) / "estimates/chain_pairs.csv").exists())
            self.assertTrue(
                (Path(tmp) / "estimates/population_parameter_pairs.csv").exists()
            )
            self.assertTrue(
                (Path(tmp) / "estimates/population_model_fit.csv").exists()
            )
            response = pl.read_csv(Path(tmp) / "estimates/response_moments.csv")
            row = frame.filter((pl.col("k_xor") == 1) & (pl.col("slices") == "S0")).row(
                0, named=True
            )
            moment = response.filter(
                (pl.col("kind") == "inter_device_same_slice")
                & (pl.col("slice_1") == "S0")
            )["polar_product_mean"].mean()
            self.assertAlmostEqual(row["inter_response_moment"], moment)
            self.assertAlmostEqual(row["alpha_squared_raw"], np.sin(np.pi * moment / 2))
            pair = frame.filter(
                (pl.col("k_xor") == 2) & (pl.col("slices") == "S0+S1")
            ).row(0, named=True)
            intra = response.filter(
                (pl.col("kind") == "intra_device_cross_slice")
                & (pl.col("slice_1") == "S0")
                & (pl.col("slice_2") == "S1")
            )["polar_product_mean"].mean()
            self.assertAlmostEqual(
                pair["alpha_squared_beta_squared_raw"], np.sin(np.pi * intra / 2)
            )

    def test_population_metric_distributions_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = fixture(root)
            result, summary = compute_population_metrics(
                data, root / "metrics", budget=120, seed=7
            )
            self.assertEqual(len(result["combinations"]), 15)
            self.assertEqual(summary.height, 90)
            self.assertTrue((root / "metrics/population_metric_distributions.npy").exists())
            self.assertTrue((root / "metrics/population_metrics_summary.csv").exists())
            self.assertTrue(
                (root / "metrics/population_metrics_null_summary.csv").exists()
            )

            single = result["combinations"]["S0"]
            responses = np.stack(
                [
                    data.responses(device, ("S0",), single["challenge_ids"])
                    for device in single["device_ids"]
                ]
            )
            np.testing.assert_allclose(single["uniformity"], responses.mean(axis=1))
            np.testing.assert_allclose(single["bitaliasing"], responses.mean(axis=0))
            np.testing.assert_allclose(
                single["UPD"], binary_shannon_entropy(responses.mean(axis=0))
            )
            np.testing.assert_allclose(
                single["UPC"], binary_shannon_entropy(responses.mean(axis=1))
            )
            expected_uniqueness = [
                np.mean(responses[i] != responses[j])
                for i, j in ((0, 1), (0, 2), (1, 2))
            ]
            np.testing.assert_allclose(single["uniqueness"], expected_uniqueness)
            np.testing.assert_allclose(
                single["H_uniqueness"],
                binary_shannon_entropy(expected_uniqueness),
            )
            np.testing.assert_allclose(
                single["Min_H_UPD"],
                binary_min_entropy(responses.mean(axis=0)),
            )
            np.testing.assert_allclose(
                single["Min_H_UPC"],
                binary_min_entropy(responses.mean(axis=1)),
            )
            self.assertEqual(len(single["uniqueness_device_pairs"]), 3)

            pair = result["combinations"]["S0+S1"]
            self.assertNotIn("A", pair["device_ids"])
            pair_summary = summarize_metric_combination(pair)
            self.assertEqual(pair_summary.height, 6)
            self.assertEqual(
                set(pair_summary["metric"]),
                {
                    "UPD",
                    "UPC",
                    "Uniqueness",
                    "H(Uniqueness)",
                    "Min_H_UPD",
                    "Min_H_UPC",
                },
            )

            loaded = np.load(
                root / "metrics/population_metric_distributions.npy",
                allow_pickle=True,
            ).item()
            np.testing.assert_array_equal(
                loaded["combinations"]["S0"]["challenge_ids"],
                single["challenge_ids"],
            )

            geometry = uniqueness_geometry(result, "S0", threshold=1.0)
            np.testing.assert_allclose(geometry["distances"], geometry["distances"].T)
            np.testing.assert_array_equal(np.diag(geometry["distances"]), 0)
            self.assertEqual(len(set(geometry["clusters"].values())), 1)
            remaining, cluster, excluded = filter_source_pool(
                geometry, "A", ["B", "C"], "leave-cluster-out"
            )
            self.assertEqual(remaining, [])
            self.assertEqual(set(excluded), {"B", "C"})
            self.assertEqual(cluster, geometry["clusters"]["A"])
            remaining, cluster, excluded = filter_source_pool(
                geometry,
                "A",
                ["B", "C"],
                "exclude-nearest",
                exclude_nearest_count=1,
            )
            self.assertEqual(len(remaining), 1)
            self.assertEqual(len(excluded), 1)
            self.assertNotIn(excluded[0], remaining)
            self.assertEqual(cluster, geometry["clusters"]["A"])
            stats = source_distance_statistics(geometry, "A", ["B", "C"])
            self.assertAlmostEqual(
                stats["target_source_min_uniqueness"],
                min(geometry["distances"][0, 1:]),
            )

            frame = pl.DataFrame(
                {
                    "k_xor": [1],
                    "slices": ["S0"],
                    "target_device": ["A"],
                    "source_devices": ['["B", "C"]'],
                    "k_pretrain": [2],
                    "m_target_budget": [8],
                    "zero_shot_acc": [0.5],
                    "transfer_acc": [0.6],
                    "acc_gain_delta": [0.1],
                }
            )
            annotated = annotate_transfer_results(frame, result, threshold=1.0)
            self.assertEqual(annotated.height, 1)
            self.assertEqual(annotated["selected_same_cluster_sources"][0], 2)

    def test_transfer_test_labels_never_select_polarity(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = fixture(root)
            kwargs = dict(
                cache_dir=str(root / "cache"),
                challenges_path=str(root / "challenges.npy"),
                slices=("S0", "S1"),
                target="A",
                source_devices=["B", "C"],
                n_base_budget=24,
                m_budgets=[8, 16],
                validation_fraction=0.25,
                epochs=2,
                batch_size=512,
                hidden_dim=8,
                embedding_dim=4,
            )
            rows = transfer_trial(**kwargs)
            diagnostic = transfer_trial(
                **kwargs,
                source_uniqueness_values=[0.2, 0.3],
                diagnostic_baselines=True,
            )
            for row, other in zip(rows, diagnostic):
                self.assertAlmostEqual(row["transfer_acc"], other["transfer_acc"])
                self.assertAlmostEqual(row["baseline_acc"], other["baseline_acc"])
                self.assertTrue(np.isfinite(other["pooled_source_acc"]))
                self.assertEqual(other["source_consensus_extra_crps"], 2 * row["test_crps"])
            entry = next(
                r
                for r in data.manifest["chains"]
                if r["device_id"] == "A" and r["slice"] == "S0"
            )
            labels = np.load(root / "cache" / entry["file"], mmap_mode="r+")
            labels[140:] = 1 - labels[140:]
            labels.flush()
            inverted = transfer_trial(**kwargs)
            for row, other in zip(rows, inverted):
                self.assertEqual(
                    row["target_total_labeled_crps"], row["m_target_budget"]
                )
                self.assertEqual(
                    row["target_training_crps"] + row["target_validation_crps"],
                    row["m_target_budget"],
                )
                self.assertAlmostEqual(row["transfer_acc"] + other["transfer_acc"], 1)
                self.assertAlmostEqual(row["baseline_acc"] + other["baseline_acc"], 1)
            load_dataset.cache_clear()

    def test_nn_updates_with_small_budget_and_transfer_freezes_backbone(self):
        torch.set_num_threads(1)
        torch.manual_seed(1)
        rng = np.random.default_rng(2)
        X = rng.normal(size=(8, 4)).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.uint8)
        model = PUFModel(4, 2, 4, 8, epochs=2, batch_size=512)
        old = model.model.fc1.weight.detach().clone()
        model.fit(X, np.zeros(8, dtype=int), y, X, np.zeros(8, dtype=int), y)
        self.assertFalse(torch.equal(old, model.model.fc1.weight))
        state = {k: v.clone() for k, v in model.model.state_dict().items()}
        model.fit_transfer(X, 1, y, X, np.ones(8, dtype=int), y)
        for name, value in model.model.state_dict().items():
            if name != "embedding.weight":
                torch.testing.assert_close(value, state[name], rtol=0, atol=0)

    def test_fixed_cohort_reuses_sources_for_all_held_out_targets(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            rows = fixed_cohort_trial(
                cache_dir=str(root / "cache"),
                challenges_path=str(root / "challenges.npy"),
                slices=("S0",),
                source_devices=["A"],
                target_devices=["B", "C"],
                n_base_budget=24,
                m_target_budget=8,
                validation_fraction=0.25,
                epochs=2,
                batch_size=512,
                hidden_dim=8,
                embedding_dim=4,
            )
            self.assertEqual({row["target_device"] for row in rows}, {"B", "C"})
            self.assertEqual({row["source_devices"] for row in rows}, {'["A"]'})
            self.assertEqual({row["held_out_target_count"] for row in rows}, {2})
            for row in rows:
                self.assertEqual(
                    row["target_training_crps"] + row["target_validation_crps"],
                    row["target_total_labeled_crps"],
                )
                self.assertEqual(row["source_total_labeled_crps"], 24)
                self.assertTrue(np.isfinite(row["transfer_acc"]))
                self.assertTrue(np.isfinite(row["zero_shot_acc"]))
            load_dataset.cache_clear()


if __name__ == "__main__":
    unittest.main()
