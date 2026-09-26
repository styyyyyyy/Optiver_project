"""Tests for grouped validation and leakage-safe target statistics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from optiver.validation import (  # noqa: E402
    GroupFold,
    assert_valid_group_folds,
    build_group_folds,
    cross_fit_stock_target_stats,
)


class GroupFoldTests(unittest.TestCase):
    def setUp(self) -> None:
        rows = []
        for time_id in [91, 3, 700, 22, 5, 410, 8, 64, 12, 999, 31]:
            for stock_id in [0, 1, 2]:
                rows.append(
                    {
                        "time_id": time_id,
                        "stock_id": stock_id,
                        "rv_pred": time_id / 1000.0 + stock_id / 10.0,
                        "target": time_id + stock_id,
                    }
                )
        self.data = pd.DataFrame(rows).sample(frac=1.0, random_state=17)
        # Non-unique labels make sure fold indices really are positional.
        self.data.index = np.arange(len(self.data)) % 7

    def test_group_kfold_is_reproducible_disjoint_and_complete(self) -> None:
        first = build_group_folds(self.data, n_splits=5, random_state=2021)
        second = build_group_folds(self.data, n_splits=5, random_state=2021)
        another_seed = build_group_folds(self.data, n_splits=5, random_state=7)

        assert_valid_group_folds(first, self.data)
        self.assertEqual(len(first), 5)
        for left, right in zip(first, second, strict=True):
            np.testing.assert_array_equal(left.train_idx, right.train_idx)
            np.testing.assert_array_equal(left.valid_idx, right.valid_idx)
            self.assertTrue(set(left.train_groups).isdisjoint(left.valid_groups))

        owners = {}
        for fold in first:
            for group in fold.valid_groups:
                self.assertNotIn(group, owners)
                owners[group] = fold.fold_id
        self.assertEqual(set(owners), set(self.data["time_id"]))
        self.assertNotEqual(
            [fold.valid_groups for fold in first],
            [fold.valid_groups for fold in another_seed],
        )

    def test_feature_balanced_protocol_is_seeded_and_keeps_remainder_groups(self) -> None:
        first = build_group_folds(
            self.data,
            n_splits=4,
            strategy="feature_balanced",
            feature_cols=["rv_pred"],
            entity_col="stock_id",
            random_state=11,
        )
        second = build_group_folds(
            self.data,
            n_splits=4,
            strategy="feature_balanced",
            feature_cols=["rv_pred"],
            entity_col="stock_id",
            random_state=11,
        )
        reordered = build_group_folds(
            self.data.sample(frac=1.0, random_state=99),
            n_splits=4,
            strategy="feature_balanced",
            feature_cols=["rv_pred"],
            entity_col="stock_id",
            random_state=11,
        )

        assert_valid_group_folds(first, self.data)
        self.assertEqual(
            [fold.valid_groups for fold in first],
            [fold.valid_groups for fold in second],
        )
        self.assertEqual(
            [set(fold.valid_groups) for fold in first],
            [set(fold.valid_groups) for fold in reordered],
        )
        sizes = [len(fold.valid_groups) for fold in first]
        self.assertEqual(sum(sizes), self.data["time_id"].nunique())
        self.assertLessEqual(max(sizes) - min(sizes), 1)

    def test_feature_balancing_rejects_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "Target column"):
            build_group_folds(
                self.data,
                n_splits=3,
                strategy="feature_balanced",
                feature_cols=["target"],
            )

    def test_seeded_group_kfold_has_an_older_sklearn_fallback(self) -> None:
        expected = build_group_folds(self.data, n_splits=5, random_state=2021)
        with patch(
            "optiver.validation.GroupKFold",
            side_effect=TypeError("shuffle is unavailable"),
        ):
            fallback = build_group_folds(
                self.data, n_splits=5, random_state=2021
            )
        self.assertEqual(
            [set(fold.valid_groups) for fold in fallback],
            [set(fold.valid_groups) for fold in expected],
        )

    def test_fold_assertion_catches_duplicate_validation_coverage(self) -> None:
        folds = build_group_folds(self.data, n_splits=3)
        source = folds[0]
        corrupted = list(folds)
        corrupted[1] = GroupFold(
            fold_id=1,
            train_idx=source.train_idx.copy(),
            valid_idx=source.valid_idx.copy(),
            train_groups=source.train_groups,
            valid_groups=source.valid_groups,
        )
        with self.assertRaises(AssertionError):
            assert_valid_group_folds(corrupted, self.data)


class CrossFittedTargetStatsTests(unittest.TestCase):
    def setUp(self) -> None:
        rows = []
        # Deliberately non-chronological and string-valued groups: time_id is
        # only an identity boundary, never something to sort or expand over.
        time_ids = ["zeta", "alpha", "mu", "beta", "omega", "gamma"]
        for group_number, time_id in enumerate(time_ids, start=1):
            rows.extend(
                [
                    {
                        "time_id": time_id,
                        "stock_id": "A",
                        "target": float(group_number),
                    },
                    {
                        "time_id": time_id,
                        "stock_id": "B",
                        "target": float(group_number * 10),
                    },
                ]
            )
        self.outer_train = pd.DataFrame(rows)
        self.outer_train.index = np.arange(len(rows)) * 10
        self.outer_valid = pd.DataFrame(
            {
                "time_id": ["held-out", "held-out", "second-held-out"],
                "stock_id": ["A", "B", "UNSEEN"],
                # These values must be ignored by the transformer.
                "target": [1e9, 2e9, 3e9],
            },
            index=[101, 55, 101],
        )

    def test_training_statistics_are_inner_oof_and_validation_uses_outer_train(self) -> None:
        encoded = cross_fit_stock_target_stats(
            self.outer_train,
            self.outer_valid,
            inner_splits=3,
            prefix="stock_rv",
        )

        self.assertEqual(
            encoded.feature_names,
            ("stock_rv_mean", "stock_rv_median", "stock_rv_std"),
        )
        self.assertEqual(encoded.train.index.tolist(), self.outer_train.index.tolist())
        self.assertEqual(encoded.valid.index.tolist(), self.outer_valid.index.tolist())

        for fold in encoded.inner_folds:
            fit_rows = self.outer_train.iloc[fold.train_idx]
            held_rows = self.outer_train.iloc[fold.valid_idx]
            expected_means = fit_rows.groupby("stock_id")["target"].mean()
            actual = encoded.train.iloc[fold.valid_idx]["stock_rv_mean"].to_numpy()
            expected = held_rows["stock_id"].map(expected_means).to_numpy()
            np.testing.assert_allclose(actual, expected)

        all_stock_means = self.outer_train.groupby("stock_id")["target"].mean()
        self.assertAlmostEqual(encoded.valid.iloc[0]["stock_rv_mean"], all_stock_means["A"])
        self.assertAlmostEqual(encoded.valid.iloc[1]["stock_rv_mean"], all_stock_means["B"])

        global_targets = self.outer_train["target"].to_numpy()
        self.assertAlmostEqual(
            encoded.valid.iloc[2]["stock_rv_mean"], float(global_targets.mean())
        )
        self.assertAlmostEqual(
            encoded.valid.iloc[2]["stock_rv_median"], float(np.median(global_targets))
        )
        self.assertAlmostEqual(
            encoded.valid.iloc[2]["stock_rv_std"], float(np.std(global_targets, ddof=0))
        )

    def test_validation_target_is_never_read(self) -> None:
        first = cross_fit_stock_target_stats(
            self.outer_train, self.outer_valid, inner_splits=3
        )
        changed = self.outer_valid.copy()
        changed["target"] = [-1.0, -2.0, -3.0]
        second = cross_fit_stock_target_stats(
            self.outer_train, changed, inner_splits=3
        )
        pd.testing.assert_frame_equal(first.validation, second.validation)

    def test_outer_groups_must_be_disjoint(self) -> None:
        invalid_valid = self.outer_valid.copy()
        invalid_valid.iloc[0, invalid_valid.columns.get_loc("time_id")] = "alpha"
        with self.assertRaisesRegex(ValueError, "disjoint"):
            cross_fit_stock_target_stats(
                self.outer_train, invalid_valid, inner_splits=3
            )


if __name__ == "__main__":
    unittest.main()
