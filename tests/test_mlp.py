from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from optiver.mlp import (  # noqa: E402
    MLPConfig,
    NumericPreprocessor,
    StockVocabulary,
    _validate_config,
    _validate_outer_folds,
    prepare_fold_features,
    run_mlp_cv,
)
from optiver.validation import build_group_folds  # noqa: E402


class NumericPreprocessorTests(unittest.TestCase):
    def test_statistics_are_fitted_only_on_training_features(self) -> None:
        train = pd.DataFrame(
            {
                "x": [1.0, 2.0, np.nan, np.inf],
                "z": [10.0, 10.0, 10.0, 10.0],
                "target": [0.01, 0.02, 0.03, 0.04],
            }
        )
        changed_targets = train.copy()
        changed_targets["target"] = [900.0, 800.0, 700.0, 600.0]
        first = NumericPreprocessor.fit(train, ["x", "z"])
        second = NumericPreprocessor.fit(changed_targets, ["x", "z"])

        np.testing.assert_allclose(first.medians, second.medians)
        np.testing.assert_allclose(first.means, second.means)
        np.testing.assert_allclose(first.scales, second.scales)
        transformed = first.transform(train)
        self.assertEqual(transformed.dtype, np.float32)
        self.assertTrue(np.isfinite(transformed).all())
        self.assertEqual(first.scales[1], 1.0)

    def test_validation_values_do_not_change_fitted_statistics(self) -> None:
        train = pd.DataFrame({"stock_id": [0, 1], "x": [1.0, 3.0]})
        valid = pd.DataFrame({"stock_id": [0, 9], "x": [5.0, 1e9]})
        prepared = prepare_fold_features(train, valid, ["x"])
        self.assertEqual(prepared.preprocessor.means[0], 2.0)
        self.assertEqual(prepared.preprocessor.scales[0], 1.0)
        self.assertAlmostEqual(float(prepared.valid_numeric[0, 0]), 3.0)
        self.assertEqual(float(prepared.valid_numeric[1, 0]), 12.0)

    def test_preprocessor_serialization_contains_fitted_arrays(self) -> None:
        preprocessor = NumericPreprocessor.fit(
            pd.DataFrame({"x": [1.0, 2.0, 3.0]}), ["x"]
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "preprocessor.npz"
            preprocessor.save(path)
            saved = np.load(path)
            self.assertEqual(saved["feature_names"].tolist(), ["x"])
            np.testing.assert_allclose(saved["medians"], [2.0])


class StockVocabularyTests(unittest.TestCase):
    def test_unknown_stock_uses_reserved_zero_index(self) -> None:
        vocabulary = StockVocabulary.fit(pd.Series([10, 2, 10]))
        encoded = vocabulary.encode(pd.Series([2, 99, 10]))
        self.assertEqual(vocabulary.size_with_unknown, 3)
        self.assertEqual(encoded[1], 0)
        self.assertTrue(np.all(encoded[[0, 2]] > 0))


class MLPValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "stock_id": [0, 1] * 6,
                "time_id": np.repeat(np.arange(6), 2),
                "target": np.linspace(0.01, 0.02, 12),
                "rv_pred": np.linspace(0.011, 0.019, 12),
            }
        )

    def test_group_folds_are_accepted_with_complete_coverage(self) -> None:
        group_folds = build_group_folds(
            self.frame, n_splits=3, random_state=7
        )
        pairs = [(fold.train_idx, fold.valid_idx) for fold in group_folds]
        validated = _validate_outer_folds(
            pairs, self.frame["time_id"].to_numpy(), len(self.frame)
        )
        self.assertEqual(len(validated), 3)

    def test_partial_pilot_folds_require_explicit_opt_in(self) -> None:
        group_folds = build_group_folds(
            self.frame, n_splits=3, random_state=7
        )
        pair = [(group_folds[0].train_idx, group_folds[0].valid_idx)]
        with self.assertRaisesRegex(ValueError, "every row"):
            _validate_outer_folds(
                pair, self.frame["time_id"].to_numpy(), len(self.frame)
            )
        accepted = _validate_outer_folds(
            pair,
            self.frame["time_id"].to_numpy(),
            len(self.frame),
            require_complete=False,
        )
        self.assertEqual(len(accepted), 1)

    def test_group_leak_is_rejected(self) -> None:
        bad = [
            (np.arange(2, 12), np.array([0, 2])),
            (np.array([0, 2]), np.arange(1, 12)),
        ]
        with self.assertRaisesRegex(ValueError, "overlaps|complement|leaks"):
            _validate_outer_folds(
                bad, self.frame["time_id"].to_numpy(), len(self.frame)
            )

    def test_target_and_legacy_cluster_features_fail_before_torch_is_needed(self) -> None:
        folds = build_group_folds(self.frame, n_splits=3, random_state=7)
        pairs = [(fold.train_idx, fold.valid_idx) for fold in folds]
        with self.assertRaisesRegex(ValueError, "target-derived|identifiers"):
            run_mlp_cv(self.frame, pairs, ["target"])

        contaminated = self.frame.assign(rv_pred_0c1=1.0)
        with self.assertRaisesRegex(ValueError, "target-derived"):
            run_mlp_cv(contaminated, pairs, ["rv_pred_0c1"])

    def test_invalid_fixed_epoch_count_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "fixed_epochs"):
            _validate_config(MLPConfig(fixed_epochs=0))


if __name__ == "__main__":
    unittest.main()
