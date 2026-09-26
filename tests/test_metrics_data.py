from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from optiver.data import (  # noqa: E402
    observable_feature_columns,
    target_derived_columns,
    validate_feature_table,
)
from optiver.feature_sets import FEATURE_GROUPS, get_feature_set  # noqa: E402
from optiver.metrics import (  # noqa: E402
    fit_rmspe_scale,
    grouped_bootstrap_rmspe_difference,
    grouped_bootstrap_rmspe,
    optimal_rmspe_constant,
    rmspe,
    rmspe_weights,
)
from optiver.baselines import evaluate_baselines  # noqa: E402


class MetricTests(unittest.TestCase):
    def test_rmspe_known_value_and_prediction_clipping(self) -> None:
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([0.0, 1.0])
        self.assertAlmostEqual(rmspe(y_true, y_pred), np.sqrt(0.625))
        self.assertAlmostEqual(rmspe(y_true, [-1.0, 1.0]), np.sqrt(0.625))

    def test_rmspe_rejects_invalid_targets(self) -> None:
        with self.assertRaises(ValueError):
            rmspe([0.0, 1.0], [0.0, 1.0])

    def test_weights_are_mean_normalized(self) -> None:
        weights = rmspe_weights([1.0, 2.0, 4.0])
        self.assertAlmostEqual(float(weights.mean()), 1.0)
        self.assertGreater(float(weights.max()), float(weights.min()))

    def test_analytical_baselines(self) -> None:
        targets = np.array([1.0, 2.0, 4.0])
        constant = optimal_rmspe_constant(targets)
        candidates = np.linspace(constant - 0.1, constant + 0.1, 101)
        scores = [rmspe(targets, np.full(3, value)) for value in candidates]
        self.assertEqual(int(np.argmin(scores)), 50)

        signal = targets / 2.0
        self.assertAlmostEqual(fit_rmspe_scale(targets, signal), 2.0)

    def test_grouped_bootstrap_returns_ordered_interval(self) -> None:
        result = grouped_bootstrap_rmspe(
            [1.0, 1.0, 2.0, 2.0],
            [0.9, 1.1, 1.8, 2.2],
            [1, 1, 2, 2],
            n_resamples=100,
            seed=7,
        )
        self.assertLessEqual(result.lower, result.estimate)
        self.assertGreaterEqual(result.upper, result.estimate)

    def test_paired_bootstrap_detects_uniformly_better_candidate(self) -> None:
        result = grouped_bootstrap_rmspe_difference(
            [1.0, 1.0, 2.0, 2.0],
            [0.95, 1.05, 1.9, 2.1],
            [0.8, 1.2, 1.6, 2.4],
            [1, 1, 2, 2],
            n_resamples=100,
            seed=7,
        )
        self.assertLess(result.estimate, 0.0)
        self.assertLess(result.upper, 0.0)


class DataTests(unittest.TestCase):
    def test_legacy_cluster_columns_are_excluded(self) -> None:
        columns = [
            "stock_id",
            "time_id",
            "rv_pred",
            "stock_cluster",
            "rv_pred_0c1",
            "target",
        ]
        self.assertEqual(
            target_derived_columns(columns),
            ["rv_pred_0c1", "stock_cluster"],
        )
        self.assertEqual(observable_feature_columns(columns), ["rv_pred"])

    def test_duplicate_keys_fail_fast(self) -> None:
        frame = pd.DataFrame(
            {
                "stock_id": [0, 0],
                "time_id": [1, 1],
                "target": [0.1, 0.2],
            }
        )
        with self.assertRaises(ValueError):
            validate_feature_table(frame)

    def test_named_feature_set_fails_when_schema_is_incomplete(self) -> None:
        with self.assertRaises(ValueError):
            get_feature_set("multiscale", ["rv_pred"], strict=True)

    def test_clean_feature_registry_is_disjoint_and_has_92_columns(self) -> None:
        flattened = [column for group in FEATURE_GROUPS.values() for column in group]
        self.assertEqual(len(flattened), 92)
        self.assertEqual(len(set(flattened)), 92)
        self.assertEqual(
            len(get_feature_set("full_clean", flattened, strict=True)),
            92,
        )


class BaselineTests(unittest.TestCase):
    def test_stock_optimal_constant_is_fit_on_training_rows_only(self) -> None:
        frame = pd.DataFrame(
            {
                "stock_id": [0, 1, 0, 1],
                "time_id": [0, 0, 1, 1],
                "target": [1.0, 4.0, 100.0, 200.0],
                "rv_pred": [1.0, 1.0, 1.0, 1.0],
            }
        )
        folds = [
            (np.array([0, 1]), np.array([2, 3])),
            (np.array([2, 3]), np.array([0, 1])),
        ]
        result = evaluate_baselines(frame, folds, seed=7)
        predictions = result.predictions["stock_optimal_constant"].to_numpy()
        np.testing.assert_allclose(predictions[[2, 3]], [1.0, 4.0])
        np.testing.assert_allclose(predictions[[0, 1]], [100.0, 200.0])


if __name__ == "__main__":
    unittest.main()
