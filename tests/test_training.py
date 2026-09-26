from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from optiver.training import (  # noqa: E402
    LightGBMConfig,
    _sanitize_features,
    run_lightgbm_cv,
)


class FeatureSanitizationTests(unittest.TestCase):
    def test_integer_columns_convert_without_lossy_assignment_error(self) -> None:
        frame = pd.DataFrame(
            {
                "integer": pd.Series([1, 2], dtype="int32"),
                "floating": [1.0, np.inf],
                "category": pd.Categorical(["a", "b"]),
            }
        )
        result = _sanitize_features(frame)
        self.assertEqual(result["integer"].dtype, np.dtype("float32"))
        self.assertEqual(result["floating"].dtype, np.dtype("float32"))
        self.assertTrue(np.isnan(result.loc[1, "floating"]))
        self.assertIsInstance(result["category"].dtype, pd.CategoricalDtype)


class NestedFeatureProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "time_id": np.repeat([10, 20, 30, 40], 2),
                "stock_id": np.tile([0, 1], 4),
                "signal": np.arange(8, dtype=float) + 1.0,
                "target": np.linspace(0.1, 0.8, 8),
            }
        )
        self.folds = [
            (np.arange(4, 8), np.arange(0, 4)),
            (np.arange(0, 4), np.arange(4, 8)),
        ]

    @staticmethod
    def _inner_split(groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
        del seed
        first_group = groups[0]
        train = np.flatnonzero(groups == first_group)
        valid = np.flatnonzero(groups != first_group)
        return train, valid

    def test_provider_is_refit_at_inner_and_outer_scopes(self) -> None:
        calls: list[dict[str, object]] = []

        def provider(train, valid, fold, scope):
            calls.append(
                {
                    "fold": fold,
                    "scope": scope,
                    "train_groups": set(train["time_id"]),
                    "valid_groups": set(valid["time_id"]),
                    "valid_has_target": "target" in valid,
                }
            )
            return train[["signal"]], valid[["signal"]], []

        class FakeModel:
            @staticmethod
            def predict(features, num_iteration=None):
                del num_iteration
                return np.full(len(features), 0.25)

            @staticmethod
            def feature_importance(importance_type="gain"):
                del importance_type
                return np.array([1.0])

        with (
            patch("optiver.training._select_boost_rounds", return_value=(1, 0.2)),
            patch("optiver.training._dataset", return_value=object()),
            patch("optiver.training.lgb.train", return_value=FakeModel()),
        ):
            result = run_lightgbm_cv(
                self.frame,
                self.folds,
                feature_provider=provider,
                inner_split_factory=self._inner_split,
                config=LightGBMConfig(max_boost_rounds=2, early_stopping_rounds=1),
            )

        self.assertEqual(
            [call["scope"] for call in calls],
            ["inner_selection", "outer_refit"] * 2,
        )
        self.assertTrue(all(not call["valid_has_target"] for call in calls))
        self.assertEqual(calls[0]["train_groups"], {30})
        self.assertEqual(calls[0]["valid_groups"], {40})
        self.assertEqual(calls[1]["train_groups"], {30, 40})
        self.assertEqual(calls[1]["valid_groups"], {10, 20})
        self.assertTrue(np.isfinite(result.oof_predictions).all())

    def test_outer_group_overlap_fails_before_provider_runs(self) -> None:
        calls = 0

        def provider(train, valid, fold, scope):
            nonlocal calls
            calls += 1
            return train[["signal"]], valid[["signal"]], []

        leaking_folds = [
            (np.array([0, 2, 4, 6]), np.array([1, 3, 5, 7])),
            (np.array([1, 3, 5, 7]), np.array([0, 2, 4, 6])),
        ]
        with self.assertRaisesRegex(ValueError, "leaks time_id groups"):
            run_lightgbm_cv(
                self.frame,
                leaking_folds,
                feature_provider=provider,
                inner_split_factory=self._inner_split,
                config=LightGBMConfig(fixed_boost_rounds=1),
            )
        self.assertEqual(calls, 0)

    def test_outer_validation_requires_exactly_once_coverage(self) -> None:
        duplicated = [self.folds[0], self.folds[0]]
        with self.assertRaisesRegex(ValueError, "exactly once"):
            run_lightgbm_cv(
                self.frame,
                duplicated,
                feature_provider=lambda train, valid, fold, scope: (
                    train[["signal"]],
                    valid[["signal"]],
                    [],
                ),
                inner_split_factory=self._inner_split,
                config=LightGBMConfig(fixed_boost_rounds=1),
            )

    def test_partial_calibration_fold_requires_explicit_opt_in(self) -> None:
        pair = [self.folds[0]]

        def provider(train, valid, fold, scope):
            del fold, scope
            return train[["signal"]], valid[["signal"]], []

        class FakeModel:
            @staticmethod
            def predict(features, num_iteration=None):
                del num_iteration
                return np.full(len(features), 0.25)

            @staticmethod
            def feature_importance(importance_type="gain"):
                del importance_type
                return np.array([1.0])

        config = LightGBMConfig(fixed_boost_rounds=1)
        with self.assertRaisesRegex(ValueError, "exactly once"):
            run_lightgbm_cv(
                self.frame,
                pair,
                feature_provider=provider,
                inner_split_factory=self._inner_split,
                config=config,
            )

        with (
            patch("optiver.training._dataset", return_value=object()),
            patch("optiver.training.lgb.train", return_value=FakeModel()),
        ):
            result = run_lightgbm_cv(
                self.frame,
                pair,
                feature_provider=provider,
                inner_split_factory=self._inner_split,
                config=config,
                allow_partial_oof=True,
            )
        self.assertTrue(result.summary["partial_run"])
        self.assertEqual(result.summary["n_oof_rows"], len(pair[0][1]))
        self.assertLess(result.summary["oof_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
