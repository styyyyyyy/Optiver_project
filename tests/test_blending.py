from __future__ import annotations

import sys
import unittest
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from optiver.blending import (  # noqa: E402
    align_oof_frames,
    cross_fitted_blend,
    fit_simplex_rmspe_weights,
)


def _component_frames() -> dict[str, pd.DataFrame]:
    time_id = np.repeat(np.arange(6), 2)
    fold_by_time = np.array([0, 0, 1, 1, 2, 2])
    fold = np.repeat(fold_by_time, 2)
    target = np.array(
        [1.0, 1.1, 1.2, 1.3, 0.9, 1.0, 1.4, 1.5, 1.1, 1.2, 1.3, 1.4]
    )
    keys = pd.DataFrame(
        {
            "stock_id": np.tile([10, 20], 6),
            "time_id": time_id,
            "target": target,
            "fold": fold,
        }
    )
    lightgbm = keys.assign(
        prediction=target
        * np.array([1.02, 0.98, 1.05, 0.97, 1.08, 0.96] * 2)
    )
    mlp = keys.assign(
        prediction=target
        * np.array([0.94, 1.07, 0.99, 1.03, 0.95, 1.06] * 2)
    )
    return {"lightgbm": lightgbm, "mlp": mlp}


class OOFAlignmentTests(unittest.TestCase):
    def test_alignment_is_by_key_and_has_complete_coverage(self) -> None:
        frames = _component_frames()
        frames["mlp"] = frames["mlp"].sample(frac=1.0, random_state=9)
        aligned = align_oof_frames(frames)
        result = cross_fitted_blend(
            aligned, bootstrap_resamples=20, seed=7
        )

        self.assertEqual(len(result.oof_predictions), len(frames["lightgbm"]))
        self.assertTrue(np.isfinite(result.oof_predictions).all())
        self.assertEqual(
            result.fold_weights.groupby("fold")["weight"].sum().round(12).tolist(),
            [1.0, 1.0, 1.0],
        )
        self.assertTrue((result.fold_weights["weight"] >= 0).all())
        self.assertAlmostEqual(
            float(result.deployment_weights["weight"].sum()), 1.0, places=12
        )
        self.assertEqual(
            result.summary["deployment_weight_fit"],
            "all OOF rows; for retrained base models on future unseen data only",
        )
        self.assertEqual(
            set(result.fold_metrics["model"]), {"blend", "lightgbm", "mlp"}
        )

        for fold_id in np.unique(aligned.fold):
            mask = aligned.fold == fold_id
            weights = (
                result.fold_weights.loc[
                    result.fold_weights["fold"] == fold_id
                ]
                .set_index("model")
                .loc[list(aligned.model_names), "weight"]
                .to_numpy()
            )
            expected = np.clip(aligned.predictions[mask], 0.0, None) @ weights
            np.testing.assert_allclose(result.oof_predictions[mask], expected)

    def test_key_target_and_fold_mismatches_are_rejected(self) -> None:
        mutations: list[
            tuple[str, str, Callable[[pd.DataFrame], pd.DataFrame]]
        ] = [
            (
                "key",
                "Key coverage differs",
                lambda x: x.assign(
                    time_id=x["time_id"].mask(x.index == 0, 999)
                ),
            ),
            (
                "target",
                "Targets differ",
                lambda x: x.assign(
                    target=x["target"].mask(x.index == 0, 9.0)
                ),
            ),
            (
                "fold",
                "Fold assignments differ",
                lambda x: x.assign(fold=x["fold"].mask(x.index == 0, 2)),
            ),
        ]
        for label, message, mutate in mutations:
            with self.subTest(label=label):
                frames = _component_frames()
                frames["mlp"] = mutate(frames["mlp"].copy())
                with self.assertRaisesRegex(ValueError, message):
                    align_oof_frames(frames)

    def test_duplicate_keys_and_partial_oof_are_rejected(self) -> None:
        frames = _component_frames()
        frames["mlp"].loc[1, ["stock_id", "time_id"]] = frames["mlp"].loc[
            0, ["stock_id", "time_id"]
        ]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            align_oof_frames(frames)

        frames = _component_frames()
        frames["mlp"].loc[0, "fold"] = -1
        with self.assertRaisesRegex(ValueError, "partial OOF"):
            align_oof_frames(frames)


class PosthocBlendTests(unittest.TestCase):
    def test_direct_heldout_target_edit_does_not_change_fold_weights(self) -> None:
        original_frames = _component_frames()
        original_aligned = align_oof_frames(original_frames)
        original = cross_fitted_blend(
            original_aligned, bootstrap_resamples=20, seed=3
        )

        changed_frames = _component_frames()
        heldout = changed_frames["lightgbm"]["fold"] == 0
        for frame in changed_frames.values():
            frame.loc[heldout, "target"] *= 50.0
        changed_aligned = align_oof_frames(changed_frames)
        changed = cross_fitted_blend(
            changed_aligned, bootstrap_resamples=20, seed=3
        )

        self.assertEqual(original.summary["method"], "posthoc_oof_blend")
        self.assertEqual(
            original.summary["validation_claim"],
            "descriptive_posthoc_not_strict_nested",
        )

        first = (
            original.fold_weights.loc[original.fold_weights["fold"] == 0]
            .sort_values("model")["weight"]
            .to_numpy()
        )
        second = (
            changed.fold_weights.loc[changed.fold_weights["fold"] == 0]
            .sort_values("model")["weight"]
            .to_numpy()
        )
        np.testing.assert_array_equal(first, second)

    def test_two_model_closed_form_finds_known_optimum(self) -> None:
        target = np.ones(4)
        first = np.array([0.8, 1.2, 0.8, 1.2])
        second = np.array([1.2, 0.8, 1.2, 0.8])
        weights = fit_simplex_rmspe_weights(
            target, np.column_stack([first, second])
        )
        np.testing.assert_allclose(weights, [0.5, 0.5], atol=1e-12)

    def test_general_solver_respects_simplex_and_improves_equal_weights(self) -> None:
        target = np.ones(6)
        predictions = np.column_stack(
            [
                np.ones(6),
                np.full(6, 1.4),
                np.full(6, 0.6),
            ]
        )
        weights = fit_simplex_rmspe_weights(target, predictions)
        self.assertTrue((weights >= 0).all())
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=12)
        fitted_error = np.mean(np.square(1.0 - predictions @ weights))
        equal_error = np.mean(np.square(1.0 - predictions.mean(axis=1)))
        self.assertLessEqual(fitted_error, equal_error + 1e-12)


if __name__ == "__main__":
    unittest.main()
