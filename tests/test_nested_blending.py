from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from optiver.blending import (  # noqa: E402
    AlignedOOF,
    align_oof_frames,
    posthoc_oof_blend,
)
from optiver.nested_blending import (  # noqa: E402
    SCHEMA_VERSION,
    hash_group_ids,
    load_nested_calibration_bundle,
    nested_calibrated_blend,
    sha256_file,
    validate_nested_calibration_frame,
)


def _outer_component_frames() -> dict[str, pd.DataFrame]:
    time_id = np.repeat(np.arange(6), 2)
    fold = np.repeat(np.array([0, 0, 1, 1, 2, 2]), 2)
    target = np.array(
        [1.0, 1.1, 1.2, 1.3, 0.9, 1.0, 1.4, 1.5, 1.1, 1.2, 1.3, 1.4]
    )
    common = pd.DataFrame(
        {
            "stock_id": np.tile([10, 20], 6),
            "time_id": time_id,
            "target": target,
            "fold": fold,
        }
    )
    lightgbm = common.assign(
        prediction=target
        * np.array([1.02, 0.98, 1.05, 0.97, 1.08, 0.96] * 2)
    )
    mlp = common.assign(
        prediction=target
        * np.array([0.94, 1.07, 0.99, 1.03, 0.95, 1.06] * 2)
    )
    return {"lightgbm": lightgbm, "mlp": mlp}


def _partition(
    aligned: AlignedOOF, outer_fold: int
) -> tuple[list[int], list[int], list[int], list[int]]:
    time_id = aligned.keys["time_id"].to_numpy()
    outer_validation = sorted(
        int(value) for value in np.unique(time_id[aligned.fold == outer_fold])
    )
    outer_train = sorted(
        int(value) for value in np.unique(time_id[aligned.fold != outer_fold])
    )
    calibration = [outer_train[0]]
    fit = [value for value in outer_train if value not in calibration]
    return outer_train, outer_validation, calibration, fit


def _calibration_frame(
    aligned: AlignedOOF, outer_fold: int
) -> pd.DataFrame:
    _, _, calibration, _ = _partition(aligned, outer_fold)
    reference = pd.DataFrame(
        {
            "stock_id": aligned.keys["stock_id"],
            "time_id": aligned.keys["time_id"],
            "target": aligned.target,
        }
    )
    reference = reference.loc[reference["time_id"].isin(calibration)].copy()
    rows = []
    for model_name in aligned.model_names:
        component = reference.copy()
        component["outer_fold"] = outer_fold
        component["model"] = model_name
        if model_name == "lightgbm":
            multiplier = np.where(component["stock_id"] == 10, 0.92, 1.04)
        else:
            multiplier = np.where(component["stock_id"] == 10, 1.10, 0.98)
        component["prediction"] = component["target"] * multiplier
        rows.append(component)
    return pd.concat(rows, ignore_index=True)


def _manifest(
    aligned: AlignedOOF,
    outer_fold: int,
    *,
    prediction_hash: str = "0" * 64,
) -> dict[str, object]:
    outer_train, outer_validation, calibration, fit = _partition(
        aligned, outer_fold
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "outer_fold": outer_fold,
        "group_column": "time_id",
        "outer_train_groups": outer_train,
        "outer_validation_groups": outer_validation,
        "calibration_groups": calibration,
        "models": {
            model_name: {
                "fit_groups": fit,
                "outer_prediction_fit_groups": outer_train,
                "training_scope": "fit_groups_only",
            }
            for model_name in aligned.model_names
        },
        "group_hashes": {
            "outer_train_groups_sha256": hash_group_ids(outer_train),
            "outer_validation_groups_sha256": hash_group_ids(outer_validation),
            "calibration_groups_sha256": hash_group_ids(calibration),
            "fit_groups_sha256_by_model": {
                model_name: hash_group_ids(fit)
                for model_name in aligned.model_names
            },
            "outer_prediction_fit_groups_sha256_by_model": {
                model_name: hash_group_ids(outer_train)
                for model_name in aligned.model_names
            },
        },
        "predictions": {
            "path": "calibration_predictions.parquet",
            "sha256": prediction_hash,
        },
    }


def _validated_calibrations(
    aligned: AlignedOOF,
) -> dict[int, object]:
    result = {}
    for outer_fold in np.unique(aligned.fold):
        fold = int(outer_fold)
        result[fold] = validate_nested_calibration_frame(
            _calibration_frame(aligned, fold),
            _manifest(aligned, fold),
            outer_oof=aligned,
        )
    return result


class StrictNestedBlendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.aligned = align_oof_frames(_outer_component_frames())

    def test_nested_blend_covers_every_row_with_simplex_weights(self) -> None:
        result = nested_calibrated_blend(
            self.aligned,
            _validated_calibrations(self.aligned),
            bootstrap_resamples=20,
            seed=9,
        )
        self.assertTrue(np.isfinite(result.oof_predictions).all())
        self.assertEqual(len(result.oof_predictions), len(self.aligned.target))
        sums = result.fold_weights.groupby("fold")["weight"].sum()
        np.testing.assert_allclose(sums.to_numpy(), np.ones(3), atol=1e-12)
        self.assertTrue((result.fold_weights["weight"] >= 0).all())
        self.assertEqual(
            result.summary["method"], "strict_nested_calibration_blend"
        )
        self.assertIn("conditional", result.summary["uncertainty_caveat"].lower())

    def test_nested_weights_are_isolated_from_indirect_outer_oof_changes(self) -> None:
        original_nested = nested_calibrated_blend(
            self.aligned,
            _validated_calibrations(self.aligned),
            bootstrap_resamples=20,
            seed=4,
        )
        original_posthoc = posthoc_oof_blend(
            self.aligned, bootstrap_resamples=20, seed=4
        )

        # Simulate the indirect pathway: changing fold-0 targets changes base
        # models trained with fold 0, hence predictions on the other OOF folds.
        leaked_frames = _outer_component_frames()
        heldout = leaked_frames["lightgbm"]["fold"] == 0
        other_folds = ~heldout
        for frame in leaked_frames.values():
            frame.loc[heldout, "target"] *= 10.0
        leaked_frames["lightgbm"].loc[other_folds, "prediction"] = (
            leaked_frames["lightgbm"].loc[other_folds, "target"]
        )
        leaked_frames["mlp"].loc[other_folds, "prediction"] = (
            leaked_frames["mlp"].loc[other_folds, "target"] * 1.8
        )
        changed_aligned = align_oof_frames(leaked_frames)
        changed_posthoc = posthoc_oof_blend(
            changed_aligned, bootstrap_resamples=20, seed=4
        )
        changed_nested = nested_calibrated_blend(
            changed_aligned,
            _validated_calibrations(changed_aligned),
            bootstrap_resamples=20,
            seed=4,
        )

        original_posthoc_fold0 = original_posthoc.fold_weights.query("fold == 0")
        changed_posthoc_fold0 = changed_posthoc.fold_weights.query("fold == 0")
        self.assertFalse(
            np.allclose(
                original_posthoc_fold0["weight"],
                changed_posthoc_fold0["weight"],
            )
        )
        original_nested_fold0 = original_nested.fold_weights.query("fold == 0")
        changed_nested_fold0 = changed_nested.fold_weights.query("fold == 0")
        np.testing.assert_array_equal(
            original_nested_fold0["weight"].to_numpy(),
            changed_nested_fold0["weight"].to_numpy(),
        )

    def test_posthoc_method_explicitly_disclaims_nested_validity(self) -> None:
        result = posthoc_oof_blend(
            self.aligned, bootstrap_resamples=20, seed=2
        )
        self.assertEqual(result.summary["method"], "posthoc_oof_blend")
        self.assertEqual(
            result.summary["validation_claim"],
            "descriptive_posthoc_not_strict_nested",
        )
        self.assertIn("indirect", result.summary["indirect_dependency_warning"])

    def test_missing_outer_fold_calibration_is_rejected(self) -> None:
        calibrations = _validated_calibrations(self.aligned)
        calibrations.pop(2)
        with self.assertRaisesRegex(ValueError, "every outer fold"):
            nested_calibrated_blend(
                self.aligned,
                calibrations,
                bootstrap_resamples=20,
            )


class CalibrationProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.aligned = align_oof_frames(_outer_component_frames())

    def test_fit_calibration_overlap_is_rejected(self) -> None:
        manifest = _manifest(self.aligned, 0)
        calibration_group = manifest["calibration_groups"][0]  # type: ignore[index]
        manifest["models"]["lightgbm"]["fit_groups"].append(  # type: ignore[index,union-attr]
            calibration_group
        )
        manifest["group_hashes"]["fit_groups_sha256_by_model"][  # type: ignore[index]
            "lightgbm"
        ] = hash_group_ids(
            manifest["models"]["lightgbm"]["fit_groups"]  # type: ignore[index,arg-type]
        )
        with self.assertRaisesRegex(ValueError, "overlap calibration"):
            validate_nested_calibration_frame(
                _calibration_frame(self.aligned, 0),
                manifest,
                outer_oof=self.aligned,
            )

    def test_group_hash_tampering_is_rejected(self) -> None:
        manifest = _manifest(self.aligned, 0)
        manifest["group_hashes"]["calibration_groups_sha256"] = "bad"  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_nested_calibration_frame(
                _calibration_frame(self.aligned, 0),
                manifest,
                outer_oof=self.aligned,
            )

    def test_outer_partition_must_match_oof_fold_map(self) -> None:
        manifest = _manifest(self.aligned, 0)
        manifest["outer_validation_groups"] = [2, 3]
        manifest["group_hashes"]["outer_validation_groups_sha256"] = (  # type: ignore[index]
            hash_group_ids([2, 3])
        )
        with self.assertRaisesRegex(ValueError, "do not match OOF fold"):
            validate_nested_calibration_frame(
                _calibration_frame(self.aligned, 0),
                manifest,
                outer_oof=self.aligned,
            )

    def test_calibration_target_mismatch_is_rejected(self) -> None:
        frame = _calibration_frame(self.aligned, 0)
        frame.loc[0, "target"] *= 2.0
        with self.assertRaisesRegex(ValueError, "targets differ"):
            validate_nested_calibration_frame(
                frame,
                _manifest(self.aligned, 0),
                outer_oof=self.aligned,
            )

    def test_bundle_loader_binds_prediction_file_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            prediction_path = bundle / "calibration_predictions.parquet"
            _calibration_frame(self.aligned, 0).to_parquet(
                prediction_path, index=False
            )
            manifest = _manifest(
                self.aligned,
                0,
                prediction_hash=sha256_file(prediction_path),
            )
            (bundle / "calibration_manifest.json").write_text(
                __import__("json").dumps(manifest), encoding="utf-8"
            )
            loaded = load_nested_calibration_bundle(
                bundle, outer_oof=self.aligned
            )
            self.assertEqual(loaded.outer_fold, 0)

            tampered = copy.deepcopy(manifest)
            tampered["predictions"]["sha256"] = "0" * 64  # type: ignore[index]
            (bundle / "calibration_manifest.json").write_text(
                __import__("json").dumps(tampered), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "file hash mismatch"):
                load_nested_calibration_bundle(bundle, outer_oof=self.aligned)


if __name__ == "__main__":
    unittest.main()
