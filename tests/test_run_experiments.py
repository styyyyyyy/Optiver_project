from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import run_experiments  # noqa: E402


class GroupwiseTargetShuffleTests(unittest.TestCase):
    def setUp(self) -> None:
        rows = []
        # The first three groups share one signature.  The final group has a
        # singleton signature and is therefore explicitly reported unchanged.
        for time_id in [10, 20, 30]:
            for stock_id in [0, 1]:
                rows.append(
                    {
                        "time_id": time_id,
                        "stock_id": stock_id,
                        "target": time_id + stock_id / 10,
                    }
                )
        rows.append({"time_id": 99, "stock_id": 7, "target": 999.0})
        self.frame = pd.DataFrame(rows)

    def test_whole_groups_move_and_targets_remain_stock_aligned(self) -> None:
        shuffled, mapping = run_experiments._groupwise_shuffle_targets(
            self.frame, seed=17
        )

        source_lookup = mapping.set_index("time_id")["source_time_id"]
        original = self.frame.set_index(["time_id", "stock_id"])["target"]
        for row in shuffled.itertuples(index=False):
            source_time = source_lookup.loc[row.time_id]
            self.assertEqual(row.target, original.loc[(source_time, row.stock_id)])

        movable = mapping[mapping["signature_group_count"] > 1]
        self.assertTrue(movable["changed"].all())
        singleton = mapping[mapping["time_id"] == 99].iloc[0]
        self.assertFalse(singleton["changed"])
        self.assertEqual(singleton["source_time_id"], 99)

    def test_shuffle_is_seeded_and_rejects_duplicate_group_stock_rows(self) -> None:
        first, first_map = run_experiments._groupwise_shuffle_targets(
            self.frame, seed=11
        )
        second, second_map = run_experiments._groupwise_shuffle_targets(
            self.frame, seed=11
        )
        pd.testing.assert_frame_equal(first, second)
        pd.testing.assert_frame_equal(first_map, second_map)

        duplicate = pd.concat([self.frame, self.frame.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "unique"):
            run_experiments._groupwise_shuffle_targets(duplicate, seed=11)


class ExperimentAuditArtifactTests(unittest.TestCase):
    def test_source_hashes_cover_runner_and_training_module(self) -> None:
        hashes = run_experiments._source_file_hashes()
        self.assertIn("scripts/run_experiments.py", hashes)
        self.assertIn("src/optiver/training.py", hashes)
        self.assertTrue(all(len(value) == 64 for value in hashes.values()))

    def test_feature_group_exclusion_is_repeatable_and_guarded(self) -> None:
        features = ["rv_pred", "rv2", "bpv", "trade_rv"]
        selected = run_experiments._exclude_feature_groups(
            features, ["jump", "trade"]
        )
        self.assertEqual(selected, ["rv_pred", "rv2"])
        with self.assertRaisesRegex(ValueError, "removed every"):
            run_experiments._exclude_feature_groups(
                ["bpv", "jump_var"], ["jump"]
            )

    def test_main_writes_completion_marker_only_after_success(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            feature_path = root / "features.parquet"
            feature_path.write_bytes(b"placeholder")
            args = argparse.Namespace(
                cluster_mode="none",
                allow_oracle=False,
                shuffle_target_groups_seed=None,
                features_path=feature_path,
                output_root=root,
                run_name="successful_run",
            )

            def successful_execute(parsed, *, feature_path, output_dir):
                del parsed, feature_path
                run_experiments._json_dump(output_dir / "run_metadata.json", {"ok": True})

            with (
                patch.object(run_experiments, "_parse_args", return_value=args),
                patch.object(run_experiments, "_execute_run", side_effect=successful_execute),
            ):
                run_experiments.main()

            output = root / "successful_run"
            completed = json.loads((output / "COMPLETED.json").read_text())
            status = json.loads((output / "status.json").read_text())
            self.assertEqual(completed["status"], "completed")
            self.assertEqual(status["status"], "completed")

    def test_main_marks_failed_partial_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            feature_path = root / "features.parquet"
            feature_path.write_bytes(b"placeholder")
            args = argparse.Namespace(
                cluster_mode="none",
                allow_oracle=False,
                shuffle_target_groups_seed=None,
                features_path=feature_path,
                output_root=root,
                run_name="failed_run",
            )
            with (
                patch.object(run_experiments, "_parse_args", return_value=args),
                patch.object(
                    run_experiments,
                    "_execute_run",
                    side_effect=RuntimeError("synthetic failure"),
                ),
                self.assertRaisesRegex(RuntimeError, "synthetic failure"),
            ):
                run_experiments.main()

            output = root / "failed_run"
            status = json.loads((output / "status.json").read_text())
            self.assertEqual(status["status"], "failed")
            self.assertFalse((output / "COMPLETED.json").exists())


if __name__ == "__main__":
    unittest.main()
