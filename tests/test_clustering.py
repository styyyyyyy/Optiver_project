from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from optiver.clustering import (  # noqa: E402
    FeatureStockClusterer,
    TargetCorrelationStockClusterer,
    add_cluster_labels,
    add_same_time_cluster_aggregates,
)


def _target_training_frame() -> pd.DataFrame:
    records: list[dict[str, float | int]] = []
    first_pattern = np.linspace(-2.0, 2.0, 12)
    second_pattern = np.array([1.0, -1.0] * 6)
    for time_id, (first, second) in enumerate(
        zip(first_pattern, second_pattern, strict=True)
    ):
        records.extend(
            [
                {"stock_id": 0, "time_id": time_id, "target": first},
                {"stock_id": 1, "time_id": time_id, "target": 1.5 * first + 0.01},
                {"stock_id": 2, "time_id": time_id, "target": second},
                {"stock_id": 3, "time_id": time_id, "target": 2.0 * second - 0.01},
            ]
        )
    return pd.DataFrame.from_records(records)


class TargetCorrelationStockClustererTests(unittest.TestCase):
    def test_fit_uses_outer_train_and_transform_does_not_read_validation_target(self) -> None:
        outer_train = _target_training_frame()
        validation_without_target = pd.DataFrame(
            {
                "stock_id": [0, 1, 2, 3, 99],
                "time_id": [100, 100, 100, 100, 100],
                "rv_pred": [0.1, 0.2, 0.3, 0.4, 0.5],
            }
        )
        clusterer = TargetCorrelationStockClusterer(
            n_clusters=2, random_state=7
        )

        train_labeled, validation_labeled = clusterer.fit_transform_split(
            outer_train, validation_without_target
        )

        self.assertEqual(clusterer.n_fit_rows_, len(outer_train))
        self.assertEqual(set(clusterer.stock_to_cluster_), {0, 1, 2, 3})
        self.assertTrue(
            isinstance(train_labeled["stock_cluster"].dtype, pd.CategoricalDtype)
        )
        self.assertEqual(
            list(validation_labeled["stock_cluster"].cat.categories), [0, 1, -1]
        )
        self.assertEqual(validation_labeled.loc[4, "stock_cluster"], -1)

        # A validation target, even if absurdly large, is never consulted by
        # transform.  This is the central outer-fold leakage regression test.
        validation_with_target = validation_without_target.assign(
            target=[1e9, -1e9, 5e8, -5e8, 123.0]
        )
        relabeled = clusterer.transform(validation_with_target)
        pd.testing.assert_series_equal(
            validation_labeled["stock_cluster"],
            relabeled["stock_cluster"],
            check_names=True,
        )

        # The synthetic stocks form two clear correlation blocks.
        mapping = clusterer.stock_to_cluster_
        self.assertEqual(mapping[0], mapping[1])
        self.assertEqual(mapping[2], mapping[3])
        self.assertNotEqual(mapping[0], mapping[2])

    def test_one_hot_schema_is_stable_and_marks_unknown_stocks(self) -> None:
        frame = pd.DataFrame({"stock_id": [10, 20, 999]})
        encoded = add_cluster_labels(
            frame,
            {10: 0, 20: 1},
            encoding="onehot",
            cluster_labels=[0, 1],
        )
        expected_columns = [
            "stock_cluster__0",
            "stock_cluster__1",
            "stock_cluster__-1",
        ]
        self.assertEqual(
            [column for column in encoded if column.startswith("stock_cluster__")],
            expected_columns,
        )
        self.assertEqual(encoded.loc[0, expected_columns].tolist(), [1, 0, 0])
        self.assertEqual(encoded.loc[2, expected_columns].tolist(), [0, 0, 1])

    def test_too_many_clusters_fails_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "only 4 stocks"):
            TargetCorrelationStockClusterer(n_clusters=5).fit(
                _target_training_frame()
            )


class FeatureStockClustererTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "stock_id": np.repeat([0, 1, 2, 3], 4),
                "time_id": list(range(4)) * 4,
                "rv_pred": [1.0, 1.1, 0.9, 1.0, 1.2, 1.1, 1.0, 1.1]
                + [9.0, 9.2, 8.8, 9.1, 10.0, 9.8, 10.1, 9.9],
                "trade_count": [10, 11, 9, 10, 12, 11, 10, 11]
                + [90, 92, 88, 91, 100, 98, 101, 99],
                "target": np.arange(16, dtype=float),
            }
        )

    def test_target_free_fit_ignores_unrequested_target_column(self) -> None:
        kwargs = {
            "feature_cols": ["rv_pred", "trade_count"],
            "n_clusters": 2,
            "aggregations": ("mean", "std"),
            "random_state": 3,
        }
        first = FeatureStockClusterer(**kwargs).fit(self.frame)
        changed_target = self.frame.assign(target=np.arange(16) ** 5)
        second = FeatureStockClusterer(**kwargs).fit(changed_target)

        self.assertEqual(first.stock_to_cluster_, second.stock_to_cluster_)
        mapping = first.stock_to_cluster_
        self.assertEqual(mapping[0], mapping[1])
        self.assertEqual(mapping[2], mapping[3])
        self.assertNotEqual(mapping[0], mapping[2])

    def test_target_column_is_rejected_as_a_feature_by_default(self) -> None:
        with self.assertRaisesRegex(ValueError, "Refusing to use target"):
            FeatureStockClusterer(
                feature_cols=["rv_pred", "target"], n_clusters=2
            ).fit(self.frame)


class SameTimeClusterAggregateTests(unittest.TestCase):
    def setUp(self) -> None:
        # Stock A deliberately has two rows.  Its contribution is their mean (2),
        # and it must still be removed as one stock, not as one row.
        self.frame = pd.DataFrame(
            {
                "time_id": [1, 1, 1, 1],
                "stock_id": ["A", "A", "B", "C"],
                "stock_cluster": [0, 0, 0, 1],
                "signal": [1.0, 3.0, 5.0, 10.0],
                "target": [0.01, 0.02, 0.03, 0.04],
            }
        )

    def test_leave_one_stock_out_uses_only_other_stocks(self) -> None:
        result = add_same_time_cluster_aggregates(
            self.frame,
            ["signal"],
            stats=("mean", "sum", "count"),
            clusters=[0, 1],
        )

        # A's stock-level mean is 2, so its only cluster-0 peer is B at 5.
        self.assertEqual(result.loc[0, "signal_cluster_0_mean"], 5.0)
        self.assertEqual(result.loc[1, "signal_cluster_0_mean"], 5.0)
        self.assertEqual(result.loc[0, "signal_cluster_0_count"], 1.0)
        # B sees A once, using A's stock-level mean rather than either row.
        self.assertEqual(result.loc[2, "signal_cluster_0_mean"], 2.0)
        self.assertEqual(result.loc[2, "signal_cluster_0_sum"], 2.0)
        # C is outside cluster 0 and therefore sees both cluster-0 stocks.
        self.assertEqual(result.loc[3, "signal_cluster_0_mean"], 3.5)
        self.assertEqual(result.loc[3, "signal_cluster_0_count"], 2.0)
        # C is the only member of cluster 1, so it has no valid peer value.
        self.assertTrue(np.isnan(result.loc[3, "signal_cluster_1_mean"]))
        # Stocks outside cluster 1 can observe C's contemporaneous feature.
        self.assertEqual(result.loc[0, "signal_cluster_1_mean"], 10.0)

    def test_non_leave_one_out_matches_full_cluster_mean(self) -> None:
        result = add_same_time_cluster_aggregates(
            self.frame,
            ["signal"],
            clusters=[0, 1],
            leave_one_stock_out=False,
        )
        np.testing.assert_allclose(result["signal_cluster_0_mean"], 3.5)
        np.testing.assert_allclose(result["signal_cluster_1_mean"], 10.0)

    def test_target_aggregate_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Refusing to aggregate target"):
            add_same_time_cluster_aggregates(
                self.frame, ["target"], clusters=[0, 1]
            )


if __name__ == "__main__":
    unittest.main()
