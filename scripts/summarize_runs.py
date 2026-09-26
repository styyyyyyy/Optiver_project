#!/usr/bin/env python3
"""Join OOF artifacts by key and compute paired grouped-bootstrap comparisons."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from optiver.metrics import (  # noqa: E402
    grouped_bootstrap_rmspe,
    grouped_bootstrap_rmspe_difference,
)


def _parse_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("run label must not be empty")
    return label, Path(raw_path)


def _parse_comparison(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("comparison must be CANDIDATE:REFERENCE")
    return tuple(value.split(":", 1))  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=_parse_spec, required=True)
    parser.add_argument("--comparison", action="append", type=_parse_comparison, default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=2021)
    args = parser.parse_args()

    frames: dict[str, pd.DataFrame] = {}
    for label, directory in args.run:
        path = directory / "oof_predictions.parquet"
        frame = pd.read_parquet(
            path,
            columns=["stock_id", "time_id", "target", "prediction"],
        )
        if frame.duplicated(["stock_id", "time_id"]).any():
            raise ValueError(f"{label} has duplicate keys")
        frames[label] = frame

    labels = list(frames)
    reference_label = labels[0]
    merged = frames[reference_label].rename(
        columns={"prediction": reference_label}
    )
    for label in labels[1:]:
        candidate = frames[label].rename(
            columns={"target": f"target__{label}", "prediction": label}
        )
        merged = merged.merge(
            candidate,
            on=["stock_id", "time_id"],
            how="inner",
            validate="one_to_one",
        )
        if len(merged) != len(frames[reference_label]):
            raise ValueError(f"Key coverage differs for {label}")
        other_target = merged.pop(f"target__{label}")
        if not other_target.equals(merged["target"]):
            raise ValueError(f"Targets differ for {label}")

    summaries = []
    for label in labels:
        interval = grouped_bootstrap_rmspe(
            merged["target"].to_numpy(),
            merged[label].to_numpy(),
            merged["time_id"].to_numpy(),
            n_resamples=args.bootstrap_resamples,
            seed=args.seed,
        )
        summaries.append(
            {
                "model": label,
                "rmspe": interval.estimate,
                "ci_lower": interval.lower,
                "ci_upper": interval.upper,
            }
        )

    comparisons = []
    for candidate, reference in args.comparison:
        if candidate not in frames or reference not in frames:
            raise KeyError(f"Unknown comparison {candidate}:{reference}")
        interval = grouped_bootstrap_rmspe_difference(
            merged["target"].to_numpy(),
            merged[candidate].to_numpy(),
            merged[reference].to_numpy(),
            merged["time_id"].to_numpy(),
            n_resamples=args.bootstrap_resamples,
            seed=args.seed,
        )
        comparisons.append(
            {
                "candidate": candidate,
                "reference": reference,
                "rmspe_difference": interval.estimate,
                "ci_lower": interval.lower,
                "ci_upper": interval.upper,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summaries).sort_values("rmspe")
    comparison = pd.DataFrame(comparisons)
    summary.to_csv(args.output_dir / "model_summary.csv", index=False)
    comparison.to_csv(args.output_dir / "paired_comparisons.csv", index=False)
    print(summary.to_string(index=False))
    if not comparison.empty:
        print("\nPaired differences (negative favors candidate):")
        print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
