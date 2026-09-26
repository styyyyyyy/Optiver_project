#!/usr/bin/env python3
"""Render the committed final model and fold comparison figure."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "results" / "final_nested_blend"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "figures" / "final_model_comparison.png"


def main() -> None:
    model_summary = pd.read_csv(RESULT_DIR / "model_summary.csv")
    fold_metrics = pd.read_csv(RESULT_DIR / "fold_metrics.csv")
    order = ["lightgbm", "mlp", "blend"]
    labels = {"lightgbm": "LightGBM", "mlp": "MLP", "blend": "Strict blend"}
    colors = {"lightgbm": "#4C78A8", "mlp": "#F58518", "blend": "#54A24B"}

    summary = model_summary.set_index("model").loc[order]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    x = range(len(order))
    values = summary["rmspe"].to_numpy()
    lower = values - summary["ci_lower"].to_numpy()
    upper = summary["ci_upper"].to_numpy() - values
    axes[0].bar(x, values, color=[colors[name] for name in order], width=0.65)
    axes[0].errorbar(
        x,
        values,
        yerr=[lower, upper],
        fmt="none",
        color="#222222",
        capsize=4,
    )
    axes[0].set_xticks(list(x), [labels[name] for name in order])
    axes[0].set_ylabel("RMSPE (lower is better)")
    axes[0].set_title("Pooled OOF performance")
    axes[0].set_ylim(0.205, 0.235)
    axes[0].grid(axis="y", alpha=0.25)

    for name in order:
        rows = fold_metrics.loc[fold_metrics["model"] == name].sort_values("fold")
        axes[1].plot(
            rows["fold"],
            rows["rmspe"],
            marker="o",
            linewidth=2,
            label=labels[name],
            color=colors[name],
        )
    axes[1].set_xticks(range(5))
    axes[1].set_xlabel("Outer fold")
    axes[1].set_ylabel("RMSPE")
    axes[1].set_title("Fold-level performance")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    figure.suptitle("Optiver development-set grouped CV", fontsize=13)
    figure.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
