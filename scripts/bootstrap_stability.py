#!/usr/bin/env python3
"""Bootstrap stability scores over prompt pairs and plot bucketed results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SMALL_PERTURBATIONS = ["noop_format", "punctuation", "synonym"]


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, samples: int) -> tuple[float, float]:
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(samples)]
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def bucket_for(mean: float) -> str:
    if mean <= 0.035:
        return "stable"
    if mean >= 0.12:
        return "brittle"
    return "middle"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("merged_summary", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/rankings/bootstrap"))
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    df = pd.read_csv(args.merged_summary)
    metric = "semantic_cosine_distance" if "semantic_cosine_distance" in df.columns else "token_edit_distance_norm"
    small = df[df["category"].isin(SMALL_PERTURBATIONS)].copy()
    rng = np.random.default_rng(args.seed)

    rows = []
    for label, group in small.groupby("run_label", sort=False):
        values = group[metric].to_numpy()
        lo, hi = bootstrap_mean(values, rng, args.samples)
        mean = float(values.mean())
        rows.append(
            {
                "run_label": label,
                "metric": metric,
                "n": len(values),
                "mean": mean,
                "sd": float(values.std(ddof=1)),
                "ci95_low": lo,
                "ci95_high": hi,
                "bucket": bucket_for(mean),
            }
        )

    result = pd.DataFrame(rows).sort_values("mean").reset_index(drop=True)
    result["rank_point_estimate"] = np.arange(1, len(result) + 1)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out_dir / "small_perturbation_bootstrap.csv", index=False)

    colors = {"stable": "#2f7d4f", "middle": "#6f7785", "brittle": "#b64848"}
    fig, ax = plt.subplots(figsize=(10.8, 6.4))
    y = np.arange(len(result))
    xerr = np.vstack(
        [
            result["mean"] - result["ci95_low"],
            result["ci95_high"] - result["mean"],
        ]
    )
    ax.barh(
        y,
        result["mean"],
        xerr=xerr,
        color=[colors[b] for b in result["bucket"]],
        alpha=0.9,
        capsize=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(result["run_label"])
    ax.invert_yaxis()
    ax.set_xlabel("Mean semantic distance over no-op + punctuation + synonym")
    ax.set_title("Small-perturbation stability with bootstrap 95% intervals", fontsize=15, pad=14)
    ax.grid(axis="x", alpha=0.25)

    handles = [
        plt.Line2D([0], [0], color=colors["stable"], lw=8, label="stable cluster"),
        plt.Line2D([0], [0], color=colors["middle"], lw=8, label="middle cluster"),
        plt.Line2D([0], [0], color=colors["brittle"], lw=8, label="brittle cluster"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(args.out_dir / "small_perturbation_bootstrap_buckets.png", dpi=220)
    plt.close(fig)

    print(result.to_string(index=False))
    print(f"Wrote bootstrap artifacts to {args.out_dir}")


if __name__ == "__main__":
    main()
