#!/usr/bin/env python3
"""Rank model runs by aggregate perturbation stability."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SMALL_PERTURBATIONS = ["noop_format", "punctuation", "synonym"]
MEANINGFUL_PERTURBATIONS = ["punctuation", "synonym", "paraphrase", "semantic_small"]


def read_run(label: str, run_dir: str) -> pd.DataFrame:
    path = Path(run_dir)
    summary = path / "summary_with_semantic.csv"
    if not summary.exists():
        summary = path / "summary.csv"
    df = pd.read_csv(summary)
    df["run_label"] = label
    return df


def mean_for(df: pd.DataFrame, categories: list[str], metric: str) -> float:
    values = df[df["category"].isin(categories)][metric]
    return float(values.mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", nargs=2, metavar=("LABEL", "DIR"), required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/rankings"))
    args = parser.parse_args()

    frames = [read_run(label, run_dir) for label, run_dir in args.run]
    rows = []
    for df in frames:
        label = str(df["run_label"].iloc[0])
        metric = "semantic_cosine_distance" if "semantic_cosine_distance" in df.columns else "token_edit_distance_norm"
        rows.append(
            {
                "run_label": label,
                "metric": metric,
                "small_perturbation_mean": mean_for(df, SMALL_PERTURBATIONS, metric),
                "meaningful_perturbation_mean": mean_for(df, MEANINGFUL_PERTURBATIONS, metric),
                "noop_mean": mean_for(df, ["noop_format"], metric),
                "punctuation_mean": mean_for(df, ["punctuation"], metric),
                "synonym_mean": mean_for(df, ["synonym"], metric),
                "paraphrase_mean": mean_for(df, ["paraphrase"], metric),
                "semantic_small_mean": mean_for(df, ["semantic_small"], metric),
                "positive_control_mean": mean_for(df, ["positive_control"], metric),
            }
        )
    ranking = pd.DataFrame(rows).sort_values("small_perturbation_mean")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(args.out_dir / "stability_rankings.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.barh(ranking["run_label"], ranking["small_perturbation_mean"])
    ax.set_title("Small-perturbation stability ranking", fontsize=18, pad=16)
    ax.set_xlabel("Mean semantic distance: no-op + punctuation + synonym")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(args.out_dir / "small_perturbation_ranking.png", dpi=220)
    plt.close(fig)

    print(ranking.to_string(index=False))
    print(f"Wrote rankings to {args.out_dir}")


if __name__ == "__main__":
    main()

