#!/usr/bin/env python3
"""Create slide-friendly plots from stability probe outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_summary(run_dir: Path) -> None:
    summary_path = run_dir / "summary.csv"
    if not summary_path.exists():
        print(f"No summary file: {summary_path}")
        return

    df = pd.read_csv(summary_path)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = df.pivot_table(
        index="category",
        columns="model_name",
        values="token_edit_distance_norm",
        aggfunc="mean",
    )
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Output divergence by perturbation category")
    ax.set_ylabel("Normalized token edit distance")
    ax.set_xlabel("Perturbation category")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Model", loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    fig.savefig(run_dir / "plot_output_divergence_by_category.png", dpi=180)
    plt.close(fig)


def plot_curves(run_dir: Path) -> None:
    curve_path = run_dir / "curves.jsonl"
    if not curve_path.exists():
        print(f"No curve file: {curve_path}")
        return

    df = pd.read_json(curve_path, lines=True)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for (model, category), part in df.groupby(["model_name", "category"]):
        mean_curve = part.groupby("t")["token_edit_distance_norm"].mean()
        ax.plot(mean_curve.index, mean_curve.values, label=f"{model} / {category}", alpha=0.85)
    ax.set_title("Divergence over generated-token prefix length")
    ax.set_ylabel("Normalized token edit distance")
    ax.set_xlabel("Generated token prefix length")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize="small")
    fig.tight_layout()
    fig.savefig(run_dir / "plot_divergence_curves.png", dpi=180)
    plt.close(fig)


def plot_hidden(run_dir: Path) -> None:
    hidden_path = run_dir / "hidden_states.jsonl"
    if not hidden_path.exists():
        print(f"No hidden-state file: {hidden_path}")
        return

    df = pd.read_json(hidden_path, lines=True)
    if df.empty:
        return

    for model_name, part in df.groupby("model_name"):
        pivot = part.pivot_table(
            index="category",
            columns="layer",
            values="last_token_cosine_distance",
            aggfunc="mean",
        )
        fig, ax = plt.subplots(figsize=(11, 4.8))
        im = ax.imshow(pivot.values, aspect="auto", interpolation="nearest")
        ax.set_title(f"Hidden-state divergence by layer: {model_name}")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Perturbation category")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=90, fontsize=7)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        fig.colorbar(im, ax=ax, label="Last-token cosine distance")
        fig.tight_layout()
        fig.savefig(run_dir / f"plot_hidden_layer_divergence_{model_name}.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    plot_summary(args.run_dir)
    plot_curves(args.run_dir)
    plot_hidden(args.run_dir)
    print(f"Wrote plots to {args.run_dir}")


if __name__ == "__main__":
    main()

