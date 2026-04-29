#!/usr/bin/env python3
"""Compare summary metrics across multiple run directories."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def ordered_categories(df: pd.DataFrame) -> list[str]:
    preferred = [
        "control_identical",
        "noop_format",
        "punctuation",
        "synonym",
        "paraphrase",
        "semantic_small",
        "positive_control",
    ]
    present = list(dict.fromkeys(df["category"].tolist()))
    return [c for c in preferred if c in present] + [c for c in present if c not in preferred]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", nargs=2, metavar=("LABEL", "DIR"), required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/comparisons"))
    args = parser.parse_args()

    frames = []
    for label, run_dir in args.run:
        run_path = Path(run_dir)
        path = run_path / "summary_with_semantic.csv"
        if not path.exists():
            path = run_path / "summary.csv"
        df = pd.read_csv(path)
        df["run_label"] = label
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_dir / "merged_summary.csv", index=False)

    categories = ordered_categories(merged)
    merged["category"] = pd.Categorical(merged["category"], categories=categories, ordered=True)

    pivot = merged.pivot_table(
        index="category",
        columns="run_label",
        values="token_edit_distance_norm",
        aggfunc="mean",
        observed=True,
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Prompt perturbation output divergence")
    ax.set_ylabel("Normalized token edit distance")
    ax.set_xlabel("Perturbation category")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Run")
    fig.tight_layout()
    fig.savefig(args.out_dir / "compare_output_divergence.png", dpi=180)
    plt.close(fig)

    pivot_lcp = merged.pivot_table(
        index="category",
        columns="run_label",
        values="common_prefix_tokens",
        aggfunc="mean",
        observed=True,
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot_lcp.plot(kind="bar", ax=ax)
    ax.set_title("How long outputs remain identical")
    ax.set_ylabel("Common generated-token prefix")
    ax.set_xlabel("Perturbation category")
    ax.legend(title="Run")
    fig.tight_layout()
    fig.savefig(args.out_dir / "compare_common_prefix.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = []
    data = []
    positions = []
    width = 0.32
    run_labels = list(dict.fromkeys(merged["run_label"].tolist()))
    for cat_idx, category in enumerate(categories):
        for run_idx, run_label in enumerate(run_labels):
            values = merged[
                (merged["category"] == category) & (merged["run_label"] == run_label)
            ]["token_edit_distance_norm"].tolist()
            if not values:
                continue
            data.append(values)
            positions.append(cat_idx + (run_idx - (len(run_labels) - 1) / 2) * width)
            labels.append(run_label)
    bp = ax.boxplot(data, positions=positions, widths=width * 0.75, patch_artist=True)
    for patch_idx, patch in enumerate(bp["boxes"]):
        patch.set_alpha(0.7)
    ax.set_title("Output divergence distributions")
    ax.set_ylabel("Normalized token edit distance")
    ax.set_xlabel("Perturbation category")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=90)
    legend_handles = []
    for run_idx, run_label in enumerate(run_labels):
        handle = plt.Line2D([0], [0], color=f"C{run_idx}", lw=6, alpha=0.7)
        legend_handles.append(handle)
        for patch_idx in range(run_idx, len(bp["boxes"]), len(run_labels)):
            bp["boxes"][patch_idx].set_facecolor(f"C{run_idx}")
    ax.legend(legend_handles, run_labels, title="Run")
    fig.tight_layout()
    fig.savefig(args.out_dir / "compare_output_divergence_boxplot.png", dpi=180)
    plt.close(fig)

    if "semantic_cosine_distance" in merged.columns:
        pivot_sem = merged.pivot_table(
            index="category",
            columns="run_label",
            values="semantic_cosine_distance",
            aggfunc="mean",
            observed=True,
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        pivot_sem.plot(kind="bar", ax=ax)
        ax.set_title("Prompt perturbation semantic divergence")
        ax.set_ylabel("Sentence-embedding cosine distance")
        ax.set_xlabel("Perturbation category")
        ax.set_ylim(0, max(1.05, float(pivot_sem.max().max()) * 1.1))
        ax.legend(title="Run")
        fig.tight_layout()
        fig.savefig(args.out_dir / "compare_semantic_divergence.png", dpi=180)
        plt.close(fig)

    hidden_frames = []
    for label, run_dir in args.run:
        hidden_path = Path(run_dir) / "hidden_states.jsonl"
        if not hidden_path.exists():
            continue
        hdf = pd.read_json(hidden_path, lines=True)
        if hdf.empty:
            continue
        final_layer = hdf["layer"].max()
        hdf = hdf[hdf["layer"] == final_layer].copy()
        hdf["run_label"] = label
        hidden_frames.append(hdf)
    if hidden_frames:
        hidden = pd.concat(hidden_frames, ignore_index=True)
        hidden["category"] = pd.Categorical(hidden["category"], categories=categories, ordered=True)
        pivot_hidden = hidden.pivot_table(
            index="category",
            columns="run_label",
            values="last_token_cosine_distance",
            aggfunc="mean",
            observed=True,
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        pivot_hidden.plot(kind="bar", ax=ax)
        ax.set_title("Final-layer prompt-state divergence")
        ax.set_ylabel("Last-token hidden cosine distance")
        ax.set_xlabel("Perturbation category")
        ax.legend(title="Run")
        fig.tight_layout()
        fig.savefig(args.out_dir / "compare_final_layer_hidden_divergence.png", dpi=180)
        plt.close(fig)

    print(f"Wrote comparison to {args.out_dir}")


if __name__ == "__main__":
    main()
