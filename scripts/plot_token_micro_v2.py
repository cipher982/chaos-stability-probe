#!/usr/bin/env python3
"""Plot aggregate token-enforced micro perturbation results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap


PAPER = "#fbfbf8"
INK = "#202124"
MUTED = "#5f6368"
GRID = "#e8eaed"
RED = "#a43d3d"
BLUE = "#2d5f9a"


def short_category(value: str) -> str:
    value = value.replace("micro_", "")
    labels = {
        "blank_line_wrap": "blank\nline",
        "double_internal_space": "double\nspace",
        "duplicate_punctuation": "duplicate\npunct.",
        "duplicate_small_word": "duplicate\nword",
        "line_wrap": "line\nwrap",
        "parenthesize_word": "paren\nword",
        "space_before_punctuation": "space\nbefore\npunct.",
        "space_after_punctuation": "space\nafter\npunct.",
        "tab_after_space": "tab after\nspace",
        "triple_internal_space": "triple\nspace",
        "trailing_space": "trailing\nspace",
        "leading_space": "leading\nspace",
        "leading_newline": "leading\nnewline",
        "trailing_newline": "trailing\nnewline",
        "tab_indent": "tab\nindent",
        "crlf_suffix": "CRLF\nsuffix",
    }
    return labels.get(value, value.replace("_", "\n"))


def load_categories(rank_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for run_dir in sorted(rank_dir.iterdir()):
        path = run_dir / "category_summary.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["model"] = run_dir.name
        rows.append(frame)
    if not rows:
        raise FileNotFoundError(f"No category_summary.csv files under {rank_dir}")
    return pd.concat(rows, ignore_index=True)


def model_bar(summary: pd.DataFrame, out_dir: Path) -> Path:
    data = summary.sort_values("semantic_mean")
    fig, ax = plt.subplots(figsize=(12.8, 6.2), facecolor=PAPER)
    ax.set_facecolor(PAPER)
    ax.barh(data["model"], data["semantic_mean"], color=BLUE, height=0.7)
    ax.set_xlabel("Mean endpoint semantic distance", color=INK)
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for y, value in enumerate(data["semantic_mean"]):
        ax.text(value + 0.002, y, f"{value:.3f}", va="center", fontsize=9, color=INK)
    fig.tight_layout(pad=0.8)
    out = out_dir / "token_micro_v2_model_bar.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def category_heatmap(categories: pd.DataFrame, out_dir: Path) -> Path:
    data = categories[~categories["category"].str.contains("control", na=False)].copy()
    matrix = data.pivot(index="model", columns="category", values="semantic_mean")
    model_order = data.groupby("model")["semantic_mean"].mean().sort_values(ascending=False).index
    category_means = data.groupby("category")["semantic_mean"].mean().sort_values(ascending=False)
    category_order = category_means.index
    matrix = matrix.reindex(index=model_order, columns=category_order)

    cmap = LinearSegmentedColormap.from_list("token_effect", ["#f7f0ea", "#d9634b", "#7f1d1d"])
    vmax = max(0.12, float(np.nanpercentile(matrix.values, 96)))
    fig = plt.figure(figsize=(16, 7.0), facecolor=PAPER)
    gs = gridspec.GridSpec(2, 1, height_ratios=[0.9, 6.0], hspace=0.06, figure=fig)
    ax_avg = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1])

    avg_values = category_means.reindex(category_order).values
    ax_avg.set_facecolor(PAPER)
    ax_avg.bar(np.arange(len(category_order)), avg_values, color="#6f7782", width=0.76)
    ax_avg.set_xlim(-0.5, len(category_order) - 0.5)
    ax_avg.set_ylim(0, max(avg_values.max() * 1.18, 0.01))
    ax_avg.set_xticks([])
    ax_avg.set_yticks([])
    ax_avg.spines[:].set_visible(False)
    ax_avg.text(
        0.0,
        1.16,
        "column avg, sorted high -> low",
        ha="left",
        va="bottom",
        fontsize=10,
        color=MUTED,
        weight="bold",
        transform=ax_avg.transAxes,
        clip_on=False,
    )
    ax_avg.annotate(
        "",
        xy=(0.98, 1.12),
        xytext=(0.22, 1.12),
        xycoords=ax_avg.transAxes,
        textcoords=ax_avg.transAxes,
        arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 1.4},
        clip_on=False,
    )

    ax.set_facecolor(PAPER)
    im = ax.imshow(matrix.values, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
    ax.set_yticks(np.arange(len(matrix.index)), matrix.index, fontsize=10)
    ax.set_xticks(np.arange(len(matrix.columns)), [short_category(c) for c in matrix.columns], fontsize=8)
    ax.tick_params(axis="both", length=0)
    ax.set_xticks(np.arange(-0.5, len(matrix.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(matrix.index), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.spines[:].set_visible(False)
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            value = matrix.iat[y, x]
            if pd.isna(value):
                continue
            color = "white" if value > vmax * 0.62 else INK
            ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=8, color=color, weight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.012)
    cbar.set_label("Mean semantic output distance", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout(pad=0.6)
    out = out_dir / "token_micro_v2_category_heatmap.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank-dir", type=Path, default=Path("runs/rankings/token_micro_v2"))
    parser.add_argument("--out-dir", type=Path, default=Path("talk/micro_visuals"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = args.rank_dir / "combined_model_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}; run scripts/summarize_token_micro_v2.py first")
    summary = pd.read_csv(summary_path)
    categories = load_categories(args.rank_dir)
    print(model_bar(summary, args.out_dir))
    print(category_heatmap(categories, args.out_dir))


if __name__ == "__main__":
    main()
