#!/usr/bin/env python3
"""Compare E07 activation-patching waves by model and rescue mechanism."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_WAVES = [
    Path("runs/rankings/activation_patch_v1"),
    Path("runs/rankings/activation_patch_v2"),
    Path("runs/rankings/activation_patch_v3"),
    Path("runs/rankings/activation_patch_v4_reverse"),
]

MODEL_LABELS = {
    "qwen35_08b": "Qwen 0.8B",
    "qwen35_2b": "Qwen 2B",
    "qwen35_4b": "Qwen 4B",
    "qwen35_9b": "Qwen 9B",
    "gemma4_e2b_it": "Gemma E2B IT",
    "gemma4_e2b_base": "Gemma E2B base",
    "gemma4_e4b_base": "Gemma E4B base",
}


def load_cases(wave_dirs: list[Path]) -> pd.DataFrame:
    frames = []
    for wave_dir in wave_dirs:
        path = wave_dir / "patch_cases_ranked.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path).copy()
        df["wave"] = wave_dir.name
        frames.append(df)
    if not frames:
        raise SystemExit("No patch_cases_ranked.csv files found")
    cases = pd.concat(frames, ignore_index=True).copy()
    for col in [
        "best_rescue_fraction",
        "prompt_lcp_token_best_rescue_fraction",
        "final_context_token_best_rescue_fraction",
    ]:
        if col not in cases.columns:
            cases[col] = pd.NA
        cases[col] = pd.to_numeric(cases[col], errors="coerce")
    if "corrupt_replay_matches_b_branch" in cases.columns:
        cases["replayable"] = cases["corrupt_replay_matches_b_branch"].fillna(False).astype(bool)
    else:
        cases["replayable"] = False
    cases["model_label"] = cases["model_name"].map(MODEL_LABELS).fillna(cases["model_name"])
    cases["prompt_lcp_full"] = cases["prompt_lcp_token_best_rescue_fraction"] >= 1.0
    cases["prompt_lcp_strong"] = cases["prompt_lcp_token_best_rescue_fraction"] >= 0.5
    cases["final_full"] = cases["final_context_token_best_rescue_fraction"] >= 1.0
    cases["final_strong"] = cases["final_context_token_best_rescue_fraction"] >= 0.5
    cases["best_full"] = cases["best_rescue_fraction"] >= 1.0
    cases["best_strong"] = cases["best_rescue_fraction"] >= 0.5
    cases["final_only_full"] = cases["final_full"] & ~cases["prompt_lcp_strong"]
    return cases


def summarize(cases: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_summary = (
        cases.groupby(["wave", "model_name", "model_label"], dropna=False)
        .agg(
            cases=("pair_id", "count"),
            replayable=("replayable", "sum"),
            mean_best_rescue=("best_rescue_fraction", "mean"),
            median_best_rescue=("best_rescue_fraction", "median"),
            best_full=("best_full", "sum"),
            best_strong=("best_strong", "sum"),
            prompt_lcp_full=("prompt_lcp_full", "sum"),
            prompt_lcp_strong=("prompt_lcp_strong", "sum"),
            final_full=("final_full", "sum"),
            final_strong=("final_strong", "sum"),
            final_only_full=("final_only_full", "sum"),
        )
        .reset_index()
    )
    class_counts = (
        cases.groupby(["wave", "model_name", "model_label", "best_position_class"], dropna=False)
        .size()
        .reset_index(name="n")
    )
    return model_summary, class_counts


def plot_position_classes(model_summary: pd.DataFrame, class_counts: pd.DataFrame, out_dir: Path) -> None:
    forward = class_counts[~class_counts["wave"].str.contains("reverse")].copy()
    if forward.empty:
        return
    pivot = (
        forward.pivot_table(
            index=["model_name", "model_label"],
            columns="best_position_class",
            values="n",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )
    ordered_models = [m for m in MODEL_LABELS if m in set(pivot["model_name"])]
    pivot = pivot.set_index("model_name").loc[ordered_models].reset_index()
    cols = [
        col
        for col in ["prompt_lcp", "aligned_prompt", "generated_prefix", "final_context"]
        if col in pivot.columns
    ]
    colors = {
        "prompt_lcp": "#b85c38",
        "aligned_prompt": "#d8a03d",
        "generated_prefix": "#4b9a88",
        "final_context": "#3b6f9e",
    }

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    y = range(len(pivot))
    left = pd.Series([0] * len(pivot), dtype=float)
    for col in cols:
        ax.barh(y, pivot[col], left=left, label=col.replace("_", " "), color=colors[col])
        left += pivot[col]
    ax.set_yticks(list(y), pivot["model_label"])
    ax.invert_yaxis()
    ax.set_xlabel("Selected cases")
    ax.set_title("E07 best rescue position class by model", loc="left", fontsize=14)
    ax.grid(axis="x", alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=len(cols),
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_dir / "e07_best_rescue_position_classes.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave-dir", type=Path, action="append", default=[])
    parser.add_argument("--out-dir", type=Path, default=Path("runs/rankings/activation_patch_comparison"))
    args = parser.parse_args()

    wave_dirs = args.wave_dir or DEFAULT_WAVES
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cases = load_cases(wave_dirs)
    model_summary, class_counts = summarize(cases)
    cases.to_csv(args.out_dir / "case_level_summary.csv", index=False)
    model_summary.to_csv(args.out_dir / "model_level_summary.csv", index=False)
    class_counts.to_csv(args.out_dir / "position_class_counts.csv", index=False)
    plot_position_classes(model_summary, class_counts, args.out_dir)
    print(model_summary.to_string(index=False))
    print(f"Wrote activation-patch comparison artifacts to {args.out_dir}")


if __name__ == "__main__":
    main()
