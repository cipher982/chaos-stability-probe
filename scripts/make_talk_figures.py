#!/usr/bin/env python3
"""Render a small set of slide-friendly figures from completed probe runs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CATEGORIES = [
    "control_identical",
    "noop_format",
    "punctuation",
    "synonym",
    "paraphrase",
    "semantic_small",
    "positive_control",
]

DISPLAY = {
    "control_identical": "Identical",
    "noop_format": "No-op format",
    "punctuation": "Punctuation",
    "synonym": "Synonym",
    "paraphrase": "Paraphrase",
    "semantic_small": "Small semantic",
    "positive_control": "Positive control",
}


def read_summary(label: str, run_dir: str) -> pd.DataFrame:
    path = Path(run_dir)
    summary = path / "summary_with_semantic.csv"
    if not summary.exists():
        summary = path / "summary.csv"
    df = pd.read_csv(summary)
    df["run_label"] = label
    return df


def first_existing(paths: list[str]) -> str | None:
    for path in paths:
        if (Path(path) / "summary_with_semantic.csv").exists() or (Path(path) / "summary.csv").exists():
            return path
    return None


def mean_table(frames: list[pd.DataFrame], metric: str) -> pd.DataFrame:
    merged = pd.concat(frames, ignore_index=True)
    merged = merged[merged["category"].isin(CATEGORIES)].copy()
    merged["category"] = pd.Categorical(merged["category"], categories=CATEGORIES, ordered=True)
    pivot = merged.pivot_table(
        index="category",
        columns="run_label",
        values=metric,
        aggfunc="mean",
        observed=True,
    )
    pivot.index = [DISPLAY[idx] for idx in pivot.index]
    return pivot


def plot_horizontal_grouped(table: pd.DataFrame, title: str, xlabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.2))
    table.plot(kind="barh", ax=ax, width=0.78)
    ax.set_title(title, fontsize=18, pad=16)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("")
    ax.set_xlim(0, max(1.05, float(table.max().max()) * 1.12))
    ax.grid(axis="x", alpha=0.25)
    ax.legend(title="", loc="lower right", frameon=True)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_qwen_hidden(out_dir: Path) -> None:
    frames = []
    for label, run_dir in [
        ("Qwen3.5 0.8B", "runs/qwen35_08b_expanded/qwen35_08b"),
        ("Qwen3.5 4B", "runs/qwen35_4b_expanded/qwen35_4b"),
    ]:
        hpath = Path(run_dir) / "hidden_states.jsonl"
        if not hpath.exists():
            continue
        df = pd.read_json(hpath, lines=True)
        if df.empty:
            continue
        df = df[df["layer"] == df["layer"].max()].copy()
        df["run_label"] = label
        frames.append(df)
    if not frames:
        return
    table = mean_table(frames, "last_token_cosine_distance")
    plot_horizontal_grouped(
        table,
        "Final-layer prompt-state divergence",
        "Last-token hidden cosine distance",
        out_dir / "qwen_hidden_state_divergence.png",
    )


def main() -> None:
    out_dir = Path("runs/talk_figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    qwen_frames = [
        read_summary("Qwen3.5 0.8B", "runs/qwen35_08b_expanded/qwen35_08b"),
        read_summary("Qwen3.5 4B", "runs/qwen35_4b_expanded/qwen35_4b"),
    ]
    qwen9_dir = first_existing(
        [
            "runs/sagemaker_artifacts/chaos-stability-qwen35-9b-20260429-001/runs/qwen35_9b",
            "runs/sagemaker_artifacts/chaos-stability-panel-20260429-001/runs/qwen35_9b",
            "runs/qwen35_9b_allpairs/qwen35_9b",
        ]
    )
    if qwen9_dir:
        qwen_frames.append(read_summary("Qwen3.5 9B", qwen9_dir))
    qwen_sem = mean_table(qwen_frames, "semantic_cosine_distance")
    plot_horizontal_grouped(
        qwen_sem,
        "Qwen3.5 stability differs sharply by size",
        "Sentence-embedding cosine distance",
        out_dir / "qwen_semantic_divergence.png",
    )

    cross_frames = [
        read_summary("Qwen3.5 0.8B", "runs/qwen35_08b_expanded/qwen35_08b"),
        read_summary("Qwen3.5 4B", "runs/qwen35_4b_expanded/qwen35_4b"),
        read_summary(
            "Gemma 4 E2B",
            "runs/sagemaker_artifacts/chaos-stability-gemma4-e2b-20260429-001/runs/gemma4_e2b_it",
        ),
        read_summary(
            "OLMo 3 7B",
            "runs/sagemaker_artifacts/chaos-stability-olmo3-7b-20260429-001/runs/olmo3_7b_instruct",
        ),
    ]
    if qwen9_dir:
        cross_frames.append(read_summary("Qwen3.5 9B", qwen9_dir))
    gemma_e4b_dir = first_existing(
        [
            "runs/sagemaker_artifacts/chaos-stability-gemma4-e4b-20260429-002/runs/gemma4_e4b_it",
            "runs/sagemaker_artifacts/chaos-stability-gemma4-e4b-20260429-001/runs/gemma4_e4b_it",
        ]
    )
    if gemma_e4b_dir:
        cross_frames.append(read_summary("Gemma 4 E4B", gemma_e4b_dir))
    cross_sem = mean_table(cross_frames, "semantic_cosine_distance")
    plot_horizontal_grouped(
        cross_sem,
        "Different models have different stability profiles",
        "Sentence-embedding cosine distance",
        out_dir / "cross_lab_semantic_divergence.png",
    )

    plot_qwen_hidden(out_dir)
    print(f"Wrote talk figures to {out_dir}")


if __name__ == "__main__":
    main()
