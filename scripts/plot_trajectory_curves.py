#!/usr/bin/env python3
"""Plot output-prefix divergence trajectories from curve JSONL artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OUT_DIR = Path("runs/trajectory_figures")


LONG_RUNS = {
    "Qwen3.5 0.8B": "runs/sagemaker_artifacts/chaos-longprobe-qwen35-08b-20260429-001/runs/qwen35_08b/curves.jsonl",
    "Qwen3.5 4B": "runs/sagemaker_artifacts/chaos-longprobe-qwen35-4b-20260429-001/runs/qwen35_4b/curves.jsonl",
    "DeepSeek R1 Qwen 7B": "runs/sagemaker_artifacts/chaos-longprobe-deepseek-r1-qwen7b-20260429-001/runs/deepseek_r1_distill_qwen_7b/curves.jsonl",
}

LONG_PAIR_LABELS = {
    "noop_stability_newline": "No-op newline",
    "punctuation_complex_system": "Punctuation",
    "synonym_small_tiny": "Synonym",
    "semantic_weather_markets": "Small semantic",
}

QWEN_THINKOFF_RUNS = {
    "Qwen3.5 0.8B": "runs/sagemaker_artifacts/chaos-scaffold-long-qwen35-08b-thinkoff-20260429-001/runs/qwen35_08b/curves.jsonl",
    "Qwen3.5 2B": "runs/sagemaker_artifacts/chaos-scaffold-long-qwen35-2b-thinkoff-20260429-001/runs/qwen35_2b/curves.jsonl",
    "Qwen3.5 4B": "runs/sagemaker_artifacts/chaos-scaffold-long-qwen35-4b-thinkoff-20260429-001/runs/qwen35_4b/curves.jsonl",
    "Qwen3.5 9B": "runs/sagemaker_artifacts/chaos-scaffold-long-qwen35-9b-thinkoff-20260429-001/runs/qwen35_9b/curves.jsonl",
}

QWEN_THINKOFF_BOOTSTRAP_LABELS = {
    "Qwen3.5 0.8B": "Qwen3.5 0.8B think-off",
    "Qwen3.5 2B": "Qwen3.5 2B think-off",
    "Qwen3.5 4B": "Qwen3.5 4B think-off",
    "Qwen3.5 9B": "Qwen3.5 9B think-off",
}

QWEN_QUANT_RUNS = {
    "Qwen3.5 0.8B": {
        "BF16": "runs/sagemaker_artifacts/chaos-stability-panel-20260429-001/runs/qwen35_08b/curves.jsonl",
        "8-bit": "runs/sagemaker_artifacts/chaos-stability-qwen35-08b-bnb8-20260429-001/runs/qwen35_08b_bnb8/curves.jsonl",
        "4-bit": "runs/sagemaker_artifacts/chaos-stability-qwen35-08b-bnb4-20260429-001/runs/qwen35_08b_bnb4/curves.jsonl",
    },
    "Qwen3.5 4B": {
        "BF16": "runs/sagemaker_artifacts/chaos-stability-panel-20260429-001/runs/qwen35_4b/curves.jsonl",
        "8-bit": "runs/sagemaker_artifacts/chaos-stability-qwen35-4b-bnb8-20260429-001/runs/qwen35_4b_bnb8/curves.jsonl",
        "4-bit": "runs/sagemaker_artifacts/chaos-stability-qwen35-4b-bnb4-20260429-001/runs/qwen35_4b_bnb4/curves.jsonl",
    },
}


def read_curves(path: str) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    return df.sort_values(["pair_id", "repeat", "t"]).reset_index(drop=True)


def plot_long_probe() -> None:
    frames = []
    for label, path in LONG_RUNS.items():
        if not Path(path).exists():
            continue
        df = read_curves(path)
        df["model_label"] = label
        frames.append(df)
    if not frames:
        return

    merged = pd.concat(frames, ignore_index=True)
    pair_ids = [pair for pair in LONG_PAIR_LABELS if pair in set(merged["pair_id"])]
    fig, axes = plt.subplots(len(pair_ids), 1, figsize=(11, 9), sharex=True)
    if len(pair_ids) == 1:
        axes = [axes]

    for ax, pair_id in zip(axes, pair_ids, strict=True):
        subset = merged[merged["pair_id"] == pair_id]
        for model_label, model_df in subset.groupby("model_label", sort=False):
            ax.plot(
                model_df["t"],
                model_df["token_edit_distance_norm"],
                linewidth=2.2,
                label=model_label,
            )
        ax.set_title(LONG_PAIR_LABELS[pair_id], loc="left", fontsize=12, pad=6)
        ax.set_ylabel("Prefix divergence")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)

    axes[0].legend(loc="upper right", frameon=True)
    axes[-1].set_xlabel("Generated token position")
    fig.suptitle(
        "Output trajectories after tiny prompt perturbations",
        fontsize=18,
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "longprobe_output_trajectory_divergence.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(len(pair_ids), 1, figsize=(11, 9), sharex=True)
    if len(pair_ids) == 1:
        axes = [axes]

    for ax, pair_id in zip(axes, pair_ids, strict=True):
        subset = merged[merged["pair_id"] == pair_id]
        for model_label, model_df in subset.groupby("model_label", sort=False):
            raw_diff = model_df["token_edit_distance_norm"] * model_df["t"]
            ax.plot(
                model_df["t"],
                raw_diff,
                linewidth=2.2,
                label=model_label,
            )
        ax.set_title(LONG_PAIR_LABELS[pair_id], loc="left", fontsize=12, pad=6)
        ax.set_ylabel("Token edits")
        ax.grid(alpha=0.25)

    axes[0].legend(loc="upper left", frameon=True)
    axes[-1].set_xlabel("Generated token position")
    fig.suptitle(
        "Output trajectories after tiny prompt perturbations",
        fontsize=18,
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "longprobe_output_trajectory_token_edits.png", dpi=220)
    plt.close(fig)


def plot_qwen_thinkoff_probe() -> None:
    frames = []
    for label, path in QWEN_THINKOFF_RUNS.items():
        if not Path(path).exists():
            continue
        df = read_curves(path)
        df = df[df["category"].isin(["noop_format", "punctuation", "synonym"])].copy()
        df = df[df["t"] <= 256].copy()
        df["model_label"] = label
        frames.append(df)
    if not frames:
        return

    merged = pd.concat(frames, ignore_index=True)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.2, 4.8),
        gridspec_kw={"width_ratios": [1.55, 1.0]},
    )

    for model_label, model_df in merged.groupby("model_label", sort=False):
        mean_curve = (
            model_df.groupby("t", as_index=False)["token_edit_distance_norm"]
            .mean()
            .sort_values("t")
        )
        axes[0].plot(
            mean_curve["t"],
            mean_curve["token_edit_distance_norm"],
            linewidth=2.4,
            label=model_label,
        )

    axes[0].set_title("Token-prefix divergence", loc="left", fontsize=13, pad=6)
    axes[0].set_xlabel("Generated token position")
    axes[0].set_ylabel("Mean prefix divergence")
    axes[0].set_ylim(-0.02, 0.86)
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="lower right", frameon=True, fontsize=9)

    bootstrap_path = OUT_DIR.parent / "rankings/scaffold_long_wave/small_perturbation_bootstrap.csv"
    if bootstrap_path.exists():
        boot = pd.read_csv(bootstrap_path).set_index("run_label")
        rows = []
        for display_label, run_label in QWEN_THINKOFF_BOOTSTRAP_LABELS.items():
            if run_label not in boot.index:
                continue
            row = boot.loc[run_label]
            rows.append(
                {
                    "display_label": display_label,
                    "mean": float(row["mean"]),
                    "low": float(row["ci95_low"]),
                    "high": float(row["ci95_high"]),
                }
            )
        semantic = pd.DataFrame(rows)
        if not semantic.empty:
            y = range(len(semantic))
            xerr = [
                semantic["mean"] - semantic["low"],
                semantic["high"] - semantic["mean"],
            ]
            axes[1].barh(
                list(y),
                semantic["mean"],
                xerr=xerr,
                color=["#3b6fb6", "#c07b32", "#50955b", "#8f62b3"][: len(semantic)],
                alpha=0.85,
                capsize=3,
            )
            axes[1].set_yticks(list(y), semantic["display_label"])
            axes[1].invert_yaxis()
            axes[1].set_xlim(0, 0.13)
            axes[1].set_xlabel("512-token semantic distance")
            axes[1].set_title("Same runs, semantic metric", loc="left", fontsize=13, pad=6)
            axes[1].grid(axis="x", alpha=0.25)

    fig.tight_layout(pad=1.2)
    fig.savefig(OUT_DIR / "qwen_thinkoff_trajectory_and_semantic.png", dpi=220)
    plt.close(fig)


def plot_quantized_qwen() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, (model_label, runs) in zip(axes, QWEN_QUANT_RUNS.items(), strict=True):
        for precision_label, path in runs.items():
            if not Path(path).exists():
                continue
            df = read_curves(path)
            df = df[df["category"].isin(["punctuation", "synonym"])].copy()
            if df.empty:
                continue
            mean_curve = (
                df.groupby("t", as_index=False)["token_edit_distance_norm"]
                .mean()
                .sort_values("t")
            )
            ax.plot(
                mean_curve["t"],
                mean_curve["token_edit_distance_norm"],
                linewidth=2.4,
                label=precision_label,
            )
        ax.set_title(model_label)
        ax.set_xlabel("Generated token position")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Mean prefix divergence\npunctuation + synonym")
    axes[1].legend(loc="lower right", frameon=True)
    fig.suptitle("Qwen precision sweep: perturbation trajectories", fontsize=17)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "qwen_quantized_output_trajectory_divergence.png", dpi=220)
    plt.close(fig)


def write_branch_summary() -> None:
    rows = []
    for label, path in LONG_RUNS.items():
        if not Path(path).exists():
            continue
        df = read_curves(path)
        for pair_id, pair_df in df.groupby("pair_id"):
            final = pair_df.sort_values("t").iloc[-1]
            rows.append(
                {
                    "model": label,
                    "pair_id": pair_id,
                    "pair": LONG_PAIR_LABELS.get(pair_id, pair_id),
                    "tokens": int(final["t"]),
                    "final_prefix_divergence": float(final["token_edit_distance_norm"]),
                    "common_prefix_tokens": int(final["common_prefix_tokens"]),
                }
            )
    pd.DataFrame(rows).to_csv(OUT_DIR / "longprobe_branch_summary.csv", index=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_long_probe()
    plot_qwen_thinkoff_probe()
    plot_quantized_qwen()
    write_branch_summary()
    print(f"Wrote trajectory figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
