#!/usr/bin/env python3
"""Plot compact E09 trajectory-event summary figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TARGET_LABELS = {
    "at_branch": "At branch",
    "branch_within_1": "Decision window <=1",
    "pre_branch_within_1": "Strict pre-branch <=1",
}

FEATURE_LABELS = {
    "min_margin_logit": "Low margin",
    "js_divergence": "JS divergence",
    "centered_logit_normalized_l2": "Centered logit L2",
    "max_effective_branching_factor": "Max BF",
}

MODEL_LABELS = {
    "qwen35_08b": "Qwen 0.8B",
    "qwen35_2b": "Qwen 2B",
    "qwen35_4b": "Qwen 4B",
    "qwen35_9b": "Qwen 9B",
    "gemma4_e2b_it": "Gemma E2B IT",
    "gemma4_e2b_base": "Gemma E2B base",
    "gemma4_e4b_it": "Gemma E4B IT",
    "gemma4_e4b_base": "Gemma E4B base",
}


def style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="x", alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_auroc(artifact_dir: Path, out_dir: Path) -> None:
    full = pd.read_csv(artifact_dir / "branch_prediction" / "branch_prediction_auc.csv")
    long = pd.read_csv(artifact_dir / "branch_prediction_long_prefix" / "branch_prediction_auc.csv")
    rows = []
    for scope, df in [("All branches", full), ("Branch t >= 5", long)]:
        subset = df[
            (df["group"] == "all")
            & (df["target"].isin(TARGET_LABELS))
            & (df["feature"].isin(FEATURE_LABELS))
        ].copy()
        subset["scope"] = scope
        rows.append(subset)
    data = pd.concat(rows, ignore_index=True)
    data["label"] = data["target"].map(TARGET_LABELS) + " / " + data["feature"].map(FEATURE_LABELS)
    order = [
        ("At branch", "Low margin"),
        ("At branch", "JS divergence"),
        ("Decision window <=1", "JS divergence"),
        ("Decision window <=1", "Low margin"),
        ("Strict pre-branch <=1", "Centered logit L2"),
        ("Strict pre-branch <=1", "JS divergence"),
    ]
    labels = [f"{target} / {feature}" for target, feature in order]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), sharey=True)
    for ax, scope in zip(axes, ["All branches", "Branch t >= 5"], strict=True):
        scope_data = data[data["scope"] == scope].set_index("label").reindex(labels).dropna(subset=["auroc"])
        y = range(len(scope_data))
        xerr = [
            scope_data["auroc"] - scope_data["auroc_ci_low"],
            scope_data["auroc_ci_high"] - scope_data["auroc"],
        ]
        ax.errorbar(
            scope_data["auroc"],
            list(y),
            xerr=xerr,
            fmt="o",
            markersize=6,
            capsize=3,
            color="#2f5d8c",
            ecolor="#7d98b5",
        )
        ax.axvline(0.5, color="#9a9a9a", linestyle="--", linewidth=1.2)
        ax.set_xlim(0.45, 1.0)
        ax.set_title(scope, loc="left", fontsize=13)
        ax.set_xlabel("AUROC")
        ax.set_yticks(list(y), scope_data.index)
        ax.invert_yaxis()
        style_axes(ax)
    fig.suptitle("E09 branch-event classifiers", fontsize=17, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "e09_branch_classifier_aurocs.png", dpi=220)
    plt.close(fig)


def plot_model_profiles(artifact_dir: Path, out_dir: Path) -> None:
    data = pd.read_csv(artifact_dir / "model_comparison" / "model_branch_summary.csv")
    data["label"] = data["model_name"].map(MODEL_LABELS).fillna(data["model_name"])
    data = data.set_index("model_name").loc[list(MODEL_LABELS)].reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1.2, 1.0]})
    y = range(len(data))
    axes[0].barh(y, data["immediate_visible_branch_rate"], label="Immediate", color="#b85c38")
    axes[0].barh(
        y,
        data["silent_logit_divergence_rate"],
        left=data["immediate_visible_branch_rate"],
        label="Delayed visible branch",
        color="#3b6f9e",
    )
    axes[0].barh(
        y,
        data["no_visible_branch_rate"],
        left=data["immediate_visible_branch_rate"] + data["silent_logit_divergence_rate"],
        label="No visible branch",
        color="#b7b7b7",
    )
    axes[0].set_yticks(list(y), data["label"])
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("Share of non-control token-certified pairs")
    axes[0].set_title("Branch profile", loc="left", fontsize=13)
    axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    style_axes(axes[0])

    axes[1].scatter(data["mean_branch_t"], data["mean_semantic"], s=70, color="#2f5d8c")
    for _, row in data.iterrows():
        axes[1].annotate(
            row["label"],
            (row["mean_branch_t"], row["mean_semantic"]),
            xytext=(5, 2),
            textcoords="offset points",
            fontsize=9,
        )
    axes[1].set_xlabel("Mean visible branch token")
    axes[1].set_ylabel("Mean semantic distance")
    axes[1].set_title("Timing vs downstream divergence", loc="left", fontsize=13)
    style_axes(axes[1])

    fig.suptitle("E09 model branch profiles", fontsize=17, y=0.98)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(out_dir / "e09_model_branch_profiles.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("runs/trajectory_artifacts/logit_token_cert_v1"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("runs/figures/e09_summary"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_auroc(args.artifact_dir, args.out_dir)
    plot_model_profiles(args.artifact_dir, args.out_dir)
    for path in sorted(args.out_dir.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
