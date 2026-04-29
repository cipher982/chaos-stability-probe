#!/usr/bin/env python3
"""Download, score, and summarize the robust prompt-ladder SageMaker wave."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROBUST_JOBS = {
    "Qwen3.5 0.8B": ("chaos-robust-qwen35-08b-20260429-002", "qwen35_08b"),
    "Qwen3.5 2B": ("chaos-robust-qwen35-2b-20260429-002", "qwen35_2b"),
    "Qwen3.5 4B": ("chaos-robust-qwen35-4b-20260429-002", "qwen35_4b"),
    "Qwen3.5 9B": ("chaos-robust-qwen35-9b-20260429-002", "qwen35_9b"),
    "Gemma4 E4B it": ("chaos-robust-gemma4-e4b-it-20260429-002", "gemma4_e4b_it"),
}

SMALL_CATEGORIES = ["noop_format", "punctuation", "synonym"]
PLANNED_CONTRASTS = [
    ("Qwen3.5 4B", "Qwen3.5 0.8B"),
    ("Qwen3.5 4B", "Qwen3.5 2B"),
    ("Qwen3.5 4B", "Qwen3.5 9B"),
    ("Qwen3.5 2B", "Qwen3.5 0.8B"),
    ("Gemma4 E4B it", "Qwen3.5 4B"),
]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_dir(job_name: str, model_name: str) -> Path:
    return Path("runs/sagemaker_artifacts") / job_name / "runs" / model_name


def ensure_artifact(label: str, job_name: str, model_name: str) -> Path | None:
    root = Path("runs/sagemaker_artifacts") / job_name
    if not root.exists():
        try:
            run(["uv", "run", "python", "scripts/download_sagemaker_artifact.py", job_name, "--extract"])
        except subprocess.CalledProcessError:
            print(f"Skipping {label}: artifact not ready")
            return None
    directory = run_dir(job_name, model_name)
    if not directory.exists():
        print(f"Skipping {label}: missing run dir {directory}")
        return None
    semantic_path = directory / "summary_with_semantic.csv"
    if not semantic_path.exists():
        run(["uv", "run", "python", "scripts/add_semantic_metrics.py", str(directory)])
    return directory


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, samples: int) -> tuple[float, float]:
    means = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(samples)])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_permutation(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, samples: int) -> tuple[float, float]:
    diff = a - b
    observed = float(diff.mean())
    signs = rng.choice(np.array([-1.0, 1.0]), size=(samples, len(diff)))
    permuted = (signs * diff).mean(axis=1)
    p_value = float((np.abs(permuted) >= abs(observed)).mean())
    return observed, p_value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/rankings/robust_wave"))
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for label, (job_name, model_name) in ROBUST_JOBS.items():
        directory = ensure_artifact(label, job_name, model_name)
        if directory is None:
            continue
        df = pd.read_csv(directory / "summary_with_semantic.csv")
        df["run_label"] = label
        frames.append(df)

    if not frames:
        raise SystemExit("No robust-wave artifacts were ready")

    merged = pd.concat(frames, ignore_index=True)
    merged.to_csv(args.out_dir / "merged_summary.csv", index=False)
    small = merged[merged["category"].isin(SMALL_CATEGORIES)].copy()

    rows = []
    rng = np.random.default_rng(args.seed)
    for label, group in small.groupby("run_label", sort=False):
        values = group["semantic_cosine_distance"].to_numpy()
        lo, hi = bootstrap_ci(values, rng, args.bootstrap_samples)
        rows.append(
            {
                "run_label": label,
                "n": len(values),
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "ci95_low": lo,
                "ci95_high": hi,
                "noop_mean": float(group[group["category"] == "noop_format"]["semantic_cosine_distance"].mean()),
                "punctuation_mean": float(
                    group[group["category"] == "punctuation"]["semantic_cosine_distance"].mean()
                ),
                "synonym_mean": float(group[group["category"] == "synonym"]["semantic_cosine_distance"].mean()),
            }
        )

    result = pd.DataFrame(rows).sort_values("mean").reset_index(drop=True)
    result.to_csv(args.out_dir / "small_perturbation_bootstrap.csv", index=False)

    contrast_rows = []
    keyed = small.pivot_table(
        index=["pair_id", "category"],
        columns="run_label",
        values="semantic_cosine_distance",
        aggfunc="mean",
        observed=True,
    )
    for left, right in PLANNED_CONTRASTS:
        if left not in keyed.columns or right not in keyed.columns:
            continue
        pairwise = keyed[[left, right]].dropna()
        if pairwise.empty:
            continue
        observed, p_value = paired_permutation(
            pairwise[left].to_numpy(),
            pairwise[right].to_numpy(),
            rng,
            args.bootstrap_samples,
        )
        contrast_rows.append(
            {
                "left": left,
                "right": right,
                "n_pairs": len(pairwise),
                "mean_difference_left_minus_right": observed,
                "paired_permutation_p_two_sided": p_value,
                "left_mean": float(pairwise[left].mean()),
                "right_mean": float(pairwise[right].mean()),
            }
        )
    contrasts = pd.DataFrame(contrast_rows)
    contrasts.to_csv(args.out_dir / "paired_permutation_tests.csv", index=False)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    y = np.arange(len(result))
    xerr = np.vstack([result["mean"] - result["ci95_low"], result["ci95_high"] - result["mean"]])
    ax.barh(y, result["mean"], xerr=xerr, color="#2f6fbb", capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(result["run_label"])
    ax.invert_yaxis()
    ax.set_xlabel("Mean semantic distance over no-op + punctuation + synonym")
    ax.set_title("Robust prompt-ladder sensitivity")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out_dir / "small_perturbation_bootstrap.png", dpi=220)
    plt.close(fig)

    print(result.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if not contrasts.empty:
        print()
        print(contrasts.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Wrote robust-wave artifacts to {args.out_dir}")


if __name__ == "__main__":
    main()
