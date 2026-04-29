#!/usr/bin/env python3
"""Download and summarize robust logit-probe SageMaker jobs."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOGIT_JOBS = {
    "Qwen3.5 0.8B": ("chaos-logit-robust-qwen35-08b-20260429-001", "qwen35_08b"),
    "Qwen3.5 2B": ("chaos-logit-robust-qwen35-2b-20260429-001", "qwen35_2b"),
    "Qwen3.5 4B": ("chaos-logit-robust-qwen35-4b-20260429-001", "qwen35_4b"),
    "Qwen3.5 9B": ("chaos-logit-robust-qwen35-9b-20260429-001", "qwen35_9b"),
    "Gemma4 E4B it": ("chaos-logit-robust-gemma4-e4b-it-20260429-001", "gemma4_e4b_it"),
    "Gemma4 E4B base": ("chaos-logit-robust-gemma4-e4b-base-20260429-001", "gemma4_e4b_base"),
    "OLMo3 7B": ("chaos-logit-robust-olmo3-20260429-001", "olmo3_7b_instruct"),
    "GPT-2 XL": ("chaos-logit-legacy-gpt2-xl-20260429-001", "gpt2_xl"),
    "OPT 6.7B": ("chaos-logit-legacy-opt-6p7b-20260429-001", "opt_6p7b"),
    "LLaMA1 7B": ("chaos-logit-legacy-llama1-7b-20260429-001", "llama1_7b_huggyllama"),
}

SMALL_CATEGORIES = ["noop_format", "punctuation", "synonym"]
DROP_HEAVY_FIELDS = {"topk_a", "topk_b"}


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
    if not (directory / "logit_probes.jsonl").exists():
        print(f"Skipping {label}: missing logit_probes.jsonl")
        return None
    return directory


def read_logit_rows(path: Path, label: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            for field in DROP_HEAVY_FIELDS:
                row.pop(field, None)
            row["run_label"] = label
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(group: pd.DataFrame) -> dict[str, Any]:
    return {
        "n": len(group),
        "js_mean": float(group["js_divergence"].mean()),
        "js_median": float(group["js_divergence"].median()),
        "kl_a_to_b_mean": float(group["kl_a_to_b"].mean()),
        "kl_b_to_a_mean": float(group["kl_b_to_a"].mean()),
        "top1_flip_rate": float((~group["top1_same"].astype(bool)).mean()),
        "mean_abs_logit_delta": float(group["mean_abs_logit_delta"].mean()),
        "rms_logit_delta": float(group["rms_logit_delta"].mean()),
        "centered_logit_l2": float(group["centered_logit_normalized_l2"].mean()),
        "a_top1_margin_logit": float(group["a_top1_margin_logit"].mean()),
        "b_top1_margin_logit": float(group["b_top1_margin_logit"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/rankings/logit_wave"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for label, (job_name, model_name) in LOGIT_JOBS.items():
        directory = ensure_artifact(label, job_name, model_name)
        if directory is None:
            continue
        frames.append(read_logit_rows(directory / "logit_probes.jsonl", label))

    if not frames:
        raise SystemExit("No logit-wave artifacts were ready")

    merged = pd.concat(frames, ignore_index=True)
    merged.to_csv(args.out_dir / "merged_logit_probes_light.csv", index=False)
    small = merged[merged["category"].isin(SMALL_CATEGORIES)].copy()

    prompt_end = small[(small["t"] == 0) & (small["anchor"] == "prompt_a_generation")].copy()
    prompt_rows = []
    for label, group in prompt_end.groupby("run_label", sort=False):
        prompt_rows.append({"run_label": label, **summarize(group)})
    prompt_summary = pd.DataFrame(prompt_rows).sort_values("js_mean").reset_index(drop=True)
    prompt_summary.to_csv(args.out_dir / "prompt_end_logit_summary.csv", index=False)

    trajectory = small[small["t"] > 0].copy()
    trajectory_rows = []
    for label, group in trajectory.groupby("run_label", sort=False):
        trajectory_rows.append({"run_label": label, **summarize(group)})
    trajectory_summary = pd.DataFrame(trajectory_rows).sort_values("js_mean").reset_index(drop=True)
    trajectory_summary.to_csv(args.out_dir / "teacher_forced_trajectory_logit_summary.csv", index=False)

    by_t = (
        trajectory.groupby(["run_label", "t"], as_index=False, observed=True)
        .agg(js_mean=("js_divergence", "mean"), top1_flip_rate=("top1_same", lambda s: float((~s.astype(bool)).mean())))
        .sort_values(["run_label", "t"])
    )
    by_t.to_csv(args.out_dir / "teacher_forced_js_by_t.csv", index=False)

    if not prompt_summary.empty:
        fig, ax = plt.subplots(figsize=(9.5, 4.8))
        ax.barh(prompt_summary["run_label"], prompt_summary["js_mean"], color="#7948a7")
        ax.invert_yaxis()
        ax.set_xlabel("Mean Jensen-Shannon divergence at prompt end")
        ax.set_title("Next-token distribution shift before decoding")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(args.out_dir / "prompt_end_js_summary.png", dpi=220)
        plt.close(fig)

    if not by_t.empty:
        fig, ax = plt.subplots(figsize=(9.5, 5.2))
        for label, group in by_t.groupby("run_label", sort=False):
            ax.plot(group["t"], group["js_mean"], label=label, linewidth=2)
        ax.set_xlabel("Teacher-forced token position")
        ax.set_ylabel("Mean Jensen-Shannon divergence")
        ax.set_title("Logit divergence along the same continuation")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.out_dir / "teacher_forced_js_by_t.png", dpi=220)
        plt.close(fig)

    print("Prompt-end logit summary")
    print(prompt_summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print()
    print("Teacher-forced trajectory logit summary")
    print(trajectory_summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"Wrote logit-wave artifacts to {args.out_dir}")


if __name__ == "__main__":
    main()
