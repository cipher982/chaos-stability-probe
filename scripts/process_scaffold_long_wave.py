#!/usr/bin/env python3
"""Download and summarize the 512-token scaffold/content SageMaker wave."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SMALL_CATEGORIES = ["noop_format", "punctuation", "synonym"]
DEFAULT_QUEUE = Path("configs/sagemaker_queue.json")
DEFAULT_OUT_DIR = Path("runs/rankings/scaffold_long_wave")


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_registry() -> dict[str, dict[str, Any]]:
    path = Path("configs/models.json")
    if not path.exists():
        return {}
    return {row["name"]: row for row in load_json(path)}


def load_annotation_labels() -> dict[str, str]:
    path = Path("runs/rankings/scaffold_analysis/model_scaffold_annotations.csv")
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["model_name"], df["run_label"]))


def label_for(model_name: str, registry: dict[str, dict[str, Any]], labels: dict[str, str]) -> str:
    if model_name in labels:
        return labels[model_name]
    row = registry.get(model_name, {})
    size = row.get("size")
    family = row.get("family")
    if family and size:
        return f"{family} {size}"
    return model_name


def job_label(job: dict[str, Any], registry: dict[str, dict[str, Any]], labels: dict[str, str]) -> str:
    return job.get("run_label") or label_for(job["model"], registry, labels)


def scaffold_long_jobs(queue_path: Path) -> list[dict[str, Any]]:
    queue = load_json(queue_path)
    jobs = []
    for job in queue:
        if not str(job.get("job_name", "")).startswith("chaos-scaffold-long"):
            continue
        jobs.append(job)
    return jobs


def run_dir(job_name: str, model_name: str) -> Path:
    return Path("runs/sagemaker_artifacts") / job_name / "runs" / model_name


def describe_training_job(job: dict[str, Any]) -> dict[str, Any] | None:
    session = boto3.Session(
        profile_name=job.get("profile", "zh-marketing-preprod-aiengineer"),
        region_name=job.get("region", "us-east-1"),
    )
    sm = session.client(
        "sagemaker",
        config=Config(connect_timeout=3, read_timeout=8, retries={"max_attempts": 1}),
    )
    try:
        return sm.describe_training_job(TrainingJobName=job["job_name"])
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code == "ValidationException":
            return None
        raise


def ensure_artifact(job: dict[str, Any]) -> Path | None:
    job_name = job["job_name"]
    model_name = job["model"]
    root = Path("runs/sagemaker_artifacts") / job_name
    if not root.exists():
        description = describe_training_job(job)
        if description is None:
            return None
        if description.get("TrainingJobStatus") != "Completed":
            return None
        cmd = [
            "uv",
            "run",
            "python",
            "scripts/download_sagemaker_artifact.py",
            job_name,
            "--extract",
            "--profile",
            job.get("profile", "zh-marketing-preprod-aiengineer"),
            "--region",
            job.get("region", "us-east-1"),
        ]
        try:
            run(cmd)
        except subprocess.CalledProcessError:
            return None
    directory = run_dir(job_name, model_name)
    if not directory.exists():
        return None
    if not (directory / "summary.csv").exists() or not (directory / "generations.jsonl").exists():
        return None
    semantic_path = directory / "summary_with_semantic.csv"
    if not semantic_path.exists():
        run(["uv", "run", "python", "scripts/add_semantic_metrics.py", str(directory), "--batch-size", "32"])
    return directory


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, samples: int) -> tuple[float, float]:
    means = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(samples)])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def metadata_rows(jobs: list[dict[str, Any]], ready_dirs: dict[str, Path]) -> pd.DataFrame:
    rows = []
    registry = load_registry()
    labels = load_annotation_labels()
    for job in jobs:
        description = describe_training_job(job)
        status = "NotLaunched" if description is None else description.get("TrainingJobStatus", "")
        secondary_status = "" if description is None else description.get("SecondaryStatus", "")
        model_name = job["model"]
        row = registry.get(model_name, {})
        behavior = row.get("observed_behavior", {})
        rows.append(
            {
                "job_name": job["job_name"],
                "model_name": model_name,
                "run_label": job_label(job, registry, labels),
                "profile": job.get("profile", "zh-marketing-preprod-aiengineer"),
                "instance_type": job.get("instance_type", "ml.g6e.2xlarge"),
                "max_new_tokens": job.get("max_new_tokens"),
                "training_status": status,
                "secondary_status": secondary_status,
                "ready": job["job_name"] in ready_dirs,
                "run_dir": str(ready_dirs.get(job["job_name"], "")),
                "dominant_prefix_kind": behavior.get("dominant_prefix_kind", "unknown"),
                "visible_reasoning_scaffold": behavior.get("visible_reasoning_scaffold"),
                "template_echo": behavior.get("template_echo"),
                "boundary_detection": behavior.get("boundary_detection", "unknown"),
            }
        )
    return pd.DataFrame(rows)


def summarize_ready(merged: pd.DataFrame, out_dir: Path, bootstrap_samples: int, seed: int) -> None:
    small = merged[merged["category"].isin(SMALL_CATEGORIES)].copy()
    rng = np.random.default_rng(seed)
    rows = []
    for label, group in small.groupby("run_label", sort=False):
        values = group["semantic_cosine_distance"].dropna().to_numpy()
        if len(values) == 0:
            continue
        lo, hi = bootstrap_ci(values, rng, bootstrap_samples)
        rows.append(
            {
                "run_label": label,
                "n_pairs": len(values),
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "ci95_low": lo,
                "ci95_high": hi,
                "noop_mean": float(group[group["category"] == "noop_format"]["semantic_cosine_distance"].mean()),
                "punctuation_mean": float(
                    group[group["category"] == "punctuation"]["semantic_cosine_distance"].mean()
                ),
                "synonym_mean": float(group[group["category"] == "synonym"]["semantic_cosine_distance"].mean()),
                "a_token_median": float(group["a_generated_tokens"].median()),
                "b_token_median": float(group["b_generated_tokens"].median()),
                "a_token_max": int(group["a_generated_tokens"].max()),
                "b_token_max": int(group["b_generated_tokens"].max()),
            }
        )
    summary = pd.DataFrame(rows).sort_values("mean").reset_index(drop=True)
    summary.to_csv(out_dir / "small_perturbation_bootstrap.csv", index=False)

    if summary.empty:
        return

    fig, ax = plt.subplots(figsize=(10.5, max(4.5, 0.34 * len(summary))))
    y = np.arange(len(summary))
    xerr = np.vstack([summary["mean"] - summary["ci95_low"], summary["ci95_high"] - summary["mean"]])
    ax.barh(y, summary["mean"], xerr=xerr, color="#3478b9", capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(summary["run_label"])
    ax.invert_yaxis()
    ax.set_xlabel("Mean semantic distance over no-op + punctuation + synonym")
    ax.set_title("512-token scaffold-long sensitivity readout")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "small_perturbation_bootstrap.png", dpi=220)
    plt.close(fig)

    token_stats = (
        merged.groupby(["run_label"], as_index=False)
        .agg(
            n_rows=("run_label", "size"),
            a_min=("a_generated_tokens", "min"),
            a_median=("a_generated_tokens", "median"),
            a_max=("a_generated_tokens", "max"),
            b_min=("b_generated_tokens", "min"),
            b_median=("b_generated_tokens", "median"),
            b_max=("b_generated_tokens", "max"),
        )
        .sort_values("run_label")
    )
    token_stats.to_csv(out_dir / "token_budget_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    registry = load_registry()
    labels = load_annotation_labels()
    jobs = scaffold_long_jobs(args.queue)
    ready_dirs: dict[str, Path] = {}
    frames = []
    for job in jobs:
        directory = ensure_artifact(job)
        if directory is None:
            continue
        ready_dirs[job["job_name"]] = directory
        df = pd.read_csv(directory / "summary_with_semantic.csv")
        model_name = job["model"]
        df["job_name"] = job["job_name"]
        df["run_label"] = job_label(job, registry, labels)
        df["source_profile"] = job.get("profile", "zh-marketing-preprod-aiengineer")
        df["source_instance_type"] = job.get("instance_type", "ml.g6e.2xlarge")
        frames.append(df)

    manifest = metadata_rows(jobs, ready_dirs)
    manifest.to_csv(args.out_dir / "job_manifest.csv", index=False)

    if not frames:
        print("No scaffold-long artifacts are ready yet.")
        print(f"Wrote manifest to {args.out_dir / 'job_manifest.csv'}")
        return

    merged = pd.concat(frames, ignore_index=True)
    merged.to_csv(args.out_dir / "merged_summary.csv", index=False)
    summarize_ready(merged, args.out_dir, args.bootstrap_samples, args.seed)

    ready_count = int(manifest["ready"].sum())
    print(f"Ready scaffold-long jobs: {ready_count}/{len(manifest)}")
    print(manifest[["run_label", "job_name", "profile", "instance_type", "ready"]].to_string(index=False))
    print(f"Wrote scaffold-long artifacts to {args.out_dir}")


if __name__ == "__main__":
    main()
