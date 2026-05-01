#!/usr/bin/env python3
"""Download and merge completed activation-patching SageMaker jobs."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import boto3
import botocore.exceptions
import pandas as pd


ACTIVE_STATUSES = {
    "InProgress": "in_progress",
    "Stopping": "stopping",
    "Stopped": "stopped",
    "Failed": "failed",
}


def load_queue(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def describe(job: dict[str, Any], region: str) -> dict[str, Any] | None:
    profile = job.get("profile", "zh-marketing-preprod-aiengineer")
    sm = boto3.Session(profile_name=profile, region_name=region).client("sagemaker")
    try:
        return sm.describe_training_job(TrainingJobName=job["job_name"])
    except sm.exceptions.ResourceNotFound:
        return None
    except botocore.exceptions.ClientError as exc:
        error = exc.response.get("Error", {})
        if error.get("Code") == "ValidationException" and "not found" in error.get("Message", "").lower():
            return None
        raise RuntimeError(f"Could not describe {job['job_name']} with profile {profile}: {exc}") from exc
    except botocore.exceptions.BotoCoreError as exc:
        raise RuntimeError(f"Could not describe {job['job_name']} with profile {profile}: {exc}") from exc


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def normalize_training_status(status: str) -> str:
    return ACTIVE_STATUSES.get(status, status.lower())


def ensure_artifact(job: dict[str, Any], region: str, artifact_dir: Path) -> tuple[str, Path | None, dict[str, Any] | None]:
    desc = describe(job, region)
    if not desc:
        return "not_found", None, None
    training_status = desc["TrainingJobStatus"]
    if training_status != "Completed":
        return normalize_training_status(training_status), None, desc
    name = job["job_name"]
    extract_dir = artifact_dir / name
    if not extract_dir.exists():
        run(
            [
                "uv",
                "run",
                "python",
                "scripts/download_sagemaker_artifact.py",
                name,
                "--profile",
                job.get("profile", "zh-marketing-preprod-aiengineer"),
                "--region",
                region,
                "--out-dir",
                str(artifact_dir),
                "--extract",
            ]
        )
    return "completed", extract_dir, desc


def find_model_dir(extract_dir: Path, model_name: str) -> Path | None:
    preferred = extract_dir / "runs" / model_name
    if preferred.exists():
        return preferred
    matches = sorted(extract_dir.glob(f"**/{model_name}__*.csv"))
    return matches[0].parent if matches else None


def metadata_for(csv_path: Path) -> dict[str, Any]:
    metadata_path = csv_path.with_suffix(".json")
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def position_class(label: str) -> str:
    if label == "prompt_lcp_token":
        return "prompt_lcp"
    if label == "final_context_token":
        return "final_context"
    if label.startswith("aligned_generated_prefix"):
        return "generated_prefix"
    if label.startswith("aligned_prompt_pos"):
        return "aligned_prompt"
    return "other"


def summarize_patch_csv(path: Path, job: dict[str, Any]) -> dict[str, Any]:
    df = pd.read_csv(path)
    meta = metadata_for(path)
    finite = df[df["rescue_fraction"].notna()].copy()
    best = df.iloc[0] if finite.empty else finite.sort_values("rescue_fraction", ascending=False).iloc[0]

    row: dict[str, Any] = {
        "job_name": job["job_name"],
        "queue_model": job["model"],
        "profile": job.get("profile"),
        "instance_type": job.get("instance_type"),
        "csv_path": str(path),
        "model_name": best["model_name"],
        "pair_id": best["pair_id"],
        "category": best["category"],
        "first_diff_token": int(best["first_diff_token"]),
        "a_branch_token": best["a_branch_token"],
        "b_branch_token": best["b_branch_token"],
        "clean_metric_a_minus_b": float(best["clean_metric_a_minus_b"]),
        "corrupt_metric_a_minus_b": float(best["corrupt_metric_a_minus_b"]),
        "corrupt_replay_matches_b_branch": meta.get("corrupt_replay_matches_b_branch"),
        "best_position_label": best["position_label"],
        "best_position_class": position_class(str(best["position_label"])),
        "best_layer": int(best["layer"]),
        "best_rescue_fraction": float(best["rescue_fraction"]) if pd.notna(best["rescue_fraction"]) else None,
        "best_metric_a_minus_b": float(best["metric_a_minus_b"]),
        "best_top1_token": best["top1_token"],
    }

    for label, group in df.groupby("position_label"):
        good = group[group["rescue_fraction"].notna()]
        if good.empty:
            continue
        label_best = good.sort_values("rescue_fraction", ascending=False).iloc[0]
        prefix = str(label)
        row[f"{prefix}_best_layer"] = int(label_best["layer"])
        row[f"{prefix}_best_rescue_fraction"] = float(label_best["rescue_fraction"])
        row[f"{prefix}_best_top1_token"] = label_best["top1_token"]
    return row


def summarize_model(summary: pd.DataFrame) -> pd.DataFrame:
    work = summary.copy()
    work["replayable"] = work["corrupt_replay_matches_b_branch"].fillna(False).astype(bool)
    work["finite_rescue"] = work["best_rescue_fraction"].notna()
    work["strong_rescue"] = work["best_rescue_fraction"].fillna(float("-inf")) >= 0.5
    work["full_or_overshoot_rescue"] = work["best_rescue_fraction"].fillna(float("-inf")) >= 1.0
    work["replayable_strong_rescue"] = work["replayable"] & work["strong_rescue"]
    work["replayable_full_or_overshoot_rescue"] = work["replayable"] & work["full_or_overshoot_rescue"]
    for cls in ["prompt_lcp", "final_context", "generated_prefix", "aligned_prompt"]:
        work[f"best_at_{cls}"] = work["best_position_class"] == cls

    return (
        work.groupby("model_name", dropna=False)
        .agg(
            pairs=("pair_id", "count"),
            replayable_pairs=("replayable", "sum"),
            finite_rescue_pairs=("finite_rescue", "sum"),
            mean_best_rescue=("best_rescue_fraction", "mean"),
            median_best_rescue=("best_rescue_fraction", "median"),
            max_best_rescue=("best_rescue_fraction", "max"),
            strong_rescue_pairs=("strong_rescue", "sum"),
            full_or_overshoot_pairs=("full_or_overshoot_rescue", "sum"),
            replayable_strong_rescue_pairs=("replayable_strong_rescue", "sum"),
            replayable_full_or_overshoot_pairs=("replayable_full_or_overshoot_rescue", "sum"),
            prompt_lcp_best_pairs=("best_at_prompt_lcp", "sum"),
            final_context_best_pairs=("best_at_final_context", "sum"),
            generated_prefix_best_pairs=("best_at_generated_prefix", "sum"),
            aligned_prompt_best_pairs=("best_at_aligned_prompt", "sum"),
        )
        .reset_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=Path("configs/sagemaker_queue_activation_patch_v1.json"))
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--artifact-dir", type=Path, default=Path("runs/sagemaker_artifacts"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/rankings/activation_patch_v1"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for job in load_queue(args.queue):
        try:
            artifact_status, extract_dir, desc = ensure_artifact(job, args.region, args.artifact_dir)
        except Exception as exc:
            manifest_rows.append(
                {
                    "job_name": job["job_name"],
                    "model_name": job["model"],
                    "status": "describe_or_download_error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue
        if extract_dir is None:
            row = {"job_name": job["job_name"], "model_name": job["model"], "status": artifact_status}
            if desc:
                row["secondary_status"] = desc.get("SecondaryStatus")
                row["failure_reason"] = desc.get("FailureReason")
            manifest_rows.append(row)
            continue
        model_dir = find_model_dir(extract_dir, job["model"])
        if model_dir is None:
            manifest_rows.append({"job_name": job["job_name"], "model_name": job["model"], "status": "missing_model_dir"})
            continue
        paths = sorted(
            path
            for path in model_dir.glob("*.csv")
            if path.name != "patch_summary.csv" and not path.name.startswith("selected_patch_targets")
        )
        if not paths:
            manifest_rows.append({"job_name": job["job_name"], "model_name": job["model"], "status": "missing_outputs"})
            continue
        rows.extend(summarize_patch_csv(path, job) for path in paths)
        manifest_rows.append(
            {
                "job_name": job["job_name"],
                "model_name": job["model"],
                "status": "processed",
                "cases": len(paths),
            }
        )

    pd.DataFrame(manifest_rows).to_csv(args.out_dir / "job_manifest.csv", index=False)
    if not rows:
        raise SystemExit("No completed activation-patching jobs were ready")

    summary = pd.DataFrame(rows).sort_values(
        ["model_name", "corrupt_replay_matches_b_branch", "best_rescue_fraction"],
        ascending=[True, False, False],
        na_position="last",
    )
    summary.to_csv(args.out_dir / "merged_patch_summary.csv", index=False)
    ranked = summary.sort_values("best_rescue_fraction", ascending=False, na_position="last")
    ranked.to_csv(args.out_dir / "patch_cases_ranked.csv", index=False)
    model_summary = summarize_model(summary)
    model_summary.to_csv(args.out_dir / "model_summary.csv", index=False)

    print(model_summary.to_string(index=False))
    print()
    print(
        ranked[
            [
                "model_name",
                "pair_id",
                "first_diff_token",
                "corrupt_replay_matches_b_branch",
                "best_position_class",
                "best_layer",
                "best_rescue_fraction",
            ]
        ].to_string(index=False)
    )
    print(f"Wrote activation-patching artifacts to {args.out_dir}")


if __name__ == "__main__":
    main()
