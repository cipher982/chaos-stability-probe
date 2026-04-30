#!/usr/bin/env python3
"""Download and merge completed silent-divergence SageMaker jobs."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import boto3
import botocore.exceptions
import pandas as pd


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
    return {
        "Completed": "completed",
        "InProgress": "in_progress",
        "Failed": "failed",
        "Stopped": "stopped",
        "Stopping": "stopping",
    }.get(status, status.lower())


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
    matches = sorted(extract_dir.glob(f"**/{model_name}_silent_divergence_summary.csv"))
    return matches[0].parent if matches else None


def read_metadata(model_dir: Path) -> dict[str, Any]:
    keys = [
        "model_name",
        "model_id",
        "backend",
        "requested_device",
        "resolved_device",
        "requested_dtype",
        "resolved_dtype",
        "torch_version",
        "transformers_version",
        "git_sha",
        "git_dirty",
    ]
    metadata_path = model_dir / "run_metadata.json"
    if not metadata_path.exists():
        return {
            "runtime_metadata_status": "missing",
            **{f"runtime_{key}": None for key in keys},
        }
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        "runtime_metadata_status": "present",
        **{f"runtime_{key}": metadata.get(key) for key in keys},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=Path("configs/sagemaker_queue_silent_divergence_pilot_v1.json"))
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--artifact-dir", type=Path, default=Path("runs/sagemaker_artifacts"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/rankings/silent_divergence_pilot_v1"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_frames = []
    layer_frames = []
    manifest_rows = []
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
        summary_path = model_dir / f"{job['model']}_silent_divergence_summary.csv"
        layer_path = model_dir / f"{job['model']}_silent_divergence_layers.csv"
        if not summary_path.exists() or not layer_path.exists():
            manifest_rows.append({"job_name": job["job_name"], "model_name": job["model"], "status": "missing_outputs"})
            continue
        summary = pd.read_csv(summary_path)
        layers = pd.read_csv(layer_path)
        metadata = read_metadata(model_dir)
        summary["job_name"] = job["job_name"]
        layers["job_name"] = job["job_name"]
        for key, value in metadata.items():
            summary[key] = value
            layers[key] = value
        summary_frames.append(summary)
        layer_frames.append(layers)
        manifest_rows.append(
            {
                "job_name": job["job_name"],
                "model_name": job["model"],
                "status": "processed",
                "runtime_metadata_status": metadata["runtime_metadata_status"],
            }
        )

    pd.DataFrame(manifest_rows).to_csv(args.out_dir / "job_manifest.csv", index=False)
    if not summary_frames:
        raise SystemExit("No completed silent-divergence jobs were ready")

    summary = pd.concat(summary_frames, ignore_index=True)
    layers = pd.concat(layer_frames, ignore_index=True)
    if "effective_branching_factor_a" in summary and "effective_branching_factor_b" in summary:
        summary["max_effective_branching_factor"] = summary[
            ["effective_branching_factor_a", "effective_branching_factor_b"]
        ].max(axis=1)
    else:
        summary["max_effective_branching_factor"] = pd.NA
    summary.to_csv(args.out_dir / "merged_silent_divergence_summary.csv", index=False)
    layers.to_csv(args.out_dir / "merged_silent_divergence_layers.csv", index=False)

    readout = (
        summary.groupby(["model_name", "pair_id"], dropna=False)
        .agg(
            branch_t=("branch_t", "first"),
            rows=("t", "count"),
            max_js=("js_divergence", "max"),
            max_effective_branching_factor=("max_effective_branching_factor", "max"),
            max_final_hidden=("final_layer_cosine_distance", "max"),
            max_any_hidden=("max_layer_cosine_distance", "max"),
            runtime_metadata_status=("runtime_metadata_status", "first"),
            runtime_backend=("runtime_backend", "first"),
            runtime_model_id=("runtime_model_id", "first"),
            runtime_resolved_device=("runtime_resolved_device", "first"),
            runtime_resolved_dtype=("runtime_resolved_dtype", "first"),
            runtime_torch_version=("runtime_torch_version", "first"),
            runtime_transformers_version=("runtime_transformers_version", "first"),
            runtime_git_sha=("runtime_git_sha", "first"),
            runtime_git_dirty=("runtime_git_dirty", "first"),
        )
        .reset_index()
    )
    readout.to_csv(args.out_dir / "silent_divergence_readout.csv", index=False)
    print(readout.to_string(index=False))
    print(f"Wrote silent-divergence artifacts to {args.out_dir}")


if __name__ == "__main__":
    main()
