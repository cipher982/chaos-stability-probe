#!/usr/bin/env python3
"""Download and process completed token-micro SageMaker jobs."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import boto3


ROOT = Path(__file__).resolve().parents[1]


def load_queue(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text())


def describe(job: dict[str, Any], region: str) -> dict[str, Any] | None:
    profile = job.get("profile", "zh-marketing-preprod-aiengineer")
    sm = boto3.Session(profile_name=profile, region_name=region).client("sagemaker")
    try:
        return sm.describe_training_job(TrainingJobName=job["job_name"])
    except Exception:
        return None


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def find_run_dir(extract_dir: Path, model_name: str) -> Path | None:
    preferred = extract_dir / "runs" / model_name
    if (preferred / "generations.jsonl").exists():
        return preferred
    matches = sorted(extract_dir.glob("**/generations.jsonl"))
    if not matches:
        return None
    return matches[0].parent


def validate_token_micro_run(run_dir: Path) -> None:
    summary_path = run_dir / "summary.csv"
    prompt_tokens_path = run_dir / "prompt_tokens.jsonl"
    skipped_path = run_dir / "skipped_pairs.jsonl"
    if not prompt_tokens_path.exists():
        raise FileNotFoundError(f"{run_dir} missing prompt_tokens.jsonl")
    if not skipped_path.exists():
        print(f"{run_dir} has no skipped_pairs.jsonl; treating as zero skipped token-identical pairs.", flush=True)
    with summary_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"category", "prompt_token_edit_distance"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{summary_path} missing columns: {sorted(missing)}")
        bad = [
            row.get("pair_id", "<unknown>")
            for row in reader
            if row["category"] != "micro_control_identical"
            and int(row["prompt_token_edit_distance"]) <= 0
        ]
    if bad:
        sample = ", ".join(bad[:5])
        raise ValueError(f"{summary_path} has token-identical non-controls: {sample}")


def process_completed(job: dict[str, Any], region: str, artifact_dir: Path, rank_dir: Path) -> bool:
    desc = describe(job, region)
    if not desc or desc["TrainingJobStatus"] != "Completed":
        return False
    name = job["job_name"]
    model_name = job["model"]
    extract_dir = artifact_dir / name
    run_dir = find_run_dir(extract_dir, model_name) if extract_dir.exists() else None
    if run_dir is None:
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
        run_dir = find_run_dir(extract_dir, model_name)
    if run_dir is None:
        print(f"No generations.jsonl found for {name} under {extract_dir}", flush=True)
        return False

    validate_token_micro_run(run_dir)
    out_dir = rank_dir / model_name
    marker = out_dir / "summary.json"
    if marker.exists():
        return False
    run(
        [
            "uv",
            "run",
            "python",
            "scripts/process_micro_sweep.py",
            str(run_dir),
            "--out-dir",
            str(out_dir),
        ]
    )
    return True


def process_completed_safe(
    job: dict[str, Any],
    region: str,
    artifact_dir: Path,
    rank_dir: Path,
    error_dir: Path,
) -> bool:
    try:
        return process_completed(job, region, artifact_dir, rank_dir)
    except Exception as exc:
        error_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "job_name": job.get("job_name"),
            "model": job.get("model"),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        path = error_dir / f"{job.get('job_name', 'unknown')}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Skipping {job.get('job_name')}: {type(exc).__name__}: {exc}", flush=True)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=Path("configs/sagemaker_queue_token_micro_v2.json"))
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--artifact-dir", type=Path, default=Path("runs/sagemaker_artifacts"))
    parser.add_argument("--rank-dir", type=Path, default=Path("runs/rankings/token_micro_v2"))
    parser.add_argument("--error-dir", type=Path, default=None)
    parser.add_argument("--sleep-s", type=int, default=900)
    parser.add_argument("--passes", type=int, default=1, help="Number of passes; 0 means forever.")
    args = parser.parse_args()
    error_dir = args.error_dir or args.rank_dir / "_processing_errors"

    queue = load_queue(args.queue)
    i = 0
    while args.passes == 0 or i < args.passes:
        i += 1
        print(f"Process pass {i} at {time.strftime('%Y-%m-%dT%H:%M:%S%z')}", flush=True)
        processed = 0
        for job in queue:
            if process_completed_safe(job, args.region, args.artifact_dir, args.rank_dir, error_dir):
                processed += 1
        if processed:
            run(
                [
                    "uv",
                    "run",
                    "python",
                    "scripts/summarize_token_micro_v2.py",
                    "--rank-dir",
                    str(args.rank_dir),
                ]
            )
            run(
                [
                    "uv",
                    "run",
                    "python",
                    "scripts/plot_token_micro_v2.py",
                    "--rank-dir",
                    str(args.rank_dir),
                    "--out-dir",
                    "talk/micro_visuals",
                ]
            )
        print(f"Processed {processed} completed jobs", flush=True)
        if args.passes != 0 and i >= args.passes:
            break
        time.sleep(args.sleep_s)


if __name__ == "__main__":
    main()
