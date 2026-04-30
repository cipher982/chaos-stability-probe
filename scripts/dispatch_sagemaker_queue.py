#!/usr/bin/env python3
"""Launch queued SageMaker chaos jobs until the active-lane limit is full."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import boto3


DEFAULT_PROFILE = "zh-marketing-preprod-aiengineer"
DEFAULT_REGION = "us-east-1"
ACTIVE_STATUSES = {"InProgress", "Stopping"}


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError("Queue config must be a JSON list")
    return data


def all_chaos_jobs(sm: Any) -> dict[str, str]:
    jobs: dict[str, str] = {}
    for status in ["InProgress", "Completed", "Failed", "Stopped", "Stopping"]:
        paginator = sm.get_paginator("list_training_jobs")
        for page in paginator.paginate(StatusEquals=status, SortBy="CreationTime", SortOrder="Descending"):
            for summary in page.get("TrainingJobSummaries", []):
                name = summary["TrainingJobName"]
                if name.startswith("chaos-"):
                    jobs[name] = summary["TrainingJobStatus"]
    return jobs


def build_launch_cmd(job: dict[str, Any]) -> list[str]:
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/launch_sagemaker_panel.py",
        "--profile",
        job.get("profile", DEFAULT_PROFILE),
        "--region",
        job.get("region", DEFAULT_REGION),
        "--job-name",
        job["job_name"],
        "--entrypoint",
        job.get("entrypoint", "panel"),
        "--prompt-pairs",
        job["prompt_pairs"],
        "--model",
        job["model"],
        "--max-new-tokens",
        str(job.get("max_new_tokens", 128)),
        "--timeout-s",
        str(job.get("timeout_s", 7200)),
        "--max-runtime-s",
        str(job.get("max_runtime_s", 28800)),
        "--instance-type",
        job.get("instance_type", "ml.g6e.2xlarge"),
    ]
    if "bucket" in job:
        cmd.extend(["--bucket", job["bucket"]])
    if "role_arn" in job:
        cmd.extend(["--role-arn", job["role_arn"]])
    if "repeats" in job:
        cmd.extend(["--repeats", str(job["repeats"])])
    for pair_id in job.get("pair_ids", []):
        cmd.extend(["--pair-id", pair_id])
    if job.get("sample"):
        cmd.append("--sample")
        cmd.extend(["--temperature", str(job.get("temperature", 0.7))])
        cmd.extend(["--top-p", str(job.get("top_p", 0.95))])
    if job.get("different_seeds_within_pair"):
        cmd.append("--different-seeds-within-pair")
    if job.get("skip_token_identical_non_controls"):
        cmd.append("--skip-token-identical-non-controls")
    if job.get("skip_hidden"):
        cmd.append("--skip-hidden")
    if "thinking_mode" in job:
        cmd.extend(["--thinking-mode", job["thinking_mode"]])
    if job.get("logit_probe"):
        cmd.append("--logit-probe")
        cmd.extend(["--logit-top-k", str(job.get("logit_top_k", 10))])
        cmd.extend(["--logit-max-steps", str(job.get("logit_max_steps", 128))])
    elif job.get("entrypoint") == "silent_divergence":
        cmd.extend(["--logit-max-steps", str(job.get("logit_max_steps", 64))])
    if job.get("no_tags"):
        cmd.append("--no-tags")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=Path("configs/sagemaker_queue.json"))
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--max-active", type=int, default=5)
    parser.add_argument(
        "--include-cross-account",
        action="store_true",
        help="Also consider queue entries whose profile differs from --profile.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep trying later queue entries if one launch fails, for mixed account/quota queues.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sess = boto3.Session(profile_name=args.profile, region_name=args.region)
    sm = sess.client("sagemaker")
    existing = all_chaos_jobs(sm)
    active = sum(1 for status in existing.values() if status in ACTIVE_STATUSES)
    slots = max(0, args.max_active - active)

    queue = load_json(args.queue)
    profile_queue = [
        job
        for job in queue
        if args.include_cross_account or job.get("profile", args.profile) == args.profile
    ]
    pending = [job for job in profile_queue if job["job_name"] not in existing]
    print(f"Active chaos jobs: {active}/{args.max_active}; open slots: {slots}; queued not launched: {len(pending)}")

    launched = 0
    for job in pending[:slots]:
        cmd = build_launch_cmd(job)
        print("+", " ".join(cmd))
        if not args.dry_run:
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as exc:
                print(f"Launch failed for {job['job_name']}: {exc}")
                if not args.continue_on_error:
                    raise
                continue
        launched += 1

    if launched == 0:
        print("No jobs launched.")
    else:
        print(f"Launched {launched} queued jobs.")


if __name__ == "__main__":
    main()
