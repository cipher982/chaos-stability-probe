#!/usr/bin/env python3
"""Keep a SageMaker queue moving across profiles and instance-family lanes."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import boto3

from dispatch_sagemaker_queue import ACTIVE_STATUSES, build_launch_cmd


DEFAULT_REGION = "us-east-1"


def load_queue(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise TypeError("Queue must be a JSON list")
    return data


def job_family(instance_type: str) -> str:
    # ml.g6e.2xlarge -> ml.g6e
    parts = instance_type.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else instance_type


def list_jobs(profile: str, region: str) -> dict[str, dict[str, Any]]:
    sm = boto3.Session(profile_name=profile, region_name=region).client("sagemaker")
    jobs: dict[str, dict[str, Any]] = {}
    for status in ["InProgress", "Stopping", "Completed", "Failed", "Stopped"]:
        paginator = sm.get_paginator("list_training_jobs")
        for page in paginator.paginate(StatusEquals=status, SortBy="CreationTime", SortOrder="Descending"):
            for summary in page.get("TrainingJobSummaries", []):
                name = summary["TrainingJobName"]
                if not name.startswith("chaos-"):
                    continue
                info: dict[str, Any] = {"status": summary["TrainingJobStatus"]}
                if summary["TrainingJobStatus"] in ACTIVE_STATUSES:
                    try:
                        desc = sm.describe_training_job(TrainingJobName=name)
                        info["instance_type"] = desc.get("ResourceConfig", {}).get("InstanceType", "")
                    except Exception:
                        info["instance_type"] = ""
                jobs[name] = info
    return jobs


def launch(job: dict[str, Any], dry_run: bool) -> bool:
    cmd = build_launch_cmd(job)
    print("+", " ".join(cmd), flush=True)
    if dry_run:
        return True
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"Launch failed for {job['job_name']}: {exc}", flush=True)
        return False


def pass_once(
    queue: list[dict[str, Any]],
    region: str,
    profile_limits: dict[str, int],
    family_limits: dict[tuple[str, str], int],
    dry_run: bool,
) -> int:
    profiles = sorted({job.get("profile", "zh-marketing-preprod-aiengineer") for job in queue})
    jobs_by_profile = {profile: list_jobs(profile, region) for profile in profiles}
    launched = 0

    active_by_profile: dict[str, int] = {}
    active_by_family: dict[tuple[str, str], int] = defaultdict(int)
    existing_names: set[str] = set()
    for profile, jobs in jobs_by_profile.items():
        active = 0
        for name, info in jobs.items():
            existing_names.add(name)
            if info["status"] not in ACTIVE_STATUSES:
                continue
            active += 1
            instance_type = info.get("instance_type") or ""
            active_by_family[(profile, job_family(instance_type))] += 1
        active_by_profile[profile] = active

    for job in queue:
        name = job["job_name"]
        if name in existing_names:
            continue
        profile = job.get("profile", "zh-marketing-preprod-aiengineer")
        family = job_family(job.get("instance_type", "ml.g6e.2xlarge"))
        profile_limit = profile_limits.get(profile, 1)
        family_limit = family_limits.get((profile, family), profile_limit)
        if active_by_profile.get(profile, 0) >= profile_limit:
            continue
        if active_by_family.get((profile, family), 0) >= family_limit:
            continue
        if launch(job, dry_run=dry_run):
            launched += 1
            existing_names.add(name)
            active_by_profile[profile] = active_by_profile.get(profile, 0) + 1
            active_by_family[(profile, family)] += 1
    print(
        "Active by profile:",
        dict(sorted(active_by_profile.items())),
        "launched:",
        launched,
        flush=True,
    )
    return launched


def parse_profile_limit(values: list[str]) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for value in values:
        profile, raw = value.split("=", 1)
        parsed[profile] = int(raw)
    return parsed


def parse_family_limit(values: list[str]) -> dict[tuple[str, str], int]:
    parsed: dict[tuple[str, str], int] = {}
    for value in values:
        key, raw = value.split("=", 1)
        profile, family = key.split(":", 1)
        parsed[(profile, family)] = int(raw)
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=Path("configs/sagemaker_queue_token_micro.json"))
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--sleep-s", type=int, default=600)
    parser.add_argument("--passes", type=int, default=1, help="Number of supervisor passes; 0 means forever.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--profile-limit",
        action="append",
        default=[],
        help="Profile active-job limit, e.g. zh-marketing-preprod-aiengineer=5",
    )
    parser.add_argument(
        "--family-limit",
        action="append",
        default=[],
        help="Per profile/family limit, e.g. zh-marketing-preprod-aiengineer:ml.g6e=5",
    )
    args = parser.parse_args()

    profile_limits = {
        "zh-marketing-preprod-aiengineer": 5,
        "zh-qa-aiengineer": 1,
        "zh-ml-productionengineer": 1,
        "zh-marketing-productionengineer": 1,
    }
    profile_limits.update(parse_profile_limit(args.profile_limit))
    family_limits = parse_family_limit(args.family_limit)

    queue = load_queue(args.queue)
    i = 0
    while args.passes == 0 or i < args.passes:
        i += 1
        print(f"Supervisor pass {i} at {time.strftime('%Y-%m-%dT%H:%M:%S%z')}", flush=True)
        pass_once(queue, args.region, profile_limits, family_limits, args.dry_run)
        if args.passes != 0 and i >= args.passes:
            break
        time.sleep(args.sleep_s)


if __name__ == "__main__":
    main()
