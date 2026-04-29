#!/usr/bin/env python3
"""Download and extract a SageMaker model artifact."""

from __future__ import annotations

import argparse
import subprocess
import tarfile
from pathlib import Path
from urllib.parse import urlparse

import boto3


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        raise ValueError(f"Not an s3 URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_name")
    parser.add_argument("--profile", default="zh-marketing-preprod-aiengineer")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/sagemaker_artifacts"))
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args()

    sess = boto3.Session(profile_name=args.profile, region_name=args.region)
    sm = sess.client("sagemaker")
    s3 = sess.client("s3")
    job = sm.describe_training_job(TrainingJobName=args.job_name)
    artifact = job.get("ModelArtifacts", {}).get("S3ModelArtifacts")
    if not artifact:
        raise SystemExit(f"No model artifact for {args.job_name}; status={job['TrainingJobStatus']}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dest = args.out_dir / f"{args.job_name}.tar.gz"
    bucket, key = parse_s3_uri(artifact)
    print(f"Downloading {artifact} -> {dest}")
    s3.download_file(bucket, key, str(dest))

    if args.extract:
        extract_dir = args.out_dir / args.job_name
        extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(dest, "r:gz") as tar:
            tar.extractall(extract_dir, filter="data")
        print(f"Extracted to {extract_dir}")
        subprocess.run(["find", str(extract_dir), "-maxdepth", "4", "-type", "f"], check=False)


if __name__ == "__main__":
    main()

