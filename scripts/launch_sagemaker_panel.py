#!/usr/bin/env python3
"""Package and launch a SageMaker GPU job for the stability probe."""

from __future__ import annotations

import argparse
import json
import tarfile
import tempfile
import time
from pathlib import Path

import boto3


DEFAULT_BUCKET = "636347950933-use1-zmp-preprod-ml-ops-sagemaker-models"
DEFAULT_IMAGE = "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.7.1-gpu-py312"
DEFAULT_PROFILE = "zh-marketing-preprod-aiengineer"
DEFAULT_REGION = "us-east-1"
DEFAULT_ROLE = "arn:aws:iam::636347950933:role/zmp-preprod-use1_sagemaker_0"


EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "runs",
}


def should_include(path: Path) -> bool:
    parts = set(path.parts)
    if parts & EXCLUDE_DIRS:
        return False
    if path.name == "raw_initial_discussion.txt":
        return False
    if path.suffix in {".pyc", ".DS_Store"}:
        return False
    return True


def make_source_tar(repo_root: Path, output_path: Path) -> None:
    with tarfile.open(output_path, "w:gz") as tar:
        for path in repo_root.rglob("*"):
            if not path.is_file() or not should_include(path.relative_to(repo_root)):
                continue
            tar.add(path, arcname=path.relative_to(repo_root))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--role-arn", default=DEFAULT_ROLE)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--instance-type", default="ml.g6e.2xlarge")
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument("--volume-size-gb", type=int, default=200)
    parser.add_argument("--max-runtime-s", type=int, default=8 * 60 * 60)
    parser.add_argument("--job-name", default="")
    parser.add_argument("--entrypoint", choices=["panel", "silent_divergence", "activation_patch"], default="panel")
    parser.add_argument("--model", action="append", default=[], help="Model name or ID. Repeatable.")
    parser.add_argument("--prompt-pairs", default="configs/prompt_pairs.json")
    parser.add_argument("--pair-id", action="append", dest="pair_ids", default=[])
    parser.add_argument("--targets-csv", default="")
    parser.add_argument("--targets-json", default="")
    parser.add_argument("--limit-pairs", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout-s", type=int, default=3600)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--different-seeds-within-pair", action="store_true")
    parser.add_argument("--skip-token-identical-non-controls", action="store_true")
    parser.add_argument("--skip-hidden", action="store_true")
    parser.add_argument("--logit-probe", action="store_true")
    parser.add_argument("--logit-top-k", type=int, default=20)
    parser.add_argument("--logit-max-steps", type=int, default=128)
    parser.add_argument("--thinking-mode", choices=["default", "enabled", "disabled"], default="default")
    parser.add_argument("--positions", choices=["final", "changed-final", "aligned", "all"], default="aligned")
    parser.add_argument("--no-tags", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path.cwd()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    job_name = args.job_name or f"chaos-stability-{timestamp}"
    s3_prefix = f"chaos-stability/{job_name}"
    source_s3 = f"s3://{args.bucket}/{s3_prefix}/source.tar.gz"
    output_s3 = f"s3://{args.bucket}/{s3_prefix}/output"

    run_args: list[str] = [
        "--timeout-s",
        str(args.timeout_s),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--prompt-pairs",
        args.prompt_pairs,
        "--device",
        "cuda",
        "--dtype",
        "auto",
        "--thinking-mode",
        args.thinking_mode,
    ]
    if args.entrypoint == "panel":
        run_args.extend(["--repeats", str(args.repeats)])
    for model in args.model:
        run_args.extend(["--model", model])
    if args.limit_pairs:
        run_args.extend(["--limit-pairs", str(args.limit_pairs)])
    if args.entrypoint == "panel" and args.sample:
        run_args.append("--sample")
        run_args.extend(["--temperature", str(args.temperature), "--top-p", str(args.top_p)])
    if args.entrypoint == "panel" and args.different_seeds_within_pair:
        run_args.append("--different-seeds-within-pair")
    if args.entrypoint == "panel" and args.skip_token_identical_non_controls:
        run_args.append("--skip-token-identical-non-controls")
    if args.entrypoint == "panel" and args.skip_hidden:
        run_args.append("--skip-hidden")
    if args.entrypoint == "panel" and args.logit_probe:
        run_args.extend(
            [
                "--logit-probe",
                "--logit-top-k",
                str(args.logit_top_k),
                "--logit-max-steps",
                str(args.logit_max_steps),
            ]
        )
    if args.entrypoint == "silent_divergence":
        run_args.extend(["--logit-max-steps", str(args.logit_max_steps)])
        for pair_id in args.pair_ids:
            run_args.extend(["--pair-id", pair_id])
    if args.entrypoint == "activation_patch":
        run_args.extend(["--positions", args.positions])
        for pair_id in args.pair_ids:
            run_args.extend(["--pair-id", pair_id])
        if args.targets_csv:
            run_args.extend(["--targets-csv", args.targets_csv])
        if args.targets_json:
            run_args.extend(["--targets-json", args.targets_json])

    sess = boto3.Session(profile_name=args.profile, region_name=args.region)
    s3 = sess.client("s3")
    sm = sess.client("sagemaker")

    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = Path(tmpdir) / "source.tar.gz"
        make_source_tar(repo_root, tar_path)
        print(f"Packaged {tar_path} ({tar_path.stat().st_size / 1024 / 1024:.2f} MiB)")
        print(f"Uploading {source_s3}")
        if not args.dry_run:
            s3.upload_file(str(tar_path), args.bucket, f"{s3_prefix}/source.tar.gz")

    request = {
        "TrainingJobName": job_name,
        "AlgorithmSpecification": {
            "TrainingImage": args.image,
            "TrainingInputMode": "File",
        },
        "RoleArn": args.role_arn,
        "OutputDataConfig": {"S3OutputPath": output_s3},
        "ResourceConfig": {
            "InstanceType": args.instance_type,
            "InstanceCount": args.instance_count,
            "VolumeSizeInGB": args.volume_size_gb,
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": args.max_runtime_s},
        "HyperParameters": {
            "sagemaker_program": "sagemaker_entry.py",
            "sagemaker_submit_directory": source_s3,
            "sagemaker_region": args.region,
            "sagemaker_container_log_level": "20",
        },
        "Environment": {
            "CHAOS_RUN_ARGS": json.dumps(run_args),
            "CHAOS_ENTRYPOINT": args.entrypoint,
        },
        "EnableNetworkIsolation": False,
    }
    if not args.no_tags:
        request["Tags"] = [
            {"Key": "Project", "Value": "chaos-stability"},
            {"Key": "Purpose", "Value": "llm-stability-probe"},
        ]

    print(json.dumps(request, indent=2))
    if args.dry_run:
        return

    sm.create_training_job(**request)
    print(f"Launched {job_name}")
    print(f"Output: {output_s3}")


if __name__ == "__main__":
    main()
