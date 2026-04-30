#!/usr/bin/env python3
"""Print compact SageMaker status for chaos stability jobs."""

from __future__ import annotations

import argparse
import json

import boto3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="zh-marketing-preprod-aiengineer")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--job-name", default="")
    parser.add_argument("--prefix", default="chaos-stability")
    parser.add_argument("--max-results", type=int, default=10)
    parser.add_argument("--details", action="store_true", help="Describe each listed job and include instance details.")
    args = parser.parse_args()

    sm = boto3.Session(profile_name=args.profile, region_name=args.region).client("sagemaker")
    if args.job_name:
        job = sm.describe_training_job(TrainingJobName=args.job_name)
        print(
            json.dumps(
                {
                    "TrainingJobName": job["TrainingJobName"],
                    "TrainingJobStatus": job["TrainingJobStatus"],
                    "SecondaryStatus": job.get("SecondaryStatus"),
                    "CreationTime": job.get("CreationTime").isoformat() if job.get("CreationTime") else None,
                    "TrainingStartTime": job.get("TrainingStartTime").isoformat()
                    if job.get("TrainingStartTime")
                    else None,
                    "TrainingEndTime": job.get("TrainingEndTime").isoformat()
                    if job.get("TrainingEndTime")
                    else None,
                    "InstanceType": job.get("ResourceConfig", {}).get("InstanceType"),
                    "FailureReason": job.get("FailureReason"),
                    "OutputPath": job.get("OutputDataConfig", {}).get("S3OutputPath"),
                    "ModelArtifacts": job.get("ModelArtifacts", {}).get("S3ModelArtifacts"),
                },
                indent=2,
            )
        )
        return

    resp = sm.list_training_jobs(
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=args.max_results,
        NameContains=args.prefix,
    )
    for summary in resp["TrainingJobSummaries"]:
        if args.details:
            job = sm.describe_training_job(TrainingJobName=summary["TrainingJobName"])
            print(
                summary["TrainingJobName"],
                job["TrainingJobStatus"],
                job.get("SecondaryStatus"),
                job.get("ResourceConfig", {}).get("InstanceType"),
                summary.get("CreationTime"),
                summary.get("TrainingEndTime"),
            )
            continue
        print(
            summary["TrainingJobName"],
            summary["TrainingJobStatus"],
            summary.get("CreationTime"),
            summary.get("TrainingEndTime"),
        )


if __name__ == "__main__":
    main()
