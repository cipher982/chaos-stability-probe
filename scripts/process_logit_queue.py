#!/usr/bin/env python3
"""Download and summarize completed logit-probe queue jobs."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

import boto3
import pandas as pd


DROP_HEAVY_FIELDS = {"topk_a", "topk_b"}


def load_queue(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def ensure_artifact(job: dict[str, Any], region: str, artifact_dir: Path) -> Path | None:
    desc = describe(job, region)
    if not desc or desc["TrainingJobStatus"] != "Completed":
        return None
    name = job["job_name"]
    model = job["model"]
    extract_dir = artifact_dir / name
    run_dir = find_run_dir(extract_dir, model) if extract_dir.exists() else None
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
        run_dir = find_run_dir(extract_dir, model)
    if run_dir is None or not (run_dir / "logit_probes.jsonl").exists():
        return None
    return run_dir


def ensure_semantic_summary(run_dir: Path, out_dir: Path) -> None:
    summary_path = run_dir / "summary.csv"
    prompt_tokens_path = run_dir / "prompt_tokens.jsonl"
    if not summary_path.exists():
        raise FileNotFoundError(f"{run_dir} missing summary.csv")
    if not prompt_tokens_path.exists():
        raise FileNotFoundError(f"{run_dir} missing prompt_tokens.jsonl")
    with summary_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "prompt_token_edit_distance" not in (reader.fieldnames or []):
            raise ValueError(f"{summary_path} missing prompt_token_edit_distance")
        bad = [
            row.get("pair_id", "<unknown>")
            for row in reader
            if row["category"] != "micro_control_identical"
            and int(row["prompt_token_edit_distance"]) <= 0
        ]
    if bad:
        raise ValueError(f"{summary_path} has token-identical non-controls: {', '.join(bad[:5])}")
    marker = out_dir / "summary.json"
    if not marker.exists():
        run(["uv", "run", "python", "scripts/process_micro_sweep.py", str(run_dir), "--out-dir", str(out_dir)])


def read_logit_rows(path: Path, job: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            for field in DROP_HEAVY_FIELDS:
                row.pop(field, None)
            row["job_name"] = job["job_name"]
            row["queue_model"] = job["model"]
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_logits(group: pd.DataFrame) -> dict[str, Any]:
    top1_same = group["top1_same"].astype(bool)
    return {
        "n_rows": int(len(group)),
        "js_mean": float(group["js_divergence"].mean()),
        "js_median": float(group["js_divergence"].median()),
        "top1_flip_rate": float((~top1_same).mean()),
        "mean_abs_logit_delta": float(group["mean_abs_logit_delta"].mean()),
        "centered_logit_l2": float(group["centered_logit_normalized_l2"].mean()),
        "mean_top1_margin_logit": float(
            pd.concat([group["a_top1_margin_logit"], group["b_top1_margin_logit"]]).mean()
        ),
        "a_top1_prob": float(group["a_top1_prob"].mean()),
        "b_top1_prob": float(group["b_top1_prob"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=Path("configs/sagemaker_queue_logit_token_cert_v1.json"))
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--artifact-dir", type=Path, default=Path("runs/sagemaker_artifacts"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/rankings/logit_token_cert_v1"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    semantic_rows = []
    for job in load_queue(args.queue):
        run_dir = ensure_artifact(job, args.region, args.artifact_dir)
        if run_dir is None:
            continue
        model_out = args.out_dir / job["model"]
        ensure_semantic_summary(run_dir, model_out)
        frames.append(read_logit_rows(run_dir / "logit_probes.jsonl", job))
        semantic = json.loads((model_out / "summary.json").read_text(encoding="utf-8"))
        semantic_rows.append({"model_name": job["model"], "job_name": job["job_name"], **semantic})

    if not frames:
        raise SystemExit("No completed logit jobs were ready")

    logits = pd.concat(frames, ignore_index=True)
    logits.to_csv(args.out_dir / "merged_logit_probes_light.csv", index=False)

    prompt_end = logits[(logits["t"] == 0) & (logits["anchor"] == "prompt_a_generation")].copy()
    prompt_rows = []
    for model, group in prompt_end.groupby("queue_model", sort=False):
        prompt_rows.append({"model_name": model, **summarize_logits(group)})
    prompt_summary = pd.DataFrame(prompt_rows).sort_values("js_mean").reset_index(drop=True)
    prompt_summary.to_csv(args.out_dir / "prompt_end_logit_summary.csv", index=False)

    trajectory = logits[logits["t"] > 0].copy()
    trajectory_rows = []
    for model, group in trajectory.groupby("queue_model", sort=False):
        trajectory_rows.append({"model_name": model, **summarize_logits(group)})
    trajectory_summary = pd.DataFrame(trajectory_rows).sort_values("js_mean").reset_index(drop=True)
    trajectory_summary.to_csv(args.out_dir / "teacher_forced_trajectory_logit_summary.csv", index=False)

    semantic_summary = pd.DataFrame(semantic_rows)
    semantic_summary.to_csv(args.out_dir / "semantic_summary.csv", index=False)
    joined = semantic_summary.merge(prompt_summary, on="model_name", how="inner")
    joined.to_csv(args.out_dir / "semantic_vs_prompt_end_logits.csv", index=False)

    corr_rows = []
    for col in ["js_mean", "top1_flip_rate", "mean_top1_margin_logit", "a_top1_prob", "centered_logit_l2"]:
        if col in joined and len(joined) >= 3:
            corr_rows.append(
                {
                    "x": col,
                    "y": "micro_semantic_mean",
                    "pearson_corr": float(joined[col].corr(joined["micro_semantic_mean"])),
                    "n_models": int(len(joined)),
                }
            )
    pd.DataFrame(corr_rows).to_csv(args.out_dir / "semantic_logit_correlations.csv", index=False)
    print(prompt_summary.to_string(index=False))
    print(f"Wrote logit queue artifacts to {args.out_dir}")


if __name__ == "__main__":
    main()
