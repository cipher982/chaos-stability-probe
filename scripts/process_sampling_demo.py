#!/usr/bin/env python3
"""Summarize sampled same-prompt vs perturbed-prompt variance demos."""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from add_semantic_metrics import cosine_distance, embed_texts


SAMPLE_JOBS = {
    "OLMo3": ("chaos-sample-demo-olmo3-20260429-001", "olmo3_7b_instruct"),
    "Qwen3.5 4B": ("chaos-sample-demo-qwen35-4b-20260429-001", "qwen35_4b"),
    "Qwen3.5 0.8B": ("chaos-sample-demo-qwen35-08b-20260429-001", "qwen35_08b"),
}


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_dir(job_name: str, model_name: str) -> Path:
    return Path("runs/sagemaker_artifacts") / job_name / "runs" / model_name


def ensure_artifact(label: str, job_name: str, model_name: str) -> Path | None:
    root = Path("runs/sagemaker_artifacts") / job_name
    if not root.exists():
        try:
            run(["uv", "run", "python", "scripts/download_sagemaker_artifact.py", job_name, "--extract"])
        except subprocess.CalledProcessError:
            print(f"Skipping {label}: artifact not ready")
            return None
    directory = run_dir(job_name, model_name)
    if not (directory / "generations.jsonl").exists():
        print(f"Skipping {label}: missing generations.jsonl")
        return None
    return directory


def pairwise(values: list[int]) -> list[tuple[int, int]]:
    return list(itertools.combinations(values, 2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/rankings/sampling_demo"))
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    distance_rows: list[dict[str, Any]] = []

    for label, (job_name, model_name) in SAMPLE_JOBS.items():
        directory = ensure_artifact(label, job_name, model_name)
        if directory is None:
            continue
        rows = [json.loads(line) for line in (directory / "generations.jsonl").read_text().splitlines()]
        texts = [row["generated_text"] for row in rows]
        embeddings = embed_texts(texts, args.embedding_model, args.batch_size)
        for row, emb in zip(rows, embeddings):
            row["_embedding"] = emb

        by_pair: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_pair.setdefault(row["pair_id"], []).append(row)

        for pair_id, pair_rows in by_pair.items():
            a_rows = [row for row in pair_rows if row["side"] == "a"]
            b_rows = [row for row in pair_rows if row["side"] == "b"]
            groups = [
                ("within_a", [(a_rows[i], a_rows[j]) for i, j in pairwise(list(range(len(a_rows))))]),
                ("within_b", [(b_rows[i], b_rows[j]) for i, j in pairwise(list(range(len(b_rows))))]),
                ("between_ab", [(a, b) for a in a_rows for b in b_rows]),
            ]
            for kind, pairs in groups:
                distances = [cosine_distance(left["_embedding"], right["_embedding"]) for left, right in pairs]
                for value in distances:
                    distance_rows.append(
                        {
                            "run_label": label,
                            "pair_id": pair_id,
                            "distance_type": kind,
                            "semantic_cosine_distance": value,
                        }
                    )
                summary_rows.append(
                    {
                        "run_label": label,
                        "pair_id": pair_id,
                        "distance_type": kind,
                        "n": len(distances),
                        "mean": float(np.mean(distances)),
                        "median": float(np.median(distances)),
                        "p90": float(np.quantile(distances, 0.9)),
                    }
                )

    if not summary_rows:
        raise SystemExit("No sampling-demo artifacts were ready")

    summary = pd.DataFrame(summary_rows)
    distances = pd.DataFrame(distance_rows)
    summary.to_csv(args.out_dir / "sampling_distance_summary.csv", index=False)
    distances.to_csv(args.out_dir / "sampling_pairwise_distances.csv", index=False)

    for pair_id, group in summary.groupby("pair_id", sort=False):
        plot = group.pivot(index="run_label", columns="distance_type", values="mean")
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        plot[["within_a", "within_b", "between_ab"]].plot(kind="bar", ax=ax)
        ax.set_title(pair_id)
        ax.set_ylabel("Mean semantic distance")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(args.out_dir / f"{pair_id}_sampling_variance.png", dpi=220)
        plt.close(fig)

    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Wrote sampling-demo artifacts to {args.out_dir}")


if __name__ == "__main__":
    main()
