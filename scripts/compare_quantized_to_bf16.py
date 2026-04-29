#!/usr/bin/env python3
"""Measure quantized output fidelity against the BF16 run for the same prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch

from add_semantic_metrics import embed_texts, cosine_distance
from run_stability_probe import levenshtein


RUNS = {
    "qwen35_08b": {
        "BF16": "runs/sagemaker_artifacts/chaos-stability-panel-20260429-001/runs/qwen35_08b",
        "8-bit": "runs/sagemaker_artifacts/chaos-stability-qwen35-08b-bnb8-20260429-001/runs/qwen35_08b_bnb8",
        "4-bit": "runs/sagemaker_artifacts/chaos-stability-qwen35-08b-bnb4-20260429-001/runs/qwen35_08b_bnb4",
    },
    "qwen35_4b": {
        "BF16": "runs/sagemaker_artifacts/chaos-stability-panel-20260429-001/runs/qwen35_4b",
        "8-bit": "runs/sagemaker_artifacts/chaos-stability-qwen35-4b-bnb8-20260429-001/runs/qwen35_4b_bnb8",
        "4-bit": "runs/sagemaker_artifacts/chaos-stability-qwen35-4b-bnb4-20260429-001/runs/qwen35_4b_bnb4",
    },
}

SMALL_CATEGORIES = {"noop_format", "punctuation", "synonym"}


def load_generations(run_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (run_dir / "generations.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["pair_id"],
        row["category"],
        row["repeat"],
        row["sample"],
        row["seed_a"],
        row["seed_b"],
        row["side"],
    )


def token_distance(a: list[int], b: list[int]) -> float:
    return levenshtein(a, b) / max(len(a), len(b), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/quantization_fidelity"))
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    compare_rows: list[dict[str, Any]] = []
    text_pairs: list[tuple[int, str, str]] = []

    for model_label, run_map in RUNS.items():
        bf16_rows = load_generations(Path(run_map["BF16"]))
        bf16_by_key = {key(row): row for row in bf16_rows}
        for precision_label, run_dir in run_map.items():
            if precision_label == "BF16":
                continue
            q_rows = load_generations(Path(run_dir))
            for row in q_rows:
                b = bf16_by_key.get(key(row))
                if not b:
                    continue
                idx = len(compare_rows)
                compare_rows.append(
                    {
                        "model": model_label,
                        "precision": precision_label,
                        "pair_id": row["pair_id"],
                        "category": row["category"],
                        "side": row["side"],
                        "repeat": row["repeat"],
                        "sample": row["sample"],
                        "token_edit_distance_norm": token_distance(
                            row["generated_tokens"],
                            b["generated_tokens"],
                        ),
                    }
                )
                text_pairs.append((idx, row["generated_text"], b["generated_text"]))

    texts: list[str] = []
    for _, a, b in text_pairs:
        texts.append(a)
        texts.append(b)
    embeddings = embed_texts(texts, args.embedding_model, args.batch_size)
    semantic_by_idx: dict[int, float] = {}
    for pair_idx, (row_idx, _, _) in enumerate(text_pairs):
        semantic_by_idx[row_idx] = cosine_distance(embeddings[pair_idx * 2], embeddings[pair_idx * 2 + 1])

    for idx, value in semantic_by_idx.items():
        compare_rows[idx]["semantic_cosine_distance"] = value

    df = pd.DataFrame(compare_rows)
    df.to_csv(args.out_dir / "qwen_quantized_vs_bf16_rows.csv", index=False)

    summary = (
        df.groupby(["model", "precision", "category"], as_index=False)
        .agg(
            semantic_cosine_distance=("semantic_cosine_distance", "mean"),
            token_edit_distance_norm=("token_edit_distance_norm", "mean"),
        )
        .sort_values(["model", "precision", "category"])
    )
    summary.to_csv(args.out_dir / "qwen_quantized_vs_bf16_summary.csv", index=False)

    small = df[df["category"].isin(SMALL_CATEGORIES)].copy()
    small_summary = (
        small.groupby(["model", "precision"], as_index=False)
        .agg(
            bf16_fidelity_semantic_distance=("semantic_cosine_distance", "mean"),
            bf16_fidelity_token_distance=("token_edit_distance_norm", "mean"),
        )
        .sort_values(["model", "precision"])
    )
    small_summary.to_csv(args.out_dir / "qwen_quantized_vs_bf16_small_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    labels = [f"{row.model}\n{row.precision}" for row in small_summary.itertuples()]
    ax.bar(labels, small_summary["bf16_fidelity_semantic_distance"], color=["#6aaed6", "#2b6ea6"] * 2)
    ax.set_ylabel("Semantic distance from BF16 output")
    ax.set_title("Quantized output fidelity to BF16 on small perturbations")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out_dir / "qwen_quantized_vs_bf16_small_semantic.png", dpi=220)
    plt.close(fig)

    print(small_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Wrote {args.out_dir}")


if __name__ == "__main__":
    main()
