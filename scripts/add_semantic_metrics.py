#!/usr/bin/env python3
"""Add sentence-embedding semantic distances to a run summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    denom = torch.linalg.norm(a) * torch.linalg.norm(b)
    if float(denom) == 0.0:
        return float("nan")
    return max(0.0, min(2.0, float(1.0 - torch.dot(a, b) / denom)))


def embed_texts(texts: list[str], model_id: str, batch_size: int) -> list[torch.Tensor]:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()
    embeddings: list[torch.Tensor] = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            outputs = model(**inputs)
            pooled = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings.extend(pooled.detach().cpu())
    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    generations_path = args.run_dir / "generations.jsonl"
    summary_path = args.run_dir / "summary.csv"
    if not generations_path.exists() or not summary_path.exists():
        raise SystemExit("Run directory must contain generations.jsonl and summary.csv")

    generation_rows: list[dict[str, Any]] = [
        json.loads(line) for line in generations_path.read_text(encoding="utf-8").splitlines()
    ]
    texts = [row["generated_text"] for row in generation_rows]
    embeddings = embed_texts(texts, args.embedding_model, args.batch_size)

    keyed: dict[tuple[Any, ...], dict[str, torch.Tensor]] = {}
    key_fields = ["model_name", "pair_id", "category", "repeat", "sample", "seed_a", "seed_b"]
    for row, embedding in zip(generation_rows, embeddings):
        key = tuple(row.get(field) for field in key_fields)
        keyed.setdefault(key, {})[row["side"]] = embedding

    sem_rows = []
    for key, sides in keyed.items():
        if "a" not in sides or "b" not in sides:
            continue
        row = dict(zip(key_fields, key))
        row["semantic_cosine_distance"] = cosine_distance(sides["a"], sides["b"])
        sem_rows.append(row)

    sem_df = pd.DataFrame(sem_rows)
    summary = pd.read_csv(summary_path)
    merged = summary.merge(sem_df, on=key_fields, how="left")
    merged.to_csv(args.run_dir / "summary_with_semantic.csv", index=False)
    print(f"Wrote {args.run_dir / 'summary_with_semantic.csv'}")


if __name__ == "__main__":
    main()

