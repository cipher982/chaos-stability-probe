#!/usr/bin/env python3
"""Compare raw vs scaffold-stripped output divergence."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from add_semantic_metrics import cosine_distance, embed_texts


SMALL_CATEGORIES = ["noop_format", "punctuation", "synonym"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_text(text: str) -> str:
    return text.replace("Ċ", "\n").replace("Ġ", " ")


def strip_boilerplate(text: str) -> tuple[str, str, str]:
    """Remove only obvious wrapper tokens, preserving scaffold content."""
    raw = normalize_text(text).lstrip()
    if raw.startswith("Thinking Process:"):
        return re.sub(r"^Thinking Process:\s*", "", raw, count=1), "thinking_process", "clean"
    if raw.startswith("<think>"):
        return re.sub(r"^<think>\s*", "", raw, count=1), "think_tag", "clean"
    if re.match(r"^(Okay|Alright|Hmm|Let me think|I need|We need)[,\\s]", raw, flags=re.I):
        return raw, "visible_cot", "none"
    if "<</SYS>>" in raw or raw.count("User:") >= 2 or raw.count("Assistant:") >= 2:
        return raw, "template_echo", "none"
    return raw, "none", "none"


def extract_after_scaffold(text: str) -> tuple[str, str]:
    """Best-effort answer-content extraction after visible reasoning/scaffold."""
    raw = normalize_text(text).lstrip()

    if raw.startswith("<think>"):
        close = raw.find("</think>")
        if close >= 0:
            return raw[close + len("</think>") :].lstrip(), "clean"
        return "", "failed_no_think_close"

    if raw.startswith("Thinking Process:"):
        markers = [
            r"\n\s*(?:Final Answer|Answer|Response)\s*:\s*",
            r"\n\s*\*\*(?:Final Answer|Answer|Response)\s*:\*\*\s*",
            r"\n\s*Here(?:'s| is)\s+",
            r"\n\s*Weather forecasts",
            r"\n\s*Regression tests",
            r"\n\s*```",
        ]
        for pattern in markers:
            match = re.search(pattern, raw, flags=re.I)
            if match:
                return raw[match.start() :].lstrip(), "heuristic"
        draft = re.search(r"\n\s*\d+\.\s+\*\*Drafting[^:]*:\*\*?\s*", raw, flags=re.I)
        if draft:
            after = raw[draft.end() :].lstrip()
            if after:
                return after, "heuristic_drafting"
        return "", "failed_no_answer_boundary"

    if re.match(r"^(Okay|Alright|Hmm|Let me think|I need|We need)[,\\s]", raw, flags=re.I):
        markers = [
            r"\n\s*(?:So,?\s*)?(?:the answer|here is|here's|a concise|in short)\\b",
            r"\n\s*```",
        ]
        for pattern in markers:
            match = re.search(pattern, raw, flags=re.I)
            if match:
                return raw[match.start() :].lstrip(), "heuristic"
        return "", "failed_visible_cot_no_boundary"

    return raw, "not_scaffolded"


def truncate_words(text: str, n_words: int) -> str:
    if n_words <= 0:
        return text
    return " ".join(text.split()[:n_words])


def keyed_generation_rows(generation_path: Path, run_label: str) -> dict[tuple[Any, ...], dict[str, Any]]:
    keyed: dict[tuple[Any, ...], dict[str, Any]] = {}
    key_fields = ["model_name", "pair_id", "category", "repeat", "sample", "seed_a", "seed_b"]
    for row in load_jsonl(generation_path):
        key = tuple(row.get(field) for field in key_fields)
        side = row["side"]
        keyed.setdefault(key, {"run_label": run_label, **dict(zip(key_fields, key))})[side] = row
    return keyed


def choose_generation_file(model_name: str) -> Path | None:
    candidates = []
    for path in Path("runs/sagemaker_artifacts").glob("**/generations.jsonl"):
        try:
            rows = load_jsonl(path)
        except Exception:
            continue
        if any(row.get("model_name") == model_name for row in rows):
            candidates.append((("robust" in str(path), len(rows)), path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def ensure_semantic_text_distances(text_pairs: list[tuple[str, str]], model_id: str, batch_size: int) -> list[float]:
    flat = []
    for a, b in text_pairs:
        flat.extend([a, b])
    embeddings = embed_texts(flat, model_id, batch_size)
    distances = []
    for i in range(0, len(embeddings), 2):
        distances.append(cosine_distance(embeddings[i], embeddings[i + 1]))
    return distances


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-summary", type=Path, default=Path("runs/rankings/final_21model_readout/merged_summary.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/rankings/content_only"))
    parser.add_argument("--truncate-words", type=int, default=96)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    merged = pd.read_csv(args.merged_summary)
    model_map = merged[["run_label", "model_name", "model_id"]].drop_duplicates()

    rows: list[dict[str, Any]] = []
    text_pairs: list[tuple[str, str]] = []
    row_text_indices: list[int] = []
    for model in model_map.to_dict("records"):
        generation_path = choose_generation_file(model["model_name"])
        if generation_path is None:
            continue
        keyed = keyed_generation_rows(generation_path, model["run_label"])
        for item in keyed.values():
            if "a" not in item or "b" not in item:
                continue
            if item["category"] not in set(SMALL_CATEGORIES):
                continue
            raw_a = normalize_text(item["a"]["generated_text"])
            raw_b = normalize_text(item["b"]["generated_text"])
            boiler_a, scaffold_kind_a, boiler_conf_a = strip_boilerplate(raw_a)
            boiler_b, scaffold_kind_b, boiler_conf_b = strip_boilerplate(raw_b)
            answer_a, answer_conf_a = extract_after_scaffold(raw_a)
            answer_b, answer_conf_b = extract_after_scaffold(raw_b)

            base = {
                "run_label": model["run_label"],
                "model_name": model["model_name"],
                "model_id": model["model_id"],
                "generation_file": str(generation_path),
                "pair_id": item["pair_id"],
                "category": item["category"],
                "scaffold_kind_a": scaffold_kind_a,
                "scaffold_kind_b": scaffold_kind_b,
                "boilerplate_confidence_a": boiler_conf_a,
                "boilerplate_confidence_b": boiler_conf_b,
                "answer_confidence_a": answer_conf_a,
                "answer_confidence_b": answer_conf_b,
                "raw_a_words": len(raw_a.split()),
                "raw_b_words": len(raw_b.split()),
                "boilerplate_stripped_a_words": len(boiler_a.split()),
                "boilerplate_stripped_b_words": len(boiler_b.split()),
                "answer_a_words": len(answer_a.split()),
                "answer_b_words": len(answer_b.split()),
                "raw_a_prefix": raw_a[:180].replace("\n", "\\n"),
                "raw_b_prefix": raw_b[:180].replace("\n", "\\n"),
                "answer_a_prefix": answer_a[:180].replace("\n", "\\n"),
                "answer_b_prefix": answer_b[:180].replace("\n", "\\n"),
            }

            for view, a_text, b_text in [
                ("raw", raw_a, raw_b),
                ("boilerplate_stripped", boiler_a, boiler_b),
                ("answer_heuristic", answer_a, answer_b),
            ]:
                a_trunc = truncate_words(a_text, args.truncate_words)
                b_trunc = truncate_words(b_text, args.truncate_words)
                row = {
                    **base,
                    "view": view,
                    "a_words_after_truncate": len(a_trunc.split()),
                    "b_words_after_truncate": len(b_trunc.split()),
                    "usable": bool(a_trunc.strip()) and bool(b_trunc.strip()),
                }
                rows.append(row)
                if row["usable"]:
                    row_text_indices.append(len(rows) - 1)
                    text_pairs.append((a_trunc, b_trunc))

    distances = ensure_semantic_text_distances(text_pairs, args.embedding_model, args.batch_size) if text_pairs else []
    for idx, distance in zip(row_text_indices, distances):
        rows[idx]["semantic_cosine_distance"] = distance

    detail = pd.DataFrame(rows)
    detail.to_csv(args.out_dir / "content_only_detail.csv", index=False)

    usable = detail[detail["usable"] & detail["semantic_cosine_distance"].notna()].copy()
    summary = (
        usable.groupby(["run_label", "view"], as_index=False, observed=True)
        .agg(
            n=("semantic_cosine_distance", "count"),
            mean=("semantic_cosine_distance", "mean"),
            median=("semantic_cosine_distance", "median"),
            answer_clean_rate=("answer_confidence_a", lambda s: float((s == "clean").mean())),
            answer_heuristic_rate=("answer_confidence_a", lambda s: float(s.astype(str).str.startswith("heuristic").mean())),
            answer_failed_rate=("answer_confidence_a", lambda s: float(s.astype(str).str.startswith("failed").mean())),
        )
        .sort_values(["view", "mean"])
    )
    summary.to_csv(args.out_dir / "content_only_summary.csv", index=False)

    pivot = summary.pivot(index="run_label", columns="view", values="mean").reset_index()
    if {"raw", "boilerplate_stripped"}.issubset(pivot.columns):
        pivot["boilerplate_delta"] = pivot["boilerplate_stripped"] - pivot["raw"]
    if {"raw", "answer_heuristic"}.issubset(pivot.columns):
        pivot["answer_delta"] = pivot["answer_heuristic"] - pivot["raw"]
    pivot.to_csv(args.out_dir / "content_only_pivot.csv", index=False)

    print(pivot.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Wrote content-only artifacts to {args.out_dir}")


if __name__ == "__main__":
    main()
