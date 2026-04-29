#!/usr/bin/env python3
"""Summarize high-N micro-perturbation runs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_stability_probe import generation_metrics


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def generation_pairs(run_dir: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in (run_dir / "generations.jsonl").open()]
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(
        index=["pair_id", "category", "repeat", "sample", "seed_a", "seed_b"],
        columns="side",
        values=["generated_text", "generated_token_count", "prompt_chars"],
        aggfunc="first",
    )
    pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
    return pivot.reset_index()


def write_summary_from_generations(run_dir: Path) -> None:
    """Build summary.csv from completed A/B rows when a run is still in flight."""
    rows = [json.loads(line) for line in (run_dir / "generations.jsonl").open()]
    by_key: dict[tuple, dict[str, dict]] = {}
    key_fields = ["model_name", "pair_id", "category", "repeat", "sample", "seed_a", "seed_b"]
    for row in rows:
        key = tuple(row.get(field) for field in key_fields)
        by_key.setdefault(key, {})[row["side"]] = row

    summary_rows = []
    for key, sides in by_key.items():
        if "a" not in sides or "b" not in sides:
            continue
        a = sides["a"]
        b = sides["b"]
        metrics = generation_metrics(
            a["generated_tokens"],
            b["generated_tokens"],
            a["generated_text"],
            b["generated_text"],
        )
        summary_rows.append({**dict(zip(key_fields, key)), **metrics})

    if not summary_rows:
        raise SystemExit("No completed A/B pairs found in generations.jsonl")
    pd.DataFrame(summary_rows).to_csv(run_dir / "summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/rankings/micro_sweep"))
    parser.add_argument("--prompt-pairs", type=Path, default=Path("configs/prompt_pairs_micro_500.json"))
    parser.add_argument("--top-n", type=int, default=40)
    args = parser.parse_args()

    summary_semantic = args.run_dir / "summary_with_semantic.csv"
    summary_csv = args.run_dir / "summary.csv"
    if not summary_csv.exists():
        write_summary_from_generations(args.run_dir)
    if not summary_semantic.exists():
        run(["uv", "run", "python", "scripts/add_semantic_metrics.py", str(args.run_dir), "--batch-size", "64"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(summary_semantic)
    pairs = generation_pairs(args.run_dir)
    merged = summary.merge(
        pairs,
        on=["pair_id", "category", "repeat", "sample", "seed_a", "seed_b"],
        how="left",
    )
    if args.prompt_pairs.exists():
        prompt_pairs = pd.DataFrame(json.loads(args.prompt_pairs.read_text(encoding="utf-8")))
        prompt_pairs = prompt_pairs[["id", "prompt_a", "prompt_b"]].rename(columns={"id": "pair_id"})
        merged = merged.merge(prompt_pairs, on="pair_id", how="left")

    category = (
        merged.groupby("category", sort=True)
        .agg(
            n=("pair_id", "count"),
            semantic_mean=("semantic_cosine_distance", "mean"),
            semantic_median=("semantic_cosine_distance", "median"),
            semantic_p90=("semantic_cosine_distance", lambda s: s.quantile(0.90)),
            token_edit_mean=("token_edit_distance_norm", "mean"),
            common_prefix_median=("common_prefix_tokens", "median"),
            a_tokens_median=("generated_token_count_a", "median"),
            b_tokens_median=("generated_token_count_b", "median"),
        )
        .reset_index()
        .sort_values("semantic_mean", ascending=False)
    )
    category.to_csv(args.out_dir / "category_summary.csv", index=False)

    worst = merged.sort_values("semantic_cosine_distance", ascending=False).head(args.top_n).copy()
    keep = [
        "pair_id",
        "category",
        "semantic_cosine_distance",
        "token_edit_distance_norm",
        "common_prefix_tokens",
        "prompt_a",
        "prompt_b",
        "generated_text_a",
        "generated_text_b",
    ]
    worst[keep].to_csv(args.out_dir / "worst_examples.csv", index=False)

    controls = merged[merged["category"] == "micro_control_identical"]
    control_mean = float(controls["semantic_cosine_distance"].mean()) if not controls.empty else float("nan")
    overall = {
        "run_dir": str(args.run_dir),
        "n_pairs": int(len(merged)),
        "control_identical_semantic_mean": control_mean,
        "micro_semantic_mean": float(
            merged[merged["category"] != "micro_control_identical"]["semantic_cosine_distance"].mean()
        ),
        "micro_semantic_p90": float(
            merged[merged["category"] != "micro_control_identical"]["semantic_cosine_distance"].quantile(0.90)
        ),
        "micro_semantic_max": float(
            merged[merged["category"] != "micro_control_identical"]["semantic_cosine_distance"].max()
        ),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(overall, indent=2) + "\n", encoding="utf-8")

    print(category.to_string(index=False))
    print(json.dumps(overall, indent=2))
    print(f"Wrote micro-sweep artifacts to {args.out_dir}")


if __name__ == "__main__":
    main()
