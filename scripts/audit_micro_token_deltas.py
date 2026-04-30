#!/usr/bin/env python3
"""Audit micro prompt pairs by effective post-template token differences."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_stability_probe import format_prompt, levenshtein


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYSTEM_PROMPT = "You are a concise, accurate assistant. Answer directly."


RUNS = {
    "qwen35_2b": ROOT / "runs/sagemaker_artifacts/chaos-micro-qwen2b-512-20260429-001/runs/qwen35_2b",
    "qwen35_4b": ROOT / "runs/sagemaker_artifacts/chaos-micro-qwen4b-thinkoff-512-20260429-001/runs/qwen35_4b",
    "qwen35_9b": ROOT / "runs/sagemaker_artifacts/chaos-micro-qwen9b-thinkoff-512-20260429-001/runs/qwen35_9b",
    "gemma4_e2b_it": ROOT / "runs/sagemaker_artifacts/chaos-micro-gemma-e2b-it-512-20260429-001/runs/gemma4_e2b_it",
    "gemma4_e4b_it": ROOT / "runs/sagemaker_artifacts/chaos-micro-gemma-e4b-it-512-20260429-001/runs/gemma4_e4b_it",
}


def load_models() -> dict[str, dict[str, Any]]:
    return {row["name"]: row for row in json.loads((ROOT / "configs/models.json").read_text())}


def common_prefix_len(a: list[int], b: list[int]) -> int:
    i = 0
    for x, y in zip(a, b):
        if x != y:
            break
        i += 1
    return i


def delta_kind(a: list[int], b: list[int], edit: int) -> str:
    if edit == 0:
        return "token_identical"
    if edit == 1 and len(b) == len(a) + 1:
        return "one_token_insert"
    if edit == 1 and len(b) == len(a) - 1:
        return "one_token_delete"
    if edit == 1 and len(b) == len(a):
        return "one_token_substitution"
    if edit <= 3:
        return "small_token_delta_2_3"
    return "multi_token_delta"


def token_context(tokenizer: Any, ids: list[int], center: int, radius: int = 4) -> str:
    lo = max(0, center - radius)
    hi = min(len(ids), center + radius + 1)
    return " | ".join(repr(tokenizer.decode([tid], skip_special_tokens=False)) for tid in ids[lo:hi])


def audit_model(
    model_name: str,
    run_dir: Path,
    prompt_pairs: list[dict[str, Any]],
    out_dir: Path,
    system_prompt: str,
) -> None:
    models = load_models()
    if model_name not in models:
        raise SystemExit(f"Unknown model in configs/models.json: {model_name}")
    entry = models[model_name]
    metadata = json.loads((run_dir / "metadata.json").read_text())
    thinking_mode = metadata.get("thinking_mode", "default")
    tokenizer = AutoTokenizer.from_pretrained(
        entry["model_id"],
        trust_remote_code=entry.get("trust_remote_code", False),
        local_files_only=True,
    )

    audit_rows = []
    for pair in prompt_pairs:
        formatted_a = format_prompt(tokenizer, pair["prompt_a"], system_prompt, thinking_mode)
        formatted_b = format_prompt(tokenizer, pair["prompt_b"], system_prompt, thinking_mode)
        ids_a = tokenizer(formatted_a)["input_ids"]
        ids_b = tokenizer(formatted_b)["input_ids"]
        edit = levenshtein(ids_a, ids_b)
        lcp = common_prefix_len(ids_a, ids_b)
        audit_rows.append(
            {
                "model_name": model_name,
                "pair_id": pair["id"],
                "category": pair["category"],
                "formatted_equal": formatted_a == formatted_b,
                "prompt_token_equal": ids_a == ids_b,
                "prompt_token_edit_distance": edit,
                "prompt_token_delta_kind": delta_kind(ids_a, ids_b, edit),
                "prompt_token_lcp": lcp,
                "input_tokens_a": len(ids_a),
                "input_tokens_b": len(ids_b),
                "input_token_delta": len(ids_b) - len(ids_a),
                "first_diff_context_a": "" if edit == 0 else token_context(tokenizer, ids_a, lcp),
                "first_diff_context_b": "" if edit == 0 else token_context(tokenizer, ids_b, lcp),
            }
        )

    audit = pd.DataFrame(audit_rows)
    summary = pd.read_csv(run_dir / "summary_with_semantic.csv")
    merged = summary.merge(
        audit,
        on=["model_name", "pair_id", "category"],
        how="left",
        validate="one_to_one",
    )

    model_out = out_dir / model_name
    model_out.mkdir(parents=True, exist_ok=True)
    audit.to_csv(model_out / "token_audit.csv", index=False)
    merged.to_csv(model_out / "summary_token_audited.csv", index=False)

    category = (
        merged.groupby(["category", "prompt_token_delta_kind"], sort=True)
        .agg(
            n=("pair_id", "count"),
            prompt_token_edit_mean=("prompt_token_edit_distance", "mean"),
            semantic_mean=("semantic_cosine_distance", "mean"),
            semantic_p90=("semantic_cosine_distance", lambda s: s.quantile(0.90)),
            token_edit_mean=("token_edit_distance_norm", "mean"),
        )
        .reset_index()
        .sort_values(["prompt_token_delta_kind", "semantic_mean"], ascending=[True, False])
    )
    category.to_csv(model_out / "token_category_summary.csv", index=False)

    valid = merged[
        (merged["category"] != "micro_control_identical")
        & (merged["prompt_token_delta_kind"] != "token_identical")
    ]
    effective = (
        valid.groupby("category", sort=True)
        .agg(
            n=("pair_id", "count"),
            prompt_token_edit_mean=("prompt_token_edit_distance", "mean"),
            semantic_mean=("semantic_cosine_distance", "mean"),
            semantic_p90=("semantic_cosine_distance", lambda s: s.quantile(0.90)),
            token_edit_mean=("token_edit_distance_norm", "mean"),
        )
        .reset_index()
        .sort_values("semantic_mean", ascending=False)
    )
    effective.to_csv(model_out / "effective_token_category_summary.csv", index=False)

    overall = {
        "model_name": model_name,
        "run_dir": str(run_dir.relative_to(ROOT)),
        "n_pairs": int(len(merged)),
        "n_token_identical_non_control": int(
            ((merged["category"] != "micro_control_identical") & (merged["prompt_token_delta_kind"] == "token_identical")).sum()
        ),
        "n_effective_non_control": int(len(valid)),
        "effective_semantic_mean": float(valid["semantic_cosine_distance"].mean()),
        "effective_semantic_p90": float(valid["semantic_cosine_distance"].quantile(0.90)),
    }
    (model_out / "summary.json").write_text(json.dumps(overall, indent=2) + "\n")
    print(json.dumps(overall, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-pairs", type=Path, default=ROOT / "configs/prompt_pairs_micro_500.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs/rankings/micro_token_audit")
    parser.add_argument("--model", action="append", choices=sorted(RUNS), help="Model name to audit; repeatable")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    args = parser.parse_args()

    prompt_pairs = json.loads(args.prompt_pairs.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected = args.model or sorted(RUNS)
    for model_name in selected:
        run_dir = RUNS[model_name]
        if not (run_dir / "summary_with_semantic.csv").exists():
            print(f"Skipping {model_name}: missing {run_dir / 'summary_with_semantic.csv'}", file=sys.stderr)
            continue
        audit_model(model_name, run_dir, prompt_pairs, args.out_dir, args.system_prompt)


if __name__ == "__main__":
    main()
