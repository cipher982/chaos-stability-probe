#!/usr/bin/env python3
"""Build model-specific micro prompt files with verified prompt-token deltas."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.make_micro_perturbation_pairs import BASE_PROMPTS, mutate
from scripts.run_stability_probe import format_prompt, levenshtein


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYSTEM_PROMPT = "You are a concise, accurate assistant. Answer directly."

# Categories that usually survive chat-template/tokenizer normalization.
TOKEN_HEAVY_KINDS = [
    "double_internal_space",
    "triple_internal_space",
    "line_wrap",
    "blank_line_wrap",
    "tab_indent",
    "tab_after_space",
    "space_before_punctuation",
    "space_after_punctuation",
    "duplicate_punctuation",
    "parenthesize_word",
    "duplicate_small_word",
    "leading_newline",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def model_registry(path: Path) -> dict[str, dict[str, Any]]:
    return {row["name"]: row for row in load_json(path)}


def candidate_pairs(count: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    kinds = list(TOKEN_HEAVY_KINDS)
    rows = []
    i = 0
    attempts = 0
    seen: set[tuple[str, str, str]] = set()
    while len(rows) < count and attempts < count * 20:
        attempts += 1
        if i % len(kinds) == 0:
            rng.shuffle(kinds)
        prompt = BASE_PROMPTS[i % len(BASE_PROMPTS)]
        kind = kinds[i % len(kinds)]
        i += 1
        mutated = mutate(prompt, kind, rng)
        key = (kind, prompt, mutated)
        if mutated == prompt or key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "id": f"token_cert_{kind}_{len(rows):04d}",
                "category": f"micro_{kind}",
                "prompt_a": prompt,
                "prompt_b": mutated,
            }
        )
    if len(rows) < count:
        raise RuntimeError(f"Only generated {len(rows)} unique candidates out of requested {count}")
    return rows


def controls(count: int) -> list[dict[str, str]]:
    rows = []
    for i in range(count):
        prompt = BASE_PROMPTS[i % len(BASE_PROMPTS)]
        rows.append(
            {
                "id": f"token_cert_control_identical_{i:03d}",
                "category": "micro_control_identical",
                "prompt_a": prompt,
                "prompt_b": prompt,
            }
        )
    return rows


def token_ids(tokenizer: Any, prompt: str, system_prompt: str, thinking_mode: str) -> list[int]:
    formatted = format_prompt(tokenizer, prompt, system_prompt, thinking_mode)
    return tokenizer(formatted)["input_ids"]


def certify_for_job(
    job: dict[str, Any],
    models: dict[str, dict[str, Any]],
    candidates: list[dict[str, str]],
    out_dir: Path,
    target_effective: int,
    control_count: int,
    system_prompt: str,
) -> dict[str, Any]:
    model_name = job["model"]
    entry = models[model_name]
    thinking_mode = job.get("thinking_mode", "default")
    tokenizer = AutoTokenizer.from_pretrained(
        entry["model_id"],
        trust_remote_code=entry.get("trust_remote_code", False),
    )

    selected: list[dict[str, str]] = []
    skipped = 0
    for row in candidates:
        ids_a = token_ids(tokenizer, row["prompt_a"], system_prompt, thinking_mode)
        ids_b = token_ids(tokenizer, row["prompt_b"], system_prompt, thinking_mode)
        if levenshtein(ids_a, ids_b) <= 0:
            skipped += 1
            continue
        selected.append(row)
        if len(selected) >= target_effective:
            break

    if len(selected) < target_effective:
        raise RuntimeError(
            f"{model_name}: only {len(selected)} effective prompt-token pairs "
            f"from {len(candidates)} candidates"
        )

    out_path = out_dir / f"{model_name}.json"
    rows = controls(control_count) + selected
    out_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "model": model_name,
        "thinking_mode": thinking_mode,
        "prompt_pairs": str(out_path.relative_to(ROOT)),
        "controls": control_count,
        "effective_non_controls": len(selected),
        "candidate_rows_seen": len(selected) + skipped,
        "token_identical_candidates_skipped": skipped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=ROOT / "configs/sagemaker_queue_token_micro_v2.json")
    parser.add_argument("--models", type=Path, default=ROOT / "configs/models.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "configs/prompt_pairs_token_certified")
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs/prompt_pairs_token_certified/manifest.json")
    parser.add_argument("--model", action="append", help="Model name to certify; repeatable. Default: every model in queue.")
    parser.add_argument("--candidate-count", type=int, default=4000)
    parser.add_argument("--target-effective", type=int, default=500)
    parser.add_argument("--controls", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    args = parser.parse_args()

    queue = load_json(args.queue)
    if args.model:
        want = set(args.model)
        queue = [job for job in queue if job["model"] in want]
    if not queue:
        raise SystemExit("No queue entries selected")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    candidates = candidate_pairs(args.candidate_count, args.seed)
    models = model_registry(args.models)
    manifest = [
        certify_for_job(
            job=job,
            models=models,
            candidates=candidates,
            out_dir=args.out_dir,
            target_effective=args.target_effective,
            control_count=args.controls,
            system_prompt=args.system_prompt,
        )
        for job in queue
    ]
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
