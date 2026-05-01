#!/usr/bin/env python3
"""Compare deterministic singleton generation against batched generation."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_stability_probe import (  # noqa: E402
    format_prompt,
    load_json,
    load_model,
    pick_device,
    pick_dtype,
    set_seed,
)


def select_model(models_path: Path, selector: str) -> dict[str, Any]:
    for entry in load_json(models_path):
        if selector in {entry["name"], entry["model_id"]}:
            return entry
    raise SystemExit(f"Unknown model selector: {selector}")


def select_prompts(prompt_pairs_path: Path, limit_pairs: int) -> tuple[list[str], list[str]]:
    pairs = load_json(prompt_pairs_path)
    if limit_pairs:
        pairs = pairs[:limit_pairs]
    labels: list[str] = []
    prompts: list[str] = []
    for pair in pairs:
        labels.append(f"{pair['id']}:a")
        prompts.append(pair["prompt_a"])
        labels.append(f"{pair['id']}:b")
        prompts.append(pair["prompt_b"])
    return labels, prompts


def generate_singleton(
    loaded: Any,
    text: str,
    max_new_tokens: int,
    seed: int,
) -> list[int]:
    set_seed(seed)
    inputs = loaded.tokenizer(text, return_tensors="pt").to(loaded.device)
    input_len = int(inputs["input_ids"].shape[1])
    with torch.inference_mode():
        output_ids = loaded.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=loaded.tokenizer.pad_token_id or loaded.tokenizer.eos_token_id,
        )
    return output_ids[0, input_len:].detach().cpu().tolist()


def generate_batches(
    loaded: Any,
    texts: list[str],
    indices: list[int],
    batch_size: int,
    max_new_tokens: int,
    seed: int,
) -> list[list[int]]:
    by_index: dict[int, list[int]] = {}
    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        set_seed(seed)
        inputs = loaded.tokenizer([texts[idx] for idx in chunk], return_tensors="pt", padding=True).to(loaded.device)
        input_len = int(inputs["input_ids"].shape[1])
        with torch.inference_mode():
            output_ids = loaded.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=loaded.tokenizer.pad_token_id or loaded.tokenizer.eos_token_id,
            )
        for row_idx, original_idx in enumerate(chunk):
            by_index[original_idx] = output_ids[row_idx, input_len:].detach().cpu().tolist()
    return [by_index[idx] for idx in range(len(indices))]


def first_mismatch(a: list[int], b: list[int]) -> int | None:
    for idx, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return idx
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path("configs/models.json"))
    parser.add_argument("--model", default="qwen35_08b")
    parser.add_argument("--prompt-pairs", type=Path, default=Path("configs/prompt_pairs_mechinterp_seed.json"))
    parser.add_argument("--limit-pairs", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--batch-size", type=int, action="append")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--thinking-mode", choices=["default", "enabled", "disabled"], default="disabled")
    parser.add_argument("--system-prompt", default="You are a concise, accurate assistant. Answer directly.")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    batch_sizes = args.batch_size or [2, 4, 8]

    labels, prompts = select_prompts(args.prompt_pairs, args.limit_pairs)
    device = pick_device(args.device)
    dtype = pick_dtype(device, args.dtype)
    loaded = load_model(select_model(args.models, args.model), device, dtype)
    loaded.tokenizer.padding_side = "left"
    formatted = [format_prompt(loaded.tokenizer, prompt, args.system_prompt, args.thinking_mode) for prompt in prompts]

    start = time.time()
    singleton = [generate_singleton(loaded, text, args.max_new_tokens, args.seed) for text in formatted]
    singleton_elapsed_s = time.time() - start

    rows: list[dict[str, Any]] = []
    scenarios: list[tuple[str, list[int], int]] = []
    for batch_size in batch_sizes:
        scenarios.append((f"ordered_bs{batch_size}", list(range(len(formatted))), batch_size))
        scenarios.append((f"reversed_bs{batch_size}", list(reversed(range(len(formatted)))), batch_size))
        shuffled = list(range(len(formatted)))
        random.Random(20260430 + batch_size).shuffle(shuffled)
        scenarios.append((f"shuffled_bs{batch_size}", shuffled, batch_size))

    for scenario, indices, batch_size in scenarios:
        start = time.time()
        batched = generate_batches(loaded, formatted, indices, batch_size, args.max_new_tokens, args.seed)
        elapsed_s = time.time() - start
        mismatches = []
        for idx, (expected, actual) in enumerate(zip(singleton, batched, strict=True)):
            mismatch_idx = first_mismatch(expected, actual)
            if mismatch_idx is not None:
                mismatches.append(
                    {
                        "index": idx,
                        "label": labels[idx],
                        "first_mismatch_token": mismatch_idx,
                        "singleton_next": expected[mismatch_idx : mismatch_idx + 5],
                        "batched_next": actual[mismatch_idx : mismatch_idx + 5],
                    }
                )
        row = {
            "scenario": scenario,
            "batch_size": batch_size,
            "elapsed_s": elapsed_s,
            "mismatch_count": len(mismatches),
            "prompt_count": len(labels),
            "mismatches": mismatches[:10],
        }
        rows.append(row)
        print(
            scenario,
            f"elapsed_s={elapsed_s:.2f}",
            f"mismatches={len(mismatches)}/{len(labels)}",
            flush=True,
        )
        for mismatch in mismatches[:5]:
            print(" ", json.dumps(mismatch), flush=True)

    result = {
        "model": args.model,
        "device": str(device),
        "dtype": str(dtype),
        "prompt_pairs": str(args.prompt_pairs),
        "prompt_count": len(labels),
        "max_new_tokens": args.max_new_tokens,
        "singleton_elapsed_s": singleton_elapsed_s,
        "scenarios": rows,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
