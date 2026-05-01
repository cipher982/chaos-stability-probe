#!/usr/bin/env python3
"""Run residual activation patching for one prompt-pair branch point."""

from __future__ import annotations

import argparse
import csv
import json
import math
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import torch

from run_stability_probe import (
    append_continuation,
    common_prefix_len,
    decode_token,
    generate_once,
    load_json,
    load_model,
    pick_device,
    pick_dtype,
    prompt_token_payload,
    tokenize_prompt,
)


def get_path(root: Any, dotted: str) -> Any | None:
    cur = root
    for part in dotted.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def find_blocks(model: Any) -> tuple[str, Any]:
    for path in [
        "model.language_model.layers",
        "model.layers",
        "model.model.layers",
        "transformer.h",
        "gpt_neox.layers",
    ]:
        blocks = get_path(model, path)
        if blocks is not None:
            return path, blocks
    raise ValueError("Could not find transformer block list on this model")


def select_model(models_path: Path, selector: str) -> dict[str, Any]:
    for entry in load_json(models_path):
        if selector in {entry["name"], entry["model_id"]}:
            return entry
    raise SystemExit(f"Unknown model selector: {selector}")


def select_pair(prompt_pairs_path: Path, pair_id: str) -> dict[str, Any]:
    for pair in load_json(prompt_pairs_path):
        if pair["id"] == pair_id:
            return pair
    raise SystemExit(f"Unknown pair_id {pair_id} in {prompt_pairs_path}")


def first_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    return output


def replace_first_tensor(output: Any, value: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (value, *output[1:])
    return value


def logits_for_inputs(loaded: Any, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    with torch.inference_mode():
        out = loaded.model(**inputs, use_cache=False)
    return out.logits[0, -1, :].detach().float().cpu()


def branch_metric(logits: torch.Tensor, a_token: int, b_token: int) -> dict[str, Any]:
    top = torch.topk(logits, k=5)
    probs = torch.softmax(logits, dim=-1)
    return {
        "metric_a_minus_b": float(logits[a_token] - logits[b_token]),
        "a_token_logit": float(logits[a_token]),
        "b_token_logit": float(logits[b_token]),
        "a_token_prob": float(probs[a_token]),
        "b_token_prob": float(probs[b_token]),
        "top1_token_id": int(top.indices[0]),
        "top1_token": None,
        "top1_prob": float(probs[int(top.indices[0])]),
    }


def cache_clean_activations(loaded: Any, blocks: Any, inputs: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    cache: list[torch.Tensor | None] = [None] * len(blocks)
    handles = []

    def make_hook(layer_idx: int):
        def hook(_module: Any, _inp: Any, output: Any) -> None:
            cache[layer_idx] = first_tensor(output).detach()

        return hook

    for idx, block in enumerate(blocks):
        handles.append(block.register_forward_hook(make_hook(idx)))
    try:
        with torch.inference_mode():
            loaded.model(**inputs, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if any(item is None for item in cache):
        raise RuntimeError("Failed to cache clean activations for every layer")
    return [item for item in cache if item is not None]


def patched_logits(
    loaded: Any,
    blocks: Any,
    corrupt_inputs: dict[str, torch.Tensor],
    clean_cache: list[torch.Tensor],
    layer_idx: int,
    clean_pos: int,
    corrupt_pos: int,
) -> torch.Tensor:
    def hook(_module: Any, _inp: Any, output: Any) -> Any:
        hidden = first_tensor(output).clone()
        source = clean_cache[layer_idx][:, clean_pos, :].to(device=hidden.device, dtype=hidden.dtype)
        hidden[:, corrupt_pos, :] = source
        return replace_first_tensor(output, hidden)

    handle = blocks[layer_idx].register_forward_hook(hook)
    try:
        return logits_for_inputs(loaded, corrupt_inputs)
    finally:
        handle.remove()


def aligned_position_pairs(
    clean_prompt_ids: list[int],
    corrupt_prompt_ids: list[int],
    common_prefix_len: int,
) -> list[tuple[str, int, int]]:
    pairs: list[tuple[str, int, int]] = []
    matcher = SequenceMatcher(a=clean_prompt_ids, b=corrupt_prompt_ids, autojunk=False)
    for _tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if _tag != "equal":
            continue
        for clean_pos, corrupt_pos in zip(range(i1, i2), range(j1, j2), strict=True):
            pairs.append((f"aligned_prompt_pos_{clean_pos}_to_{corrupt_pos}", clean_pos, corrupt_pos))
    clean_prompt_len = len(clean_prompt_ids)
    corrupt_prompt_len = len(corrupt_prompt_ids)
    for idx in range(common_prefix_len):
        pairs.append(
            (
                f"aligned_generated_prefix_pos_{idx}",
                clean_prompt_len + idx,
                corrupt_prompt_len + idx,
            )
        )
    return pairs


def position_pairs(
    mode: str,
    clean_len: int,
    corrupt_len: int,
    prompt_lcp: int,
    clean_prompt_ids: list[int],
    corrupt_prompt_ids: list[int],
    generated_prefix_len: int,
) -> list[tuple[str, int, int]]:
    final = ("final_context_token", clean_len - 1, corrupt_len - 1)
    changed = ("prompt_lcp_token", min(prompt_lcp, clean_len - 1), min(prompt_lcp, corrupt_len - 1))
    if mode == "final":
        return [final]
    if mode == "changed-final":
        return [changed, final] if changed != final else [final]
    if mode == "aligned":
        pairs = aligned_position_pairs(clean_prompt_ids, corrupt_prompt_ids, generated_prefix_len)
        if changed not in pairs:
            pairs.append(changed)
        if final not in pairs:
            pairs.append(final)
        return pairs
    if mode == "all":
        pairs = [(f"pos_{i}", i, i) for i in range(min(clean_len, corrupt_len))]
        if final not in pairs:
            pairs.append(final)
        return pairs
    raise ValueError(f"Unknown position mode: {mode}")


def rescue_fraction(clean_metric: float, corrupt_metric: float, patched_metric: float) -> float:
    den = clean_metric - corrupt_metric
    if math.isclose(den, 0.0, abs_tol=1e-9):
        return float("nan")
    return (patched_metric - corrupt_metric) / den


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path("configs/models.json"))
    parser.add_argument("--model", default="qwen35_08b")
    parser.add_argument("--prompt-pairs", type=Path, default=Path("configs/prompt_pairs_mechinterp_seed.json"))
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/mechinterp_patch"))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--thinking-mode", choices=["default", "enabled", "disabled"], default="disabled")
    parser.add_argument("--positions", choices=["final", "changed-final", "aligned", "all"], default="changed-final")
    parser.add_argument("--system-prompt", default="You are a concise, accurate assistant. Answer directly.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    entry = select_model(args.models, args.model)
    pair = select_pair(args.prompt_pairs, args.pair_id)
    device = pick_device(args.device)
    dtype = pick_dtype(device, args.dtype)
    loaded = load_model(entry, device, dtype)
    block_path, blocks = find_blocks(loaded.model)

    gen_a = generate_once(
        loaded,
        pair["prompt_a"],
        args.system_prompt,
        args.max_new_tokens,
        False,
        0.7,
        0.95,
        1234,
        args.thinking_mode,
    )
    gen_b = generate_once(
        loaded,
        pair["prompt_b"],
        args.system_prompt,
        args.max_new_tokens,
        False,
        0.7,
        0.95,
        1234,
        args.thinking_mode,
    )
    first_diff = common_prefix_len(gen_a["generated_tokens"], gen_b["generated_tokens"])
    if first_diff >= min(len(gen_a["generated_tokens"]), len(gen_b["generated_tokens"])):
        raise SystemExit("Generated continuations did not branch within the captured token budget")

    a_branch_token = int(gen_a["generated_tokens"][first_diff])
    b_branch_token = int(gen_b["generated_tokens"][first_diff])
    common_prefix = gen_a["generated_tokens"][:first_diff]

    clean_prompt_inputs = tokenize_prompt(loaded, pair["prompt_a"], args.system_prompt, args.thinking_mode)
    corrupt_prompt_inputs = tokenize_prompt(loaded, pair["prompt_b"], args.system_prompt, args.thinking_mode)
    clean_prompt_ids = clean_prompt_inputs["input_ids"][0].tolist()
    corrupt_prompt_ids = corrupt_prompt_inputs["input_ids"][0].tolist()
    clean_inputs = append_continuation(clean_prompt_inputs, common_prefix)
    corrupt_inputs = append_continuation(corrupt_prompt_inputs, common_prefix)
    clean_len = int(clean_inputs["input_ids"].shape[1])
    corrupt_len = int(corrupt_inputs["input_ids"].shape[1])
    prompt_delta = prompt_token_payload(loaded, pair["prompt_a"], pair["prompt_b"], args.system_prompt, args.thinking_mode)

    clean_logits = logits_for_inputs(loaded, clean_inputs)
    corrupt_logits = logits_for_inputs(loaded, corrupt_inputs)
    clean_base = branch_metric(clean_logits, a_branch_token, b_branch_token)
    corrupt_base = branch_metric(corrupt_logits, a_branch_token, b_branch_token)
    clean_base["top1_token"] = decode_token(loaded.tokenizer, int(clean_base["top1_token_id"]))
    corrupt_base["top1_token"] = decode_token(loaded.tokenizer, int(corrupt_base["top1_token_id"]))
    clean_metric = clean_base["metric_a_minus_b"]
    corrupt_metric = corrupt_base["metric_a_minus_b"]
    if int(corrupt_base["top1_token_id"]) != b_branch_token:
        print(
            "Warning: corrupt replay top-1 does not match the observed B branch token; "
            "this pair may be a weak patching target.",
            flush=True,
        )

    clean_cache = cache_clean_activations(loaded, blocks, clean_inputs)
    rows = []
    for label, clean_pos, corrupt_pos in position_pairs(
        args.positions,
        clean_len,
        corrupt_len,
        int(prompt_delta["prompt_token_lcp"]),
        clean_prompt_ids,
        corrupt_prompt_ids,
        len(common_prefix),
    ):
        for layer_idx in range(len(blocks)):
            logits = patched_logits(loaded, blocks, corrupt_inputs, clean_cache, layer_idx, clean_pos, corrupt_pos)
            metrics = branch_metric(logits, a_branch_token, b_branch_token)
            top1_token_id = int(metrics["top1_token_id"])
            metrics["top1_token"] = decode_token(loaded.tokenizer, top1_token_id)
            rows.append(
                {
                    "model_name": loaded.name,
                    "pair_id": pair["id"],
                    "category": pair["category"],
                    "block_path": block_path,
                    "layer": layer_idx,
                    "position_label": label,
                    "clean_pos": clean_pos,
                    "corrupt_pos": corrupt_pos,
                    "first_diff_token": first_diff,
                    "a_branch_token_id": a_branch_token,
                    "a_branch_token": decode_token(loaded.tokenizer, a_branch_token),
                    "b_branch_token_id": b_branch_token,
                    "b_branch_token": decode_token(loaded.tokenizer, b_branch_token),
                    "clean_metric_a_minus_b": clean_metric,
                    "corrupt_metric_a_minus_b": corrupt_metric,
                    "rescue_fraction": rescue_fraction(clean_metric, corrupt_metric, metrics["metric_a_minus_b"]),
                    **metrics,
                }
            )

    stem = f"{loaded.name}__{pair['id']}"
    out_csv = args.out_dir / f"{stem}.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "model": entry,
        "pair": pair,
        "block_path": block_path,
        "first_diff_token": first_diff,
        "a_branch_token_id": a_branch_token,
        "a_branch_token": decode_token(loaded.tokenizer, a_branch_token),
        "b_branch_token_id": b_branch_token,
        "b_branch_token": decode_token(loaded.tokenizer, b_branch_token),
        "clean_metric_a_minus_b": clean_metric,
        "corrupt_metric_a_minus_b": corrupt_metric,
        "clean_replay_top1_token_id": int(clean_base["top1_token_id"]),
        "clean_replay_top1_token": clean_base["top1_token"],
        "corrupt_replay_top1_token_id": int(corrupt_base["top1_token_id"]),
        "corrupt_replay_top1_token": corrupt_base["top1_token"],
        "corrupt_replay_matches_b_branch": int(corrupt_base["top1_token_id"]) == b_branch_token,
        "prompt_delta": {k: v for k, v in prompt_delta.items() if not k.startswith("prompt_token_ids_")},
        "generated_text_a": gen_a["generated_text"],
        "generated_text_b": gen_b["generated_text"],
    }
    (args.out_dir / f"{stem}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
