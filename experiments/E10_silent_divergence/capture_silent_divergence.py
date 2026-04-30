#!/usr/bin/env python3
"""Capture hidden/logit trajectories through a paired common-prefix window."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import transformers

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_stability_probe import (
    append_continuation,
    common_prefix_len,
    cosine_distance,
    generate_once,
    load_json,
    load_model,
    logit_distribution_metrics,
    normalized_l2,
    pick_device,
    pick_dtype,
    tokenize_prompt,
)


DEFAULT_SYSTEM_PROMPT = "You are a concise, accurate assistant. Answer directly."


def current_git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def current_git_dirty() -> bool | None:
    try:
        return bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except Exception:
        return None


def device_name(device: torch.device) -> str | None:
    if device.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_name(device.index or 0)
    if device.type == "mps":
        return "mps"
    if device.type == "cpu":
        return platform.processor() or "cpu"
    return None


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def metadata_subset(metadata: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "model_name",
        "model_id",
        "backend",
        "requested_device",
        "resolved_device",
        "requested_dtype",
        "resolved_dtype",
        "torch_version",
        "transformers_version",
        "git_sha",
        "git_dirty",
    ]
    return {f"runtime_{key}": metadata.get(key) for key in keys}


def build_run_metadata(
    args: argparse.Namespace,
    entry: dict[str, Any],
    loaded: Any,
    device: torch.device,
    dtype: torch.dtype,
    pair_ids: list[str],
) -> dict[str, Any]:
    model = loaded.model
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "backend": "transformers",
        "model_name": args.model,
        "model_id": entry["model_id"],
        "model_class": type(model).__name__,
        "tokenizer_class": type(loaded.tokenizer).__name__,
        "hf_device_map": getattr(model, "hf_device_map", None),
        "requested_device": args.device,
        "resolved_device": str(device),
        "device_name": device_name(device),
        "requested_dtype": args.dtype,
        "resolved_dtype": dtype_name(dtype),
        "thinking_mode": args.thinking_mode,
        "system_prompt": args.system_prompt,
        "max_new_tokens": args.max_new_tokens,
        "logit_max_steps": args.logit_max_steps,
        "prompt_pairs": str(args.prompt_pairs),
        "pair_ids": pair_ids,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_sha": current_git_sha(),
        "git_dirty": current_git_dirty(),
        "argv": sys.argv,
    }


def select_model(models_path: Path, selector: str) -> dict[str, Any]:
    for entry in load_json(models_path):
        if selector in {entry["name"], entry["model_id"]}:
            return entry
    raise SystemExit(f"Unknown model selector: {selector}")


def prompt_pairs_by_id(path: Path) -> dict[str, dict[str, Any]]:
    return {pair["id"]: pair for pair in load_json(path)}


def pair_ids_from_file(path: Path, model_name: str, limit: int | None) -> list[str]:
    rows = pd.read_csv(path)
    if "model_name" in rows.columns:
        rows = rows[rows["model_name"] == model_name]
    if "best_rescue_fraction" in rows.columns:
        rows = rows.sort_values("best_rescue_fraction", ascending=False)
    ids = [str(v) for v in rows["pair_id"].dropna().drop_duplicates()]
    return ids[:limit] if limit else ids


def trajectory_rows_for_pair(
    loaded: Any,
    pair: dict[str, Any],
    system_prompt: str,
    max_new_tokens: int,
    logit_max_steps: int,
    thinking_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gen_a = generate_once(
        loaded,
        pair["prompt_a"],
        system_prompt,
        max_new_tokens,
        False,
        0.7,
        0.95,
        1234,
        thinking_mode,
    )
    gen_b = generate_once(
        loaded,
        pair["prompt_b"],
        system_prompt,
        max_new_tokens,
        False,
        0.7,
        0.95,
        1234,
        thinking_mode,
    )
    branch_t = common_prefix_len(gen_a["generated_tokens"], gen_b["generated_tokens"])
    if branch_t >= min(len(gen_a["generated_tokens"]), len(gen_b["generated_tokens"])):
        branch_t = None

    inputs_a = tokenize_prompt(loaded, pair["prompt_a"], system_prompt, thinking_mode)
    inputs_b = tokenize_prompt(loaded, pair["prompt_b"], system_prompt, thinking_mode)
    if branch_t is None:
        common_prefix_len_tokens = min(len(gen_a["generated_tokens"]), len(gen_b["generated_tokens"]))
    else:
        common_prefix_len_tokens = branch_t
    common_tokens = gen_a["generated_tokens"][:common_prefix_len_tokens]
    max_t = min(logit_max_steps, len(common_tokens))
    if branch_t is not None:
        max_t = min(max_t, branch_t)

    summary_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []

    for t in range(max_t + 1):
        suffix = common_tokens[:t]
        full_a = append_continuation(inputs_a, suffix)
        full_b = append_continuation(inputs_b, suffix)
        with torch.inference_mode():
            out_a = loaded.model(**full_a, output_hidden_states=True, use_cache=False)
            out_b = loaded.model(**full_b, output_hidden_states=True, use_cache=False)

        logits_a = out_a.logits[0, -1, :].detach().cpu()
        logits_b = out_b.logits[0, -1, :].detach().cpu()
        logit_metrics = logit_distribution_metrics(logits_a, logits_b)

        layer_metrics = []
        for layer_idx, (ha, hb) in enumerate(zip(out_a.hidden_states, out_b.hidden_states)):
            last_a = ha[0, -1, :].detach().cpu()
            last_b = hb[0, -1, :].detach().cpu()
            cos = cosine_distance(last_a, last_b)
            l2 = normalized_l2(last_a, last_b)
            layer_metrics.append((layer_idx, cos, l2))
            layer_rows.append(
                {
                    "pair_id": pair["id"],
                    "category": pair.get("category"),
                    "t": t,
                    "branch_t": branch_t,
                    "tokens_until_branch": None if branch_t is None else branch_t - t,
                    "layer": layer_idx,
                    "last_token_cosine_distance": cos,
                    "last_token_normalized_l2": l2,
                }
            )

        final_layer = layer_metrics[-1]
        max_layer_cos = max(item[1] for item in layer_metrics)
        max_layer_l2 = max(item[2] for item in layer_metrics)
        summary_rows.append(
            {
                "pair_id": pair["id"],
                "category": pair.get("category"),
                "t": t,
                "branch_t": branch_t,
                "tokens_until_branch": None if branch_t is None else branch_t - t,
                "generated_prefix_len": t,
                "js_divergence": logit_metrics["js_divergence"],
                "top1_same": logit_metrics["top1_same"],
                "entropy_a": logit_metrics["entropy_a"],
                "entropy_b": logit_metrics["entropy_b"],
                "effective_branching_factor_a": logit_metrics["effective_branching_factor_a"],
                "effective_branching_factor_b": logit_metrics["effective_branching_factor_b"],
                "a_top1_margin_logit": logit_metrics["a_top1_margin_logit"],
                "b_top1_margin_logit": logit_metrics["b_top1_margin_logit"],
                "final_layer_cosine_distance": final_layer[1],
                "final_layer_normalized_l2": final_layer[2],
                "max_layer_cosine_distance": max_layer_cos,
                "max_layer_normalized_l2": max_layer_l2,
            }
        )

    return summary_rows, layer_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path("configs/models.json"))
    parser.add_argument("--model", default="qwen35_2b")
    parser.add_argument("--prompt-pairs", type=Path, default=Path("configs/prompt_pairs_mechinterp_seed.json"))
    parser.add_argument("--pair-id", action="append", dest="pair_ids")
    parser.add_argument("--pairs-from", type=Path, default=Path("runs/mechinterp_patch/selected_patch_targets_aligned.csv"))
    parser.add_argument("--limit-pairs", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/silent_divergence_pilot"))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--logit-max-steps", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--thinking-mode", choices=["default", "enabled", "disabled"], default="disabled")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    entry = select_model(args.models, args.model)
    pairs = prompt_pairs_by_id(args.prompt_pairs)
    pair_ids = args.pair_ids or pair_ids_from_file(args.pairs_from, args.model, args.limit_pairs)
    pair_ids = [pid for pid in pair_ids if pid in pairs]
    if not pair_ids:
        raise SystemExit("No selected pair IDs were found in the prompt-pair file")

    device = pick_device(args.device)
    dtype = pick_dtype(device, args.dtype)
    loaded = load_model(entry, device, dtype)
    run_metadata = build_run_metadata(args, entry, loaded, device, dtype, pair_ids)
    runtime_cols = metadata_subset(run_metadata)

    summary_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    for pair_id in pair_ids:
        print(f"Capturing {args.model} {pair_id}", flush=True)
        srows, lrows = trajectory_rows_for_pair(
            loaded,
            pairs[pair_id],
            args.system_prompt,
            args.max_new_tokens,
            args.logit_max_steps,
            args.thinking_mode,
        )
        for row in srows:
            row["model_name"] = args.model
            row.update(runtime_cols)
        for row in lrows:
            row["model_name"] = args.model
            row.update(runtime_cols)
        summary_rows.extend(srows)
        layer_rows.extend(lrows)

    summary_path = args.out_dir / f"{args.model}_silent_divergence_summary.csv"
    layer_path = args.out_dir / f"{args.model}_silent_divergence_layers.csv"
    metadata_path = args.out_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(run_metadata, indent=2, sort_keys=True), encoding="utf-8")
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    with layer_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(layer_rows[0].keys()))
        writer.writeheader()
        writer.writerows(layer_rows)
    print(f"Wrote {summary_path}")
    print(f"Wrote {layer_path}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
