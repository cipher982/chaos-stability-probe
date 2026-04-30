#!/usr/bin/env python3
"""Extract Qwen-Scope SAE features at branch-patching positions.

This is intentionally narrow: it uses the Qwen3.5 2B residual-stream SAE
release and reads features at the same clean/corrupt positions used by the
branch activation-patching experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download

from activation_patch_branch import (
    find_blocks,
    first_tensor,
    select_model,
    select_pair,
)
from run_stability_probe import (
    append_continuation,
    common_prefix_len,
    decode_token,
    generate_once,
    load_model,
    pick_device,
    pick_dtype,
    prompt_token_payload,
    tokenize_prompt,
)


def capture_layer_residual(loaded: Any, blocks: Any, inputs: dict[str, torch.Tensor], layer: int) -> torch.Tensor:
    captured: dict[str, torch.Tensor] = {}

    def hook(_module: Any, _inp: Any, output: Any) -> None:
        captured["residual"] = first_tensor(output).detach().float().cpu()

    handle = blocks[layer].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            loaded.model(**inputs, use_cache=False)
    finally:
        handle.remove()
    if "residual" not in captured:
        raise RuntimeError(f"Failed to capture residual at layer {layer}")
    return captured["residual"][0]


def load_sae(repo_id: str, layer: int) -> dict[str, torch.Tensor]:
    path = hf_hub_download(repo_id=repo_id, filename=f"layer{layer}.sae.pt")
    sae = torch.load(path, map_location="cpu")
    return {key: value.detach().float().cpu() for key, value in sae.items()}


def feature_acts(sae: dict[str, torch.Tensor], residual: torch.Tensor, top_k: int) -> list[tuple[int, float]]:
    pre_acts = residual @ sae["W_enc"].T + sae["b_enc"]
    vals, idxs = pre_acts.topk(top_k)
    return [(int(idx), float(val)) for idx, val in zip(idxs.tolist(), vals.tolist(), strict=True)]


def selected_positions(
    clean_len: int,
    corrupt_len: int,
    prompt_lcp: int,
    generated_prefix_len: int,
) -> list[tuple[str, int, int]]:
    positions = [
        ("prompt_lcp_token", min(prompt_lcp, clean_len - 1), min(prompt_lcp, corrupt_len - 1)),
        ("final_context_token", clean_len - 1, corrupt_len - 1),
    ]
    if generated_prefix_len:
        positions.append(
            (
                f"last_generated_prefix_token_{generated_prefix_len - 1}",
                clean_len - 1,
                corrupt_len - 1,
            )
        )
    deduped = []
    seen = set()
    for item in positions:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path("configs/models.json"))
    parser.add_argument("--model", default="qwen35_2b")
    parser.add_argument("--prompt-pairs", type=Path, default=Path("configs/prompt_pairs_mechinterp_seed.json"))
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--sae-repo", default="Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_100")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 1, 23])
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/mechinterp_sae"))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--thinking-mode", choices=["default", "enabled", "disabled"], default="disabled")
    parser.add_argument("--system-prompt", default="You are a concise, accurate assistant. Answer directly.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    entry = select_model(args.models, args.model)
    pair = select_pair(args.prompt_pairs, args.pair_id)
    device = pick_device(args.device)
    dtype = pick_dtype(device, args.dtype)
    loaded = load_model(entry, device, dtype)
    _block_path, blocks = find_blocks(loaded.model)

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

    common_prefix = gen_a["generated_tokens"][:first_diff]
    clean_prompt_inputs = tokenize_prompt(loaded, pair["prompt_a"], args.system_prompt, args.thinking_mode)
    corrupt_prompt_inputs = tokenize_prompt(loaded, pair["prompt_b"], args.system_prompt, args.thinking_mode)
    clean_inputs = append_continuation(clean_prompt_inputs, common_prefix)
    corrupt_inputs = append_continuation(corrupt_prompt_inputs, common_prefix)
    clean_len = int(clean_inputs["input_ids"].shape[1])
    corrupt_len = int(corrupt_inputs["input_ids"].shape[1])
    prompt_delta = prompt_token_payload(loaded, pair["prompt_a"], pair["prompt_b"], args.system_prompt, args.thinking_mode)
    positions = selected_positions(clean_len, corrupt_len, int(prompt_delta["prompt_token_lcp"]), len(common_prefix))

    rows: list[dict[str, Any]] = []
    for layer in args.layers:
        sae = load_sae(args.sae_repo, layer)
        clean_residual = capture_layer_residual(loaded, blocks, clean_inputs, layer)
        corrupt_residual = capture_layer_residual(loaded, blocks, corrupt_inputs, layer)
        for label, clean_pos, corrupt_pos in positions:
            for side, residual, pos in [
                ("clean", clean_residual, clean_pos),
                ("corrupt", corrupt_residual, corrupt_pos),
            ]:
                token_id = int((clean_inputs if side == "clean" else corrupt_inputs)["input_ids"][0, pos])
                for rank, (feature_id, activation) in enumerate(feature_acts(sae, residual[pos], args.top_k), start=1):
                    rows.append(
                        {
                            "model_name": loaded.name,
                            "sae_repo": args.sae_repo,
                            "pair_id": pair["id"],
                            "category": pair["category"],
                            "layer": layer,
                            "position_label": label,
                            "side": side,
                            "pos": pos,
                            "token_id": token_id,
                            "token_text": decode_token(loaded.tokenizer, token_id),
                            "rank": rank,
                            "feature_id": feature_id,
                            "activation": activation,
                        }
                    )

    stem = f"{loaded.name}__{pair['id']}__sae_features"
    out_csv = args.out_dir / f"{stem}.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "model": entry,
        "pair": pair,
        "sae_repo": args.sae_repo,
        "layers": args.layers,
        "top_k": args.top_k,
        "first_diff_token": first_diff,
        "a_branch_token": decode_token(loaded.tokenizer, int(gen_a["generated_tokens"][first_diff])),
        "b_branch_token": decode_token(loaded.tokenizer, int(gen_b["generated_tokens"][first_diff])),
        "prompt_delta": {key: value for key, value in prompt_delta.items() if not key.startswith("prompt_token_ids_")},
        "positions": [
            {"label": label, "clean_pos": clean_pos, "corrupt_pos": corrupt_pos}
            for label, clean_pos, corrupt_pos in positions
        ],
    }
    (args.out_dir / f"{stem}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
