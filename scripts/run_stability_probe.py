#!/usr/bin/env python3
"""Run prompt and hidden-state stability probes for local/open LLMs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@dataclass
class LoadedModel:
    name: str
    model_id: str
    tokenizer: Any
    model: Any
    device: torch.device


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device: torch.device, requested: str) -> torch.dtype:
    if requested != "auto":
        return {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[requested]
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def load_model(entry: dict[str, Any], device: torch.device, dtype: torch.dtype) -> LoadedModel:
    model_id = entry["model_id"]
    trust_remote_code = bool(entry.get("trust_remote_code", False))
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict[str, Any] = {
        "dtype": dtype,
        "trust_remote_code": trust_remote_code,
    }
    quantization = entry.get("quantization")
    if quantization:
        if device.type != "cuda":
            raise ValueError(f"Quantization {quantization!r} requires CUDA in this harness")
        if quantization == "bnb_8bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif quantization == "bnb_4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type=entry.get("bnb_4bit_quant_type", "nf4"),
            )
        else:
            raise ValueError(f"Unknown quantization mode: {quantization}")

    if device.type == "cuda":
        device_map = entry.get("device_map", "auto")
        if device_map == "single":
            kwargs["device_map"] = {"": 0}
        elif device_map != "none":
            kwargs["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if device.type != "cuda" or entry.get("device_map") == "none":
        model = model.to(device)
    model.eval()

    return LoadedModel(
        name=entry["name"],
        model_id=model_id,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )


def chat_template_kwargs(thinking_mode: str) -> dict[str, Any]:
    if thinking_mode == "disabled":
        return {"enable_thinking": False}
    if thinking_mode == "enabled":
        return {"enable_thinking": True}
    return {}


def format_prompt(tokenizer: Any, prompt: str, system_prompt: str | None, thinking_mode: str) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs(thinking_mode),
        )
    if system_prompt:
        return f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
    return prompt


def tokenize_prompt(
    loaded: LoadedModel,
    prompt: str,
    system_prompt: str | None,
    thinking_mode: str,
) -> dict[str, torch.Tensor]:
    text = format_prompt(loaded.tokenizer, prompt, system_prompt, thinking_mode)
    inputs = loaded.tokenizer(text, return_tensors="pt")
    return {k: v.to(loaded.device) for k, v in inputs.items()}


def prompt_token_payload(
    loaded: LoadedModel,
    prompt_a: str,
    prompt_b: str,
    system_prompt: str | None,
    thinking_mode: str,
) -> dict[str, Any]:
    formatted_a = format_prompt(loaded.tokenizer, prompt_a, system_prompt, thinking_mode)
    formatted_b = format_prompt(loaded.tokenizer, prompt_b, system_prompt, thinking_mode)
    ids_a = loaded.tokenizer(formatted_a)["input_ids"]
    ids_b = loaded.tokenizer(formatted_b)["input_ids"]
    edit = levenshtein(ids_a, ids_b)
    lcp = common_prefix_len(ids_a, ids_b)
    if edit == 0:
        kind = "token_identical"
    elif edit == 1 and len(ids_b) == len(ids_a) + 1:
        kind = "one_token_insert"
    elif edit == 1 and len(ids_b) == len(ids_a) - 1:
        kind = "one_token_delete"
    elif edit == 1 and len(ids_b) == len(ids_a):
        kind = "one_token_substitution"
    elif edit <= 3:
        kind = "small_token_delta_2_3"
    else:
        kind = "multi_token_delta"
    return {
        "formatted_prompt_equal": formatted_a == formatted_b,
        "prompt_token_equal": ids_a == ids_b,
        "prompt_token_edit_distance": edit,
        "prompt_token_delta_kind": kind,
        "prompt_token_lcp": lcp,
        "prompt_input_tokens_a": len(ids_a),
        "prompt_input_tokens_b": len(ids_b),
        "prompt_input_token_delta": len(ids_b) - len(ids_a),
        "prompt_token_ids_a": ids_a,
        "prompt_token_ids_b": ids_b,
    }


def generate_once(
    loaded: LoadedModel,
    prompt: str,
    system_prompt: str | None,
    max_new_tokens: int,
    sample: bool,
    temperature: float,
    top_p: float,
    seed: int,
    thinking_mode: str,
) -> dict[str, Any]:
    set_seed(seed)
    inputs = tokenize_prompt(loaded, prompt, system_prompt, thinking_mode)
    input_len = int(inputs["input_ids"].shape[1])
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": sample,
        "pad_token_id": loaded.tokenizer.pad_token_id or loaded.tokenizer.eos_token_id,
    }
    if sample:
        generation_kwargs.update({"temperature": temperature, "top_p": top_p})

    start = time.time()
    with torch.inference_mode():
        output_ids = loaded.model.generate(**inputs, **generation_kwargs)
    elapsed = time.time() - start

    generated_ids = output_ids[0, input_len:].detach().cpu().tolist()
    generated_text = loaded.tokenizer.decode(generated_ids, skip_special_tokens=True)
    return {
        "prompt_chars": len(prompt),
        "input_tokens": input_len,
        "generated_tokens": generated_ids,
        "generated_token_count": len(generated_ids),
        "generated_text": generated_text,
        "elapsed_s": elapsed,
    }


def decode_token(tokenizer: Any, token_id: int) -> str:
    try:
        return tokenizer.decode([int(token_id)], skip_special_tokens=False)
    except Exception:
        return ""


def topk_logits_payload(logits: torch.Tensor, tokenizer: Any, k: int) -> list[dict[str, Any]]:
    k = min(k, int(logits.numel()))
    values, indices = torch.topk(logits.float(), k=k)
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    probs = torch.exp(log_probs[indices])
    rows = []
    for rank, (token_id, logit, prob, log_prob) in enumerate(zip(indices, values, probs, log_probs[indices]), 1):
        tid = int(token_id)
        rows.append(
            {
                "rank": rank,
                "token_id": tid,
                "token": decode_token(tokenizer, tid),
                "logit": float(logit),
                "prob": float(prob),
                "log_prob": float(log_prob),
            }
        )
    return rows


def rank_of_token(logits: torch.Tensor, token_id: int) -> int:
    value = logits[int(token_id)]
    return int((logits > value).sum().item()) + 1


def logit_distribution_metrics(logits_a: torch.Tensor, logits_b: torch.Tensor) -> dict[str, Any]:
    a = logits_a.float()
    b = logits_b.float()
    logp_a = torch.log_softmax(a, dim=-1)
    logp_b = torch.log_softmax(b, dim=-1)
    p_a = torch.exp(logp_a)
    p_b = torch.exp(logp_b)
    m = 0.5 * (p_a + p_b)
    log_m = torch.log(torch.clamp(m, min=torch.finfo(torch.float32).tiny))

    top2_a = torch.topk(a, k=2)
    top2_b = torch.topk(b, k=2)
    a_top1 = int(top2_a.indices[0])
    b_top1 = int(top2_b.indices[0])
    centered_a = a - a.mean()
    centered_b = b - b.mean()
    entropy_a = float(-(p_a * logp_a).sum())
    entropy_b = float(-(p_b * logp_b).sum())

    return {
        "kl_a_to_b": float((p_a * (logp_a - logp_b)).sum()),
        "kl_b_to_a": float((p_b * (logp_b - logp_a)).sum()),
        "js_divergence": float(
            0.5 * (p_a * (logp_a - log_m)).sum() + 0.5 * (p_b * (logp_b - log_m)).sum()
        ),
        "entropy_a": entropy_a,
        "entropy_b": entropy_b,
        "effective_branching_factor_a": float(math.exp(entropy_a)),
        "effective_branching_factor_b": float(math.exp(entropy_b)),
        "top1_same": a_top1 == b_top1,
        "a_top1_token_id": a_top1,
        "b_top1_token_id": b_top1,
        "a_top1_prob": float(p_a[a_top1]),
        "b_top1_prob": float(p_b[b_top1]),
        "a_top1_margin_logit": float(top2_a.values[0] - top2_a.values[1]),
        "b_top1_margin_logit": float(top2_b.values[0] - top2_b.values[1]),
        "a_top1_rank_in_b": rank_of_token(b, a_top1),
        "b_top1_rank_in_a": rank_of_token(a, b_top1),
        "a_top1_logit_delta": float(b[a_top1] - a[a_top1]),
        "b_top1_logit_delta": float(b[b_top1] - a[b_top1]),
        "mean_abs_logit_delta": float(torch.mean(torch.abs(b - a))),
        "rms_logit_delta": float(torch.sqrt(torch.mean((b - a) ** 2))),
        "max_abs_logit_delta": float(torch.max(torch.abs(b - a))),
        "centered_logit_cosine_distance": cosine_distance(centered_a, centered_b),
        "centered_logit_normalized_l2": normalized_l2(centered_a, centered_b),
    }


def model_logits_for_input_ids(
    loaded: LoadedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    with torch.inference_mode():
        out = loaded.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    return out.logits[0].detach().cpu()


def append_continuation(inputs: dict[str, torch.Tensor], continuation_ids: list[int]) -> dict[str, torch.Tensor]:
    if not continuation_ids:
        return inputs
    suffix = torch.tensor([continuation_ids], dtype=inputs["input_ids"].dtype, device=inputs["input_ids"].device)
    suffix_mask = torch.ones_like(suffix, dtype=inputs["attention_mask"].dtype)
    return {
        "input_ids": torch.cat([inputs["input_ids"], suffix], dim=1),
        "attention_mask": torch.cat([inputs["attention_mask"], suffix_mask], dim=1),
    }


def logit_probe_rows(
    loaded: LoadedModel,
    prompt_a: str,
    prompt_b: str,
    system_prompt: str | None,
    gen_a_ids: list[int],
    gen_b_ids: list[int],
    top_k: int,
    max_steps: int,
    thinking_mode: str,
) -> list[dict[str, Any]]:
    inputs_a = tokenize_prompt(loaded, prompt_a, system_prompt, thinking_mode)
    inputs_b = tokenize_prompt(loaded, prompt_b, system_prompt, thinking_mode)
    prompt_len_a = int(inputs_a["input_ids"].shape[1])
    prompt_len_b = int(inputs_b["input_ids"].shape[1])

    rows = []
    anchors = [
        ("prompt_a_generation", gen_a_ids[:max_steps]),
        ("prompt_b_generation", gen_b_ids[:max_steps]),
    ]
    for anchor_name, continuation in anchors:
        full_a = append_continuation(inputs_a, continuation)
        full_b = append_continuation(inputs_b, continuation)
        logits_a_all = model_logits_for_input_ids(loaded, full_a["input_ids"], full_a["attention_mask"])
        logits_b_all = model_logits_for_input_ids(loaded, full_b["input_ids"], full_b["attention_mask"])

        for t in range(len(continuation) + 1):
            pos_a = prompt_len_a + t - 1
            pos_b = prompt_len_b + t - 1
            logits_a = logits_a_all[pos_a]
            logits_b = logits_b_all[pos_b]
            metrics = logit_distribution_metrics(logits_a, logits_b)
            rows.append(
                {
                    "anchor": anchor_name,
                    "t": t,
                    "teacher_forced_token_id": None if t == 0 else int(continuation[t - 1]),
                    "teacher_forced_token": None if t == 0 else decode_token(loaded.tokenizer, int(continuation[t - 1])),
                    **metrics,
                    "topk_a": topk_logits_payload(logits_a, loaded.tokenizer, top_k),
                    "topk_b": topk_logits_payload(logits_b, loaded.tokenizer, top_k),
                }
            )
    return rows


def levenshtein(a: list[Any], b: list[Any]) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (ca != cb)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def common_prefix_len(a: list[Any], b: list[Any]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def generation_metrics(a_ids: list[int], b_ids: list[int], a_text: str, b_text: str) -> dict[str, Any]:
    token_edit = levenshtein(a_ids, b_ids)
    token_den = max(len(a_ids), len(b_ids), 1)
    char_edit = levenshtein(list(a_text), list(b_text))
    char_den = max(len(a_text), len(b_text), 1)
    lcp = common_prefix_len(a_ids, b_ids)
    return {
        "a_generated_tokens": len(a_ids),
        "b_generated_tokens": len(b_ids),
        "common_prefix_tokens": lcp,
        "first_diff_token": None if lcp == min(len(a_ids), len(b_ids)) else lcp,
        "token_edit_distance": token_edit,
        "token_edit_distance_norm": token_edit / token_den,
        "char_edit_distance": char_edit,
        "char_edit_distance_norm": char_edit / char_den,
    }


def curve_rows(a_ids: list[int], b_ids: list[int]) -> list[dict[str, Any]]:
    max_len = max(len(a_ids), len(b_ids))
    rows = []
    for t in range(1, max_len + 1):
        a_prefix = a_ids[: min(t, len(a_ids))]
        b_prefix = b_ids[: min(t, len(b_ids))]
        den = max(len(a_prefix), len(b_prefix), 1)
        rows.append(
            {
                "t": t,
                "token_edit_distance_norm": levenshtein(a_prefix, b_prefix) / den,
                "common_prefix_tokens": common_prefix_len(a_prefix, b_prefix),
            }
        )
    return rows


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    denom = torch.linalg.norm(a) * torch.linalg.norm(b)
    if float(denom) == 0.0:
        return math.nan
    value = float(1.0 - torch.dot(a, b) / denom)
    return max(0.0, min(2.0, value))


def normalized_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    denom = 0.5 * (torch.linalg.norm(a) + torch.linalg.norm(b))
    if float(denom) == 0.0:
        return math.nan
    return float(torch.linalg.norm(a - b) / denom)


def hidden_state_rows(
    loaded: LoadedModel,
    prompt_a: str,
    prompt_b: str,
    system_prompt: str | None,
    thinking_mode: str,
) -> list[dict[str, Any]]:
    inputs_a = tokenize_prompt(loaded, prompt_a, system_prompt, thinking_mode)
    inputs_b = tokenize_prompt(loaded, prompt_b, system_prompt, thinking_mode)

    with torch.inference_mode():
        out_a = loaded.model(**inputs_a, output_hidden_states=True, use_cache=False)
        out_b = loaded.model(**inputs_b, output_hidden_states=True, use_cache=False)

    rows = []
    for layer_idx, (ha, hb) in enumerate(zip(out_a.hidden_states, out_b.hidden_states)):
        last_a = ha[0, -1, :].detach().cpu()
        last_b = hb[0, -1, :].detach().cpu()
        mean_a = ha[0].mean(dim=0).detach().cpu()
        mean_b = hb[0].mean(dim=0).detach().cpu()
        rows.append(
            {
                "layer": layer_idx,
                "last_token_cosine_distance": cosine_distance(last_a, last_b),
                "last_token_normalized_l2": normalized_l2(last_a, last_b),
                "mean_pool_cosine_distance": cosine_distance(mean_a, mean_b),
                "mean_pool_normalized_l2": normalized_l2(mean_a, mean_b),
            }
        )
    return rows


def select_models(model_entries: list[dict[str, Any]], names: list[str], smoke: bool) -> list[dict[str, Any]]:
    if names:
        wanted = set(names)
        selected = [m for m in model_entries if m["name"] in wanted or m["model_id"] in wanted]
        missing = wanted - {m["name"] for m in selected} - {m["model_id"] for m in selected}
        if missing:
            raise SystemExit(f"Unknown model selectors: {sorted(missing)}")
        return selected
    if smoke:
        selected = [m for m in model_entries if m.get("enabled_for_smoke")]
        if selected:
            return selected
    return model_entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path("configs/models.json"))
    parser.add_argument("--prompt-pairs", type=Path, default=Path("configs/prompt_pairs.json"))
    parser.add_argument("--model", action="append", default=[], help="Model name or model_id. Repeatable.")
    parser.add_argument("--smoke", action="store_true", help="Use the smoke-test model selection.")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/local_smoke"))
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--limit-pairs", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--sample", action="store_true", help="Use sampled decoding instead of deterministic.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--different-seeds-within-pair",
        action="store_true",
        help="Use different seeds for prompt A/B. Default keeps pair seeds equal where possible.",
    )
    parser.add_argument("--skip-hidden", action="store_true")
    parser.add_argument(
        "--logit-probe",
        action="store_true",
        help="Capture full-vocab divergence metrics and top-k logit snapshots under teacher forcing.",
    )
    parser.add_argument("--logit-top-k", type=int, default=20)
    parser.add_argument("--logit-max-steps", type=int, default=128)
    parser.add_argument(
        "--thinking-mode",
        choices=["default", "enabled", "disabled"],
        default="default",
        help="Pass enable_thinking to chat templates that support it. Use disabled for Qwen thinking-off controls.",
    )
    parser.add_argument(
        "--skip-token-identical-non-controls",
        action="store_true",
        help="Skip non-control prompt pairs whose formatted prompt token IDs are identical.",
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a concise, accurate assistant. Answer directly.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    generation_path = args.out_dir / "generations.jsonl"
    summary_path = args.out_dir / "summary.csv"
    curve_path = args.out_dir / "curves.jsonl"
    hidden_path = args.out_dir / "hidden_states.jsonl"
    logit_path = args.out_dir / "logit_probes.jsonl"
    prompt_token_path = args.out_dir / "prompt_tokens.jsonl"
    skipped_path = args.out_dir / "skipped_pairs.jsonl"
    failures_path = args.out_dir / "failures.jsonl"
    for path in [generation_path, curve_path, hidden_path, logit_path, prompt_token_path, skipped_path, failures_path]:
        if path.exists():
            path.unlink()

    model_entries = select_models(load_json(args.models), args.model, args.smoke)
    prompt_pairs = load_json(args.prompt_pairs)
    if args.limit_pairs:
        prompt_pairs = prompt_pairs[: args.limit_pairs]

    device = pick_device(args.device)
    dtype = pick_dtype(device, args.dtype)
    metadata = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": str(device),
        "dtype": str(dtype),
        "torch": torch.__version__,
        "models": model_entries,
        "max_new_tokens": args.max_new_tokens,
        "sample": args.sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "repeats": args.repeats,
        "logit_probe": args.logit_probe,
        "logit_top_k": args.logit_top_k,
        "logit_max_steps": args.logit_max_steps,
        "thinking_mode": args.thinking_mode,
        "skip_token_identical_non_controls": args.skip_token_identical_non_controls,
        "system_prompt": args.system_prompt,
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    summary_rows: list[dict[str, Any]] = []
    for entry in model_entries:
        try:
            loaded = load_model(entry, device, dtype)
        except Exception as exc:
            write_jsonl(failures_path, {"stage": "load_model", "model": entry, "error": repr(exc)})
            continue

        for pair_idx, pair in enumerate(prompt_pairs):
            try:
                prompt_delta = prompt_token_payload(
                    loaded,
                    pair["prompt_a"],
                    pair["prompt_b"],
                    args.system_prompt,
                    args.thinking_mode,
                )
                prompt_token_row = {
                    "model_name": loaded.name,
                    "model_id": loaded.model_id,
                    "pair_id": pair["id"],
                    "category": pair["category"],
                    **prompt_delta,
                }
                write_jsonl(prompt_token_path, prompt_token_row)
                prompt_delta_scalars = {
                    k: v for k, v in prompt_delta.items() if not k.startswith("prompt_token_ids_")
                }
                if (
                    args.skip_token_identical_non_controls
                    and pair["category"] != "micro_control_identical"
                    and prompt_delta["prompt_token_edit_distance"] == 0
                ):
                    write_jsonl(skipped_path, {**prompt_token_row, "reason": "token_identical_non_control"})
                    continue

                for repeat_idx in range(args.repeats):
                    seed_a = args.seed + pair_idx * 1000 + repeat_idx
                    seed_b = seed_a if not args.different_seeds_within_pair else seed_a + 100_000
                    row_base = {
                        "model_name": loaded.name,
                        "model_id": loaded.model_id,
                        "pair_id": pair["id"],
                        "category": pair["category"],
                        "repeat": repeat_idx,
                        "sample": args.sample,
                        "seed_a": seed_a,
                        "seed_b": seed_b,
                        **prompt_delta_scalars,
                    }
                    gen_a = generate_once(
                        loaded,
                        pair["prompt_a"],
                        args.system_prompt,
                        args.max_new_tokens,
                        args.sample,
                        args.temperature,
                        args.top_p,
                        seed_a,
                        args.thinking_mode,
                    )
                    gen_b = generate_once(
                        loaded,
                        pair["prompt_b"],
                        args.system_prompt,
                        args.max_new_tokens,
                        args.sample,
                        args.temperature,
                        args.top_p,
                        seed_b,
                        args.thinking_mode,
                    )
                    write_jsonl(generation_path, {**row_base, "side": "a", **gen_a})
                    write_jsonl(generation_path, {**row_base, "side": "b", **gen_b})
                    metrics = generation_metrics(
                        gen_a["generated_tokens"],
                        gen_b["generated_tokens"],
                        gen_a["generated_text"],
                        gen_b["generated_text"],
                    )
                    summary_row = {**row_base, **metrics}
                    summary_rows.append(summary_row)
                    for curve in curve_rows(gen_a["generated_tokens"], gen_b["generated_tokens"]):
                        write_jsonl(curve_path, {**row_base, **curve})

                    if args.logit_probe:
                        for lrow in logit_probe_rows(
                            loaded,
                            pair["prompt_a"],
                            pair["prompt_b"],
                            args.system_prompt,
                            gen_a["generated_tokens"],
                            gen_b["generated_tokens"],
                            args.logit_top_k,
                            args.logit_max_steps,
                            args.thinking_mode,
                        ):
                            write_jsonl(logit_path, {**row_base, **lrow})

                if not args.skip_hidden:
                    for hrow in hidden_state_rows(
                        loaded,
                        pair["prompt_a"],
                        pair["prompt_b"],
                        args.system_prompt,
                        args.thinking_mode,
                    ):
                        write_jsonl(
                            hidden_path,
                            {
                                "model_name": loaded.name,
                                "model_id": loaded.model_id,
                                "pair_id": pair["id"],
                                "category": pair["category"],
                                "sample": args.sample,
                                **hrow,
                            },
                        )

            except Exception as exc:
                write_jsonl(
                    failures_path,
                    {
                        "model_name": loaded.name,
                        "model_id": loaded.model_id,
                        "pair_id": pair["id"],
                        "category": pair["category"],
                        "stage": "pair",
                        "error": repr(exc),
                    },
                )

        del loaded.model
        del loaded.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if summary_rows:
        with summary_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"Wrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
