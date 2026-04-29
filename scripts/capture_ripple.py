#!/usr/bin/env python3
"""Capture per-layer / per-token internal state from a small Qwen model
under prompt perturbations, for the "ripple map" visualization.

For each prompt pair (A, B) we greedy-decode both sides from the same seed,
and at every decoding step we collect, per transformer block:
    - post-block residual hidden state
    - attention sublayer output
    - MLP sublayer output
Plus top-K LM-head logits at the final layer.

All norms/cosines are computed in float32. For all layers we store L2 norms
and A-vs-B cosine distances. For layer 0, mid, and final we also store the
raw float16 vectors (both A and B) for two priority pairs so downstream viz
can run PCA/UMAP.

Output:
    runs/ripple_qwen35_2b/
        metadata.json
        ripple.jsonl  (one line per pair_id x token_idx)
        raw/<pair_id>_<A|B>_layer<L>_tok<T>.npy (priority pairs only)
        manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# -----------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------

DEFAULT_MODEL = "Qwen/Qwen3.5-2B"
FALLBACK_MODEL = "Qwen/Qwen3.5-0.8B"
SEED = 1234
TOPK_K = 50
PRIORITY_CATEGORIES = {"paraphrase", "semantic_small"}  # store raw vectors for these


# -----------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    a32 = a.detach().to(torch.float32).flatten()
    b32 = b.detach().to(torch.float32).flatten()
    denom = (a32.norm() * b32.norm()).item()
    if denom == 0.0:
        return 0.0
    cos = (a32 @ b32).item() / denom
    return float(1.0 - cos)


def l2_norm(x: torch.Tensor) -> float:
    return float(x.detach().to(torch.float32).flatten().norm().item())


# -----------------------------------------------------------------------
# Model discovery
# -----------------------------------------------------------------------


def find_decoder_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    """Find the transformer decoder layer list.

    Handles plain CausalLM (model.model.layers) and multimodal conditional
    generation wrappers that expose a nested language_model submodule.
    """
    candidates = []

    def visit(mod: torch.nn.Module, path: str) -> None:
        # ModuleList of decoder-layer-like modules
        if isinstance(mod, torch.nn.ModuleList) and len(mod) > 0:
            first = mod[0]
            has_attn = any("attn" in n.lower() for n, _ in first.named_children())
            has_mlp = any(n.lower() in ("mlp", "feed_forward", "ffn") for n, _ in first.named_children())
            if has_attn and has_mlp:
                candidates.append((path, mod))
        for name, child in mod.named_children():
            visit(child, f"{path}.{name}" if path else name)

    visit(model, "")
    if not candidates:
        raise RuntimeError("Could not locate decoder layers in model")
    # Pick deepest / longest ModuleList (main transformer stack)
    candidates.sort(key=lambda x: (-len(x[1]), len(x[0])))
    return candidates[0]  # (path, ModuleList)


def find_attn_mlp(layer: torch.nn.Module) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Find the self-attention and MLP submodules inside a decoder layer."""
    attn = None
    mlp = None
    for name, child in layer.named_children():
        lname = name.lower()
        if "attn" in lname and attn is None:
            attn = child
        if lname in ("mlp", "feed_forward", "ffn") and mlp is None:
            mlp = child
    if attn is None or mlp is None:
        raise RuntimeError(
            f"Layer has children {[n for n,_ in layer.named_children()]}; "
            "could not find attn/mlp"
        )
    return attn, mlp


# -----------------------------------------------------------------------
# Hook capture
# -----------------------------------------------------------------------


class LayerCapture:
    """Registers forward hooks on each decoder layer's attn, mlp, and the
    layer itself; exposes a per-step snapshot dict."""

    def __init__(self, layers: list[torch.nn.Module]):
        self.layers = layers
        self.n_layers = len(layers)
        self.attn_out: list[torch.Tensor | None] = [None] * self.n_layers
        self.mlp_out: list[torch.Tensor | None] = [None] * self.n_layers
        self.block_out: list[torch.Tensor | None] = [None] * self.n_layers
        self._handles: list[Any] = []

    def _make_output_hook(self, store: list[torch.Tensor | None], idx: int):
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                t = output[0]
            else:
                t = output
            # Take last-token vector: shape [batch, seq, hidden] -> [hidden]
            store[idx] = t[:, -1, :].detach().to("cpu")
        return hook

    def register(self) -> None:
        for i, layer in enumerate(self.layers):
            attn, mlp = find_attn_mlp(layer)
            self._handles.append(attn.register_forward_hook(self._make_output_hook(self.attn_out, i)))
            self._handles.append(mlp.register_forward_hook(self._make_output_hook(self.mlp_out, i)))
            self._handles.append(layer.register_forward_hook(self._make_output_hook(self.block_out, i)))

    def clear_step(self) -> None:
        for i in range(self.n_layers):
            self.attn_out[i] = None
            self.mlp_out[i] = None
            self.block_out[i] = None

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


# -----------------------------------------------------------------------
# Generation with capture
# -----------------------------------------------------------------------


def build_input_ids(tokenizer, prompt: str, device: torch.device) -> torch.Tensor:
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        text = prompt
    enc = tokenizer(text, return_tensors="pt")
    return enc.input_ids.to(device)


@torch.no_grad()
def generate_with_capture(
    model: torch.nn.Module,
    tokenizer,
    capture: LayerCapture,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
) -> dict:
    """Greedy decode and, at each step, record per-layer snapshots and top-K logits."""
    input_ids = build_input_ids(tokenizer, prompt, device)
    generated_ids: list[int] = []
    generated_text: list[str] = []

    # per-step snapshots
    attn_per_step: list[list[torch.Tensor]] = []  # [T][L] -> [hidden]
    mlp_per_step: list[list[torch.Tensor]] = []
    block_per_step: list[list[torch.Tensor]] = []
    topk_per_step: list[dict] = []

    past_key_values = None
    cur_ids = input_ids

    for step in range(max_new_tokens):
        capture.clear_step()
        out = model(
            input_ids=cur_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = out.past_key_values
        logits = out.logits[:, -1, :]  # [1, V]

        # Top-K logits
        topk = torch.topk(logits[0].to(torch.float32), k=TOPK_K)
        topk_per_step.append(
            {"indices": topk.indices.tolist(), "values": topk.values.tolist()}
        )

        # Greedy next token
        next_id = int(torch.argmax(logits, dim=-1).item())
        generated_ids.append(next_id)
        generated_text.append(tokenizer.decode([next_id], skip_special_tokens=False))

        # Snapshot captures
        attn_per_step.append([t.clone() if t is not None else None for t in capture.attn_out])
        mlp_per_step.append([t.clone() if t is not None else None for t in capture.mlp_out])
        block_per_step.append([t.clone() if t is not None else None for t in capture.block_out])

        # Feed next token
        cur_ids = torch.tensor([[next_id]], device=device, dtype=input_ids.dtype)

        # Early stop on EOS
        if tokenizer.eos_token_id is not None and next_id == tokenizer.eos_token_id:
            break

    return {
        "generated_ids": generated_ids,
        "generated_text": generated_text,
        "attn": attn_per_step,
        "mlp": mlp_per_step,
        "block": block_per_step,
        "topk": topk_per_step,
    }


# -----------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------


def topk_kl(a_topk: dict, b_topk: dict) -> float:
    """KL(softmax(a) || softmax(b)) restricted to the union of top-K indices.
    Logits for missing indices treated as -inf for that side.
    """
    a_idx = a_topk["indices"]; a_val = a_topk["values"]
    b_idx = b_topk["indices"]; b_val = b_topk["values"]
    union = sorted(set(a_idx) | set(b_idx))
    a_map = dict(zip(a_idx, a_val))
    b_map = dict(zip(b_idx, b_val))
    a_logits = torch.tensor([a_map.get(i, -1e9) for i in union], dtype=torch.float32)
    b_logits = torch.tensor([b_map.get(i, -1e9) for i in union], dtype=torch.float32)
    a_logp = F.log_softmax(a_logits, dim=-1)
    b_logp = F.log_softmax(b_logits, dim=-1)
    a_p = a_logp.exp()
    kl = (a_p * (a_logp - b_logp)).sum().item()
    return float(kl)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------


def select_pairs(all_pairs: list[dict]) -> list[dict]:
    """First pair per tier, plus 2 extra synonym pairs (if available)."""
    by_cat: dict[str, list[dict]] = {}
    for p in all_pairs:
        by_cat.setdefault(p["category"], []).append(p)
    selected = []
    order = [
        "control_identical",
        "noop_format",
        "punctuation",
        "synonym",
        "paraphrase",
        "semantic_small",
        "positive_control",
    ]
    for cat in order:
        if cat in by_cat:
            selected.append(by_cat[cat][0])
    # Extra synonym pairs (index 1 and 2 if present)
    syn = by_cat.get("synonym", [])
    for extra in syn[1:3]:
        selected.append(extra)
    return selected


def try_load_model(model_id: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device.type in ("mps", "cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    return tokenizer, model, dtype


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--output-dir", default="runs/ripple_qwen35_2b")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--pairs-file", default="configs/prompt_pairs.json")
    global TOPK_K
    ap.add_argument("--topk", type=int, default=TOPK_K)
    args = ap.parse_args()
    TOPK_K = args.topk

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    ripple_path = out_dir / "ripple.jsonl"
    if ripple_path.exists():
        ripple_path.unlink()

    pairs_all = json.loads(Path(args.pairs_file).read_text())
    pairs = select_pairs(pairs_all)
    print(f"Selected {len(pairs)} pairs: {[p['id'] for p in pairs]}")

    device = pick_device()
    print(f"Device: {device}")
    set_seed(SEED)

    t0 = time.time()
    model_id = args.model
    fallback_used = False
    try:
        tokenizer, model, dtype = try_load_model(model_id, device)
    except Exception as e:
        print(f"Failed to load {model_id}: {e}")
        print(f"Falling back to {FALLBACK_MODEL}")
        model_id = FALLBACK_MODEL
        fallback_used = True
        tokenizer, model, dtype = try_load_model(model_id, device)
    load_time = time.time() - t0
    print(f"Loaded {model_id} in {load_time:.1f}s, dtype={dtype}")

    # Locate decoder layers
    path, layer_list = find_decoder_layers(model)
    n_layers = len(layer_list)
    # Determine hidden_dim from config
    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, "text_config"):
        hidden_dim = cfg.text_config.hidden_size
    else:
        hidden_dim = cfg.hidden_size if cfg is not None else None
    print(f"Decoder stack at: {path}  |  layers={n_layers}  hidden_dim={hidden_dim}")

    capture = LayerCapture(list(layer_list))
    capture.register()

    manifest_rows = []
    priority_layer_indices = sorted({0, n_layers // 2, n_layers - 1})
    print(f"Priority raw-vector layers: {priority_layer_indices}")

    try:
        for pair in pairs:
            pid = pair["id"]
            cat = pair["category"]
            t_pair = time.time()
            try:
                set_seed(SEED)
                capA = generate_with_capture(
                    model, tokenizer, capture, pair["prompt_a"], args.max_new_tokens, device
                )
                set_seed(SEED)
                capB = generate_with_capture(
                    model, tokenizer, capture, pair["prompt_b"], args.max_new_tokens, device
                )
            except Exception as e:
                print(f"[{pid}] ERROR: {e}")
                manifest_rows.append({"pair_id": pid, "category": cat, "tokens": 0, "error": str(e)})
                continue

            T = min(len(capA["generated_ids"]), len(capB["generated_ids"]))
            store_raw = cat in PRIORITY_CATEGORIES

            for t_idx in range(T):
                layer_rows = []
                for L in range(n_layers):
                    bA = capA["block"][t_idx][L]
                    bB = capB["block"][t_idx][L]
                    aA = capA["attn"][t_idx][L]
                    aB = capB["attn"][t_idx][L]
                    mA = capA["mlp"][t_idx][L]
                    mB = capB["mlp"][t_idx][L]
                    if any(x is None for x in (bA, bB, aA, aB, mA, mB)):
                        # hook didn't fire for this sublayer; record NaN-ish
                        layer_rows.append({
                            "layer": L,
                            "resid_cos": None, "attn_cos": None, "mlp_cos": None,
                            "resid_norm_a": None, "resid_norm_b": None,
                            "attn_norm_a": None, "attn_norm_b": None,
                            "mlp_norm_a": None, "mlp_norm_b": None,
                        })
                        continue
                    layer_rows.append({
                        "layer": L,
                        "resid_cos": cosine_distance(bA, bB),
                        "attn_cos": cosine_distance(aA, aB),
                        "mlp_cos": cosine_distance(mA, mB),
                        "resid_norm_a": l2_norm(bA), "resid_norm_b": l2_norm(bB),
                        "attn_norm_a": l2_norm(aA), "attn_norm_b": l2_norm(aB),
                        "mlp_norm_a": l2_norm(mA), "mlp_norm_b": l2_norm(mB),
                    })
                    # Save raw vectors for priority pairs at priority layers
                    if store_raw and L in priority_layer_indices:
                        np.save(
                            raw_dir / f"{pid}_A_resid_L{L}_T{t_idx}.npy",
                            bA.to(torch.float16).numpy(),
                        )
                        np.save(
                            raw_dir / f"{pid}_B_resid_L{L}_T{t_idx}.npy",
                            bB.to(torch.float16).numpy(),
                        )

                row = {
                    "pair_id": pid,
                    "category": cat,
                    "token_idx": t_idx,
                    "token_id_a": capA["generated_ids"][t_idx],
                    "token_id_b": capB["generated_ids"][t_idx],
                    "token_text_a": capA["generated_text"][t_idx],
                    "token_text_b": capB["generated_text"][t_idx],
                    "layers": layer_rows,
                    "topk_logits_a": capA["topk"][t_idx],
                    "topk_logits_b": capB["topk"][t_idx],
                    "topk_kl_ab": topk_kl(capA["topk"][t_idx], capB["topk"][t_idx]),
                }
                with ripple_path.open("a") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            dt = time.time() - t_pair
            print(f"[{pid}] category={cat} tokens={T} time={dt:.1f}s")
            manifest_rows.append(
                {"pair_id": pid, "category": cat, "tokens": T, "error": ""}
            )
    finally:
        capture.remove()

    # metadata
    meta = {
        "model_id": model_id,
        "requested_model": args.model,
        "fallback_used": fallback_used,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "decoder_path": path,
        "max_new_tokens": args.max_new_tokens,
        "topk": TOPK_K,
        "seed": SEED,
        "device": str(device),
        "dtype": str(dtype),
        "torch_version": torch.__version__,
        "priority_layers_raw": priority_layer_indices,
        "priority_categories_raw": sorted(PRIORITY_CATEGORIES),
        "pair_ids": [p["id"] for p in pairs],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "load_time_s": load_time,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    # manifest
    with (out_dir / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pair_id", "category", "tokens", "error"])
        w.writeheader()
        w.writerows(manifest_rows)

    # -------------- validation -----------------
    print("\n=== VALIDATION ===")
    # Reload the jsonl and summarize
    rows = [json.loads(l) for l in ripple_path.open()]
    by_cat: dict[str, list[dict]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)

    print(f"{'category':<22} mean_resid_cos(final, tok<=10)")
    for cat, rs in by_cat.items():
        vals = []
        for r in rs:
            if r["token_idx"] <= 10 and r["layers"]:
                v = r["layers"][-1]["resid_cos"]
                if v is not None:
                    vals.append(v)
        if vals:
            print(f"{cat:<22} {np.mean(vals):.6f}   (n={len(vals)})")

    # control should be ~0
    ctrl = by_cat.get("control_identical", [])
    if ctrl:
        ctrl_max = max(
            (lr["resid_cos"] or 0.0)
            for r in ctrl for lr in r["layers"]
        )
        print(f"control_identical MAX resid_cos across all layers/tokens: {ctrl_max:.2e}")

    pos = by_cat.get("positive_control", [])
    if pos:
        pos_vals = [lr["resid_cos"] for r in pos for lr in r["layers"] if lr["resid_cos"] is not None]
        if pos_vals:
            print(f"positive_control mean resid_cos across all layers/tokens: {np.mean(pos_vals):.4f}")

    total = time.time() - t0
    print(f"\nTotal wall time: {total:.1f}s")
    print(f"Output: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
