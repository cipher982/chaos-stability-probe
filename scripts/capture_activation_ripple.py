#!/usr/bin/env python3
"""Capture activation-perturbation ripple data.

Single prompt, no generation. For each injection layer in a schedule, inject
Gaussian noise = 1% of the activation L2 norm at one token position, then
record normalized residual delta at every downstream layer x token.

Output:
    runs/activation_ripple_qwen35_2b/
        metadata.json
        activation_ripple.json
    talk/data/activation_ripple.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3.5-2B"
FALLBACK = "Qwen/Qwen3.5-0.8B"
SEED = 1234
PERTURB_FRAC = 0.01
INJECT_LAYERS = [0, 4, 8, 12, 16, 20, 23]
PROMPT_PAIR_ID = "control_identical_weather"


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def find_decoder_layers(model):
    cands = []

    def visit(mod, path):
        if isinstance(mod, torch.nn.ModuleList) and len(mod) > 0:
            first = mod[0]
            has_attn = any("attn" in n.lower() for n, _ in first.named_children())
            has_mlp = any(n.lower() in ("mlp", "feed_forward", "ffn") for n, _ in first.named_children())
            if has_attn and has_mlp:
                cands.append((path, mod))
        for name, child in mod.named_children():
            visit(child, f"{path}.{name}" if path else name)

    visit(model, "")
    cands.sort(key=lambda x: (-len(x[1]), len(x[0])))
    return cands[0]


def find_embed(model):
    """Find the input token embedding module."""
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Embedding):
            # Heuristic: the big vocab-sized embedding, not positional
            if mod.num_embeddings > 1000:
                return name, mod
    raise RuntimeError("No embedding found")


def main():
    out_dir = Path("runs/activation_ripple_qwen35_2b")
    out_dir.mkdir(parents=True, exist_ok=True)
    talk_data_dir = Path("talk/data")
    talk_data_dir.mkdir(parents=True, exist_ok=True)

    # Load prompt
    pairs = json.loads(Path("configs/prompt_pairs.json").read_text())
    pair = next(p for p in pairs if p["id"] == PROMPT_PAIR_ID)
    prompt = pair["prompt_a"]

    device = pick_device()
    print(f"Device: {device}")

    torch.manual_seed(SEED)

    model_id = MODEL_ID
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float16, low_cpu_mem_usage=True,
        )
    except Exception as e:
        print(f"Falling back to {FALLBACK}: {e}")
        model_id = FALLBACK
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float16, low_cpu_mem_usage=True,
        )
    model.to(device).eval()

    # Build input using chat template
    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
    else:
        text = prompt
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    seq_len = input_ids.shape[1]
    token_strs = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]
    print(f"Seq len: {seq_len}")

    inject_token = seq_len // 2
    print(f"Injection token: {inject_token} -> {token_strs[inject_token]!r}")

    decoder_path, layers = find_decoder_layers(model)
    n_layers = len(layers)
    print(f"Decoder: {decoder_path} layers={n_layers}")

    # We'll hook each decoder layer's forward to capture the post-layer residual
    # (i.e. residual AFTER block L; "layer L output"). For "layer 0" we capture
    # the embedding (pre-layer-0 residual) via an embedding hook.
    # So we have n_layers+1 residual states indexed 0..n_layers.
    #   resid[0] = embedding output (pre-layer-0)
    #   resid[L] for L>=1 = output of decoder layer L-1

    # Control flag for injection
    # State for a given forward pass:
    # - inject_target_layer: int in 0..n_layers, or None
    # - noise_tensor: fixed tensor to add at that injection point
    state = {
        "inject_target": None,   # index into resid (0..n_layers)
        "noise": None,           # [hidden] tensor on device
        "captured": [None] * (n_layers + 1),  # each is cpu float32 [seq, hidden]
    }

    def embed_hook(module, inputs, output):
        # output: [batch, seq, hidden]
        x = output
        if state["inject_target"] == 0 and state["noise"] is not None:
            x = x.clone()
            x[0, inject_token, :] = x[0, inject_token, :] + state["noise"].to(x.dtype)
            state["captured"][0] = x[0].detach().to("cpu").to(torch.float32)
            return x
        state["captured"][0] = x[0].detach().to("cpu").to(torch.float32)
        return output

    def make_layer_hook(layer_idx):
        # layer_idx is 0..n_layers-1; corresponds to resid index layer_idx+1
        resid_idx = layer_idx + 1

        def hook(module, inputs, output):
            if isinstance(output, tuple):
                t = output[0]
                rest = output[1:]
            else:
                t = output
                rest = None
            if state["inject_target"] == resid_idx and state["noise"] is not None:
                t = t.clone()
                t[0, inject_token, :] = t[0, inject_token, :] + state["noise"].to(t.dtype)
                state["captured"][resid_idx] = t[0].detach().to("cpu").to(torch.float32)
                if rest is not None:
                    return (t,) + rest
                return t
            state["captured"][resid_idx] = t[0].detach().to("cpu").to(torch.float32)
            return output
        return hook

    # Find embedding module
    embed_name, embed_mod = find_embed(model)
    print(f"Embedding: {embed_name}")

    handles = [embed_mod.register_forward_hook(embed_hook)]
    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_layer_hook(i)))

    t0 = time.time()

    # -------- Clean forward pass --------
    state["inject_target"] = None
    state["noise"] = None
    state["captured"] = [None] * (n_layers + 1)
    with torch.no_grad():
        model(input_ids=input_ids, use_cache=False)
    clean = [c.clone() for c in state["captured"]]
    assert all(c is not None for c in clean), "Missing clean capture"
    print(f"Clean pass done ({time.time()-t0:.1f}s). Resid shapes: {clean[0].shape}")

    # -------- Perturbed forward passes --------
    rng = np.random.default_rng(SEED)
    injections = []
    validation_rows = []

    for L_inject in INJECT_LAYERS:
        # Build noise scaled to 1% of clean resid norm at (L_inject, inject_token)
        vec = clean[L_inject][inject_token]  # float32 [hidden]
        vec_norm = float(vec.norm().item())
        noise_np = rng.standard_normal(vec.shape[0]).astype(np.float32)
        noise_np = noise_np / (np.linalg.norm(noise_np) + 1e-9) * (PERTURB_FRAC * vec_norm)
        noise = torch.from_numpy(noise_np).to(device)

        state["inject_target"] = L_inject
        state["noise"] = noise
        state["captured"] = [None] * (n_layers + 1)
        with torch.no_grad():
            model(input_ids=input_ids, use_cache=False)
        perturbed = state["captured"]

        # Compute normalized delta at every downstream layer (>= L_inject)
        ripple = []
        for Ld in range(L_inject, n_layers + 1):
            pert = perturbed[Ld]
            base = clean[Ld]
            # Per-token: L2(pert - base) / L2(base)
            diff = (pert - base).to(torch.float32)
            base_norms = base.to(torch.float32).norm(dim=-1)  # [seq]
            diff_norms = diff.norm(dim=-1)  # [seq]
            rel = (diff_norms / (base_norms + 1e-9)).numpy().astype(np.float32)
            ripple.append(rel.tolist())

        ripple_arr = np.array(ripple)  # [downstream_layers, seq]
        mean_at_inject = float(ripple_arr[0].mean())
        mean_at_final = float(ripple_arr[-1].mean())
        max_at_final = float(ripple_arr[-1].max())
        validation_rows.append({
            "L_inject": L_inject,
            "mean_at_inject": mean_at_inject,
            "mean_at_final": mean_at_final,
            "max_at_final": max_at_final,
        })
        print(f"  L_inject={L_inject:2d}  mean@inject={mean_at_inject:.4f}  mean@final={mean_at_final:.4f}  max@final={max_at_final:.4f}")

        injections.append({
            "inject_layer": L_inject,
            "inject_token": inject_token,
            "ripple": ripple,
        })

    for h in handles:
        h.remove()

    total_time = time.time() - t0

    # -------- Validation report --------
    print("\n=== VALIDATION ===")
    print(f"{'L_inj':>6} {'mean@inj':>10} {'mean@final':>12} {'max@final':>10}")
    for r in validation_rows:
        print(f"{r['L_inject']:>6} {r['mean_at_inject']:>10.4f} {r['mean_at_final']:>12.4f} {r['max_at_final']:>10.4f}")

    shallow = [r for r in validation_rows if r["L_inject"] <= 4]
    deep = [r for r in validation_rows if r["L_inject"] >= 16 and r["L_inject"] < n_layers]
    shallow_avg = float(np.mean([r["mean_at_final"] for r in shallow])) if shallow else 0.0
    deep_avg = float(np.mean([r["mean_at_final"] for r in deep])) if deep else 0.0
    print(f"\nShallow (L<=4) avg mean@final: {shallow_avg:.4f}   (expect < 0.01 for contraction)")
    print(f"Deep    (L>=16) avg mean@final: {deep_avg:.4f}   (expect > 0.01 for amplification)")
    if shallow_avg < PERTURB_FRAC and deep_avg > PERTURB_FRAC:
        print("Li et al. pattern observed: shallow contracts, deep amplifies.")
    else:
        print("WARNING: Li et al. pattern NOT observed in this single-prompt / single-token setup.")

    # -------- Save outputs --------
    out = {
        "model": model_id,
        "n_layers": n_layers,
        "seq_len": seq_len,
        "prompt": prompt,
        "token_strs": token_strs,
        "injection_token": inject_token,
        "perturb_frac": PERTURB_FRAC,
        "inject_layers": INJECT_LAYERS,
        "validation": validation_rows,
        "injections": injections,
    }

    main_path = out_dir / "activation_ripple.json"
    main_path.write_text(json.dumps(out))
    (talk_data_dir / "activation_ripple.json").write_text(json.dumps(out))

    meta = {
        "model_id": model_id,
        "n_layers": n_layers,
        "hidden_dim": int(clean[0].shape[-1]),
        "seq_len": seq_len,
        "prompt_pair_id": PROMPT_PAIR_ID,
        "prompt_text": prompt,
        "injection_token": inject_token,
        "injection_layers": INJECT_LAYERS,
        "perturb_frac": PERTURB_FRAC,
        "seed": SEED,
        "device": str(device),
        "decoder_path": decoder_path,
        "runtime_s": round(total_time, 1),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    size_kb = main_path.stat().st_size / 1024
    print(f"\nWrote {main_path} ({size_kb:.1f} KB)")
    print(f"Wrote {talk_data_dir / 'activation_ripple.json'}")
    print(f"Total wall time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
