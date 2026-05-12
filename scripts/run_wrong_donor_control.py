#!/usr/bin/env python3
"""Wrong-donor patching control (E11 Phase 4 ancillary).

Tests donor-specificity: at each target case's winning (position class,
best layer), substitute A's clean-cache activation with a donor case's
clean-cache activation at the same layer and same position-class slot.
Donor cases are drawn from the same model but have different A-branch
tokens. If rescue fraction stays near 0 and strict replay fails, the
patching signatures rely on donor-specific information. If donors
often rescue, the signatures are less about branch-specific content
and more about position-sensitive perturbation.

Minimal: one model (qwen35_2b), ~12 cases from the panel, 3 random
donors per case. Local MPS.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[1]


def pick_device(req: str) -> torch.device:
    if req == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(req)


def find_blocks(model: Any) -> Any:
    for path in ("model.language_model.layers", "model.layers", "model.model.layers"):
        cur = model
        ok = True
        for part in path.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok:
            return cur
    raise RuntimeError("no transformer blocks found")


def first_tensor(x: Any) -> torch.Tensor:
    return x[0] if isinstance(x, tuple) else x


def replace_first(x: Any, v: torch.Tensor) -> Any:
    return (v, *x[1:]) if isinstance(x, tuple) else v


def format_prompt(tok: Any, prompt: str, sys_prompt: str | None) -> str:
    msgs = []
    if sys_prompt:
        msgs.append({"role": "system", "content": sys_prompt})
    msgs.append({"role": "user", "content": prompt})
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    return (f"{sys_prompt}\n\n" if sys_prompt else "") + f"User: {prompt}\nAssistant:"


def cache_clean_activations(model, blocks, input_ids) -> list[torch.Tensor]:
    cache: list[torch.Tensor | None] = [None] * len(blocks)
    handles = []

    def make_hook(i):
        def hook(_m, _i, output):
            cache[i] = first_tensor(output).detach()
        return hook

    for i, blk in enumerate(blocks):
        handles.append(blk.register_forward_hook(make_hook(i)))
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return [c for c in cache if c is not None]


def patched_last_logits(model, blocks, input_ids, donor_vec, layer_idx, corrupt_pos):
    def hook(_m, _i, output):
        hidden = first_tensor(output).clone()
        src = donor_vec.to(device=hidden.device, dtype=hidden.dtype)
        hidden[:, corrupt_pos, :] = src
        return replace_first(output, hidden)
    handle = blocks[layer_idx].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            out = model(input_ids=input_ids, use_cache=False)
        return out.logits[0, -1, :].detach().float().cpu()
    finally:
        handle.remove()


def greedy_decode(model, tokenizer, input_ids, max_new_tokens):
    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return out[0, input_ids.shape[1]:].detach().cpu().tolist()


def load_prompt_pair(repo: Path, model_key: str, pair_id: str) -> tuple[str, str]:
    pairs = json.loads((repo / f"configs/prompt_pairs_token_certified/{model_key}.json").read_text())
    for row in pairs:
        if row["id"] == pair_id:
            return row["prompt_a"], row["prompt_b"]
    raise KeyError(pair_id)


def position_for_class(
    position_class: str,
    clean_prompt_len: int,
    corrupt_prompt_len: int,
    clean_prompt_ids: list[int],
    corrupt_prompt_ids: list[int],
    common_prefix_len: int,
) -> tuple[int, int] | None:
    """Resolve (clean_pos, corrupt_pos) for the requested class.

    For prompt_lcp we use the first differing prompt token index.
    For final_context we use the last position before branch, which
    equals (prompt_len + common_prefix_len - 1) on each side after the
    shared generated prefix is appended. For generated_prefix we pick
    the position of the last shared generated token.
    """
    if position_class == "prompt_lcp":
        lcp = 0
        n = min(len(clean_prompt_ids), len(corrupt_prompt_ids))
        for i in range(n):
            if clean_prompt_ids[i] != corrupt_prompt_ids[i]:
                lcp = i
                break
        else:
            lcp = n
        return lcp, lcp
    if position_class == "final_context":
        # Full input = prompt + common_prefix; last index on each side.
        return (
            clean_prompt_len + common_prefix_len - 1,
            corrupt_prompt_len + common_prefix_len - 1,
        )
    if position_class == "generated_prefix":
        if common_prefix_len <= 0:
            return None
        # Last shared generated token.
        return (
            clean_prompt_len + common_prefix_len - 1,
            corrupt_prompt_len + common_prefix_len - 1,
        )
    return None


def rescue_fraction(clean_m: float, corrupt_m: float, patched_m: float) -> float:
    den = clean_m - corrupt_m
    if abs(den) < 1e-9:
        return float("nan")
    return (patched_m - corrupt_m) / den


def select_qwen2b_targets(repo: Path) -> pd.DataFrame:
    panel = pd.read_csv(
        repo / "runs/rankings/activation_patch_comparison/case_level_summary.csv",
        low_memory=False,
    )
    ev = pd.read_csv(
        repo / "runs/trajectory_events/logit_token_cert_v1/trajectory_events.csv",
        low_memory=False,
    )
    m = panel.merge(
        ev[["model_name", "pair_id", "event_kind"]].drop_duplicates(["model_name", "pair_id"]),
        on=["model_name", "pair_id"],
        how="left",
    )
    m = m[(m["model_name"] == "qwen35_2b") & (~m["pair_id"].str.endswith("__reverse"))].copy()
    m = m[m["best_position_class"].isin(["prompt_lcp", "generated_prefix", "final_context"])]
    # Only keep ones where strict replay worked — otherwise the "winning" position
    # is not a clean reference point.
    m = m[m["replayable"] == True]

    def signature(r):
        if bool(r["prompt_lcp_full"]):
            return "boundary_rescue"
        if r["event_kind"] == "immediate_visible_branch":
            return "tok_shift_immediate"
        return "gen_prefix_after_silent"

    m["signature"] = m.apply(signature, axis=1)
    return m[["pair_id", "signature", "best_position_class", "best_layer"]].reset_index(drop=True)


def run(args):
    repo = REPO
    device = pick_device(args.device)
    models = {m["name"]: m for m in json.loads((repo / "configs/models.json").read_text())}
    entry = models[args.model]
    dtype = torch.float16 if device.type == "mps" else torch.bfloat16
    print(f"device={device} dtype={dtype} model={entry['name']}", flush=True)
    tok = AutoTokenizer.from_pretrained(entry["model_id"], trust_remote_code=bool(entry.get("trust_remote_code")))
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        entry["model_id"], dtype=dtype, trust_remote_code=bool(entry.get("trust_remote_code"))
    ).to(device)
    model.eval()
    blocks = find_blocks(model)

    cases = select_qwen2b_targets(repo)
    if args.limit:
        cases = cases.head(args.limit)
    print(f"candidates: {len(cases)}  signatures: {cases['signature'].value_counts().to_dict()}", flush=True)

    sys_prompt = args.system_prompt

    # Pre-compute clean caches and metadata for ALL candidate cases (so any can
    # serve as donor).
    case_ctx: dict[str, dict] = {}
    for i, row in cases.iterrows():
        pair_id = row["pair_id"]
        try:
            prompt_a, prompt_b = load_prompt_pair(repo, args.model, pair_id)
        except KeyError:
            continue
        fa = format_prompt(tok, prompt_a, sys_prompt)
        fb = format_prompt(tok, prompt_b, sys_prompt)
        ids_a = tok(fa, return_tensors="pt").input_ids.to(device)
        ids_b = tok(fb, return_tensors="pt").input_ids.to(device)
        gen_a = greedy_decode(model, tok, ids_a, args.decode_len)
        gen_b = greedy_decode(model, tok, ids_b, args.decode_len)
        # first differing generated token
        lcp = 0
        n = min(len(gen_a), len(gen_b))
        while lcp < n and gen_a[lcp] == gen_b[lcp]:
            lcp += 1
        if lcp >= n:
            print(f"  skip {pair_id}: no local branch in {args.decode_len} tokens", flush=True)
            continue
        a_branch = gen_a[lcp]
        b_branch = gen_b[lcp]
        shared = gen_a[:lcp]
        clean_input = torch.tensor([ids_a[0].tolist() + shared], dtype=torch.long, device=device)
        corrupt_input = torch.tensor([ids_b[0].tolist() + shared], dtype=torch.long, device=device)
        clean_cache = cache_clean_activations(model, blocks, clean_input)
        # base metrics
        with torch.inference_mode():
            lg_clean = model(input_ids=clean_input, use_cache=False).logits[0, -1, :].detach().float().cpu()
            lg_corrupt = model(input_ids=corrupt_input, use_cache=False).logits[0, -1, :].detach().float().cpu()
        clean_m = float(lg_clean[a_branch] - lg_clean[b_branch])
        corrupt_m = float(lg_corrupt[a_branch] - lg_corrupt[b_branch])
        case_ctx[pair_id] = {
            "row": row.to_dict(),
            "clean_input": clean_input,
            "corrupt_input": corrupt_input,
            "clean_cache": clean_cache,
            "clean_prompt_len": int(ids_a.shape[1]),
            "corrupt_prompt_len": int(ids_b.shape[1]),
            "clean_prompt_ids": ids_a[0].tolist(),
            "corrupt_prompt_ids": ids_b[0].tolist(),
            "common_prefix_len": lcp,
            "a_branch": a_branch,
            "b_branch": b_branch,
            "clean_m": clean_m,
            "corrupt_m": corrupt_m,
        }

    pair_ids = list(case_ctx.keys())
    print(f"loaded {len(pair_ids)} cases with local branches", flush=True)

    rng = random.Random(args.seed)
    rows = []
    t0 = time.time()
    for target_pid in pair_ids:
        ctx = case_ctx[target_pid]
        row = ctx["row"]
        pos_class = row["best_position_class"]
        layer = int(row["best_layer"])
        tgt_pos = position_for_class(
            pos_class,
            ctx["clean_prompt_len"],
            ctx["corrupt_prompt_len"],
            ctx["clean_prompt_ids"],
            ctx["corrupt_prompt_ids"],
            ctx["common_prefix_len"],
        )
        if tgt_pos is None:
            continue
        clean_pos, corrupt_pos = tgt_pos
        # --- self-patch reference (using target's own clean cache). Should rescue.
        self_donor_vec = ctx["clean_cache"][layer][:, clean_pos, :]
        self_logits = patched_last_logits(
            model, blocks, ctx["corrupt_input"], self_donor_vec, layer, corrupt_pos
        )
        self_m = float(self_logits[ctx["a_branch"]] - self_logits[ctx["b_branch"]])
        self_rescue = rescue_fraction(ctx["clean_m"], ctx["corrupt_m"], self_m)
        self_top1 = int(torch.argmax(self_logits).item())

        # --- donor pool: other pairs with different a_branch_token_id.
        donor_pool = [
            p for p in pair_ids
            if p != target_pid
            and case_ctx[p]["a_branch"] != ctx["a_branch"]
            and case_ctx[p]["clean_cache"][layer].shape[1] > 0
        ]
        rng.shuffle(donor_pool)
        donors = donor_pool[: args.donors_per_case]

        donor_rescues = []
        donor_strict_matches = 0
        for donor_pid in donors:
            dctx = case_ctx[donor_pid]
            # Donor position within the donor's own clean cache at the *same* class.
            d_pos = position_for_class(
                pos_class,
                dctx["clean_prompt_len"],
                dctx["corrupt_prompt_len"],
                dctx["clean_prompt_ids"],
                dctx["corrupt_prompt_ids"],
                dctx["common_prefix_len"],
            )
            if d_pos is None:
                continue
            d_clean_pos, _ = d_pos
            dcache = dctx["clean_cache"][layer]
            if d_clean_pos >= dcache.shape[1]:
                continue
            donor_vec = dcache[:, d_clean_pos, :]
            logits = patched_last_logits(
                model, blocks, ctx["corrupt_input"], donor_vec, layer, corrupt_pos
            )
            patched_m = float(logits[ctx["a_branch"]] - logits[ctx["b_branch"]])
            rf = rescue_fraction(ctx["clean_m"], ctx["corrupt_m"], patched_m)
            top1 = int(torch.argmax(logits).item())
            donor_rescues.append(rf)
            if top1 == ctx["a_branch"]:
                donor_strict_matches += 1
            rows.append({
                "target_pair_id": target_pid,
                "signature": row["signature"],
                "position_class": pos_class,
                "layer": layer,
                "kind": "donor",
                "donor_pair_id": donor_pid,
                "rescue_fraction": rf,
                "patched_top1_token_id": top1,
                "target_a_branch_token_id": ctx["a_branch"],
                "donor_a_branch_token_id": dctx["a_branch"],
                "strict_replay": int(top1 == ctx["a_branch"]),
            })

        rows.append({
            "target_pair_id": target_pid,
            "signature": row["signature"],
            "position_class": pos_class,
            "layer": layer,
            "kind": "self",
            "donor_pair_id": target_pid,
            "rescue_fraction": self_rescue,
            "patched_top1_token_id": self_top1,
            "target_a_branch_token_id": ctx["a_branch"],
            "donor_a_branch_token_id": ctx["a_branch"],
            "strict_replay": int(self_top1 == ctx["a_branch"]),
        })

        print(
            f"  {target_pid} [{row['signature']}] pos={pos_class} L{layer}  "
            f"self_rescue={self_rescue:.2f} self_match={self_top1==ctx['a_branch']}  "
            f"donors_n={len(donor_rescues)} donor_mean={sum(donor_rescues)/max(len(donor_rescues),1):.2f}  "
            f"donor_strict={donor_strict_matches}/{len(donor_rescues)}",
            flush=True,
        )

    out_dir = repo / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = out_dir / "donor_control.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path} (elapsed {round(time.time()-t0,1)}s)")

    if len(df):
        donor = df[df["kind"] == "donor"]
        self_df = df[df["kind"] == "self"]
        print("\n=== self-patch (sanity) ===")
        print(self_df.groupby("signature")[["rescue_fraction", "strict_replay"]].mean())
        print("\n=== wrong-donor patch ===")
        print(donor.groupby("signature")[["rescue_fraction", "strict_replay"]].agg(["mean", "count"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen35_2b")
    ap.add_argument("--decode-len", type=int, default=80)
    ap.add_argument("--donors-per-case", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260512)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--system-prompt",
        default="You are a concise, accurate assistant. Answer directly.",
    )
    ap.add_argument("--out-dir", default="runs/wrong_donor_control/qwen2b")
    args = ap.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
