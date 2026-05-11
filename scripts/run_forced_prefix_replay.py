#!/usr/bin/env python3
"""Forced-prefix replay (E11 Phase 3, minimal).

For each (model, pair_id):
  1. Greedy-decode A and A'. Find local branch_t' = first differing token.
  2. Force A' through A's tokens a_0 .. a_{branch_t'} (inclusive of branch).
  3. Free-decode K more tokens. Measure token LCP with A's continuation.
  4. Also log p_{A'}(a_{branch_t'}) and its rank in A' logits at branch_t'.

Writes one CSV row per case plus a JSONL with full continuations.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


@dataclass
class Loaded:
    name: str
    model_id: str
    tokenizer: Any
    model: Any
    device: torch.device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(req: str) -> torch.device:
    if req == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(req)


def load_model(entry: dict, device: torch.device) -> Loaded:
    dtype = torch.float16 if device.type == "mps" else torch.bfloat16
    tok = AutoTokenizer.from_pretrained(
        entry["model_id"], trust_remote_code=bool(entry.get("trust_remote_code", False))
    )
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        entry["model_id"],
        dtype=dtype,
        trust_remote_code=bool(entry.get("trust_remote_code", False)),
    ).to(device)
    model.eval()
    return Loaded(entry["name"], entry["model_id"], tok, model, device)


def format_prompt(tok: Any, prompt: str, system_prompt: str | None) -> str:
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": prompt})
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    prefix = f"{system_prompt}\n\n" if system_prompt else ""
    return f"{prefix}User: {prompt}\nAssistant:"


def tokenize(loaded: Loaded, text: str) -> torch.Tensor:
    ids = loaded.tokenizer(text, return_tensors="pt").input_ids.to(loaded.device)
    return ids


def greedy_decode(
    loaded: Loaded, input_ids: torch.Tensor, max_new_tokens: int
) -> list[int]:
    with torch.inference_mode():
        out = loaded.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=loaded.tokenizer.pad_token_id or loaded.tokenizer.eos_token_id,
        )
    input_len = int(input_ids.shape[1])
    return out[0, input_len:].detach().cpu().tolist()


def forward_last_logits(loaded: Loaded, input_ids: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        out = loaded.model(input_ids=input_ids)
    return out.logits[0, -1, :].float()


def prob_and_rank_of_token(logits: torch.Tensor, token_id: int) -> tuple[float, int]:
    probs = torch.softmax(logits, dim=-1)
    p = float(probs[token_id])
    rank = int((logits > logits[token_id]).sum().item()) + 1
    return p, rank


def first_diff_index(a: list[int], b: list[int]) -> int | None:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return None


def token_lcp(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def load_qwen35_2b_cases(repo: Path) -> pd.DataFrame:
    """Select cases: all qwen35_2b pairs with regime assignments from E07 panel,
    plus the Phase 1 hero (0434) which is outside the panel."""
    panel = pd.read_csv(
        repo / "runs/rankings/activation_patch_comparison/case_level_summary.csv",
        low_memory=False,
    )
    ev = pd.read_csv(
        repo / "runs/trajectory_events/logit_token_cert_v1/trajectory_events.csv",
        low_memory=False,
    )
    m = panel.merge(
        ev[["model_name", "pair_id", "event_kind", "branch_t"]].drop_duplicates(
            ["model_name", "pair_id"]
        ),
        on=["model_name", "pair_id"],
        how="left",
    )
    m = m[m["model_name"] == "qwen35_2b"].copy()
    # Skip __reverse synthetic rows (no trajectory event).
    m = m[~m["pair_id"].str.endswith("__reverse")]

    def regime(r: pd.Series) -> str:
        if bool(r["prompt_lcp_full"]):
            return "edit_boundary"
        if r["event_kind"] == "immediate_visible_branch":
            return "prompt_accumulation"
        return "trajectory_migration"

    m["regime"] = m.apply(regime, axis=1)
    out = m[["pair_id", "regime", "event_kind", "branch_t"]].drop_duplicates("pair_id")

    # Phase 1 hero: qwen35_2b parenthesize_word_0434 — not in the panel. Regime from
    # aligned patching was edit_boundary-borderline; treat as edit_boundary here.
    if "token_cert_parenthesize_word_0434" not in set(out["pair_id"]):
        hero_row = ev[
            (ev["model_name"] == "qwen35_2b")
            & (ev["pair_id"] == "token_cert_parenthesize_word_0434")
        ]
        if not hero_row.empty:
            r = hero_row.iloc[0]
            out = pd.concat(
                [
                    out,
                    pd.DataFrame(
                        [
                            {
                                "pair_id": "token_cert_parenthesize_word_0434",
                                "regime": "edit_boundary",
                                "event_kind": r["event_kind"],
                                "branch_t": r["branch_t"],
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
    return out.reset_index(drop=True)


def load_prompt_pair(repo: Path, model_key: str, pair_id: str) -> tuple[str, str]:
    path = repo / f"configs/prompt_pairs_token_certified/{model_key}.json"
    pairs = json.loads(path.read_text())
    for row in pairs:
        if row["id"] == pair_id:
            return row["prompt_a"], row["prompt_b"]
    raise KeyError(f"{pair_id} not in {path}")


def run_case(
    loaded: Loaded,
    pair_id: str,
    prompt_a: str,
    prompt_b: str,
    system_prompt: str | None,
    decode_len: int,
    post_force_len: int,
) -> dict:
    # --- Tokenize both prompts with chat template.
    fa = format_prompt(loaded.tokenizer, prompt_a, system_prompt)
    fb = format_prompt(loaded.tokenizer, prompt_b, system_prompt)
    ids_a = tokenize(loaded, fa)
    ids_b = tokenize(loaded, fb)

    # --- Free decode both.
    set_seed(1234)
    gen_a = greedy_decode(loaded, ids_a, decode_len)
    set_seed(1234)
    gen_b = greedy_decode(loaded, ids_b, decode_len)

    # --- Find local branch_t' on generated stream.
    local_branch_t = first_diff_index(gen_a, gen_b)
    if local_branch_t is None:
        # Generations identical; nothing to replay.
        return {
            "pair_id": pair_id,
            "local_branch_t": None,
            "a_branch_token_id": None,
            "b_branch_token_id": None,
            "a_branch_token_text": None,
            "b_branch_token_text": None,
            "p_A_prime_of_a_branch_token": None,
            "rank_a_branch_token_in_A_prime": None,
            "forced_rejoin_lcp": None,
            "forced_continuation_text": None,
            "a_continuation_text": None,
            "note": "no_local_branch",
        }

    a_branch_id = gen_a[local_branch_t]
    b_branch_id = gen_b[local_branch_t]

    # --- p_{A'}(a_branch_token) at branch_t'.
    prefix_len_b = int(ids_b.shape[1])
    shared_prefix = gen_a[:local_branch_t]
    # Input to get A's logits-at-branch under A' prompt:
    # (A' prompt) + shared pre-branch tokens — model will predict at position branch_t'.
    b_probe_ids = torch.tensor(
        [ids_b[0].tolist() + shared_prefix], dtype=torch.long, device=loaded.device
    )
    logits_b_at_branch = forward_last_logits(loaded, b_probe_ids)
    p_a_branch, rank_a_branch = prob_and_rank_of_token(logits_b_at_branch, a_branch_id)

    # --- Forced-prefix replay: force A' through a_0..a_{branch_t'} (inclusive),
    # then free-decode post_force_len more tokens.
    forced_ids = torch.tensor(
        [ids_b[0].tolist() + gen_a[: local_branch_t + 1]],
        dtype=torch.long,
        device=loaded.device,
    )
    set_seed(1234)
    forced_cont = greedy_decode(loaded, forced_ids, post_force_len)

    # --- Rejoin metric: token LCP of (A's post-branch tail) vs (forced continuation).
    a_tail = gen_a[local_branch_t + 1 : local_branch_t + 1 + post_force_len]
    rejoin = token_lcp(a_tail, forced_cont)

    return {
        "pair_id": pair_id,
        "local_branch_t": local_branch_t,
        "a_branch_token_id": a_branch_id,
        "b_branch_token_id": b_branch_id,
        "a_branch_token_text": loaded.tokenizer.decode(
            [a_branch_id], skip_special_tokens=False
        ),
        "b_branch_token_text": loaded.tokenizer.decode(
            [b_branch_id], skip_special_tokens=False
        ),
        "p_A_prime_of_a_branch_token": p_a_branch,
        "rank_a_branch_token_in_A_prime": rank_a_branch,
        "forced_rejoin_lcp": rejoin,
        "forced_continuation_text": loaded.tokenizer.decode(
            forced_cont, skip_special_tokens=True
        ),
        "a_continuation_text": loaded.tokenizer.decode(
            a_tail, skip_special_tokens=True
        ),
        "note": "ok",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen35_2b")
    ap.add_argument(
        "--system-prompt",
        default="You are a concise, accurate assistant. Answer directly.",
    )
    ap.add_argument("--decode-len", type=int, default=64)
    ap.add_argument("--post-force-len", type=int, default=10)
    ap.add_argument("--device", default="auto")
    ap.add_argument(
        "--out-dir",
        default="runs/forced_prefix_replay/phase3_qwen2b",
    )
    ap.add_argument("--limit", type=int, default=0, help="0 = all cases")
    args = ap.parse_args()

    repo = REPO
    models = {m["name"]: m for m in json.loads((repo / "configs/models.json").read_text())}
    if args.model not in models:
        raise SystemExit(f"unknown model {args.model}")
    entry = models[args.model]
    device = pick_device(args.device)
    print(f"device={device} dtype={'fp16' if device.type=='mps' else 'bf16'} model={args.model}", flush=True)
    loaded = load_model(entry, device)

    cases = load_qwen35_2b_cases(repo)
    if args.limit:
        cases = cases.head(args.limit)
    print(f"cases: {len(cases)}  regimes: {cases['regime'].value_counts().to_dict()}", flush=True)

    out_dir = repo / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    detail_path = out_dir / "detail.jsonl"
    detail_f = detail_path.open("w")

    t0 = time.time()
    for i, case in cases.iterrows():
        pair_id = case["pair_id"]
        try:
            prompt_a, prompt_b = load_prompt_pair(repo, args.model, pair_id)
        except KeyError as e:
            print(f"[skip] {pair_id}: {e}", flush=True)
            continue
        t_case = time.time()
        res = run_case(
            loaded,
            pair_id,
            prompt_a,
            prompt_b,
            args.system_prompt,
            args.decode_len,
            args.post_force_len,
        )
        res["regime"] = case["regime"]
        res["event_kind"] = case["event_kind"]
        res["sagemaker_branch_t"] = case["branch_t"]
        res["elapsed_s"] = round(time.time() - t_case, 2)
        rows.append(res)
        detail_f.write(json.dumps(res) + "\n")
        detail_f.flush()
        marker = (
            f"rejoin={res['forced_rejoin_lcp']}"
            if res["forced_rejoin_lcp"] is not None
            else f"note={res['note']}"
        )
        print(
            f"[{i+1}/{len(cases)}] {pair_id} regime={res['regime']} "
            f"local_t={res['local_branch_t']} {marker} ({res['elapsed_s']}s)",
            flush=True,
        )
    detail_f.close()

    df = pd.DataFrame(rows)
    csv_path = out_dir / "summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path}  (elapsed {round(time.time()-t0,1)}s)")

    if len(df):
        agg = (
            df[df["forced_rejoin_lcp"].notna()]
            .groupby("regime")["forced_rejoin_lcp"]
            .agg(["count", "mean", "median", "min", "max"])
        )
        print("\nregime x rejoin_lcp:\n", agg)

    return 0


if __name__ == "__main__":
    sys.exit(main())
