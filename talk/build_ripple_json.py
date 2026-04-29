#!/usr/bin/env python3
"""Build viz-friendly ripple.json from the raw ripple.jsonl.

Shape: per-category grid of layer x token metrics for the ripple map viz.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "runs" / "ripple_qwen35_2b" / "ripple.jsonl"
META = ROOT / "runs" / "ripple_qwen35_2b" / "metadata.json"
OUT = ROOT / "talk" / "data" / "ripple.json"

CATEGORY_ORDER = [
    "control_identical",
    "noop_format",
    "punctuation",
    "synonym",
    "paraphrase",
    "semantic_small",
    "positive_control",
]


def main() -> None:
    meta = json.loads(META.read_text())
    n_layers = int(meta["n_layers"])

    rows_by_pair: dict[str, list[dict]] = defaultdict(list)
    pair_category: dict[str, str] = {}
    pair_id_by_category: dict[str, str] = {}

    with SRC.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pid = row["pair_id"]
            cat = row["category"]
            pair_category[pid] = cat
            pair_id_by_category.setdefault(cat, pid)
            rows_by_pair[pid].append(row)

    pairs_out: dict[str, dict] = {}
    for cat in CATEGORY_ORDER:
        pid = pair_id_by_category.get(cat)
        if pid is None:
            continue
        rows = sorted(rows_by_pair[pid], key=lambda r: r["token_idx"])
        n_tokens = len(rows)

        def grid(key: str) -> list[list[float]]:
            # [n_layers][n_tokens]
            return [
                [float(r["layers"][L][key]) for r in rows]
                for L in range(n_layers)
            ]

        tokens_a = [r["token_text_a"] for r in rows]
        tokens_b = [r["token_text_b"] for r in rows]
        topk_kl = [float(r.get("topk_kl_ab", 0.0) or 0.0) for r in rows]

        pairs_out[cat] = {
            "pair_id": pid,
            "n_tokens": n_tokens,
            "tokens_a": tokens_a,
            "tokens_b": tokens_b,
            "top_token_a": tokens_a,
            "top_token_b": tokens_b,
            "resid_cos": grid("resid_cos"),
            "attn_cos": grid("attn_cos"),
            "mlp_cos": grid("mlp_cos"),
            "resid_norm_a": grid("resid_norm_a"),
            "resid_norm_b": grid("resid_norm_b"),
            "topk_kl": topk_kl,
        }

    out = {
        "model": "Qwen3.5-2B",
        "n_layers": n_layers,
        "hidden_dim": meta.get("hidden_dim"),
        "categories": [c for c in CATEGORY_ORDER if c in pairs_out],
        "pairs": pairs_out,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out))
    size_kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT} ({size_kb:.1f} KB, {len(pairs_out)} categories)")


if __name__ == "__main__":
    main()
