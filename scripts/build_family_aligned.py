"""Build talk/data/family_aligned.json.

For each scaffold-long run we:
- detect the token-index answer-start boundary on each generation
- pair sides a/b by (pair_id, category)
- compute per-token edit distance on:
  * "raw" alignment (token 0 = first generated token) — same as family_curves
  * "aligned" alignment (token 0 = answer-start) — reasoning only; others dropped

Output schema mirrors family_curves.json but with `alignments` keyed by
"raw" / "aligned". Each panel entry also carries drop_rate and scaffold stats.
"""
from __future__ import annotations

import json
import statistics as stats
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from detect_answer_start import detect_answer_start


ROOT = Path("/Users/davidrose/git/chaos")
ART = ROOT / "runs/sagemaker_artifacts"

# One entry per model we want in the panel.
#   group: "reasoning" (reasoning model, scaffolded) |
#          "reasoning-thinkoff" (same reasoning model, think mode off) |
#          "non-reasoning"
MODELS = [
    # Reasoning
    ("Qwen3.5 4B",       "reasoning",          "chaos-scaffold-long-qwen35-4b-20260429-001/runs/qwen35_4b"),
    ("Qwen3.5 9B",       "reasoning",          "chaos-scaffold-long-qwen35-9b-20260429-001/runs/qwen35_9b"),
    ("DeepSeek-R1 7B",   "reasoning",          "chaos-scaffold-long-deepseek-r1-qwen7b-20260429-001/runs/deepseek_r1_distill_qwen_7b"),
    ("Phi-4 reasoning+", "reasoning",          "chaos-scaffold-long-phi4-reasoning-plus-20260429-001/runs/phi4_reasoning_plus"),
    ("SmolLM3 3B",       "reasoning",          "chaos-scaffold-long-smollm3-3b-20260429-001/runs/smollm3_3b"),
    # Thinkoff (natively no scaffold)
    ("Qwen3.5 0.8B think-off", "reasoning-thinkoff", "chaos-scaffold-long-qwen35-08b-thinkoff-20260429-001/runs/qwen35_08b"),
    ("Qwen3.5 2B think-off",   "reasoning-thinkoff", "chaos-scaffold-long-qwen35-2b-thinkoff-20260429-001/runs/qwen35_2b"),
    ("Qwen3.5 4B think-off",   "reasoning-thinkoff", "chaos-scaffold-long-qwen35-4b-thinkoff-20260429-001/runs/qwen35_4b"),
    ("Qwen3.5 9B think-off",   "reasoning-thinkoff", "chaos-scaffold-long-qwen35-9b-thinkoff-20260429-001/runs/qwen35_9b"),
    # Non-reasoning
    ("Qwen3.5 0.8B",     "non-reasoning",      "chaos-scaffold-long-qwen35-08b-20260429-001/runs/qwen35_08b"),
    ("Qwen3.5 2B",       "non-reasoning",      "chaos-scaffold-long-qwen35-2b-20260429-001/runs/qwen35_2b"),
    ("Mistral 7B v0.3",  "non-reasoning",      "chaos-scaffold-long-mistral7b-v03-20260429-001/runs/mistral7b_instruct_v03"),
    ("OLMo-3 7B",        "non-reasoning",      "chaos-scaffold-long-olmo3-7b-20260429-001/runs/olmo3_7b_instruct"),
    ("Granite 3.3 8B",   "non-reasoning",      "chaos-scaffold-long-granite33-8b-20260429-001/runs/granite33_8b_instruct"),
    ("Gemma 4 E4B it",   "non-reasoning",      "chaos-scaffold-long-gemma4-e4b-it-20260429-001/runs/gemma4_e4b_it"),
]

CATEGORIES = [
    "control_identical", "noop_format", "punctuation",
    "synonym", "paraphrase", "semantic_small", "positive_control",
]

T_MAX = 128
DRAFT_MARKERS = {"**Drafting final"}


def load_generations(run_dir: Path) -> list[dict]:
    path = run_dir / "generations.jsonl"
    if not path.exists():
        return []
    return [json.loads(x) for x in path.open()]


def detect_final_answer_start(text: str, token_ids: list[int]) -> tuple[Optional[int], Optional[str]]:
    """Detect a final-answer boundary, not an intermediate drafting section.

    `detect_answer_start` intentionally recognizes some draft headings for
    inspection, but this slide claims x=0 is where the answer starts. Drafting
    headings are still scaffold/deliberation, so this visualization drops them.
    """
    idx, marker = detect_answer_start(text, token_ids)
    if marker in DRAFT_MARKERS:
        return None, None
    return idx, marker


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


def edit_distance_prefix(a: list[int], b: list[int]) -> list[int]:
    """Return per-t token-level Levenshtein distance of prefixes a[:t] vs b[:t].

    This intentionally matches scripts/run_stability_probe.py::curve_rows.
    A same-index mismatch count falsely treats a one-token insertion/deletion as
    sustained divergence; Levenshtein lets offset-only artifacts decay.
    """
    n = min(max(len(a), len(b)), T_MAX)
    out = []
    for t in range(1, n + 1):
        a_prefix = a[: min(t, len(a))]
        b_prefix = b[: min(t, len(b))]
        out.append(levenshtein(a_prefix, b_prefix))
    return out  # length n, out[t-1] = edits through position t


def build_for_model(name: str, group: str, rel: Path) -> dict:
    run_dir = ART / rel
    rows = load_generations(run_dir)
    if not rows:
        return None  # type: ignore[return-value]

    # Group rows by (pair_id, category) -> {side: row}
    pairs: dict[tuple, dict] = defaultdict(dict)
    for r in rows:
        key = (r.get("pair_id"), r.get("category"))
        pairs[key][r.get("side")] = r

    # Decide whether to attempt answer-start detection for this model.
    attempt_align = group == "reasoning"

    # Per-category collections
    # raw[cat] = list of per-t edit arrays (one per pair)
    raw_acc: dict[str, list[list[int]]] = defaultdict(list)
    aligned_acc: dict[str, list[list[int]]] = defaultdict(list)
    # scaffold_acc[cat] = list of [{x: negative_int, y: cumulative_edits_at_x}]
    # where x = raw_t - pair_scaffold_end, so end-of-scaffold = 0 for each pair
    scaffold_acc: dict[str, list[list[dict]]] = defaultdict(list)
    scaffold_lens: list[int] = []
    pairs_total = 0
    pairs_aligned = 0

    for (pair_id, cat), sides in pairs.items():
        if "a" not in sides or "b" not in sides:
            continue
        pairs_total += 1
        ra, rb = sides["a"], sides["b"]
        ta = ra.get("generated_tokens") or []
        tb = rb.get("generated_tokens") or []
        txa = ra.get("generated_text") or ""
        txb = rb.get("generated_text") or ""
        if not ta or not tb:
            continue

        # Raw alignment: compare from token 0
        raw_curve = edit_distance_prefix(ta, tb)
        raw_acc[cat].append(raw_curve)

        if attempt_align:
            ia, _ = detect_final_answer_start(txa, ta)
            ib, _ = detect_final_answer_start(txb, tb)
            if ia is not None and ib is not None:
                pairs_aligned += 1
                scaffold_lens.append(ia)
                scaffold_lens.append(ib)
                # Answer side: slice at each pair's detected answer-start
                sa = ta[ia:]
                sb = tb[ib:]
                if sa and sb:
                    aligned_acc[cat].append(edit_distance_prefix(sa, sb))
                # Scaffold side: plot raw edit curve on x = raw_t - min(ia, ib)
                # so that each pair's scaffold-end lands at (or near) x=0.
                # Use min(ia, ib) because that's where the first pair finishes
                # scaffolding and we stop comparing scaffold-scaffold tokens.
                anchor = min(ia, ib)
                scaffold_pts = []
                for ti, val in enumerate(raw_curve):
                    t = ti + 1
                    if t <= anchor:
                        x = t - anchor  # negative or zero
                        scaffold_pts.append({"x": x, "y": val / t})
                if scaffold_pts:
                    scaffold_acc[cat].append(scaffold_pts)
        else:
            # For thinkoff / non-reasoning: aligned == raw (scaffold = 0)
            pairs_aligned += 1
            aligned_acc[cat].append(raw_curve)

    def make_cat_curves(acc: dict[str, list[list[int]]]) -> dict:
        out = {}
        for cat in CATEGORIES:
            arrs = acc.get(cat, [])
            if not arrs:
                out[cat] = []
                continue
            # mean per-t, normalized by t
            max_len = max(len(a) for a in arrs)
            pts = []
            for ti in range(max_len):
                vals = [a[ti] for a in arrs if len(a) > ti]
                if not vals:
                    continue
                t = ti + 1
                edit_mean = (sum(vals) / len(vals)) / t
                pts.append({"t": t, "edit_mean": edit_mean, "n": len(vals)})
            out[cat] = pts
        return out

    def make_scaffold_curves(acc: dict[str, list[list[dict]]]) -> dict:
        """Aggregate per-pair scaffold-aligned points into a mean curve per x.

        Every pair in a category contributes a list of {x, y} points (x negative,
        with end-of-scaffold landing at x=0 for THAT pair). We bin by integer x
        and compute mean y across pairs that have at least one point at that x.
        """
        out = {}
        for cat in CATEGORIES:
            pair_curves = acc.get(cat, [])
            if not pair_curves:
                out[cat] = []
                continue
            by_x: dict[int, list[float]] = defaultdict(list)
            for pc in pair_curves:
                for p in pc:
                    by_x[int(p["x"])].append(p["y"])
            # Require at least ceil(n_pairs / 3) contributions for trustworthy mean
            need = max(1, (len(pair_curves) + 2) // 3)
            pts = []
            for x in sorted(by_x.keys()):
                vals = by_x[x]
                if len(vals) >= need:
                    pts.append({"x": x, "edit_mean": sum(vals) / len(vals), "n": len(vals)})
            out[cat] = pts
        return out

    median_scaffold = stats.median(scaffold_lens) if scaffold_lens else 0
    p25 = stats.quantiles(scaffold_lens, n=4)[0] if len(scaffold_lens) >= 4 else 0
    p75 = stats.quantiles(scaffold_lens, n=4)[2] if len(scaffold_lens) >= 4 else 0
    drop_rate = 1.0 - (pairs_aligned / pairs_total) if pairs_total else 0.0

    return {
        "name": name,
        "group": group,
        "family": "reasoning" if group != "non-reasoning" else "non-reasoning",
        "raw_categories": make_cat_curves(raw_acc),
        "aligned_categories": make_cat_curves(aligned_acc),
        "scaffold_categories": make_scaffold_curves(scaffold_acc),
        "meta": {
            "pairs_total": pairs_total,
            "pairs_aligned": pairs_aligned,
            "drop_rate": drop_rate,
            "median_scaffold_tokens": median_scaffold,
            "scaffold_p25": p25,
            "scaffold_p75": p75,
        },
    }


def main():
    panel = []
    for name, group, rel in MODELS:
        entry = build_for_model(name, group, Path(rel))
        if entry is None:
            print(f"  [skip] {name} (no data at {rel})")
            continue
        m = entry["meta"]
        print(f"  {name:28s} [{group:20s}] pairs={m['pairs_total']:2d} aligned={m['pairs_aligned']:2d} drop={m['drop_rate']:.0%} scaffold~{m['median_scaffold_tokens']}")
        panel.append(entry)

    out = {
        "panel": panel,
        "tMax": T_MAX,
        "categories": CATEGORIES,
    }
    out_path = ROOT / "talk/data/family_aligned.json"
    out_path.write_text(json.dumps(out, indent=1))
    print(f"\nWrote {out_path} ({out_path.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
