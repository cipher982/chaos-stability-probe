"""Rebuild talk/data/branching.json with all 7 perturbation tiers for Qwen 0.8B/4B/9B."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "0.8B": ROOT / "runs/qwen35_08b_allpairs_v2",
    "4B":   ROOT / "runs/qwen35_4b_allpairs/qwen35_4b",
    "9B":   ROOT / "runs/qwen35_9b_allpairs/qwen35_9b",
}
OUT = ROOT / "talk/data/branching.json"

CATEGORY_ORDER = [
    "control_identical", "noop_format", "punctuation", "synonym",
    "paraphrase", "semantic_small", "positive_control",
]


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def build_model(run_dir):
    gens = load_jsonl(run_dir / "generations.jsonl")
    curves = load_jsonl(run_dir / "curves.jsonl")

    # Key: (pair_id, category, repeat=0) -> {a: text, b: text}
    texts = {}
    for g in gens:
        if g.get("repeat", 0) != 0 or g.get("sample"):
            continue
        k = (g["pair_id"], g["category"])
        texts.setdefault(k, {})[g["side"]] = g["generated_text"]

    # curves keyed same way (merging a/b rows — curves are pair-level, not per-side)
    curves_by_key = {}
    for c in curves:
        if c.get("repeat", 0) != 0 or c.get("sample"):
            continue
        k = (c["pair_id"], c["category"])
        curves_by_key.setdefault(k, []).append(c)

    out = {}
    for (pair_id, category), sides in texts.items():
        if "a" not in sides or "b" not in sides:
            continue
        cv = sorted(curves_by_key.get((pair_id, category), []), key=lambda c: c["t"])
        if not cv:
            continue
        out[category] = {
            "pair_id": pair_id,
            "text_a": sides["a"],
            "text_b": sides["b"],
            "curve": [
                {"t": c["t"],
                 "common_prefix": c["common_prefix_tokens"],
                 "edit": c["token_edit_distance_norm"]}
                for c in cv
            ],
        }
    return out


def main():
    result = {}
    for model, run_dir in RUNS.items():
        by_cat = build_model(run_dir)
        # Order categories
        ordered = {k: by_cat[k] for k in CATEGORY_ORDER if k in by_cat}
        result[model] = ordered
        print(f"{model}: {list(ordered.keys())}")

    OUT.write_text(json.dumps(result, indent=None, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
