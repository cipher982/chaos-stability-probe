"""Rebuild talk data files for the 4 fixed charts (scaffold-confound-free).

Sources use thinkoff variants for Qwen 4B/9B so every model emits
answer-first tokens. Outputs:
  - talk/data/scrubber.json     (Qwen ladder: 0.8B, 2B, 4B-thinkoff, 9B-thinkoff)
  - talk/data/family_curves.json (11 non-reasoning-or-thinkoff models, 3 groups)
  - talk/data/branching.json    (0.8B vs 4B-thinkoff)
"""
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path("/Users/davidrose/git/chaos")
ART = ROOT / "runs/sagemaker_artifacts"

CATEGORIES = [
    "control_identical", "noop_format", "punctuation",
    "synonym", "paraphrase", "semantic_small", "positive_control",
]

# ------------------------ SOURCES ------------------------
# (display_name, group, artifact_subpath)
QWEN_LADDER = [
    ("0.8B", 0.8, "chaos-scaffold-long-qwen35-08b-20260429-001/runs/qwen35_08b"),
    ("2B",   2.0, "chaos-scaffold-long-qwen35-2b-20260429-001/runs/qwen35_2b"),
    ("4B",   4.0, "chaos-scaffold-long-qwen35-4b-thinkoff-20260429-001/runs/qwen35_4b"),
    ("9B",   9.0, "chaos-scaffold-long-qwen35-9b-thinkoff-20260429-001/runs/qwen35_9b"),
]

FAMILY_PANEL = [
    # Group A: Qwen ladder (thinkoff for 4B/9B)
    ("Qwen3.5 0.8B",           "qwen_ladder", "chaos-scaffold-long-qwen35-08b-20260429-001/runs/qwen35_08b"),
    ("Qwen3.5 2B",             "qwen_ladder", "chaos-scaffold-long-qwen35-2b-20260429-001/runs/qwen35_2b"),
    ("Qwen3.5 4B (thinkoff)",  "qwen_ladder", "chaos-scaffold-long-qwen35-4b-thinkoff-20260429-001/runs/qwen35_4b"),
    ("Qwen3.5 9B (thinkoff)",  "qwen_ladder", "chaos-scaffold-long-qwen35-9b-thinkoff-20260429-001/runs/qwen35_9b"),
    # Group B: Non-reasoning instruct
    ("Mistral 7B v0.3",        "instruct",    "chaos-scaffold-long-mistral7b-v03-20260429-001/runs/mistral7b_instruct_v03"),
    ("OLMo-3 7B",              "instruct",    "chaos-scaffold-long-olmo3-7b-20260429-001/runs/olmo3_7b_instruct"),
    ("Granite 3.3 8B",         "instruct",    "chaos-scaffold-long-granite33-8b-20260429-001/runs/granite33_8b_instruct"),
    ("Gemma 4 E4B it",         "instruct",    "chaos-scaffold-long-gemma4-e4b-it-20260429-001/runs/gemma4_e4b_it"),
    ("Falcon 3 10B",           "instruct",    "chaos-scaffold-long-falcon3-10b-20260429-001/runs/falcon3_10b_instruct"),
    # Group C: Legacy architectures
    ("GPT-2 XL",               "legacy",      "chaos-scaffold-long-gpt2-xl-g5x-20260429-001/runs/gpt2_xl"),
    ("GPT-J 6B",               "legacy",      "chaos-scaffold-long-gptj-6b-g5-20260429-001/runs/gptj_6b"),
    ("OPT 6.7B",               "legacy",      "chaos-scaffold-long-opt-6p7b-qa-ai-g5-20260429-001/runs/opt_6p7b"),
    ("Pythia 6.9B",            "legacy",      "chaos-scaffold-long-pythia-6p9b-ml-g5-20260429-001/runs/pythia_6p9b"),
    ("LLaMA-1 7B",             "legacy",      "chaos-scaffold-long-llama1-7b-prod-g5-20260429-001/runs/llama1_7b"),
]

BRANCHING = {
    "0.8B": "chaos-scaffold-long-qwen35-08b-20260429-001/runs/qwen35_08b",
    "4B":   "chaos-scaffold-long-qwen35-4b-thinkoff-20260429-001/runs/qwen35_4b",
}

T_MAX = 64  # align with existing charts' x-axis window


# ------------------------ HELPERS ------------------------
def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open()]


def resolve_run_dir(subpath: str) -> Path:
    """Artifact layout has .../runs/<model_slug>/ — the slug may not match
    what's in MODELS. Resolve by finding the single subdir under .../runs/."""
    p = ART / subpath
    if p.exists():
        return p
    parent = (ART / subpath).parent
    if not parent.exists():
        raise FileNotFoundError(parent)
    subs = [d for d in parent.iterdir() if d.is_dir()]
    if len(subs) == 1:
        return subs[0]
    raise FileNotFoundError(f"Could not find unique run dir at {p}; parent has {subs}")


# ------------------------ SCRUBBER ------------------------
def build_scrubber() -> dict:
    models_out = []
    for name, params, sub in QWEN_LADDER:
        run_dir = resolve_run_dir(sub)
        csv_path = run_dir / "summary_with_semantic.csv"
        by_cat = defaultdict(list)
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("sample", "False") == "True":
                    continue
                if row.get("repeat", "0") != "0":
                    continue
                cat = row["category"]
                sem = row.get("semantic_cosine_distance", "")
                if sem == "" or sem is None:
                    continue
                by_cat[cat].append(float(sem))
        model_entry = {"name": name, "params": params, "by_category": {}}
        for cat in CATEGORIES:
            vals = by_cat.get(cat, [])
            if not vals:
                model_entry["by_category"][cat] = {"mean": 0.0, "values": []}
            else:
                model_entry["by_category"][cat] = {
                    "mean": statistics.fmean(vals),
                    "values": vals,
                }
        models_out.append(model_entry)

    # Sanity checks
    for m in models_out:
        ci = m["by_category"]["control_identical"]["mean"]
        pc = m["by_category"]["positive_control"]["mean"]
        print(f"  scrubber {m['name']}: control_identical={ci:.4f}  pos_control={pc:.3f}  "
              f"n(syn)={len(m['by_category']['synonym']['values'])}")

    return {"categories": CATEGORIES, "models": models_out}


# ------------------------ FAMILY_CURVES ------------------------
def build_family_curves() -> dict:
    panel = []
    for name, group, sub in FAMILY_PANEL:
        run_dir = resolve_run_dir(sub)
        curves = load_jsonl(run_dir / "curves.jsonl")
        if not curves:
            print(f"  [skip] {name}: no curves.jsonl at {run_dir}")
            continue
        # Group by (category, t) -> list of normalized edit distances
        # curves.jsonl has one row per (pair, category, t) — average across pairs.
        by_cat_t = defaultdict(lambda: defaultdict(list))
        by_cat_t_cp = defaultdict(lambda: defaultdict(list))
        for r in curves:
            if r.get("sample", False):
                continue
            if r.get("repeat", 0) != 0:
                continue
            cat = r.get("category")
            t = r.get("t")
            if cat is None or t is None or t > T_MAX:
                continue
            ed = r.get("token_edit_distance_norm")
            cp = r.get("common_prefix_tokens")
            if ed is not None:
                by_cat_t[cat][t].append(ed)
            if cp is not None:
                by_cat_t_cp[cat][t].append(cp)
        categories = {}
        for cat in CATEGORIES:
            pts = []
            for t in range(1, T_MAX + 1):
                evs = by_cat_t[cat].get(t, [])
                cvs = by_cat_t_cp[cat].get(t, [])
                if not evs:
                    continue
                pts.append({
                    "t": t,
                    "edit_mean": statistics.fmean(evs),
                    "cp_mean": statistics.fmean(cvs) if cvs else 0.0,
                    "n": len(evs),
                })
            categories[cat] = pts
        panel.append({
            "name": name,
            "family": group,  # "qwen_ladder" | "instruct" | "legacy"
            "categories": categories,
        })
        # Sanity
        if categories.get("control_identical"):
            ci_last = categories["control_identical"][-1]["edit_mean"]
        else:
            ci_last = float("nan")
        if categories.get("positive_control"):
            pc_last = categories["positive_control"][-1]["edit_mean"]
        else:
            pc_last = float("nan")
        print(f"  family {name} [{group}]: ctl_ident@t64={ci_last:.3f}  pos_ctl@t64={pc_last:.3f}")
    return {"panel": panel, "tMax": T_MAX}


# ------------------------ BRANCHING ------------------------
def build_branching() -> dict:
    result = {}
    for label, sub in BRANCHING.items():
        run_dir = resolve_run_dir(sub)
        gens = load_jsonl(run_dir / "generations.jsonl")
        curves = load_jsonl(run_dir / "curves.jsonl")

        texts = {}
        for g in gens:
            if g.get("repeat", 0) != 0 or g.get("sample"):
                continue
            k = (g["pair_id"], g["category"])
            texts.setdefault(k, {})[g["side"]] = g["generated_text"]

        curves_by_key = defaultdict(list)
        for c in curves:
            if c.get("repeat", 0) != 0 or c.get("sample"):
                continue
            k = (c["pair_id"], c["category"])
            curves_by_key[k].append(c)

        # Pick one pair per category (first with both sides). For consistency
        # across models, prefer the same pair_id as the other panel when possible.
        # We emit per-category dicts — the client picks one.
        by_cat = {}
        for (pair_id, category), sides in texts.items():
            if "a" not in sides or "b" not in sides:
                continue
            if category in by_cat:
                continue  # first-wins (jsonl order is deterministic)
            cv = sorted(curves_by_key.get((pair_id, category), []), key=lambda c: c["t"])
            if not cv:
                continue
            by_cat[category] = {
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
        ordered = {k: by_cat[k] for k in CATEGORIES if k in by_cat}
        result[label] = ordered
        print(f"  branching {label}: {list(ordered.keys())}")
    return result


def main():
    print("Building scrubber.json...")
    scrubber = build_scrubber()
    (ROOT / "talk/data/scrubber.json").write_text(
        json.dumps(scrubber, separators=(",", ":"))
    )

    print("\nBuilding family_curves.json...")
    family = build_family_curves()
    (ROOT / "talk/data/family_curves.json").write_text(
        json.dumps(family, separators=(",", ":"))
    )

    print("\nBuilding branching.json...")
    branching = build_branching()
    (ROOT / "talk/data/branching.json").write_text(
        json.dumps(branching, separators=(",", ":"))
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
