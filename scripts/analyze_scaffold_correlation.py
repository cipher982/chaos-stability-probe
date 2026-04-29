#!/usr/bin/env python3
"""Annotate observed reasoning scaffolds and compare them with divergence scores."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SMALL_CATEGORIES = ["noop_format", "punctuation", "synonym"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def model_registry_by_name(path: Path = Path("configs/models.json")) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {row["name"]: row for row in rows}


def metadata_sample_false(path: Path) -> bool:
    meta = path.parent / "metadata.json"
    if not meta.exists():
        return True
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
    except Exception:
        return True
    return not bool(data.get("sample"))


def normalized_text(text: str) -> str:
    return text.replace("Ċ", "\n").replace("Ġ", " ")


def scaffold_kind(text: str) -> str:
    raw = normalized_text(text).lstrip()
    if raw.startswith("Thinking Process:"):
        return "thinking_process"
    if raw.startswith("<think>"):
        return "think_tag"
    if re.match(r"^(Okay|Alright|We need|I need|Let me think|Hmm)[,\\s]", raw, flags=re.I):
        return "visible_cot"
    if "<</SYS>>" in raw or raw.count("User:") >= 2 or raw.count("Assistant:") >= 2:
        return "template_echo"
    return "none"


def generation_files_by_model(model_names: set[str]) -> dict[str, list[Path]]:
    files: dict[str, list[Path]] = {name: [] for name in model_names}
    for path in Path("runs/sagemaker_artifacts").glob("**/generations.jsonl"):
        if not metadata_sample_false(path):
            continue
        try:
            rows = load_jsonl(path)
        except Exception:
            continue
        names = {row.get("model_name") for row in rows}
        for name in names & model_names:
            files[name].append(path)
    return files


def choose_generation_file(paths: list[Path]) -> Path | None:
    if not paths:
        return None

    def score(path: Path) -> tuple[int, int]:
        rows = load_jsonl(path)
        robust = 1 if "robust" in str(path) else 0
        return (robust, len(rows))

    return max(paths, key=score)


def bootstrap_group_diff(a: np.ndarray, b: np.ndarray, seed: int = 123, samples: int = 20_000) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(samples):
        diffs.append(rng.choice(a, size=len(a), replace=True).mean() - rng.choice(b, size=len(b), replace=True).mean())
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return float(np.mean(diffs)), float(lo), float(hi)


def main() -> None:
    out_dir = Path("runs/rankings/scaffold_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    merged = pd.read_csv("runs/rankings/final_21model_readout/merged_summary.csv")
    ranking = pd.read_csv("runs/rankings/final_21model_readout/small_perturbation_bootstrap.csv")
    model_map = merged[["run_label", "model_name", "model_id"]].drop_duplicates()
    model_names = set(model_map["model_name"])
    files_by_model = generation_files_by_model(model_names)
    registry = model_registry_by_name()

    annotation_rows = []
    for row in model_map.to_dict("records"):
        label = row["run_label"]
        model_name = row["model_name"]
        registry_behavior = registry.get(model_name, {}).get("observed_behavior", {})
        path = choose_generation_file(files_by_model.get(model_name, []))
        kinds: list[str] = []
        examples: list[str] = []
        n = 0
        if path is not None:
            for gen in load_jsonl(path):
                if gen.get("side") != "a":
                    continue
                kind = scaffold_kind(gen.get("generated_text", ""))
                kinds.append(kind)
                n += 1
                if kind != "none" and len(examples) < 2:
                    examples.append(normalized_text(gen.get("generated_text", "")).lstrip()[:180].replace("\n", "\\n"))
        counts = pd.Series(kinds).value_counts().to_dict() if kinds else {}
        observed_reasoning_rate = sum(counts.get(k, 0) for k in ["thinking_process", "think_tag", "visible_cot"]) / max(n, 1)
        template_echo_rate = counts.get("template_echo", 0) / max(n, 1)
        dominant_kind = max(counts.items(), key=lambda kv: kv[1])[0] if counts else "unknown"
        annotation_rows.append(
            {
                **row,
                "generation_file": "" if path is None else str(path),
                "n_side_a_checked": n,
                "dominant_scaffold_kind": dominant_kind,
                "observed_reasoning_scaffold_rate": observed_reasoning_rate,
                "template_echo_rate": template_echo_rate,
                "has_observed_reasoning_scaffold": observed_reasoning_rate >= 0.5,
                "has_template_echo": template_echo_rate >= 0.5,
                "manual_reasoning_label": bool(registry_behavior.get("manual_reasoning_model_label", False)),
                "example_scaffold_prefix": " || ".join(examples),
            }
        )

    annotations = pd.DataFrame(annotation_rows)
    annotations.to_csv(out_dir / "model_scaffold_annotations.csv", index=False)

    if "run_label" not in ranking.columns and "model" in ranking.columns:
        ranking = ranking.rename(columns={"model": "run_label", "ci_low": "ci95_low", "ci_high": "ci95_high"})
    joined = ranking.merge(annotations, on="run_label", how="left")
    joined.to_csv(out_dir / "scaffold_vs_stability.csv", index=False)

    small = merged[merged["category"].isin(SMALL_CATEGORIES)].merge(
        annotations[["run_label", "has_observed_reasoning_scaffold", "dominant_scaffold_kind"]],
        on="run_label",
        how="left",
    )
    group_rows = []
    for key, group in small.groupby("has_observed_reasoning_scaffold", dropna=False):
        group_rows.append(
            {
                "has_observed_reasoning_scaffold": key,
                "n_models": group["run_label"].nunique(),
                "n_prompt_pairs": len(group),
                "mean_semantic_distance": float(group["semantic_cosine_distance"].mean()),
                "median_semantic_distance": float(group["semantic_cosine_distance"].median()),
                "mean_token_edit_norm": float(group["token_edit_distance_norm"].mean()),
                "mean_common_prefix_tokens": float(group["common_prefix_tokens"].mean()),
            }
        )
    group_summary = pd.DataFrame(group_rows)
    group_summary.to_csv(out_dir / "scaffold_group_summary.csv", index=False)

    scaffold_values = small[small["has_observed_reasoning_scaffold"] == True]["semantic_cosine_distance"].to_numpy()
    non_values = small[small["has_observed_reasoning_scaffold"] == False]["semantic_cosine_distance"].to_numpy()
    if len(scaffold_values) and len(non_values):
        mean_diff, lo, hi = bootstrap_group_diff(scaffold_values, non_values)
        diff = pd.DataFrame(
            [
                {
                    "metric": "semantic_cosine_distance",
                    "mean_scaffold_minus_non_scaffold": mean_diff,
                    "ci95_low": lo,
                    "ci95_high": hi,
                }
            ]
        )
        diff.to_csv(out_dir / "scaffold_bootstrap_difference.csv", index=False)

    print(joined[[
        "run_label",
        "mean",
        "ci95_low",
        "ci95_high",
        "dominant_scaffold_kind",
        "observed_reasoning_scaffold_rate",
        "template_echo_rate",
        "has_observed_reasoning_scaffold",
    ]].sort_values("mean").to_string(index=False))
    print()
    print(group_summary.to_string(index=False))
    if len(scaffold_values) and len(non_values):
        print()
        print(diff.to_string(index=False))
    print(f"Wrote scaffold analysis to {out_dir}")


if __name__ == "__main__":
    main()
