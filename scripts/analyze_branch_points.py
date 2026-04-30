#!/usr/bin/env python3
"""Summarize logit probes around the first generated-token branch point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DROP_HEAVY_FIELDS = {"topk_a", "topk_b"}
KEY_COLS = ["model_name", "pair_id", "category", "repeat"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            for field in DROP_HEAVY_FIELDS:
                row.pop(field, None)
            rows.append(row)
    return rows


def clean_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if pd.isna(value):
        return None
    return int(float(value))


def load_summary(run_dir: Path, semantic_rows: Path | None) -> pd.DataFrame:
    summary_path = run_dir / "summary_with_semantic.csv"
    if not summary_path.exists():
        summary_path = run_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"{run_dir} missing summary.csv")
    summary = pd.read_csv(summary_path)
    if "semantic_cosine_distance" in summary.columns or semantic_rows is None:
        return summary
    if not semantic_rows.exists():
        return summary
    sem = pd.read_csv(semantic_rows)
    cols = KEY_COLS + ["semantic_cosine_distance"]
    sem = sem[[c for c in cols if c in sem.columns]].drop_duplicates(KEY_COLS)
    return summary.merge(sem, on=KEY_COLS, how="left")


def load_generation_tokens(run_dir: Path) -> dict[tuple[Any, ...], dict[str, Any]]:
    path = run_dir / "generations.jsonl"
    if not path.exists():
        return {}
    rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in read_jsonl(path):
        key = tuple(row.get(col) for col in KEY_COLS)
        side = row.get("side")
        rows.setdefault(key, {})[side] = row
    return rows


def token_at(row: dict[str, Any] | None, idx: int | None) -> tuple[int | None, str | None]:
    if row is None or idx is None:
        return None, None
    ids = row.get("generated_tokens") or []
    if idx < 0 or idx >= len(ids):
        return None, None
    token_id = int(ids[idx])
    text = str(row.get("generated_text", ""))
    return token_id, text[:180].replace("\n", " / ")


def summarize_logits(group: pd.DataFrame) -> dict[str, Any]:
    top1_same = group["top1_same"].astype(bool)
    return {
        "n_rows": int(len(group)),
        "js_mean": float(group["js_divergence"].mean()),
        "js_median": float(group["js_divergence"].median()),
        "top1_flip_rate": float((~top1_same).mean()),
        "mean_top1_margin_logit": float(
            pd.concat([group["a_top1_margin_logit"], group["b_top1_margin_logit"]]).mean()
        ),
        "a_top1_prob": float(group["a_top1_prob"].mean()),
        "b_top1_prob": float(group["b_top1_prob"].mean()),
        "centered_logit_l2": float(group["centered_logit_normalized_l2"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument(
        "--semantic-rows",
        type=Path,
        default=Path("runs/rankings/token_micro_v3/stats/panel_rows.csv"),
        help="Optional processed rows with semantic_cosine_distance to merge by model/pair/repeat.",
    )
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    out_dir = args.out_dir or args.run_dir / "branch_points"
    out_dir.mkdir(parents=True, exist_ok=True)

    logit_path = args.run_dir / "logit_probes.jsonl"
    if not logit_path.exists():
        raise FileNotFoundError(f"{args.run_dir} missing logit_probes.jsonl")

    summary = load_summary(args.run_dir, args.semantic_rows)
    logits = pd.DataFrame(read_jsonl(logit_path))
    generations = load_generation_tokens(args.run_dir)

    indexed = logits.set_index(KEY_COLS + ["anchor", "t"], drop=False)
    detail_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

    for _, summary_row in summary.iterrows():
        if summary_row.get("category") == "micro_control_identical":
            continue
        first_diff = clean_int(summary_row.get("first_diff_token"))
        if first_diff is None:
            continue

        base_key = tuple(summary_row.get(col) for col in KEY_COLS)
        gen_sides = generations.get(base_key, {})
        a_token_id, a_preview = token_at(gen_sides.get("a"), first_diff)
        b_token_id, b_preview = token_at(gen_sides.get("b"), first_diff)

        candidate = {col: summary_row.get(col) for col in KEY_COLS}
        for col in [
            "semantic_cosine_distance",
            "token_edit_distance_norm",
            "common_prefix_tokens",
            "prompt_token_edit_distance",
            "prompt_token_delta_kind",
            "prompt_token_lcp",
        ]:
            if col in summary_row:
                candidate[col] = summary_row.get(col)
        candidate.update(
            {
                "first_diff_token": first_diff,
                "a_branch_token_id": a_token_id,
                "b_branch_token_id": b_token_id,
                "generated_text_a_preview": a_preview,
                "generated_text_b_preview": b_preview,
            }
        )
        candidate_rows.append(candidate)

        for window, t in [
            ("pre_branch", max(0, first_diff - 1)),
            ("branch", first_diff),
            ("post_branch", first_diff + 1),
        ]:
            for anchor in ["prompt_a_generation", "prompt_b_generation"]:
                key = base_key + (anchor, t)
                if key not in indexed.index:
                    continue
                match = indexed.loc[key]
                if isinstance(match, pd.DataFrame):
                    match = match.iloc[0]
                item = match.to_dict()
                item["branch_window"] = window
                item["first_diff_token"] = first_diff
                item["a_branch_token_id"] = a_token_id
                item["b_branch_token_id"] = b_token_id
                for col in ["semantic_cosine_distance", "common_prefix_tokens", "token_edit_distance_norm"]:
                    if col in summary_row:
                        item[col] = summary_row.get(col)
                detail_rows.append(item)

    candidates = pd.DataFrame(candidate_rows)
    if not candidates.empty:
        sort_col = "semantic_cosine_distance" if "semantic_cosine_distance" in candidates else "token_edit_distance_norm"
        candidates = candidates.sort_values(sort_col, ascending=False).reset_index(drop=True)
    candidates.to_csv(out_dir / "branch_candidates.csv", index=False)
    candidates.head(args.top_n).to_csv(out_dir / "top_branch_candidates.csv", index=False)

    detail = pd.DataFrame(detail_rows)
    detail.to_csv(out_dir / "branch_window_logit_rows.csv", index=False)
    if detail.empty:
        print(f"No branch-window rows found. Wrote candidates to {out_dir}")
        return

    summary_rows = []
    for keys, group in detail.groupby(["model_name", "category", "branch_window"], sort=False):
        model_name, category, branch_window = keys
        summary_rows.append(
            {
                "model_name": model_name,
                "category": category,
                "branch_window": branch_window,
                **summarize_logits(group),
                "semantic_mean": float(group["semantic_cosine_distance"].mean())
                if "semantic_cosine_distance" in group
                else None,
                "common_prefix_mean": float(group["common_prefix_tokens"].mean())
                if "common_prefix_tokens" in group
                else None,
            }
        )
    pd.DataFrame(summary_rows).to_csv(out_dir / "branch_window_logit_summary.csv", index=False)
    print(f"Wrote branch-point artifacts to {out_dir}")


if __name__ == "__main__":
    main()
