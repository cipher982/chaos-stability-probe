#!/usr/bin/env python3
"""Compare trajectory branch timing and event types across models."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_QWEN_ORDER = ["qwen35_08b", "qwen35_2b", "qwen35_4b", "qwen35_9b"]


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).lower() in {"true", "1", "yes"}


def model_summary(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in events.groupby("model_name", sort=False):
        non_control = group[~group["is_control"].map(bool_value)] if "is_control" in group else group
        n = len(non_control)
        event_counts = non_control["event_kind"].value_counts(normalize=True)
        rows.append(
            {
                "model_name": model,
                "n": n,
                "immediate_visible_branch_rate": float(event_counts.get("immediate_visible_branch", 0.0)),
                "silent_logit_divergence_rate": float(event_counts.get("silent_logit_divergence", 0.0)),
                "no_visible_branch_rate": float(event_counts.get("no_visible_branch", 0.0)),
                "persistent_rate": float(non_control["persistent_branch"].map(bool_value).mean()),
                "mean_branch_t": float(non_control["branch_t"].mean()),
                "median_branch_t": float(non_control["branch_t"].median()),
                "mean_semantic": float(non_control["semantic_cosine_distance"].mean()),
                "mean_branch_js": float(non_control["branch_js"].mean()),
            }
        )
    return pd.DataFrame(rows)


def branch_pivot(events: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    keep = events[events["model_name"].isin(models)].copy()
    if "is_control" in keep:
        keep = keep[~keep["is_control"].map(bool_value)]
    index_cols = ["pair_id", "category", "repeat"]
    keep = keep.sort_values(index_cols + ["model_name"])
    branch = keep.groupby(index_cols + ["model_name"], dropna=False)["branch_t"].first().unstack("model_name")
    kind = keep.groupby(index_cols + ["model_name"], dropna=False)["event_kind"].first().unstack("model_name")
    branch.columns = [f"{col}_branch_t" for col in branch.columns]
    kind.columns = [f"{col}_event_kind" for col in kind.columns]
    return pd.concat([branch, kind], axis=1).reset_index()


def qwen_ladder_rows(pivot: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows = []
    branch_cols = [f"{model}_branch_t" for model in models]
    for _, row in pivot.iterrows():
        vals = [row.get(col) for col in branch_cols]
        numeric = [None if pd.isna(v) else float(v) for v in vals]
        observed = [v for v in numeric if v is not None]
        monotonic_nonincreasing = all(
            a is None or b is None or b <= a for a, b in zip(numeric, numeric[1:])
        )
        monotonic_nondecreasing = all(
            a is None or b is None or b >= a for a, b in zip(numeric, numeric[1:])
        )
        rows.append(
            {
                "pair_id": row["pair_id"],
                "category": row["category"],
                "repeat": row["repeat"],
                **{col: row.get(col) for col in pivot.columns if col.endswith("_branch_t")},
                "observed_branch_t_min": min(observed) if observed else None,
                "observed_branch_t_max": max(observed) if observed else None,
                "observed_branch_t_span": (max(observed) - min(observed)) if len(observed) >= 2 else None,
                "monotonic_nonincreasing": monotonic_nonincreasing,
                "monotonic_nondecreasing": monotonic_nondecreasing,
                "branch_signature": " -> ".join("" if v is None else str(int(v)) for v in numeric),
            }
        )
    return pd.DataFrame(rows).sort_values("observed_branch_t_span", ascending=False, na_position="last")


def pairwise_deltas(pivot: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows = []
    for left, right in combinations(models, 2):
        left_col = f"{left}_branch_t"
        right_col = f"{right}_branch_t"
        if left_col not in pivot or right_col not in pivot:
            continue
        both = pivot.dropna(subset=[left_col, right_col])
        if both.empty:
            continue
        delta = both[right_col].astype(float) - both[left_col].astype(float)
        rows.append(
            {
                "left_model": left,
                "right_model": right,
                "n_pairs": int(len(both)),
                "mean_delta_right_minus_left": float(delta.mean()),
                "median_delta_right_minus_left": float(delta.median()),
                "right_branches_earlier_rate": float((delta < 0).mean()),
                "right_branches_later_rate": float((delta > 0).mean()),
                "same_branch_t_rate": float((delta == 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/trajectory_model_comparison"))
    parser.add_argument("--model", action="append", default=[])
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(args.events)
    models = args.model or [model for model in DEFAULT_QWEN_ORDER if model in set(events["model_name"])]
    summary = model_summary(events)
    pivot = branch_pivot(events, models)
    ladder = qwen_ladder_rows(pivot, models)
    deltas = pairwise_deltas(pivot, models)

    summary.to_csv(args.out_dir / "model_branch_summary.csv", index=False)
    pivot.to_csv(args.out_dir / "paired_branch_events.csv", index=False)
    ladder.to_csv(args.out_dir / "qwen_ladder_branch_timing.csv", index=False)
    deltas.to_csv(args.out_dir / "pairwise_branch_t_deltas.csv", index=False)

    print(summary.to_string(index=False))
    if not deltas.empty:
        print()
        print(deltas.to_string(index=False))
    print(f"Wrote model branch comparison artifacts to {args.out_dir}")


if __name__ == "__main__":
    main()
