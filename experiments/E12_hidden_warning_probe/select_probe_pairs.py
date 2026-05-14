#!/usr/bin/env python3
"""Select a compact hidden-warning recapture set from E09 events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def round_robin(df: pd.DataFrame, group_cols: list[str], limit: int) -> pd.DataFrame:
    if limit <= 0 or len(df) <= limit:
        return df
    buckets = [group for _, group in df.groupby(group_cols, sort=False, observed=True)]
    rows = []
    while buckets and len(rows) < limit:
        next_buckets = []
        for bucket in buckets:
            if bucket.empty:
                continue
            rows.append(bucket.iloc[0])
            rest = bucket.iloc[1:]
            if not rest.empty:
                next_buckets.append(rest)
            if len(rows) >= limit:
                break
        buckets = next_buckets
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=Path("runs/trajectory_events/logit_token_cert_v1/trajectory_events.csv"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--min-branch-t", type=int, default=10)
    parser.add_argument("--visible", type=int, default=48)
    parser.add_argument("--no-visible", type=int, default=12)
    args = parser.parse_args()

    events = pd.read_csv(args.events)
    events = events[(events["model_name"] == args.model) & (~events["is_control"].astype(bool))].copy()
    events["branch_t"] = pd.to_numeric(events["branch_t"], errors="coerce")

    visible = events[events["branch_t"].notna() & (events["branch_t"] >= args.min_branch_t)].copy()
    visible["branch_bucket"] = pd.cut(
        visible["branch_t"],
        bins=[args.min_branch_t - 1, 15, 32, 64, float("inf")],
        labels=["10_15", "16_32", "33_64", "65_plus"],
    )
    visible = visible.sort_values(["branch_bucket", "category", "branch_t", "pair_id"])
    visible = round_robin(visible, ["branch_bucket", "category"], args.visible)
    visible["selection_kind"] = "visible_long_prefix"

    no_visible = events[events["event_kind"] == "no_visible_branch"].copy()
    no_visible = no_visible.sort_values(["category", "pair_id"]).head(args.no_visible)
    no_visible["selection_kind"] = "no_visible_branch"

    selected = pd.concat([visible, no_visible], ignore_index=True)
    selected = selected.drop_duplicates("pair_id")
    pair_ids = selected["pair_id"].astype(str).tolist()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    selected[
        ["model_name", "pair_id", "category", "event_kind", "branch_t", "selection_kind"]
    ].to_csv(args.out_csv, index=False)
    args.out_json.write_text(json.dumps(pair_ids, indent=2) + "\n", encoding="utf-8")
    print(f"Selected {len(pair_ids)} pairs for {args.model}")
    print(selected["selection_kind"].value_counts().to_string())
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
