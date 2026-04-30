#!/usr/bin/env python3
"""Summarize activation-patching CSVs into one compact table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def metadata_for(csv_path: Path) -> dict[str, Any]:
    path = csv_path.with_suffix(".json")
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_file(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path)
    meta = metadata_for(path)
    finite = df[df["rescue_fraction"].notna()].copy()
    if finite.empty:
        best = df.iloc[0]
    else:
        best = finite.sort_values("rescue_fraction", ascending=False).iloc[0]

    out: dict[str, Any] = {
        "csv_path": str(path),
        "model_name": best["model_name"],
        "pair_id": best["pair_id"],
        "category": best["category"],
        "first_diff_token": int(best["first_diff_token"]),
        "a_branch_token": best["a_branch_token"],
        "b_branch_token": best["b_branch_token"],
        "clean_metric_a_minus_b": float(best["clean_metric_a_minus_b"]),
        "corrupt_metric_a_minus_b": float(best["corrupt_metric_a_minus_b"]),
        "corrupt_replay_matches_b_branch": meta.get("corrupt_replay_matches_b_branch"),
        "best_position_label": best["position_label"],
        "best_layer": int(best["layer"]),
        "best_rescue_fraction": float(best["rescue_fraction"]) if pd.notna(best["rescue_fraction"]) else None,
        "best_metric_a_minus_b": float(best["metric_a_minus_b"]),
        "best_top1_token": best["top1_token"],
    }

    for label, group in df.groupby("position_label"):
        good = group[group["rescue_fraction"].notna()]
        if good.empty:
            continue
        row = good.sort_values("rescue_fraction", ascending=False).iloc[0]
        prefix = str(label)
        out[f"{prefix}_best_layer"] = int(row["layer"])
        out[f"{prefix}_best_rescue_fraction"] = float(row["rescue_fraction"])
        out[f"{prefix}_best_top1_token"] = row["top1_token"]

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("patch_dir", type=Path, nargs="?", default=Path("runs/mechinterp_patch"))
    parser.add_argument("--out", type=Path, default=Path("runs/mechinterp_patch/patch_summary.csv"))
    args = parser.parse_args()

    paths = sorted(
        path
        for path in args.patch_dir.glob("*.csv")
        if path.name not in {"selected_patch_targets.csv", "patch_summary.csv"}
    )
    if not paths:
        raise SystemExit(f"No patch CSVs found in {args.patch_dir}")

    rows = [summarize_file(path) for path in paths]
    summary = pd.DataFrame(rows)
    summary = summary.sort_values(
        ["corrupt_replay_matches_b_branch", "best_rescue_fraction"],
        ascending=[False, False],
        na_position="last",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    print(f"Wrote {len(summary)} patch summaries to {args.out}")
    print(
        summary[
            [
                "model_name",
                "pair_id",
                "first_diff_token",
                "corrupt_replay_matches_b_branch",
                "best_position_label",
                "best_layer",
                "best_rescue_fraction",
                "prompt_lcp_token_best_rescue_fraction",
                "final_context_token_best_rescue_fraction",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
