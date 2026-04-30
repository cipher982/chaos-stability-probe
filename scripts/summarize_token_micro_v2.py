#!/usr/bin/env python3
"""Build aggregate readouts for token-enforced micro perturbation runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def summarize_model(run_dir: Path) -> dict[str, object] | None:
    category_path = run_dir / "category_summary.csv"
    summary_path = run_dir / "summary.json"
    if not category_path.exists() or not summary_path.exists():
        return None
    categories = pd.read_csv(category_path)
    non_control = categories[~categories["category"].str.contains("control", na=False)].copy()
    if non_control.empty:
        return None
    n_effective = int(non_control["n"].sum())
    weighted_mean = float((non_control["semantic_mean"] * non_control["n"]).sum() / n_effective)
    top = non_control.sort_values("semantic_mean", ascending=False).iloc[0]
    payload = json.loads(summary_path.read_text())
    return {
        "model": run_dir.name,
        "n_effective": n_effective,
        "semantic_mean": weighted_mean,
        "semantic_p90": payload.get("micro_semantic_p90"),
        "semantic_max": payload.get("micro_semantic_max"),
        "top_category": top["category"],
        "top_category_mean": float(top["semantic_mean"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank-dir", type=Path, default=Path("runs/rankings/token_micro_v2"))
    args = parser.parse_args()

    rows = [
        row
        for run_dir in sorted(args.rank_dir.iterdir()) if run_dir.is_dir()
        for row in [summarize_model(run_dir)]
        if row is not None
    ]
    if not rows:
        raise SystemExit(f"No processed token-micro runs found under {args.rank_dir}")
    frame = pd.DataFrame(rows).sort_values("semantic_mean", ascending=False)
    out = args.rank_dir / "combined_model_summary.csv"
    frame.to_csv(out, index=False)
    print(frame.to_string(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
