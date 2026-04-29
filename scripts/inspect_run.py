#!/usr/bin/env python3
"""Print a compact textual summary of a stability-probe run."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    summary_path = args.run_dir / "summary.csv"
    if not summary_path.exists():
        raise SystemExit(f"Missing {summary_path}")

    df = pd.read_csv(summary_path)
    cols = [
        "model_name",
        "pair_id",
        "category",
        "common_prefix_tokens",
        "token_edit_distance_norm",
        "char_edit_distance_norm",
    ]
    print(df[cols].to_string(index=False))

    failures = args.run_dir / "failures.jsonl"
    if failures.exists() and failures.stat().st_size:
        print(f"\nFailures recorded in {failures}")


if __name__ == "__main__":
    main()

