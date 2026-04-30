#!/usr/bin/env python3
"""Compare E10 silent-divergence readouts across runtimes/backends."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--left-label", default="left")
    parser.add_argument("--right-label", default="right")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    left = pd.read_csv(args.left)
    right = pd.read_csv(args.right)
    if "runtime_metadata_status" in right.columns:
        right = right[right["runtime_metadata_status"] == "present"].copy()

    paired = left.merge(right, on=["model_name", "pair_id"], suffixes=(f"_{args.left_label}", f"_{args.right_label}"))
    left_branch = f"branch_t_{args.left_label}"
    right_branch = f"branch_t_{args.right_label}"
    paired[f"branch_delta_{args.right_label}_minus_{args.left_label}"] = paired[right_branch] - paired[left_branch]

    summary_rows = []
    delta_col = f"branch_delta_{args.right_label}_minus_{args.left_label}"
    for model_name, group in paired.groupby("model_name", dropna=False, sort=False):
        comparable = group.dropna(subset=[left_branch, right_branch])
        summary_rows.append(
            {
                "model_name": model_name,
                "n_pairs": int(len(group)),
                "n_comparable_branch_t": int(len(comparable)),
                "mean_abs_branch_delta": float(comparable[delta_col].abs().mean()) if not comparable.empty else None,
                "max_abs_branch_delta": float(comparable[delta_col].abs().max()) if not comparable.empty else None,
                "same_branch_t_rate": float((comparable[delta_col] == 0).mean()) if not comparable.empty else None,
            }
        )

    summary = pd.DataFrame(summary_rows)
    paired.to_csv(args.out_dir / "paired_backend_readout.csv", index=False)
    summary.to_csv(args.out_dir / "backend_branch_t_summary.csv", index=False)

    print(summary.to_string(index=False))
    print(f"Wrote backend comparison to {args.out_dir}")


if __name__ == "__main__":
    main()
