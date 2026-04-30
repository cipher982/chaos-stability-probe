#!/usr/bin/env python3
"""Statistical summaries for token-certified micro-perturbation panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_panel(rank_dir: Path) -> pd.DataFrame:
    rows = []
    for summary_path in sorted(rank_dir.glob("*/summary.json")):
        model = summary_path.parent.name
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        run_dir = Path(summary["run_dir"])
        semantic_path = run_dir / "summary_with_semantic.csv"
        if not semantic_path.exists():
            continue
        df = pd.read_csv(semantic_path)
        df["panel_model"] = model
        rows.append(df)
    if not rows:
        raise SystemExit(f"No semantic summaries found under {rank_dir}")
    return pd.concat(rows, ignore_index=True)


def bootstrap_mean(values: np.ndarray, iterations: int, rng: np.random.Generator) -> tuple[float, float, float]:
    if values.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    means = np.empty(iterations)
    for i in range(iterations):
        means[i] = rng.choice(values, size=values.size, replace=True).mean()
    return (float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def paired_permutation(left: pd.DataFrame, right: pd.DataFrame, iterations: int, rng: np.random.Generator) -> dict:
    cols = ["pair_id", "semantic_cosine_distance"]
    merged = left[cols].merge(right[cols], on="pair_id", suffixes=("_left", "_right"))
    if merged.empty:
        return {"n_pairs": 0, "mean_difference_left_minus_right": float("nan"), "p_two_sided": float("nan")}
    diffs = (
        merged["semantic_cosine_distance_left"].to_numpy()
        - merged["semantic_cosine_distance_right"].to_numpy()
    )
    observed = float(diffs.mean())
    null = np.empty(iterations)
    for i in range(iterations):
        signs = rng.choice([-1.0, 1.0], size=diffs.size)
        null[i] = (diffs * signs).mean()
    p = float((np.abs(null) >= abs(observed)).mean())
    return {"n_pairs": int(diffs.size), "mean_difference_left_minus_right": observed, "p_two_sided": p}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank-dir", type=Path, default=Path("runs/rankings/token_micro_v3"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    out_dir = args.out_dir or args.rank_dir / "stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    panel = load_panel(args.rank_dir)
    panel.to_csv(out_dir / "panel_rows.csv", index=False)
    non_control = panel[panel["category"] != "micro_control_identical"].copy()

    model_rows = []
    for model, group in non_control.groupby("panel_model", sort=False):
        values = group["semantic_cosine_distance"].dropna().to_numpy()
        mean, lo, hi = bootstrap_mean(values, args.iterations, rng)
        model_rows.append(
            {
                "model": model,
                "n_effective": int(len(values)),
                "semantic_mean": mean,
                "ci95_low": lo,
                "ci95_high": hi,
                "semantic_p90": float(np.quantile(values, 0.90)),
                "semantic_max": float(np.max(values)),
            }
        )
    model_summary = pd.DataFrame(model_rows).sort_values("semantic_mean", ascending=False)
    model_summary.to_csv(out_dir / "model_bootstrap_ci.csv", index=False)

    category_rows = []
    for (model, category), group in non_control.groupby(["panel_model", "category"], sort=False):
        values = group["semantic_cosine_distance"].dropna().to_numpy()
        mean, lo, hi = bootstrap_mean(values, args.iterations, rng)
        category_rows.append(
            {
                "model": model,
                "category": category,
                "n": int(len(values)),
                "semantic_mean": mean,
                "ci95_low": lo,
                "ci95_high": hi,
            }
        )
    pd.DataFrame(category_rows).sort_values(["model", "semantic_mean"], ascending=[True, False]).to_csv(
        out_dir / "category_bootstrap_ci.csv", index=False
    )

    paired_rows = []
    models = sorted(non_control["panel_model"].unique())
    by_model = {model: non_control[non_control["panel_model"] == model] for model in models}
    for i, left_name in enumerate(models):
        for right_name in models[i + 1 :]:
            result = paired_permutation(by_model[left_name], by_model[right_name], args.iterations, rng)
            paired_rows.append({"left": left_name, "right": right_name, **result})
    pd.DataFrame(paired_rows).sort_values("p_two_sided").to_csv(out_dir / "paired_permutation_tests.csv", index=False)

    print(model_summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"Wrote token micro panel stats to {out_dir}")


if __name__ == "__main__":
    main()
