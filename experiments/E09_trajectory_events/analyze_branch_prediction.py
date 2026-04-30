#!/usr/bin/env python3
"""Score simple branch-window predictors from trajectory event rows."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = {
    "js_divergence": "higher",
    "centered_logit_normalized_l2": "higher",
    "top1_flip": "higher",
    "max_entropy": "higher",
    "min_margin_logit": "lower",
}
BOOTSTRAP_KEY_COLS = ["model_name", "pair_id", "repeat"]


def target_specs(df: pd.DataFrame) -> list[tuple[str, str, int | None]]:
    specs: list[tuple[str, str, int | None]] = []
    if "at_branch" in df.columns:
        specs.append(("at_branch", "at_branch", None))
    for prefix, label in [
        ("pre_branch_within_", "pre_branch_within"),
        ("branch_within_", "branch_within"),
    ]:
        horizons = sorted(
            int(col.removeprefix(prefix))
            for col in df.columns
            if col.startswith(prefix)
        )
        specs.extend((f"{prefix}{horizon}", label, horizon) for horizon in horizons)
    return specs


def auroc(labels: pd.Series, scores: pd.Series) -> float | None:
    data = pd.DataFrame({"label": labels.astype(bool), "score": scores}).dropna()
    if data.empty:
        return None
    n_pos = int(data["label"].sum())
    n_neg = int((~data["label"]).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = data["score"].rank(method="average")
    pos_rank_sum = float(ranks[data["label"]].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def score_feature(group: pd.DataFrame, target: str, feature: str, direction: str) -> float | None:
    scores = group[feature]
    if direction == "lower":
        scores = -scores
    return auroc(group[target], scores)


def bootstrap_auroc(
    group: pd.DataFrame,
    target: str,
    feature: str,
    direction: str,
    samples: int,
    seed: int,
) -> dict[str, float | None]:
    key_cols = [col for col in BOOTSTRAP_KEY_COLS if col in group.columns]
    if not key_cols or samples <= 0:
        return {"auroc_boot_mean": None, "auroc_ci_low": None, "auroc_ci_high": None}
    usable = group[key_cols + [target, feature]].dropna(subset=[target, feature])
    if usable.empty:
        return {"auroc_boot_mean": None, "auroc_ci_low": None, "auroc_ci_high": None}
    grouped = [item for _, item in usable.groupby(key_cols, dropna=False, sort=False)]
    if len(grouped) < 2:
        return {"auroc_boot_mean": None, "auroc_ci_low": None, "auroc_ci_high": None}

    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        sampled = [grouped[int(idx)] for idx in rng.integers(0, len(grouped), size=len(grouped))]
        boot = pd.concat(sampled, ignore_index=True)
        value = score_feature(boot, target, feature, direction)
        if value is not None:
            values.append(value)
    if not values:
        return {"auroc_boot_mean": None, "auroc_ci_low": None, "auroc_ci_high": None}
    arr = np.array(values, dtype=float)
    return {
        "auroc_boot_mean": float(arr.mean()),
        "auroc_ci_low": float(np.quantile(arr, 0.025)),
        "auroc_ci_high": float(np.quantile(arr, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_windows", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    parser.add_argument("--bootstrap-seed", type=int, default=12345)
    parser.add_argument(
        "--bootstrap-scope",
        choices=["all", "groups"],
        default="all",
        help="Bootstrap only the aggregate panel by default; use groups for per-model CIs.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or args.prediction_windows.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.prediction_windows)

    rows = []
    targets = target_specs(df)
    groups = [("all", df)]
    if "model_name" in df.columns:
        groups.extend((model, group) for model, group in df.groupby("model_name", sort=False))

    for group_name, group in groups:
        for target, target_kind, horizon in targets:
            if target not in group:
                continue
            labels = group[target]
            for feature, direction in FEATURES.items():
                if feature not in group:
                    continue
                value = score_feature(group, target, feature, direction)
                row = {
                    "group": group_name,
                    "target": target,
                    "target_kind": target_kind,
                    "horizon": horizon,
                    "feature": feature,
                    "direction": direction,
                    "auroc": value,
                    "n_rows": int(group[[target, feature]].dropna().shape[0]),
                    "n_prompt_pairs": int(group[[col for col in BOOTSTRAP_KEY_COLS if col in group.columns]].drop_duplicates().shape[0]),
                    "positive_rate": float(labels.mean()),
                }
                should_bootstrap = args.bootstrap_samples and (
                    args.bootstrap_scope == "groups" or group_name == "all"
                )
                if should_bootstrap:
                    seed = args.bootstrap_seed + len(rows) * 9973
                    row.update(bootstrap_auroc(group, target, feature, direction, args.bootstrap_samples, seed))
                rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "branch_prediction_auc.csv", index=False)
    printable = out.dropna(subset=["auroc"]).sort_values(
        ["group", "target_kind", "horizon", "auroc"],
        ascending=[True, True, True, False],
        na_position="first",
    )
    print(printable.to_string(index=False))
    print(f"Wrote {out_dir / 'branch_prediction_auc.csv'}")


if __name__ == "__main__":
    main()
