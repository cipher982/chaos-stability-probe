#!/usr/bin/env python3
"""Score simple branch-window predictors from trajectory event rows."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


FEATURES = {
    "js_divergence": "higher",
    "centered_logit_normalized_l2": "higher",
    "top1_flip": "higher",
    "max_entropy": "higher",
    "min_margin_logit": "lower",
}


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_windows", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
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
                scores = group[feature]
                if direction == "lower":
                    scores = -scores
                value = auroc(labels, scores)
                rows.append(
                    {
                        "group": group_name,
                        "target": target,
                        "target_kind": target_kind,
                        "horizon": horizon,
                        "feature": feature,
                        "direction": direction,
                        "auroc": value,
                        "n_rows": int(group[[target, feature]].dropna().shape[0]),
                        "positive_rate": float(labels.mean()),
                    }
                )

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
