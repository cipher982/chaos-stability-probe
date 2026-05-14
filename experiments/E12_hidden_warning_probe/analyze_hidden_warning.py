#!/usr/bin/env python3
"""Score hidden-state warning metrics around visible branch events."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BOOTSTRAP_KEY_COLS = ["model_name", "pair_id"]

SUMMARY_FEATURES = {
    "js_divergence": "higher",
    "final_layer_cosine_distance": "higher",
    "final_layer_normalized_l2": "higher",
    "final_layer_abs_norm_delta": "higher",
    "final_layer_rel_norm_delta": "higher",
    "max_layer_cosine_distance": "higher",
    "max_layer_normalized_l2": "higher",
    "max_layer_abs_norm_delta": "higher",
    "max_layer_rel_norm_delta": "higher",
    "max_effective_branching_factor": "higher",
}

LAYER_FEATURES = {
    "last_token_cosine_distance": "higher",
    "last_token_normalized_l2": "higher",
    "last_token_abs_norm_delta": "higher",
    "last_token_rel_norm_delta": "higher",
}


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


def add_targets(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    out = df.copy()
    out["branch_t"] = pd.to_numeric(out["branch_t"], errors="coerce")
    out["tokens_until_branch"] = pd.to_numeric(out["tokens_until_branch"], errors="coerce")
    has_branch = out["branch_t"].notna()
    out["at_branch"] = has_branch & (out["tokens_until_branch"] == 0)
    for horizon in horizons:
        out[f"pre_branch_within_{horizon}"] = (
            has_branch
            & (out["tokens_until_branch"] > 0)
            & (out["tokens_until_branch"] <= horizon)
        )
        out[f"pre_branch_exact_{horizon}"] = (
            has_branch & (out["tokens_until_branch"] == horizon)
        )
    return out


def target_specs(horizons: list[int]) -> list[tuple[str, str, int | None]]:
    specs = [("at_branch", "at_branch", None)]
    for horizon in horizons:
        specs.append((f"pre_branch_within_{horizon}", "strict_pre_branch_warning_window", horizon))
        specs.append((f"pre_branch_exact_{horizon}", "strict_pre_branch_exact_offset", horizon))
    return specs


def feature_rows(
    df: pd.DataFrame,
    features: dict[str, str],
    source: str,
    horizons: list[int],
    group_cols: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if group_cols:
        groups = df.groupby(group_cols, dropna=False, sort=False)
    else:
        groups = [(("all",), df)]
        group_cols = ["model_name"]
    for group_key, group in groups:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        group_values = dict(zip(group_cols, group_key, strict=False))
        for target, target_kind, horizon in target_specs(horizons):
            if target not in group:
                continue
            labels = group[target]
            for feature, direction in features.items():
                if feature not in group:
                    continue
                scores = pd.to_numeric(group[feature], errors="coerce")
                if direction == "lower":
                    scores = -scores
                data = pd.DataFrame({"label": labels.astype(bool), "score": scores}).dropna()
                rows.append(
                    {
                        **group_values,
                        "source": source,
                        "target": target,
                        "target_kind": target_kind,
                        "horizon": horizon,
                        "feature": feature,
                        "direction": direction,
                        "auroc": auroc(data["label"], data["score"]) if not data.empty else None,
                        "n_rows": int(len(data)),
                        "n_positive": int(data["label"].sum()) if not data.empty else 0,
                        "positive_rate": float(data["label"].mean()) if not data.empty else np.nan,
                        "n_prompt_pairs": int(
                            group[[col for col in BOOTSTRAP_KEY_COLS if col in group.columns]]
                            .drop_duplicates()
                            .shape[0]
                        ),
                    }
                )
    return rows


def add_summary_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {"effective_branching_factor_a", "effective_branching_factor_b"} <= set(out.columns):
        out["max_effective_branching_factor"] = out[
            ["effective_branching_factor_a", "effective_branching_factor_b"]
        ].max(axis=1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument("--layers", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, action="append", default=[1, 2, 5, 10])
    parser.add_argument(
        "--min-branch-t",
        type=int,
        default=None,
        help="Keep branched cases with branch_t >= this value; no-branch rows are retained.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    horizons = sorted(set(args.horizon))

    if len(args.summary) != len(args.layers):
        raise SystemExit("--summary and --layers must be provided the same number of times")
    summary = add_targets(
        add_summary_features(pd.concat((pd.read_csv(path) for path in args.summary), ignore_index=True)),
        horizons,
    )
    layers = add_targets(pd.concat((pd.read_csv(path) for path in args.layers), ignore_index=True), horizons)

    if args.min_branch_t is not None:
        summary = summary[summary["branch_t"].isna() | (summary["branch_t"] >= args.min_branch_t)].copy()
        layers = layers[layers["branch_t"].isna() | (layers["branch_t"] >= args.min_branch_t)].copy()

    rows: list[dict[str, object]] = []
    rows.extend(feature_rows(summary, SUMMARY_FEATURES, "summary", horizons, ["model_name"]))
    rows.extend(feature_rows(summary, SUMMARY_FEATURES, "summary", horizons, []))
    rows.extend(feature_rows(layers, LAYER_FEATURES, "layer", horizons, ["model_name", "layer"]))
    rows.extend(feature_rows(layers, LAYER_FEATURES, "layer", horizons, ["layer"]))

    out = pd.DataFrame(rows)
    if "model_name" not in out.columns:
        out["model_name"] = "all"
    out["model_name"] = out["model_name"].fillna("all")
    out = out.sort_values(
        ["source", "target_kind", "horizon", "model_name", "feature", "auroc"],
        ascending=[True, True, True, True, True, False],
        na_position="last",
    )
    out.to_csv(args.out_dir / "hidden_warning_auc.csv", index=False)

    best = (
        out[out["source"] == "layer"]
        .dropna(subset=["auroc"])
        .sort_values(["target", "model_name", "feature", "auroc"], ascending=[True, True, True, False])
        .groupby(["target", "model_name", "feature"], dropna=False)
        .head(1)
    )
    best.to_csv(args.out_dir / "hidden_warning_best_layers.csv", index=False)

    printable = out[
        (out["source"] == "summary")
        & (out["model_name"] == "all")
        & out["target"].isin(["at_branch", "pre_branch_within_1", "pre_branch_within_2", "pre_branch_within_5", "pre_branch_within_10"])
    ].dropna(subset=["auroc"])
    print(printable.sort_values(["target", "auroc"], ascending=[True, False]).to_string(index=False))
    print(f"Wrote {args.out_dir / 'hidden_warning_auc.csv'}")
    print(f"Wrote {args.out_dir / 'hidden_warning_best_layers.csv'}")


if __name__ == "__main__":
    main()
