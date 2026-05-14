#!/usr/bin/env python3
"""Nested-layer residual-delta probes for E12 hidden-warning artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_hidden_vectors import (
    auroc,
    group_vectors,
    load_artifacts,
    normalize_rows,
    oof_mean_diff_scores,
    stable_fold,
    weighted_auroc,
)


def fit_scores(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    normalize: bool,
) -> np.ndarray | None:
    if normalize:
        x_train = normalize_rows(x_train)
        x_test = normalize_rows(x_test)
    if y_train.sum() == 0 or (~y_train).sum() == 0:
        return None
    pos_mean = x_train[y_train].mean(axis=0, dtype=np.float64)
    neg_mean = x_train[~y_train].mean(axis=0, dtype=np.float64)
    direction = pos_mean - neg_mean
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm <= 1e-12:
        return None
    return x_test.astype(np.float64) @ (direction / norm)


def pair_weights(pair_ids: pd.Series) -> np.ndarray:
    return 1.0 / pair_ids.groupby(pair_ids).transform("size").to_numpy(dtype=np.float64)


def inner_layer_score(
    layer_group: pd.DataFrame,
    vectors: dict[int, np.ndarray],
    label: pd.Series,
    fold_ids: np.ndarray,
    outer_fold: object,
    normalize: bool,
) -> float | None:
    train_group = layer_group[fold_ids[layer_group.index.to_numpy()] != outer_fold]
    if train_group.empty:
        return None
    y_train = label.loc[train_group.index].to_numpy(dtype=bool)
    if y_train.sum() == 0 or (~y_train).sum() == 0:
        return None
    inner_folds = fold_ids[train_group.index.to_numpy()]
    if pd.unique(inner_folds).size < 2:
        return None
    x_train = group_vectors(train_group, vectors).astype(np.float32, copy=False)
    scores, _ = oof_mean_diff_scores(x_train, y_train, inner_folds, normalize=normalize)
    weights = pair_weights(train_group["pair_id"])
    return weighted_auroc(y_train, scores, weights)


def score_nested(
    model_group: pd.DataFrame,
    vectors: dict[int, np.ndarray],
    label: pd.Series,
    fold_ids: np.ndarray,
    normalize: bool,
) -> dict[str, object] | None:
    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_pair_ids: list[pd.Series] = []
    selected_layers: Counter[int] = Counter()

    layer_groups = {
        int(layer): group
        for layer, group in model_group.groupby("layer", sort=False)
    }
    for outer_fold in pd.unique(fold_ids[model_group.index.to_numpy()]):
        best_layer = None
        best_score = -np.inf
        for layer, layer_group in layer_groups.items():
            score = inner_layer_score(
                layer_group,
                vectors,
                label,
                fold_ids,
                outer_fold,
                normalize=normalize,
            )
            if score is not None and np.isfinite(score) and score > best_score:
                best_layer = layer
                best_score = float(score)
        if best_layer is None:
            continue

        group = layer_groups[best_layer]
        local_folds = fold_ids[group.index.to_numpy()]
        train = local_folds != outer_fold
        test = local_folds == outer_fold
        if not train.any() or not test.any():
            continue
        y = label.loc[group.index].to_numpy(dtype=bool)
        scores = fit_scores(
            group_vectors(group[train], vectors).astype(np.float32, copy=False),
            y[train],
            group_vectors(group[test], vectors).astype(np.float32, copy=False),
            normalize=normalize,
        )
        if scores is None:
            continue
        all_scores.append(scores)
        all_labels.append(y[test])
        all_pair_ids.append(group.loc[test, "pair_id"])
        selected_layers[best_layer] += 1

    if not all_scores:
        return None

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    pair_ids = pd.concat(all_pair_ids, ignore_index=True)
    weights = pair_weights(pair_ids)
    return {
        "probe": "unit_mean_diff_nested_layer" if normalize else "raw_mean_diff_nested_layer",
        "auroc": auroc(labels, scores),
        "pair_weighted_auroc": weighted_auroc(labels, scores, weights),
        "n_rows": int(labels.size),
        "n_positive": int(labels.sum()),
        "positive_rate": float(labels.mean()),
        "n_prompt_pairs": int(pair_ids.nunique()),
        "used_folds": int(sum(selected_layers.values())),
        "selected_layers": ";".join(
            f"{layer}:{count}" for layer, count in sorted(selected_layers.items())
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, action="append", default=[1, 2, 5, 10])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--split-mode",
        choices=["pair_hash", "category_holdout"],
        action="append",
        default=["pair_hash"],
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta, vectors = load_artifacts(args.artifact_dir)
    meta["branch_t"] = pd.to_numeric(meta["branch_t"], errors="coerce")
    meta["tokens_until_branch"] = pd.to_numeric(meta["tokens_until_branch"], errors="coerce")
    meta = meta[meta["branch_t"].isna() | (meta["tokens_until_branch"] >= 0)].copy()
    meta = meta.reset_index(drop=True)

    split_ids: dict[str, np.ndarray] = {
        "pair_hash": np.array([stable_fold(pair_id, args.folds) for pair_id in meta["pair_id"]]),
        "category_holdout": meta["category"].fillna("unknown").astype(str).to_numpy(),
    }

    rows: list[dict[str, object]] = []
    for split_mode in list(dict.fromkeys(args.split_mode)):
        fold_ids = split_ids[split_mode]
        for horizon in sorted(set(args.horizon)):
            label = (
                meta["branch_t"].notna()
                & (meta["tokens_until_branch"] > 0)
                & (meta["tokens_until_branch"] <= horizon)
            )
            for model_name, model_group in meta.groupby("model_name", sort=False):
                for normalize in [False, True]:
                    result = score_nested(
                        model_group,
                        vectors,
                        label,
                        fold_ids,
                        normalize=normalize,
                    )
                    if result is None:
                        continue
                    rows.append(
                        {
                            "model_name": model_name,
                            "target": f"pre_branch_within_{horizon}",
                            "target_kind": "strict_pre_branch_warning_window",
                            "horizon": horizon,
                            "split_mode": split_mode,
                            **result,
                        }
                    )

    out = pd.DataFrame(rows)
    out.to_csv(args.out_dir / "hidden_vector_warning_nested_layer_auc.csv", index=False)
    if not out.empty:
        print(out.sort_values(["split_mode", "horizon", "model_name", "probe"]).to_string(index=False))
    print(f"Wrote {args.out_dir / 'hidden_vector_warning_nested_layer_auc.csv'}")


if __name__ == "__main__":
    main()
