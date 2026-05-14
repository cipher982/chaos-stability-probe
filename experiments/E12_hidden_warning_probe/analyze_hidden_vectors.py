#!/usr/bin/env python3
"""Train simple residual-delta probes from captured hidden-vector artifacts."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    valid = np.isfinite(scores)
    labels = labels[valid].astype(bool)
    scores = scores[valid]
    if labels.size == 0:
        return None
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)

    sorted_scores = scores[order]
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = ranks[order[start:end]].mean()
        start = end

    pos_rank_sum = float(ranks[labels].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def weighted_auroc(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> float | None:
    valid = np.isfinite(scores) & np.isfinite(weights) & (weights > 0)
    labels = labels[valid].astype(bool)
    scores = scores[valid]
    weights = weights[valid].astype(np.float64)
    if labels.size == 0:
        return None
    pos = labels
    neg = ~labels
    pos_weight = float(weights[pos].sum())
    neg_weight = float(weights[neg].sum())
    if pos_weight <= 0 or neg_weight <= 0:
        return None
    pos_scores = scores[pos]
    neg_scores = scores[neg]
    pos_weights = weights[pos]
    neg_weights = weights[neg]
    total = 0.0
    for score, weight in zip(pos_scores, pos_weights, strict=True):
        total += weight * float(
            neg_weights[score > neg_scores].sum()
            + 0.5 * neg_weights[score == neg_scores].sum()
        )
    return total / (pos_weight * neg_weight)


def stable_fold(pair_id: object, folds: int) -> int:
    digest = hashlib.sha1(str(pair_id).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % folds


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def oof_mean_diff_scores(
    x: np.ndarray,
    y: np.ndarray,
    fold_ids: np.ndarray,
    normalize: bool,
) -> tuple[np.ndarray, int]:
    if normalize:
        x = normalize_rows(x)
    scores = np.full(y.shape[0], np.nan, dtype=np.float64)
    used_folds = 0
    for fold in pd.unique(fold_ids):
        train = fold_ids != fold
        test = fold_ids == fold
        if not train.any() or not test.any():
            continue
        if y[train].sum() == 0 or (~y[train]).sum() == 0:
            continue
        pos_mean = x[train & y].mean(axis=0, dtype=np.float64)
        neg_mean = x[train & ~y].mean(axis=0, dtype=np.float64)
        direction = pos_mean - neg_mean
        direction_norm = np.linalg.norm(direction)
        if not np.isfinite(direction_norm) or direction_norm <= 1e-12:
            continue
        direction = direction / direction_norm
        scores[test] = x[test].astype(np.float64) @ direction
        used_folds += 1
    return scores, used_folds


def artifact_pairs(artifact_dirs: list[Path]) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for artifact_dir in artifact_dirs:
        for meta_path in sorted(artifact_dir.glob("*_hidden_vector_features.csv")):
            npz_path = meta_path.with_suffix(".npz")
            if not npz_path.exists():
                raise FileNotFoundError(f"Missing vector array for {meta_path}: {npz_path}")
            pairs.append((meta_path, npz_path))
    return pairs


def load_artifacts(artifact_dirs: list[Path]) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    frames = []
    arrays: dict[int, np.ndarray] = {}
    for artifact_id, (meta_path, npz_path) in enumerate(artifact_pairs(artifact_dirs)):
        meta = pd.read_csv(meta_path)
        delta = np.load(npz_path)["delta"]
        if len(meta) != delta.shape[0]:
            raise ValueError(f"{meta_path} has {len(meta)} rows but {npz_path} has {delta.shape[0]} vectors")
        meta = meta.copy()
        meta["artifact_id"] = artifact_id
        meta["artifact_meta_path"] = str(meta_path)
        meta["artifact_npz_path"] = str(npz_path)
        frames.append(meta)
        arrays[artifact_id] = delta
    if not frames:
        raise SystemExit("No *_hidden_vector_features.csv artifacts found")
    return pd.concat(frames, ignore_index=True), arrays


def group_vectors(group: pd.DataFrame, arrays: dict[int, np.ndarray]) -> np.ndarray:
    group = group.copy()
    group["_row_position"] = np.arange(len(group))
    dim = None
    out = None
    for artifact_id, artifact_group in group.groupby("artifact_id", sort=False):
        local_idx = artifact_group["vector_index"].to_numpy(dtype=int)
        chunk = arrays[int(artifact_id)][local_idx]
        if dim is None:
            dim = chunk.shape[1]
            out = np.empty((len(group), dim), dtype=chunk.dtype)
        elif chunk.shape[1] != dim:
            raise ValueError("Grouped vector artifacts have mixed hidden dimensions")
        out[artifact_group["_row_position"].to_numpy(dtype=int)] = chunk
    if out is None:
        raise ValueError("Empty vector group")
    return out


def score_group(
    group: pd.DataFrame,
    vectors: dict[int, np.ndarray],
    target: str,
    target_kind: str,
    horizon: int,
    label: pd.Series,
    fold_ids: np.ndarray,
    split_mode: str,
) -> list[dict[str, object]]:
    x = group_vectors(group, vectors).astype(np.float32, copy=False)
    y = label.loc[group.index].to_numpy(dtype=bool)
    local_fold_ids = fold_ids[group.index.to_numpy()]
    row_weights = (
        1.0
        / group.groupby("pair_id")["pair_id"].transform("size").to_numpy(dtype=np.float64)
    )
    rows = []
    if y.sum() == 0 or (~y).sum() == 0:
        return rows
    for normalize in [False, True]:
        scores, used_folds = oof_mean_diff_scores(
            x,
            y,
            local_fold_ids,
            normalize=normalize,
        )
        rows.append(
            {
                "model_name": group["model_name"].iloc[0],
                "layer": int(group["layer"].iloc[0]),
                "target": target,
                "target_kind": target_kind,
                "horizon": horizon,
                "split_mode": split_mode,
                "probe": "unit_mean_diff" if normalize else "raw_mean_diff",
                "auroc": auroc(y, scores),
                "pair_weighted_auroc": weighted_auroc(y, scores, row_weights),
                "n_rows": int(len(group)),
                "n_positive": int(y.sum()),
                "positive_rate": float(y.mean()),
                "n_prompt_pairs": int(group["pair_id"].nunique()),
                "used_folds": used_folds,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, action="append", default=[0, 1, 2, 5, 10])
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
    eligible = meta["branch_t"].isna() | (meta["tokens_until_branch"] >= 0)
    meta = meta[eligible].copy()
    meta = meta.reset_index(drop=True)
    split_modes = list(dict.fromkeys(args.split_mode))
    split_ids: dict[str, np.ndarray] = {}
    split_ids["pair_hash"] = np.array([stable_fold(pair_id, args.folds) for pair_id in meta["pair_id"]])
    split_ids["category_holdout"] = meta["category"].fillna("unknown").astype(str).to_numpy()

    rows: list[dict[str, object]] = []
    for split_mode in split_modes:
        for horizon in sorted(set(args.horizon)):
            if horizon == 0:
                specs = [
                    (
                        meta,
                        "at_branch",
                        "at_branch",
                        meta["branch_t"].notna() & (meta["tokens_until_branch"] == 0),
                    )
                ]
            else:
                exact_subset = (
                    (
                        meta["branch_t"].notna()
                        & (meta["tokens_until_branch"] == horizon)
                    )
                    | (
                        meta["branch_t"].isna()
                        & (meta["t"] == horizon)
                    )
                )
                specs = [
                    (
                        meta,
                        f"pre_branch_within_{horizon}",
                        "strict_pre_branch_warning_window",
                        meta["branch_t"].notna()
                        & (meta["tokens_until_branch"] > 0)
                        & (meta["tokens_until_branch"] <= horizon),
                    ),
                    (
                        meta[exact_subset].copy(),
                        f"pre_branch_exact_{horizon}",
                        "strict_pre_branch_exact_offset",
                        meta.loc[exact_subset, "branch_t"].notna()
                        & (meta.loc[exact_subset, "tokens_until_branch"] == horizon),
                    ),
                ]
            for spec_meta, target, target_kind, label in specs:
                for _, group in spec_meta.groupby(["model_name", "layer"], sort=False):
                    rows.extend(
                        score_group(
                            group,
                            vectors,
                            target,
                            target_kind,
                            horizon,
                            label,
                            split_ids[split_mode],
                            split_mode,
                        )
                    )

    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(
            columns=[
                "model_name",
                "layer",
                "target",
                "target_kind",
                "horizon",
                "split_mode",
                "probe",
                "auroc",
                "pair_weighted_auroc",
                "n_rows",
                "n_positive",
                "positive_rate",
                "n_prompt_pairs",
                "used_folds",
            ]
        )
    out = out.sort_values(
        ["split_mode", "target_kind", "horizon", "model_name", "probe", "auroc"],
        ascending=[True, True, True, True, True, False],
    )
    out.to_csv(args.out_dir / "hidden_vector_warning_auc.csv", index=False)
    best = (
        out.dropna(subset=["auroc"])
        .sort_values(["split_mode", "target", "model_name", "probe", "auroc"], ascending=[True, True, True, True, False])
        .groupby(["split_mode", "target", "model_name", "probe"], dropna=False)
        .head(1)
    )
    best.to_csv(args.out_dir / "hidden_vector_warning_best_layers.csv", index=False)
    print(best.to_string(index=False))
    print(f"Wrote {args.out_dir / 'hidden_vector_warning_auc.csv'}")
    print(f"Wrote {args.out_dir / 'hidden_vector_warning_best_layers.csv'}")


if __name__ == "__main__":
    main()
