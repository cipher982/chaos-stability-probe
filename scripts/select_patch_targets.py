#!/usr/bin/env python3
"""Select branch-point candidates for activation patching.

This consumes the artifacts written by analyze_branch_points.py and turns them
into a ranked patch queue. The goal is to prefer cases where a tiny prompt edit
does little before the branch, but flips a low-confidence next-token decision at
the branch and produces visible semantic/output divergence.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd


BRANCH_DIR = "branch_points"


def finite(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def infer_model_name(run_dir: Path, candidates: pd.DataFrame) -> str:
    if "model_name" in candidates.columns and not candidates.empty:
        return str(candidates["model_name"].iloc[0])
    return run_dir.name


def load_run(run_dir: Path) -> pd.DataFrame:
    branch_dir = run_dir / BRANCH_DIR
    candidates_path = branch_dir / "branch_candidates.csv"
    rows_path = branch_dir / "branch_window_logit_rows.csv"
    if not candidates_path.exists():
        raise FileNotFoundError(f"{candidates_path} missing; run analyze_branch_points.py first")
    if not rows_path.exists():
        raise FileNotFoundError(f"{rows_path} missing; run analyze_branch_points.py first")

    candidates = pd.read_csv(candidates_path)
    rows = pd.read_csv(rows_path)
    if candidates.empty:
        return candidates

    group_cols = ["model_name", "pair_id", "category", "repeat", "branch_window"]
    by_window = (
        rows.groupby(group_cols, dropna=False)
        .agg(
            js_mean=("js_divergence", "mean"),
            top1_flip_rate=("top1_same", lambda s: float((~s.astype(bool)).mean())),
            top1_margin_logit=("a_top1_margin_logit", "mean"),
            a_top1_prob=("a_top1_prob", "mean"),
            b_top1_prob=("b_top1_prob", "mean"),
        )
        .reset_index()
    )
    wide = by_window.pivot_table(
        index=["model_name", "pair_id", "category", "repeat"],
        columns="branch_window",
        values=["js_mean", "top1_flip_rate", "top1_margin_logit", "a_top1_prob", "b_top1_prob"],
        aggfunc="first",
    )
    wide.columns = [f"{window}_{metric}" for metric, window in wide.columns]
    wide = wide.reset_index()

    merged = candidates.merge(wide, on=["model_name", "pair_id", "category", "repeat"], how="left")
    merged["run_dir"] = str(run_dir)
    merged["model_selector"] = infer_model_name(run_dir, candidates)
    return merged


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        item = row.to_dict()
        semantic = finite(row.get("semantic_cosine_distance"))
        token_edit = finite(row.get("token_edit_distance_norm"))
        first_diff = finite(row.get("first_diff_token"), 999.0)
        prompt_edit = finite(row.get("prompt_token_edit_distance"))
        branch_flip = finite(row.get("branch_top1_flip_rate"))
        branch_js = finite(row.get("branch_js_mean"))
        pre_js = finite(row.get("pre_branch_js_mean"))
        branch_margin = finite(row.get("branch_top1_margin_logit"))

        early_branch_bonus = 1.0 / (1.0 + max(first_diff, 0.0))
        small_prompt_bonus = 1.0 / (1.0 + max(prompt_edit, 0.0))
        quiet_before_branch = max(0.0, branch_js - pre_js)
        low_margin_bonus = 1.0 / (1.0 + abs(branch_margin))

        item["patch_target_score"] = (
            2.8 * semantic
            + 1.2 * branch_flip
            + 0.8 * quiet_before_branch
            + 0.5 * low_margin_bonus
            + 0.3 * early_branch_bonus
            + 0.2 * small_prompt_bonus
            + 0.2 * token_edit
        )
        item["target_reason"] = "; ".join(
            [
                f"semantic={semantic:.3f}",
                f"branch_flip={branch_flip:.2f}",
                f"branch_js={branch_js:.3f}",
                f"pre_js={pre_js:.3f}",
                f"first_diff={int(first_diff) if first_diff < 999 else 'na'}",
            ]
        )
        rows.append(item)
    return pd.DataFrame(rows)


def existing_patch_stems(patch_dir: Path) -> set[str]:
    if not patch_dir.exists():
        return set()
    return {path.stem for path in patch_dir.glob("*.csv")}


def command_for(row: pd.Series, args: argparse.Namespace) -> str:
    return (
        "uv run python scripts/activation_patch_branch.py "
        f"--model {row['model_selector']} "
        f"--pair-id {row['pair_id']} "
        f"--prompt-pairs {args.prompt_pairs} "
        f"--out-dir {args.patch_dir} "
        f"--max-new-tokens {args.max_new_tokens} "
        f"--thinking-mode {args.thinking_mode} "
        f"--positions {args.positions}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=Path("runs/mechinterp_patch/selected_patch_targets.csv"))
    parser.add_argument("--patch-dir", type=Path, default=Path("runs/mechinterp_patch"))
    parser.add_argument("--prompt-pairs", type=Path, default=Path("configs/prompt_pairs_mechinterp_seed.json"))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--thinking-mode", choices=["default", "enabled", "disabled"], default="disabled")
    parser.add_argument("--positions", choices=["final", "changed-final", "aligned", "all"], default="changed-final")
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--min-semantic", type=float, default=0.03)
    parser.add_argument("--max-first-diff", type=int, default=64)
    parser.add_argument("--min-branch-flip-rate", type=float, default=0.5)
    parser.add_argument("--include-existing", action="store_true")
    args = parser.parse_args()

    frames = [load_run(run_dir) for run_dir in args.run_dirs]
    candidates = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    if candidates.empty:
        raise SystemExit("No branch candidates found")

    scored = add_scores(candidates)
    if not args.include_existing:
        done = existing_patch_stems(args.patch_dir)
        scored["already_patched"] = scored.apply(
            lambda row: f"{row['model_selector']}__{row['pair_id']}" in done,
            axis=1,
        )
    else:
        scored["already_patched"] = False

    filtered = scored[
        (scored["semantic_cosine_distance"].fillna(0.0) >= args.min_semantic)
        & (scored["first_diff_token"].fillna(999999) <= args.max_first_diff)
        & (scored["branch_top1_flip_rate"].fillna(0.0) >= args.min_branch_flip_rate)
        & (~scored["already_patched"])
    ].copy()
    filtered = filtered.sort_values("patch_target_score", ascending=False).head(args.top_n)
    if filtered.empty:
        print("No unpatched candidates passed filters")
        scored.sort_values("patch_target_score", ascending=False).to_csv(args.out, index=False)
        return

    filtered["activation_patch_command"] = filtered.apply(lambda row: command_for(row, args), axis=1)
    keep = [
        "patch_target_score",
        "model_selector",
        "pair_id",
        "category",
        "semantic_cosine_distance",
        "token_edit_distance_norm",
        "first_diff_token",
        "prompt_token_edit_distance",
        "branch_top1_flip_rate",
        "branch_js_mean",
        "pre_branch_js_mean",
        "branch_top1_margin_logit",
        "a_branch_token_id",
        "b_branch_token_id",
        "target_reason",
        "activation_patch_command",
        "run_dir",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    filtered[[col for col in keep if col in filtered.columns]].to_csv(args.out, index=False)
    print(f"Wrote {len(filtered)} selected patch targets to {args.out}")
    for command in filtered["activation_patch_command"]:
        print(command)


if __name__ == "__main__":
    main()
