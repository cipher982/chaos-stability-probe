#!/usr/bin/env python3
"""Summarize logit probes from the 512-token scaffold-long wave."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SMALL_CATEGORIES = ["noop_format", "punctuation", "synonym"]
DROP_HEAVY_FIELDS = {"topk_a", "topk_b"}
DEFAULT_WAVE_DIR = Path("runs/rankings/scaffold_long_wave")
DEFAULT_OUT_DIR = Path("runs/rankings/scaffold_long_logits")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            for field in DROP_HEAVY_FIELDS:
                row.pop(field, None)
            rows.append(row)
    return rows


def summarize_logits(group: pd.DataFrame) -> dict[str, Any]:
    top1_same = group["top1_same"].astype(bool)
    return {
        "n_rows": int(len(group)),
        "js_mean": float(group["js_divergence"].mean()),
        "js_median": float(group["js_divergence"].median()),
        "kl_a_to_b_mean": float(group["kl_a_to_b"].mean()),
        "kl_b_to_a_mean": float(group["kl_b_to_a"].mean()),
        "top1_flip_rate": float((~top1_same).mean()),
        "mean_abs_logit_delta": float(group["mean_abs_logit_delta"].mean()),
        "rms_logit_delta": float(group["rms_logit_delta"].mean()),
        "centered_logit_l2": float(group["centered_logit_normalized_l2"].mean()),
        "a_top1_margin_logit": float(group["a_top1_margin_logit"].mean()),
        "b_top1_margin_logit": float(group["b_top1_margin_logit"].mean()),
        "mean_top1_margin_logit": float(
            pd.concat([group["a_top1_margin_logit"], group["b_top1_margin_logit"]]).mean()
        ),
        "a_top1_prob": float(group["a_top1_prob"].mean()),
        "b_top1_prob": float(group["b_top1_prob"].mean()),
    }


def clean_first_diff(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if value == "":
        return None
    return int(value)


def first_diff_logits(logits: pd.DataFrame, semantic: pd.DataFrame) -> pd.DataFrame:
    rows = []
    key_cols = ["model_name", "pair_id", "category", "repeat"]
    indexed = logits.set_index(key_cols + ["anchor", "t"], drop=False)
    small = semantic[semantic["category"].isin(SMALL_CATEGORIES)].copy()
    for _, row in small.iterrows():
        first_diff = clean_first_diff(row.get("first_diff_token"))
        if first_diff is None:
            continue
        # t indexes the next-token distribution after teacher-forcing t tokens.
        # At t=0 we are at prompt end; at t=first_diff we are at the branch point.
        candidates = []
        for offset_name, t in [
            ("pre_branch", max(0, first_diff - 1)),
            ("branch", max(0, first_diff)),
            ("post_branch", max(0, first_diff + 1)),
        ]:
            for anchor in ["prompt_a_generation", "prompt_b_generation"]:
                key = (row["model_name"], row["pair_id"], row["category"], row["repeat"], anchor, t)
                if key in indexed.index:
                    match = indexed.loc[key]
                    if isinstance(match, pd.DataFrame):
                        match = match.iloc[0]
                    item = match.to_dict()
                    item["branch_window"] = offset_name
                    item["first_diff_token"] = first_diff
                    item["semantic_cosine_distance"] = row["semantic_cosine_distance"]
                    item["common_prefix_tokens"] = row["common_prefix_tokens"]
                    candidates.append(item)
        rows.extend(candidates)
    return pd.DataFrame(rows)


def scatter_plot(
    df: pd.DataFrame,
    out_path: Path,
    x: str,
    y: str,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    if df.empty or x not in df or y not in df:
        return
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    colors = {
        "thinking_process": "#3478b9",
        "visible_cot": "#3d9a57",
        "think_tag": "#7b59b5",
        "template_echo": "#c84c3c",
        "none": "#5c6770",
    }
    for prefix, group in df.groupby("dominant_prefix_kind", dropna=False, sort=False):
        prefix_label = str(prefix)
        ax.scatter(
            group[x],
            group[y],
            s=74,
            alpha=0.88,
            label=prefix_label,
            color=colors.get(prefix_label, "#777777"),
            edgecolor="white",
            linewidth=0.8,
        )
        for _, row in group.iterrows():
            ax.annotate(
                row["run_label"],
                (row[x], row[y]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8,
                alpha=0.82,
            )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(title="prefix kind", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave-dir", type=Path, default=DEFAULT_WAVE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.wave_dir / "job_manifest.csv"
    semantic_path = args.wave_dir / "merged_summary.csv"
    if not manifest_path.exists() or not semantic_path.exists():
        raise SystemExit("Run scripts/process_scaffold_long_wave.py first")

    manifest = pd.read_csv(manifest_path)
    semantic = pd.read_csv(semantic_path)
    ready = manifest[manifest["ready"].astype(bool)].copy()

    frames = []
    for _, job in ready.iterrows():
        run_dir = Path(str(job["run_dir"]))
        path = run_dir / "logit_probes.jsonl"
        if not path.exists():
            continue
        df = pd.DataFrame(read_jsonl(path))
        df["job_name"] = job["job_name"]
        df["run_label"] = job["run_label"]
        df["dominant_prefix_kind"] = job["dominant_prefix_kind"]
        df["visible_reasoning_scaffold"] = job["visible_reasoning_scaffold"]
        df["template_echo"] = job["template_echo"]
        df["boundary_detection"] = job["boundary_detection"]
        frames.append(df)

    if not frames:
        raise SystemExit("No ready scaffold-long logit probes found")

    logits = pd.concat(frames, ignore_index=True)
    logits.to_csv(args.out_dir / "merged_logit_probes_light.csv", index=False)

    small = logits[logits["category"].isin(SMALL_CATEGORIES)].copy()
    prompt_end = small[(small["anchor"] == "prompt_a_generation") & (small["t"] == 0)].copy()

    rows = []
    for label, group in prompt_end.groupby("run_label", sort=False):
        meta = group.iloc[0]
        rows.append(
            {
                "run_label": label,
                "model_name": meta["model_name"],
                "dominant_prefix_kind": meta["dominant_prefix_kind"],
                "visible_reasoning_scaffold": meta["visible_reasoning_scaffold"],
                "template_echo": meta["template_echo"],
                **summarize_logits(group),
            }
        )
    prompt_summary = pd.DataFrame(rows)
    if not prompt_summary.empty:
        prompt_summary = prompt_summary.sort_values("mean_top1_margin_logit", ascending=False).reset_index(drop=True)
    prompt_summary.to_csv(args.out_dir / "prompt_end_logit_summary.csv", index=False)

    trajectory = small[small["t"] > 0].copy()
    rows = []
    for label, group in trajectory.groupby("run_label", sort=False):
        meta = group.iloc[0]
        rows.append(
            {
                "run_label": label,
                "model_name": meta["model_name"],
                "dominant_prefix_kind": meta["dominant_prefix_kind"],
                "visible_reasoning_scaffold": meta["visible_reasoning_scaffold"],
                "template_echo": meta["template_echo"],
                **summarize_logits(group),
            }
        )
    trajectory_summary = pd.DataFrame(rows).sort_values("js_mean").reset_index(drop=True)
    trajectory_summary.to_csv(args.out_dir / "teacher_forced_trajectory_logit_summary.csv", index=False)

    by_t = (
        trajectory.groupby(["run_label", "dominant_prefix_kind", "t"], as_index=False, observed=True)
        .agg(
            js_mean=("js_divergence", "mean"),
            top1_flip_rate=("top1_same", lambda s: float((~s.astype(bool)).mean())),
            mean_top1_margin_logit=(
                "a_top1_margin_logit",
                "mean",
            ),
        )
        .sort_values(["run_label", "t"])
    )
    by_t.to_csv(args.out_dir / "teacher_forced_js_by_t.csv", index=False)

    branch = first_diff_logits(logits, semantic)
    if not branch.empty:
        branch_summary_rows = []
        for keys, group in branch.groupby(["run_label", "branch_window"], sort=False):
            label, window = keys
            meta = group.iloc[0]
            branch_summary_rows.append(
                {
                    "run_label": label,
                    "branch_window": window,
                    "dominant_prefix_kind": meta["dominant_prefix_kind"],
                    **summarize_logits(group),
                    "semantic_mean": float(group["semantic_cosine_distance"].mean()),
                    "common_prefix_mean": float(group["common_prefix_tokens"].mean()),
                }
            )
        branch.to_csv(args.out_dir / "branch_window_logit_rows.csv", index=False)
        branch_summary = pd.DataFrame(branch_summary_rows)
        branch_summary.to_csv(args.out_dir / "branch_window_logit_summary.csv", index=False)

    semantic_small = pd.read_csv(args.wave_dir / "small_perturbation_bootstrap.csv")
    joined = semantic_small.merge(prompt_summary, on="run_label", how="inner", suffixes=("_semantic", "_logit"))
    joined = joined.rename(columns={"mean": "semantic_mean"})
    joined.to_csv(args.out_dir / "semantic_vs_prompt_end_logits.csv", index=False)

    corr_cols = [
        "semantic_mean",
        "js_mean",
        "top1_flip_rate",
        "mean_top1_margin_logit",
        "a_top1_prob",
        "mean_abs_logit_delta",
        "centered_logit_l2",
    ]
    corr_rows = []
    for col in corr_cols[1:]:
        if col in joined and len(joined[[corr_cols[0], col]].dropna()) >= 3:
            corr_rows.append(
                {
                    "x": col,
                    "y": "semantic_mean",
                    "pearson_corr": float(joined["semantic_mean"].corr(joined[col])),
                    "n_models": int(len(joined[[corr_cols[0], col]].dropna())),
                }
            )
    pd.DataFrame(corr_rows).to_csv(args.out_dir / "semantic_logit_correlations.csv", index=False)

    scatter_plot(
        joined,
        args.out_dir / "semantic_vs_prompt_end_margin.png",
        "mean_top1_margin_logit",
        "semantic_mean",
        "512-token semantic divergence vs prompt-end top-token margin",
        "Mean prompt-end top-1 logit margin",
        "Mean semantic distance over small perturbations",
    )
    scatter_plot(
        joined,
        args.out_dir / "semantic_vs_prompt_end_flip_rate.png",
        "top1_flip_rate",
        "semantic_mean",
        "512-token semantic divergence vs prompt-end top-1 flip rate",
        "Prompt-end top-1 flip rate",
        "Mean semantic distance over small perturbations",
    )

    print("Ready logit models:", len(prompt_summary))
    print()
    print("Prompt-end logit summary")
    print(prompt_summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print()
    if corr_rows:
        print("Correlations with 512-token semantic mean")
        print(pd.DataFrame(corr_rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"Wrote scaffold-long logit artifacts to {args.out_dir}")


if __name__ == "__main__":
    main()
