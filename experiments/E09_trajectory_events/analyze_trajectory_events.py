#!/usr/bin/env python3
"""Mine structured branch events from paired logit probes.

The event unit is a prompt pair whose generated continuations share a visible
prefix and then branch. During the common-prefix window, the two logit streams
are evaluated under the same generated history, so logit/hidden differences are
attributable to the prompt perturbation rather than different generated tokens.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


KEY_COLS = ["model_name", "pair_id", "category", "repeat"]
DROP_HEAVY_FIELDS = {"topk_a", "topk_b"}
HORIZONS = [1, 2, 5, 10]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            for field in DROP_HEAVY_FIELDS:
                row.pop(field, None)
            rows.append(row)
    return rows


def load_summary(run_dir: Path) -> pd.DataFrame:
    summary_path = run_dir / "summary_with_semantic.csv"
    if not summary_path.exists():
        summary_path = run_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"{run_dir} missing summary.csv")
    return pd.read_csv(summary_path)


def clean_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value):
        return None
    return int(float(value))


def run_label(run_dir: Path) -> str:
    return run_dir.name


def threshold_from_controls(logits: pd.DataFrame, column: str, fallback: float) -> float:
    if "category" not in logits.columns or column not in logits.columns:
        return fallback
    controls = logits[logits["category"] == "micro_control_identical"][column].dropna()
    if controls.empty:
        return fallback
    return max(fallback, float(controls.quantile(0.999)) * 10.0)


def row_metric(row: pd.Series, field: str) -> float | None:
    value = row.get(field)
    if value is None or pd.isna(value):
        return None
    return float(value)


def min_margin(row: pd.Series) -> float | None:
    a = row_metric(row, "a_top1_margin_logit")
    b = row_metric(row, "b_top1_margin_logit")
    vals = [v for v in [a, b] if v is not None]
    return min(vals) if vals else None


def max_entropy(row: pd.Series) -> float | None:
    a = row_metric(row, "entropy_a")
    b = row_metric(row, "entropy_b")
    vals = [v for v in [a, b] if v is not None]
    return max(vals) if vals else None


def effective_branching_factor(row: pd.Series, side: str) -> float | None:
    field = f"effective_branching_factor_{side}"
    value = row_metric(row, field)
    if value is not None:
        return value
    entropy = row_metric(row, f"entropy_{side}")
    if entropy is None:
        return None
    return math.exp(entropy)


def max_branching_factor(row: pd.Series) -> float | None:
    a = effective_branching_factor(row, "a")
    b = effective_branching_factor(row, "b")
    vals = [v for v in [a, b] if v is not None]
    return max(vals) if vals else None


def min_branching_factor(row: pd.Series) -> float | None:
    a = effective_branching_factor(row, "a")
    b = effective_branching_factor(row, "b")
    vals = [v for v in [a, b] if v is not None]
    return min(vals) if vals else None


def first_warning_t(window: pd.DataFrame, js_threshold: float, l2_threshold: float) -> int | None:
    if window.empty:
        return None
    for _, row in window.sort_values("t").iterrows():
        js = row_metric(row, "js_divergence") or 0.0
        l2 = row_metric(row, "centered_logit_normalized_l2") or 0.0
        top1_same = bool(row.get("top1_same", True))
        if (not top1_same) or js >= js_threshold or l2 >= l2_threshold:
            return int(row["t"])
    return None


def event_kind(
    branch_t: int | None,
    warning_t: int | None,
    branch_top1_flip: bool | None,
    branch_margin: float | None,
    branch_bf: float | None,
) -> str:
    if branch_t is None:
        return "no_visible_branch"
    if branch_t == 0:
        return "immediate_visible_branch"
    if warning_t is not None and warning_t < branch_t:
        return "silent_logit_divergence"
    if branch_margin is not None and branch_margin >= 2.0 and branch_bf is not None and branch_bf <= 3.0:
        return "high_confidence_basin_switch"
    if branch_top1_flip:
        return "branch_top1_flip"
    if branch_margin is not None and branch_margin <= 0.75:
        return "low_margin_branch"
    return "delayed_visible_branch"


def analyze_run(run_dir: Path, js_threshold: float | None, l2_threshold: float | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = load_summary(run_dir)
    logit_path = run_dir / "logit_probes.jsonl"
    if not logit_path.exists():
        raise FileNotFoundError(f"{run_dir} missing logit_probes.jsonl")
    logits = pd.DataFrame(read_jsonl(logit_path))
    if logits.empty:
        return pd.DataFrame(), pd.DataFrame()

    js_thresh = js_threshold if js_threshold is not None else threshold_from_controls(logits, "js_divergence", 1e-3)
    l2_thresh = (
        l2_threshold
        if l2_threshold is not None
        else threshold_from_controls(logits, "centered_logit_normalized_l2", 0.02)
    )

    events: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    grouped_logits = {
        key: group.sort_values("t")
        for key, group in logits[logits["anchor"] == "prompt_a_generation"].groupby(KEY_COLS, sort=False)
    }

    for _, srow in summary.iterrows():
        key = tuple(srow.get(col) for col in KEY_COLS)
        group = grouped_logits.get(key)
        if group is None or group.empty:
            continue

        branch_t = clean_int(srow.get("first_diff_token"))
        is_control = srow.get("category") == "micro_control_identical"
        common_window = group if branch_t is None else group[group["t"] <= branch_t]
        pre_visible = group if branch_t is None else group[group["t"] < branch_t]
        warning_t = first_warning_t(pre_visible, js_thresh, l2_thresh)

        branch_row = None
        if branch_t is not None:
            match = group[group["t"] == branch_t]
            if not match.empty:
                branch_row = match.iloc[0]

        branch_top1_flip = None
        branch_js = None
        branch_l2 = None
        branch_margin = None
        branch_entropy = None
        branch_bf = None
        if branch_row is not None:
            branch_top1_flip = not bool(branch_row.get("top1_same", True))
            branch_js = row_metric(branch_row, "js_divergence")
            branch_l2 = row_metric(branch_row, "centered_logit_normalized_l2")
            branch_margin = min_margin(branch_row)
            branch_entropy = max_entropy(branch_row)
            branch_bf = max_branching_factor(branch_row)

        semantic = srow.get("semantic_cosine_distance")
        token_edit_norm = srow.get("token_edit_distance_norm")
        persistent = False
        if semantic is not None and not pd.isna(semantic):
            persistent = persistent or float(semantic) >= 0.05
        if token_edit_norm is not None and not pd.isna(token_edit_norm):
            persistent = persistent or float(token_edit_norm) >= 0.10

        max_pre_js = float(pre_visible["js_divergence"].max()) if not pre_visible.empty else None
        max_pre_l2 = (
            float(pre_visible["centered_logit_normalized_l2"].max())
            if not pre_visible.empty and "centered_logit_normalized_l2" in pre_visible
            else None
        )
        min_pre_margin = None
        max_pre_bf = None
        min_pre_bf = None
        if not pre_visible.empty:
            margins = [min_margin(row) for _, row in pre_visible.iterrows()]
            margins = [m for m in margins if m is not None]
            min_pre_margin = min(margins) if margins else None
            bfs = [max_branching_factor(row) for _, row in pre_visible.iterrows()]
            bfs = [bf for bf in bfs if bf is not None]
            max_pre_bf = max(bfs) if bfs else None
            min_pre_bf = min(bfs) if bfs else None

        events.append(
            {
                "run_label": run_label(run_dir),
                "run_dir": str(run_dir),
                **{col: srow.get(col) for col in KEY_COLS},
                "is_control": is_control,
                "prompt_token_edit_distance": srow.get("prompt_token_edit_distance"),
                "prompt_token_delta_kind": srow.get("prompt_token_delta_kind"),
                "prompt_token_lcp": srow.get("prompt_token_lcp"),
                "branch_t": branch_t,
                "warning_t": warning_t,
                "silent_logit_lead": None
                if branch_t is None or warning_t is None or warning_t >= branch_t
                else branch_t - warning_t,
                "event_kind": event_kind(branch_t, warning_t, branch_top1_flip, branch_margin, branch_bf),
                "branch_top1_flip": branch_top1_flip,
                "branch_js": branch_js,
                "branch_centered_l2": branch_l2,
                "branch_min_margin_logit": branch_margin,
                "branch_max_entropy": branch_entropy,
                "branch_max_effective_branching_factor": branch_bf,
                "max_pre_branch_js": max_pre_js,
                "max_pre_branch_centered_l2": max_pre_l2,
                "min_pre_branch_margin_logit": min_pre_margin,
                "max_pre_branch_effective_branching_factor": max_pre_bf,
                "min_pre_branch_effective_branching_factor": min_pre_bf,
                "persistent_branch": persistent,
                "semantic_cosine_distance": semantic,
                "token_edit_distance_norm": token_edit_norm,
                "common_prefix_tokens": srow.get("common_prefix_tokens"),
                "js_warning_threshold": js_thresh,
                "l2_warning_threshold": l2_thresh,
            }
        )

        for _, lrow in common_window.iterrows():
            t = int(lrow["t"])
            if branch_t is not None and t > branch_t:
                continue
            pred = {
                "run_label": run_label(run_dir),
                "run_dir": str(run_dir),
                **{col: srow.get(col) for col in KEY_COLS},
                "is_control": is_control,
                "t": t,
                "branch_t": branch_t,
                "tokens_until_branch": None if branch_t is None else branch_t - t,
                "at_branch": branch_t is not None and branch_t == t,
                "js_divergence": lrow.get("js_divergence"),
                "centered_logit_normalized_l2": lrow.get("centered_logit_normalized_l2"),
                "top1_flip": not bool(lrow.get("top1_same", True)),
                "min_margin_logit": min_margin(lrow),
                "max_entropy": max_entropy(lrow),
                "max_effective_branching_factor": max_branching_factor(lrow),
                "min_effective_branching_factor": min_branching_factor(lrow),
                "persistent_branch": persistent,
            }
            for horizon in HORIZONS:
                pred[f"branch_within_{horizon}"] = (
                    branch_t is not None and 0 <= branch_t - t <= horizon
                )
                pred[f"pre_branch_within_{horizon}"] = (
                    branch_t is not None and 1 <= branch_t - t <= horizon
                )
            prediction_rows.append(pred)

    return pd.DataFrame(events), pd.DataFrame(prediction_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/trajectory_events"))
    parser.add_argument("--js-threshold", type=float, default=None)
    parser.add_argument("--l2-threshold", type=float, default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    event_frames = []
    prediction_frames = []
    for run_dir in args.run_dirs:
        events, predictions = analyze_run(run_dir, args.js_threshold, args.l2_threshold)
        if not events.empty:
            event_frames.append(events)
        if not predictions.empty:
            prediction_frames.append(predictions)

    if not event_frames:
        raise SystemExit("No trajectory events found")

    events = pd.concat(event_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    events.to_csv(args.out_dir / "trajectory_events.csv", index=False)
    predictions.to_csv(args.out_dir / "branch_prediction_windows.csv", index=False)

    summary = (
        events.groupby(["model_name", "event_kind"], dropna=False)
        .agg(
            n=("pair_id", "count"),
            persistent_rate=("persistent_branch", "mean"),
            mean_branch_t=("branch_t", "mean"),
            mean_silent_logit_lead=("silent_logit_lead", "mean"),
            mean_branch_js=("branch_js", "mean"),
            mean_branch_margin=("branch_min_margin_logit", "mean"),
            mean_branch_bf=("branch_max_effective_branching_factor", "mean"),
            mean_semantic=("semantic_cosine_distance", "mean"),
        )
        .reset_index()
        .sort_values(["model_name", "n"], ascending=[True, False])
    )
    summary.to_csv(args.out_dir / "trajectory_event_summary.csv", index=False)

    print(summary.to_string(index=False))
    print(f"Wrote trajectory event artifacts to {args.out_dir}")


if __name__ == "__main__":
    main()
