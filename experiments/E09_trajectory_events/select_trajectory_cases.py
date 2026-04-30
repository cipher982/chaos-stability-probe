#!/usr/bin/env python3
"""Select representative trajectory cases for inspection and figures."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


KEY_COLS = ["model_name", "pair_id", "repeat"]
ARCHETYPE_ORDER = [
    "strict_pre_branch_warning",
    "low_margin_branch_cliff",
    "low_bf_basin_switch",
    "top1_flip_branch",
    "long_shared_prefix_branch",
    "high_confidence_basin_switch",
    "e10_cross_model_timing",
    "immediate_branch_control",
]


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).lower() in {"true", "1", "yes"}


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None or pd.isna(value):
        return default
    return float(value)


def load_optional_csv(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    return pd.read_csv(path)


def captured_branch_features(windows: pd.DataFrame) -> pd.DataFrame:
    if windows.empty:
        return pd.DataFrame(columns=KEY_COLS)
    rows = []
    for key, group in windows.groupby(KEY_COLS, dropna=False, sort=False):
        branch_rows = group[group["at_branch"].map(bool_value)] if "at_branch" in group else pd.DataFrame()
        pre1_rows = group[group["pre_branch_within_1"].map(bool_value)] if "pre_branch_within_1" in group else pd.DataFrame()
        pre5_rows = group[group["pre_branch_within_5"].map(bool_value)] if "pre_branch_within_5" in group else pd.DataFrame()
        row = dict(zip(KEY_COLS, key, strict=True))
        row.update(
            {
                "captured_branch": not branch_rows.empty,
                "captured_pre1": not pre1_rows.empty,
                "captured_pre5": not pre5_rows.empty,
                "at_branch_js": branch_rows["js_divergence"].iloc[0] if not branch_rows.empty else None,
                "at_branch_l2": branch_rows["centered_logit_normalized_l2"].iloc[0] if not branch_rows.empty else None,
                "at_branch_margin": branch_rows["min_margin_logit"].iloc[0] if not branch_rows.empty else None,
                "at_branch_bf": branch_rows["max_effective_branching_factor"].iloc[0]
                if not branch_rows.empty and "max_effective_branching_factor" in branch_rows
                else None,
                "pre1_js": pre1_rows["js_divergence"].iloc[-1] if not pre1_rows.empty else None,
                "pre1_l2": pre1_rows["centered_logit_normalized_l2"].iloc[-1] if not pre1_rows.empty else None,
                "pre1_margin": pre1_rows["min_margin_logit"].iloc[-1] if not pre1_rows.empty else None,
                "pre1_bf": pre1_rows["max_effective_branching_factor"].iloc[-1]
                if not pre1_rows.empty and "max_effective_branching_factor" in pre1_rows
                else None,
                "max_window_js": group["js_divergence"].max(),
                "max_window_l2": group["centered_logit_normalized_l2"].max(),
                "min_window_margin": group["min_margin_logit"].min(),
                "max_window_bf": group["max_effective_branching_factor"].max()
                if "max_effective_branching_factor" in group
                else None,
                "min_window_bf": group["min_effective_branching_factor"].min()
                if "min_effective_branching_factor" in group
                else None,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def add_e10_features(cases: pd.DataFrame, silent_summary: pd.DataFrame | None) -> pd.DataFrame:
    cases = cases.copy()
    cases["e10_available"] = False
    cases["e10_model_count"] = 0
    cases["e10_branch_t_min"] = None
    cases["e10_branch_t_max"] = None
    cases["e10_branch_t_span"] = None
    if silent_summary is None or silent_summary.empty or "pair_id" not in silent_summary:
        return cases
    grouped = silent_summary.groupby("pair_id", dropna=False)
    for idx, row in cases.iterrows():
        pair_id = row["pair_id"]
        if pair_id not in grouped.groups:
            continue
        group = grouped.get_group(pair_id)
        branch_ts = group["branch_t"].dropna() if "branch_t" in group else pd.Series(dtype=float)
        cases.at[idx, "e10_available"] = True
        cases.at[idx, "e10_model_count"] = int(group["model_name"].nunique()) if "model_name" in group else int(len(group))
        if not branch_ts.empty:
            cases.at[idx, "e10_branch_t_min"] = float(branch_ts.min())
            cases.at[idx, "e10_branch_t_max"] = float(branch_ts.max())
            cases.at[idx, "e10_branch_t_span"] = float(branch_ts.max() - branch_ts.min())
    return cases


def label_archetypes(row: pd.Series) -> list[str]:
    labels: list[str] = []
    captured = bool_value(row.get("captured_branch"))
    branch_t = safe_float(row.get("branch_t"), -1)
    margin = safe_float(row.get("branch_min_margin_logit"), 999)
    bf = safe_float(row.get("branch_max_effective_branching_factor"), 999)
    semantic = safe_float(row.get("semantic_cosine_distance"))
    if row.get("event_kind") == "silent_logit_divergence" and captured and bool_value(row.get("captured_pre1")):
        labels.append("strict_pre_branch_warning")
    if captured and margin <= 0.25:
        labels.append("low_margin_branch_cliff")
    if captured and bf <= 3.0 and margin >= 2.0 and semantic >= 0.04:
        labels.append("low_bf_basin_switch")
    if captured and bool_value(row.get("branch_top1_flip")):
        labels.append("top1_flip_branch")
    if captured and branch_t >= 16:
        labels.append("long_shared_prefix_branch")
    if captured and margin >= 2.0 and semantic >= 0.04:
        labels.append("high_confidence_basin_switch")
    if bool_value(row.get("e10_available")) and safe_float(row.get("e10_branch_t_span")) >= 8:
        labels.append("e10_cross_model_timing")
    if row.get("event_kind") == "immediate_visible_branch" and semantic >= 0.05:
        labels.append("immediate_branch_control")
    return labels


def score_case(row: pd.Series) -> float:
    score = 0.0
    score += 10.0 if bool_value(row.get("captured_branch")) else 0.0
    score += 4.0 if bool_value(row.get("captured_pre1")) else 0.0
    score += 3.0 if bool_value(row.get("persistent_branch")) else 0.0
    score += min(safe_float(row.get("semantic_cosine_distance")) * 40.0, 6.0)
    score += min(safe_float(row.get("silent_logit_lead")) / 8.0, 5.0)
    score += min(safe_float(row.get("branch_js")) * 8.0, 4.0)
    score += 2.0 if safe_float(row.get("branch_max_effective_branching_factor"), 999) <= 3.0 else 0.0
    score += 3.0 if bool_value(row.get("e10_available")) else 0.0
    score += min(safe_float(row.get("e10_branch_t_span")) / 6.0, 3.0)
    if bool_value(row.get("is_control")):
        score -= 20.0
    return score


def build_candidates(events: pd.DataFrame, windows: pd.DataFrame, silent_summary: pd.DataFrame | None) -> pd.DataFrame:
    cases = events.copy()
    if "is_control" in cases:
        cases = cases[~cases["is_control"].map(bool_value)]
    features = captured_branch_features(windows)
    cases = cases.merge(features, on=KEY_COLS, how="left")
    for col in ["captured_branch", "captured_pre1", "captured_pre5"]:
        if col in cases:
            cases[col] = cases[col].fillna(False).map(bool_value)
    cases = add_e10_features(cases, silent_summary)
    cases["archetypes"] = cases.apply(lambda row: ";".join(label_archetypes(row)), axis=1)
    cases["case_score"] = cases.apply(score_case, axis=1)
    return cases.sort_values("case_score", ascending=False)


def select_recommendations(candidates: pd.DataFrame, per_archetype: int) -> pd.DataFrame:
    selected_rows = []
    seen: set[tuple[str, str, Any]] = set()
    for archetype in ARCHETYPE_ORDER:
        subset = candidates[candidates["archetypes"].str.contains(archetype, regex=False, na=False)]
        archetype_count = 0
        for _, row in subset.iterrows():
            key = (row["model_name"], row["pair_id"], row.get("repeat"))
            if key in seen:
                continue
            out = row.copy()
            out["selected_for"] = archetype
            selected_rows.append(out)
            seen.add(key)
            archetype_count += 1
            if archetype_count >= per_archetype:
                break
    if not selected_rows:
        return pd.DataFrame()
    return pd.DataFrame(selected_rows).sort_values(["selected_for", "case_score"], ascending=[True, False])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--silent-summary", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/trajectory_case_selection"))
    parser.add_argument("--per-archetype", type=int, default=3)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(args.events)
    windows = pd.read_csv(args.windows)
    silent_summary = load_optional_csv(args.silent_summary)
    candidates = build_candidates(events, windows, silent_summary)
    recommendations = select_recommendations(candidates, args.per_archetype)

    candidates.to_csv(args.out_dir / "case_candidates.csv", index=False)
    recommendations.to_csv(args.out_dir / "recommended_cases.csv", index=False)
    counts = (
        recommendations.groupby("selected_for", dropna=False)
        .size()
        .rename("n")
        .reset_index()
        .sort_values("selected_for")
    )
    counts.to_csv(args.out_dir / "recommendation_summary.csv", index=False)
    print(counts.to_string(index=False))
    print(f"Wrote case selection artifacts to {args.out_dir}")


if __name__ == "__main__":
    main()
