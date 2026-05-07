"""Load patch rescue detail for a (model, pair_id) from comparison + aligned."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

_POS_PAT = re.compile(r"aligned_prompt_pos_(\d+)_to_(\d+)_best_rescue_fraction$")


def load_case_summary(
    summary_csv: Path, model_name: str, pair_id: str, wave: str | None = None
) -> dict | None:
    df = pd.read_csv(summary_csv, low_memory=False)
    mask = (df["model_name"] == model_name) & (df["pair_id"] == pair_id)
    if wave:
        mask &= df["wave"] == wave
    hit = df[mask]
    if hit.empty:
        return None
    row = hit.iloc[0]
    return {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}


def aligned_prompt_positions(row: dict) -> list[tuple[int, float]]:
    """Return sorted (position, rescue_fraction) pairs from a case row."""
    out = []
    for k, v in row.items():
        m = _POS_PAT.match(k)
        if not m or v is None:
            continue
        out.append((int(m.group(1)), float(v)))
    out.sort()
    return out


def aligned_prompt_best(row: dict) -> tuple[int, float] | None:
    poses = aligned_prompt_positions(row)
    if not poses:
        return None
    return max(poses, key=lambda pv: pv[1])


def aligned_prompt_stats(row: dict) -> dict:
    poses = aligned_prompt_positions(row)
    if not poses:
        return {"n": 0, "max": None, "full_count": 0, "best_position": None}
    best_pos, best_val = max(poses, key=lambda pv: pv[1])
    full_count = sum(1 for _, v in poses if v >= 1.0)
    return {
        "n": len(poses),
        "max": best_val,
        "full_count": full_count,
        "best_position": best_pos,
    }


def load_aligned_json(aligned_dir: Path, model_name: str, pair_id: str) -> dict | None:
    p = aligned_dir / f"{model_name}__{pair_id}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def infer_regime(row: dict, event_kind: str) -> str:
    """Apply the 3-regime taxonomy.

    Rules (spec §3 rev 2):
      - EDIT   (prompt_lcp_full == True)          -> "edit_boundary"
      - OTHER & silent                            -> "trajectory_migration"
      - OTHER & immediate (branch_t == 0)         -> "prompt_accumulation"
    """
    plcp_full = bool(row.get("prompt_lcp_full"))
    if plcp_full:
        return "edit_boundary"
    if event_kind == "immediate_visible_branch":
        return "prompt_accumulation"
    return "trajectory_migration"
