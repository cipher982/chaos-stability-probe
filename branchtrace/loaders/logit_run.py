"""Load branch/logit probe data for a pair from trajectory artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_trajectory_row(
    events_csv: Path, model_name: str, pair_id: str, repeat: int = 0
) -> dict | None:
    df = pd.read_csv(events_csv, low_memory=False)
    mask = (df["model_name"] == model_name) & (df["pair_id"] == pair_id)
    if "repeat" in df.columns:
        mask &= df["repeat"] == repeat
    hit = df[mask]
    if hit.empty:
        return None
    return hit.iloc[0].to_dict()


def load_logit_probe_row(
    logit_probes_path: Path, pair_id: str, repeat: int = 0
) -> dict | None:
    """Scan logit_probes.jsonl; return first row matching (pair_id, repeat)."""
    with logit_probes_path.open() as f:
        for line in f:
            row = json.loads(line)
            if row.get("pair_id") == pair_id and row.get("repeat", 0) == repeat:
                return row
    return None
