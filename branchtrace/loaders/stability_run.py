"""Load generation + metadata side-by-side from a stability-probe run dir."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


def iter_generations(run_dir: Path) -> Iterator[dict]:
    path = run_dir / "generations.jsonl"
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_metadata(run_dir: Path) -> dict:
    return json.loads((run_dir / "metadata.json").read_text())


def find_pair(run_dir: Path, pair_id: str, repeat: int = 0) -> tuple[dict, dict]:
    """Return (a, b) generation rows for the given pair_id."""
    a = b = None
    for row in iter_generations(run_dir):
        if row.get("pair_id") != pair_id or row.get("repeat") != repeat:
            continue
        if row.get("side") == "a":
            a = row
        elif row.get("side") == "b":
            b = row
        if a and b:
            break
    if a is None or b is None:
        raise ValueError(f"pair {pair_id!r} (repeat={repeat}) not found in {run_dir}")
    return a, b


def load_prompt_tokens(run_dir: Path, pair_id: str) -> dict | None:
    """Return prompt_tokens.jsonl row for pair_id, containing token id lists."""
    path = run_dir / "prompt_tokens.jsonl"
    if not path.exists():
        return None
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            if row.get("pair_id") == pair_id:
                return row
    return None
