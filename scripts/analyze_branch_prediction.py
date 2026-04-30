#!/usr/bin/env python3
"""Compatibility wrapper for the E09 branch-prediction analysis."""

from __future__ import annotations

import runpy
from pathlib import Path


TARGET = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "E09_trajectory_events"
    / "analyze_branch_prediction.py"
)

runpy.run_path(str(TARGET), run_name="__main__")
