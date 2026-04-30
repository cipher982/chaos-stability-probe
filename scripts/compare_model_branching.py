#!/usr/bin/env python3
"""Compatibility wrapper for E09 model branch comparison."""

from __future__ import annotations

import runpy
from pathlib import Path


TARGET = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "E09_trajectory_events"
    / "compare_model_branching.py"
)

runpy.run_path(str(TARGET), run_name="__main__")
