#!/usr/bin/env python3
"""Compatibility wrapper for E09 trajectory case selection."""

from __future__ import annotations

import runpy
from pathlib import Path


TARGET = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "E09_trajectory_events"
    / "select_trajectory_cases.py"
)

runpy.run_path(str(TARGET), run_name="__main__")
