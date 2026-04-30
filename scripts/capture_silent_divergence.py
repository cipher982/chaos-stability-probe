#!/usr/bin/env python3
"""Compatibility wrapper for the E10 silent-divergence capture."""

from __future__ import annotations

import runpy
from pathlib import Path


TARGET = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "E10_silent_divergence"
    / "capture_silent_divergence.py"
)

runpy.run_path(str(TARGET), run_name="__main__")
