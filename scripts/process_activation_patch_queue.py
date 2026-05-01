#!/usr/bin/env python3
"""Compatibility wrapper for the E07 activation-patching queue processor."""

from __future__ import annotations

import runpy
from pathlib import Path


TARGET = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "E07_mech_branch_patching"
    / "process_activation_patch_queue.py"
)

runpy.run_path(str(TARGET), run_name="__main__")
