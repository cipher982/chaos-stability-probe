#!/usr/bin/env python3
"""Compatibility wrapper for the E07 activation-patching panel."""

from __future__ import annotations

import runpy
from pathlib import Path


TARGET = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "E07_mech_branch_patching"
    / "run_activation_patch_panel.py"
)


if __name__ == "__main__":
    runpy.run_path(str(TARGET), run_name="__main__")
