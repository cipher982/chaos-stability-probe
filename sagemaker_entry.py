#!/usr/bin/env python3
"""SageMaker entrypoint for the LLM stability probe."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def install_gpt_oss_runtime_deps(extra_args: list[object]) -> None:
    if "gpt_oss_20b" not in {str(arg) for arg in extra_args}:
        return
    cmd = [sys.executable, "-m", "pip", "install", "--no-deps", "triton==3.4.0"]
    print("Installing gpt-oss runtime dependency:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def main() -> None:
    model_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    model_dir.mkdir(parents=True, exist_ok=True)

    raw_args = os.environ.get("CHAOS_RUN_ARGS", "[]")
    extra_args = json.loads(raw_args)
    if not isinstance(extra_args, list):
        raise TypeError("CHAOS_RUN_ARGS must be a JSON list")
    install_gpt_oss_runtime_deps(extra_args)

    out_root = model_dir / "runs"
    entrypoint = os.environ.get("CHAOS_ENTRYPOINT", "panel")
    if entrypoint == "panel":
        script = "scripts/run_panel.py"
    elif entrypoint == "silent_divergence":
        script = "scripts/run_silent_divergence_panel.py"
    else:
        raise ValueError(f"Unknown CHAOS_ENTRYPOINT: {entrypoint}")

    cmd = [sys.executable, script, "--out-root", str(out_root), *[str(arg) for arg in extra_args]]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
