#!/usr/bin/env python3
"""Build E09 branch-prediction, case-selection, casebook, and comparison artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def command_output(cmd: list[str]) -> str:
    result = subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.PIPE, text=True)
    return result.stdout.strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_metadata(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def git_metadata() -> dict[str, object]:
    try:
        commit = command_output(["git", "rev-parse", "HEAD"])
        dirty = bool(command_output(["git", "status", "--porcelain"]))
    except subprocess.CalledProcessError:
        return {"available": False}
    return {"available": True, "commit": commit, "dirty": dirty}


def write_run_metadata(args: argparse.Namespace, out_root: Path, events: Path, prediction_windows: Path) -> None:
    inputs = {
        "trajectory_events": input_metadata(events),
        "branch_prediction_windows": input_metadata(prediction_windows),
    }
    if args.silent_summary:
        inputs["silent_summary"] = input_metadata(args.silent_summary)
    payload = {
        "schema_version": 1,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "artifact": "E09_trajectory_artifacts",
        "trajectory_dir": str(args.trajectory_dir),
        "out_root": str(out_root),
        "inputs": inputs,
        "args": {
            "bootstrap_samples": args.bootstrap_samples,
            "per_archetype": args.per_archetype,
            "case_limit": args.case_limit,
            "comparison_model": args.comparison_model,
        },
        "git": git_metadata(),
    }
    (out_root / "run_metadata.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=Path("runs/trajectory_artifacts"))
    parser.add_argument("--name", default="")
    parser.add_argument("--silent-summary", type=Path, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=100)
    parser.add_argument("--per-archetype", type=int, default=2)
    parser.add_argument("--case-limit", type=int, default=24)
    parser.add_argument(
        "--comparison-model",
        action="append",
        default=[],
        help="Model name to include in the cross-model branch comparison. Defaults to the Qwen ladder.",
    )
    args = parser.parse_args()

    name = args.name or args.trajectory_dir.name
    out_root = args.out_root / name
    prediction_windows = args.trajectory_dir / "branch_prediction_windows.csv"
    events = args.trajectory_dir / "trajectory_events.csv"
    if not prediction_windows.exists():
        raise SystemExit(f"Missing {prediction_windows}")
    if not events.exists():
        raise SystemExit(f"Missing {events}")

    branch_dir = out_root / "branch_prediction"
    selection_dir = out_root / "case_selection"
    casebook_dir = out_root / "casebook"
    figures_dir = casebook_dir / "figures"
    comparison_dir = out_root / "model_comparison"
    branch_dir.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)
    write_run_metadata(args, out_root, events, prediction_windows)

    run(
        [
            sys.executable,
            "scripts/analyze_branch_prediction.py",
            str(prediction_windows),
            "--out-dir",
            str(branch_dir),
            "--bootstrap-samples",
            str(args.bootstrap_samples),
        ]
    )

    select_cmd = [
        sys.executable,
        "scripts/select_trajectory_cases.py",
        "--events",
        str(events),
        "--windows",
        str(prediction_windows),
        "--out-dir",
        str(selection_dir),
        "--per-archetype",
        str(args.per_archetype),
    ]
    if args.silent_summary:
        select_cmd.extend(["--silent-summary", str(args.silent_summary)])
    run(select_cmd)

    render_cmd = [
        sys.executable,
        "scripts/render_trajectory_casebook.py",
        "--events",
        str(events),
        "--windows",
        str(prediction_windows),
        "--cases-from",
        str(selection_dir / "recommended_cases.csv"),
        "--out-dir",
        str(casebook_dir),
        "--figure-dir",
        str(figures_dir),
        "--limit",
        str(args.case_limit),
    ]
    if args.silent_summary:
        render_cmd.extend(["--silent-summary", str(args.silent_summary)])
    run(render_cmd)

    comparison_cmd = [
        sys.executable,
        "scripts/compare_model_branching.py",
        "--events",
        str(events),
        "--out-dir",
        str(comparison_dir),
    ]
    for model in args.comparison_model:
        comparison_cmd.extend(["--model", model])
    run(comparison_cmd)
    print(f"Wrote E09 artifact bundle to {out_root}")


if __name__ == "__main__":
    main()
