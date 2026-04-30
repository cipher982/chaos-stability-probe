#!/usr/bin/env python3
"""Merge local E10 silent-divergence captures into the standard readout schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


METADATA_KEYS = [
    "model_name",
    "model_id",
    "backend",
    "requested_device",
    "resolved_device",
    "requested_dtype",
    "resolved_dtype",
    "torch_version",
    "transformers_version",
    "git_sha",
    "git_dirty",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_metadata(model_dir: Path) -> dict[str, Any]:
    metadata_path = model_dir / "run_metadata.json"
    if not metadata_path.exists():
        return {
            "runtime_metadata_status": "missing",
            **{f"runtime_{key}": None for key in METADATA_KEYS},
        }
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        "runtime_metadata_status": "present",
        **{f"runtime_{key}": metadata.get(key) for key in METADATA_KEYS},
    }


def model_dirs(capture_root: Path) -> list[Path]:
    dirs = []
    for path in sorted(capture_root.iterdir()):
        if path.is_dir() and list(path.glob("*_silent_divergence_summary.csv")):
            dirs.append(path)
    return dirs


def add_missing_metadata(df: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    for key, value in metadata.items():
        if key not in df.columns:
            df[key] = value
    return df


def read_model(model_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    summary_paths = sorted(model_dir.glob("*_silent_divergence_summary.csv"))
    layer_paths = sorted(model_dir.glob("*_silent_divergence_layers.csv"))
    if not summary_paths or not layer_paths:
        raise FileNotFoundError(f"{model_dir} is missing silent-divergence CSVs")
    summary = pd.read_csv(summary_paths[0])
    layers = pd.read_csv(layer_paths[0])
    metadata = read_metadata(model_dir)
    summary = add_missing_metadata(summary, metadata)
    layers = add_missing_metadata(layers, metadata)
    summary["source_capture_dir"] = str(model_dir)
    layers["source_capture_dir"] = str(model_dir)
    manifest_row = {
        "model_dir": str(model_dir),
        "status": "processed",
        "runtime_metadata_status": metadata["runtime_metadata_status"],
        "summary_path": str(summary_paths[0]),
        "layers_path": str(layer_paths[0]),
    }
    if "model_name" in summary.columns and not summary.empty:
        manifest_row["model_name"] = summary["model_name"].iloc[0]
    return summary, layers, manifest_row


def build_readout(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.groupby(["model_name", "pair_id"], dropna=False)
        .agg(
            branch_t=("branch_t", "first"),
            rows=("t", "count"),
            max_js=("js_divergence", "max"),
            max_final_hidden=("final_layer_cosine_distance", "max"),
            max_any_hidden=("max_layer_cosine_distance", "max"),
            runtime_metadata_status=("runtime_metadata_status", "first"),
            runtime_backend=("runtime_backend", "first"),
            runtime_model_id=("runtime_model_id", "first"),
            runtime_resolved_device=("runtime_resolved_device", "first"),
            runtime_resolved_dtype=("runtime_resolved_dtype", "first"),
            runtime_torch_version=("runtime_torch_version", "first"),
            runtime_transformers_version=("runtime_transformers_version", "first"),
            runtime_git_sha=("runtime_git_sha", "first"),
            runtime_git_dirty=("runtime_git_dirty", "first"),
        )
        .reset_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or Path("runs/rankings") / args.capture_root.name
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    layers = []
    manifest_rows = []
    for model_dir in model_dirs(args.capture_root):
        summary, layer, manifest_row = read_model(model_dir)
        summaries.append(summary)
        layers.append(layer)
        manifest_rows.append(manifest_row)

    if not summaries:
        raise SystemExit(f"No local silent-divergence captures found under {args.capture_root}")

    manifest_rows.extend(
        {"model_name": row.get("model_name"), "status": row.get("status"), "panel_out_dir": row.get("out_dir")}
        for row in read_jsonl(args.capture_root / "silent_divergence_manifest.jsonl")
    )
    summary = pd.concat(summaries, ignore_index=True)
    layer_rows = pd.concat(layers, ignore_index=True)
    readout = build_readout(summary)

    pd.DataFrame(manifest_rows).to_csv(out_dir / "local_manifest.csv", index=False)
    summary.to_csv(out_dir / "merged_silent_divergence_summary.csv", index=False)
    layer_rows.to_csv(out_dir / "merged_silent_divergence_layers.csv", index=False)
    readout.to_csv(out_dir / "silent_divergence_readout.csv", index=False)

    print(readout.to_string(index=False))
    print(f"Wrote local silent-divergence readout to {out_dir}")


if __name__ == "__main__":
    main()
