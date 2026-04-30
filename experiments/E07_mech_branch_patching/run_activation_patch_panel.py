#!/usr/bin/env python3
"""Run residual activation-patching jobs model-by-model."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def select_model_names(models_path: Path, selectors: list[str]) -> list[str]:
    entries = load_json(models_path)
    if not selectors:
        return [str(entry["name"]) for entry in entries]
    wanted = set(selectors)
    return [
        str(entry["name"])
        for entry in entries
        if entry["name"] in wanted or entry["model_id"] in wanted
    ]


def pair_ids_from_targets(path: Path, model_name: str) -> list[str]:
    df = pd.read_csv(path)
    if df.empty:
        return []
    model_col = "model_selector" if "model_selector" in df.columns else "model_name"
    if model_col not in df.columns or "pair_id" not in df.columns:
        raise ValueError(f"{path} must contain pair_id and {model_col}")
    rows = df[df[model_col] == model_name]
    return list(dict.fromkeys(str(pair_id) for pair_id in rows["pair_id"].dropna()))


def pair_ids_from_targets_json(path: Path, model_name: str) -> list[str]:
    data = load_json(path)
    if isinstance(data, dict):
        raw = data.get(model_name, [])
    elif isinstance(data, list):
        raw = []
        for item in data:
            if not isinstance(item, dict) or item.get("model") != model_name:
                continue
            raw.extend(item.get("pair_ids", []))
    else:
        raise TypeError(f"{path} must be a JSON object or list")
    if not isinstance(raw, list):
        raise TypeError(f"{path} target list for {model_name} must be a list")
    return list(dict.fromkeys(str(pair_id) for pair_id in raw))


def read_pair_ids(args: argparse.Namespace, model_name: str) -> list[str]:
    pair_ids = list(args.pair_ids)
    if args.targets_csv is not None:
        pair_ids.extend(pair_ids_from_targets(args.targets_csv, model_name))
    if args.targets_json is not None:
        pair_ids.extend(pair_ids_from_targets_json(args.targets_json, model_name))
    return list(dict.fromkeys(pair_ids))


def summarize_patch_dir(model_out: Path, out_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for csv_path in sorted(model_out.glob("*.csv")):
        if csv_path.name.startswith("patch_summary"):
            continue
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            best: dict[str, Any] | None = None
            for row in reader:
                try:
                    rescue = float(row.get("rescue_fraction", "nan"))
                except ValueError:
                    continue
                if best is None or rescue > float(best.get("rescue_fraction", "-inf")):
                    best = row
        if best is not None:
            rows.append(
                {
                    "artifact": str(csv_path),
                    "model_name": best.get("model_name"),
                    "pair_id": best.get("pair_id"),
                    "best_layer": best.get("layer"),
                    "best_position_label": best.get("position_label"),
                    "best_rescue_fraction": best.get("rescue_fraction"),
                    "first_diff_token": best.get("first_diff_token"),
                }
            )
    if not rows:
        return
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path("configs/models.json"))
    parser.add_argument("--model", action="append", default=[], help="Model name or model_id. Repeatable.")
    parser.add_argument("--prompt-pairs", type=Path, default=Path("configs/prompt_pairs_mechinterp_seed.json"))
    parser.add_argument("--pair-id", action="append", dest="pair_ids", default=[])
    parser.add_argument("--targets-csv", type=Path)
    parser.add_argument("--targets-json", type=Path)
    parser.add_argument("--out-root", type=Path, default=Path("runs/mechinterp_patch_panel"))
    parser.add_argument("--timeout-s", type=int, default=3600)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--thinking-mode", choices=["default", "enabled", "disabled"], default="disabled")
    parser.add_argument("--positions", choices=["final", "changed-final", "aligned", "all"], default="aligned")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_root / "activation_patch_manifest.jsonl"
    if manifest_path.exists():
        manifest_path.unlink()

    model_names = select_model_names(args.models, args.model)
    for model_name in model_names:
        pair_ids = read_pair_ids(args, model_name)
        if not pair_ids:
            write_jsonl(
                manifest_path,
                {
                    "model_name": model_name,
                    "status": "skipped",
                    "reason": "no pair ids selected",
                    "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                },
            )
            continue

        model_out = args.out_root / model_name
        model_out.mkdir(parents=True, exist_ok=True)
        for pair_id in pair_ids:
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "activation_patch_branch.py"),
                "--model",
                model_name,
                "--pair-id",
                pair_id,
                "--prompt-pairs",
                str(args.prompt_pairs),
                "--out-dir",
                str(model_out),
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--thinking-mode",
                args.thinking_mode,
                "--positions",
                args.positions,
                "--device",
                args.device,
                "--dtype",
                args.dtype,
            ]
            started = time.time()
            row = {
                "model_name": model_name,
                "pair_id": pair_id,
                "out_dir": str(model_out),
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "timeout_s": args.timeout_s,
                "cmd": cmd,
            }
            try:
                result = subprocess.run(
                    cmd,
                    cwd=ROOT,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=args.timeout_s,
                    check=False,
                )
                row.update(
                    {
                        "status": "ok" if result.returncode == 0 else "failed",
                        "returncode": result.returncode,
                        "elapsed_s": time.time() - started,
                        "stdout_tail": result.stdout[-2000:],
                        "stderr_tail": result.stderr[-4000:],
                    }
                )
            except subprocess.TimeoutExpired as exc:
                row.update(
                    {
                        "status": "timeout",
                        "returncode": None,
                        "elapsed_s": time.time() - started,
                        "stdout_tail": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
                        "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
                    }
                )
            write_jsonl(manifest_path, row)
            print(json.dumps(row, indent=2), flush=True)
        summarize_patch_dir(model_out, model_out / "patch_summary.csv")


if __name__ == "__main__":
    main()
