#!/usr/bin/env python3
"""Run common-prefix hidden/logit divergence captures model-by-model."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path("configs/models.json"))
    parser.add_argument("--prompt-pairs", type=Path, default=Path("configs/prompt_pairs_mechinterp_seed.json"))
    parser.add_argument("--model", action="append", default=[], help="Model name or model_id. Repeatable.")
    parser.add_argument("--pair-id", action="append", dest="pair_ids", default=[])
    parser.add_argument("--out-root", type=Path, default=Path("runs/silent_divergence_panel"))
    parser.add_argument("--timeout-s", type=int, default=3600)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--logit-max-steps", type=int, default=64)
    parser.add_argument("--limit-pairs", type=int, default=0)
    parser.add_argument("--thinking-mode", choices=["default", "enabled", "disabled"], default="disabled")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    args = parser.parse_args()

    model_entries = load_json(args.models)
    if args.model:
        wanted = set(args.model)
        model_entries = [m for m in model_entries if m["name"] in wanted or m["model_id"] in wanted]

    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_root / "silent_divergence_manifest.jsonl"
    if manifest_path.exists():
        manifest_path.unlink()

    for entry in model_entries:
        model_out = args.out_root / entry["name"]
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "capture_silent_divergence.py"),
            "--model",
            entry["name"],
            "--prompt-pairs",
            str(args.prompt_pairs),
            "--out-dir",
            str(model_out),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--logit-max-steps",
            str(args.logit_max_steps),
            "--device",
            args.device,
            "--dtype",
            args.dtype,
            "--thinking-mode",
            args.thinking_mode,
        ]
        if args.limit_pairs:
            cmd.extend(["--limit-pairs", str(args.limit_pairs)])
        for pair_id in args.pair_ids:
            cmd.extend(["--pair-id", pair_id])

        started = time.time()
        row = {
            "model_name": entry["name"],
            "model_id": entry["model_id"],
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
        print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
