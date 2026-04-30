#!/usr/bin/env python3
"""Run the stability probe one model at a time with timeout isolation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path("configs/models.json"))
    parser.add_argument("--prompt-pairs", type=Path, default=Path("configs/prompt_pairs.json"))
    parser.add_argument("--model", action="append", default=[], help="Model name or model_id. Repeatable.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out-root", type=Path, default=Path("runs/panel"))
    parser.add_argument("--timeout-s", type=int, default=1800)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--limit-pairs", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--different-seeds-within-pair", action="store_true")
    parser.add_argument("--skip-token-identical-non-controls", action="store_true")
    parser.add_argument("--skip-hidden", action="store_true")
    parser.add_argument("--logit-probe", action="store_true")
    parser.add_argument("--logit-top-k", type=int, default=20)
    parser.add_argument("--logit-max-steps", type=int, default=128)
    parser.add_argument("--thinking-mode", choices=["default", "enabled", "disabled"], default="default")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    args = parser.parse_args()

    model_entries = load_json(args.models)
    if args.model:
        wanted = set(args.model)
        model_entries = [m for m in model_entries if m["name"] in wanted or m["model_id"] in wanted]
    elif args.smoke:
        model_entries = [m for m in model_entries if m.get("enabled_for_smoke")]

    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_root / "panel_manifest.jsonl"
    if manifest_path.exists():
        manifest_path.unlink()

    for entry in model_entries:
        model_out = args.out_root / entry["name"]
        cmd = [
            sys.executable,
            "scripts/run_stability_probe.py",
            "--model",
            entry["name"],
            "--prompt-pairs",
            str(args.prompt_pairs),
            "--out-dir",
            str(model_out),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--repeats",
            str(args.repeats),
            "--device",
            args.device,
            "--dtype",
            args.dtype,
            "--thinking-mode",
            args.thinking_mode,
        ]
        if args.limit_pairs:
            cmd.extend(["--limit-pairs", str(args.limit_pairs)])
        if args.sample:
            cmd.append("--sample")
            cmd.extend(["--temperature", str(args.temperature), "--top-p", str(args.top_p)])
        if args.different_seeds_within_pair:
            cmd.append("--different-seeds-within-pair")
        if args.skip_token_identical_non_controls:
            cmd.append("--skip-token-identical-non-controls")
        if args.skip_hidden:
            cmd.append("--skip-hidden")
        if args.logit_probe:
            cmd.extend(
                [
                    "--logit-probe",
                    "--logit-top-k",
                    str(args.logit_top_k),
                    "--logit-max-steps",
                    str(args.logit_max_steps),
                ]
            )

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
                cwd=Path.cwd(),
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
        print(f"{entry['name']}: {row['status']} ({row['elapsed_s']:.1f}s)")


if __name__ == "__main__":
    main()
