#!/usr/bin/env python3
"""Render the Marp deck to per-slide PNGs and HTML.

Pure rendering. No visual QA happens here — that's the qa-slides skill's job,
which spawns a vision subagent to look at the output. Pixel heuristics don't
work; see .agents/skills/qa-slides/SKILL.md for the full protocol.

Usage:
  uv run --with Pillow python talk/qa_slides.py
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "slides.md"
THEME = HERE / "theme.css"
IMG_DIR = HERE / "slide_images"


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=HERE, check=True)


def main() -> None:
    if IMG_DIR.exists():
        shutil.rmtree(IMG_DIR)
    IMG_DIR.mkdir(parents=True)

    run([
        "npx", "@marp-team/marp-cli@latest",
        "--theme", str(THEME),
        "--allow-local-files",
        "--images", "png",
        "--image-scale", "2",
        str(SRC),
        "-o", str(IMG_DIR / "slide.png"),
    ])
    run([
        "npx", "@marp-team/marp-cli@latest",
        "--theme", str(THEME),
        "--allow-local-files",
        str(SRC),
        "-o", str(HERE / "slides.html"),
    ])
    pngs = sorted(IMG_DIR.glob("slide.*.png"))
    print(f"Rendered {len(pngs)} slides to {IMG_DIR}")
    print(f"Also wrote: {HERE / 'slides.html'}")
    print()
    print("Next step: run the qa-slides skill or spawn a vision subagent on")
    print(f"the PNGs in {IMG_DIR}. See .agents/skills/qa-slides/SKILL.md.")


if __name__ == "__main__":
    main()
