#!/usr/bin/env python3
"""Verify talk/slides.md slide markers match their position and the PNG filenames.

Each slide in slides.md should have a marker of the form:
    <!-- SLIDE N / slide_images/slide.0NN.png / "Title" -->

This script:
  - parses slides.md, splits on `---` slide separators (skipping frontmatter)
  - checks that slide index in marker matches positional index
  - checks that referenced PNG exists under talk/slide_images/
  - exits non-zero on any mismatch
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLIDES_MD = ROOT / "talk" / "slides.md"
IMAGES_DIR = ROOT / "talk" / "slide_images"

MARKER_RE = re.compile(
    r'<!--\s*SLIDE\s+(\d+)\s*/\s*slide_images/slide\.(\d+)\.png\s*/\s*"([^"]*)"\s*-->'
)


def split_slides(text: str) -> list[str]:
    # Strip YAML frontmatter delimited by leading `---\n...\n---\n`.
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end == -1:
            raise SystemExit("slides.md frontmatter is unterminated")
        text = text[end + len("\n---\n"):]
    # Split on slide separators (a line that is exactly `---`).
    parts = re.split(r'(?m)^---\s*$', text)
    return [p for p in parts if p.strip()]


def main() -> int:
    text = SLIDES_MD.read_text()
    slides = split_slides(text)
    errors: list[str] = []

    for i, body in enumerate(slides, start=1):
        m = MARKER_RE.search(body)
        if not m:
            errors.append(f"slide {i}: missing SLIDE marker")
            continue
        marker_n, marker_png, title = m.group(1), m.group(2), m.group(3)
        if int(marker_n) != i:
            errors.append(
                f"slide {i}: marker says SLIDE {marker_n} (title: {title!r})"
            )
        if int(marker_png) != i:
            errors.append(
                f"slide {i}: marker references slide.{marker_png}.png"
            )
        png_path = IMAGES_DIR / f"slide.{i:03d}.png"
        if not png_path.exists():
            errors.append(f"slide {i}: {png_path.relative_to(ROOT)} not found")

    expected_pngs = sorted(IMAGES_DIR.glob("slide.*.png"))
    if len(expected_pngs) != len(slides):
        errors.append(
            f"slide count mismatch: {len(slides)} slides in slides.md vs "
            f"{len(expected_pngs)} PNGs in slide_images/"
        )

    if errors:
        print("slide numbering check FAILED:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print(f"OK: {len(slides)} slides, all markers match positions and PNGs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
