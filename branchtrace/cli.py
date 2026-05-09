"""branchtrace CLI — `build` + `render` commands."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import SCHEMA_VERSION
from .build_card import (
    build_hero_qwen2b_parenthesize_0434,
    build_silent_gemma_e2b_blank_line_wrap_0212,
)


def _cmd_build(args: argparse.Namespace) -> int:
    repo = Path(args.repo_root).resolve()
    if args.hero == "qwen2b_parenthesize_0434":
        card = build_hero_qwen2b_parenthesize_0434(repo)
    elif args.hero == "gemma_e2b_blank_line_wrap_0212":
        card = build_silent_gemma_e2b_blank_line_wrap_0212(repo)
    else:
        raise SystemExit(f"Unknown --hero: {args.hero!r}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(card.model_dump_json(indent=2))
    print(f"wrote {out} ({SCHEMA_VERSION})")
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    from .render_card import render_card_html  # lazy: jinja2 only required here.

    card_json = Path(args.card)
    repo = Path(args.repo_root).resolve()
    out = Path(args.out)
    html = render_card_html(json.loads(card_json.read_text()), repo_root=repo)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="branchtrace")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Build a Branch Card JSON from artifacts.")
    b.add_argument(
        "--hero",
        default="qwen2b_parenthesize_0434",
        help=(
            "Hero case shortcut. Supported: qwen2b_parenthesize_0434, "
            "gemma_e2b_blank_line_wrap_0212."
        ),
    )
    b.add_argument("--repo-root", default=".")
    b.add_argument("--out", required=True)
    b.set_defaults(fn=_cmd_build)

    r = sub.add_parser("render", help="Render a Branch Card HTML view.")
    r.add_argument("card")
    r.add_argument("--repo-root", default=".")
    r.add_argument("--out", required=True)
    r.set_defaults(fn=_cmd_render)

    ns = p.parse_args(argv)
    return ns.fn(ns)


if __name__ == "__main__":
    sys.exit(main())
