"""Render a Branch Card JSON dict to self-contained HTML."""

from __future__ import annotations

import base64
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _patch_rows(card: dict) -> list[dict]:
    pe = card["patch_evidence"]
    items = [
        ("prompt_lcp (edit boundary)", pe.get("prompt_lcp_rescue_fraction"), "best causal edit-boundary handle"),
        ("generated_prefix", pe.get("generated_prefix_rescue_fraction"), "shared generated tokens before branch"),
        ("aligned_prompt_control (max)", pe.get("aligned_prompt_control_max"), "strongest non-LCP prompt position"),
        ("final_context", pe.get("final_context_rescue_fraction"), "last position before branch (broad overwrite)"),
    ]
    out = []
    for label, v, note in items:
        if v is None:
            out.append({"label": label, "val": "—", "band": "weak", "bar_px": 0, "note": note})
            continue
        v = float(v)
        band = "full" if v >= 1.0 else ("strong" if v >= 0.85 else ("mid" if v >= 0.5 else "weak"))
        out.append(
            {
                "label": label,
                "val": f"{v:.3f}",
                "band": band,
                "bar_px": int(min(v, 1.3) * 200),
                "note": note,
            }
        )
    return out


def _heatmap_data_uri(card: dict, repo_root: Path) -> str | None:
    path = card["patch_evidence"].get("heatmap_path")
    if not path:
        return None
    p = repo_root / path
    if not p.exists():
        return None
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def render_card_html(card: dict, repo_root: Path) -> str:
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "j2"]),
    )
    tpl = env.get_template("card.html.j2")
    return tpl.render(
        card=card,
        patch_rows=_patch_rows(card),
        heatmap_data_uri=_heatmap_data_uri(card, repo_root),
    )
