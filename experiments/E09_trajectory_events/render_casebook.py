#!/usr/bin/env python3
"""Render a static HTML casebook for trajectory branch events."""

from __future__ import annotations

import argparse
import difflib
import html
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


KEY_COLS = ["model_name", "pair_id", "repeat"]
EVENT_SORT_COLS = [
    "persistent_branch",
    "silent_logit_lead",
    "semantic_cosine_distance",
    "branch_js",
    "max_pre_branch_js",
]
TIMELINE_COLS = [
    "t",
    "tokens_until_branch",
    "at_branch",
    "pre_branch_within_1",
    "pre_branch_within_5",
    "js_divergence",
    "centered_logit_normalized_l2",
    "min_margin_logit",
    "max_entropy",
    "top1_flip",
]


def load_prompt_pairs(paths: list[Path]) -> dict[str, dict[str, Any]]:
    pairs: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        for row in json.loads(path.read_text(encoding="utf-8")):
            if not isinstance(row, dict) or "id" not in row:
                continue
            pairs[str(row["id"])] = row
    return pairs


def default_prompt_pair_paths() -> list[Path]:
    configs = Path("configs")
    paths = list(configs.glob("prompt_pairs*.json"))
    paths.extend(path for path in (configs / "prompt_pairs_token_certified").glob("*.json") if path.name != "manifest.json")
    return sorted(path for path in paths if path.exists())


def clean_token(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    if num.is_integer() and abs(num) < 10_000:
        return str(int(num))
    return f"{num:.{digits}g}"


def slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-")


def case_id_for_event(event: pd.Series) -> str:
    return slug(f"{event.get('model_name')}-{event.get('pair_id')}-{event.get('repeat')}")


def inline_diff(a: str, b: str) -> tuple[str, str]:
    matcher = difflib.SequenceMatcher(None, a, b)
    left: list[str] = []
    right: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        a_part = html.escape(a[i1:i2])
        b_part = html.escape(b[j1:j2])
        if tag == "equal":
            left.append(a_part)
            right.append(b_part)
        elif tag == "delete":
            left.append(f'<mark class="del">{a_part}</mark>')
        elif tag == "insert":
            right.append(f'<mark class="add">{b_part}</mark>')
        else:
            left.append(f'<mark class="chg">{a_part}</mark>')
            right.append(f'<mark class="chg">{b_part}</mark>')
    return "".join(left), "".join(right)


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).lower() in {"true", "1", "yes"}


def select_events(events: pd.DataFrame, windows: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    selected = events.copy()
    if args.model:
        selected = selected[selected["model_name"].isin(args.model)]
    if args.pair_id:
        selected = selected[selected["pair_id"].isin(args.pair_id)]
    if not args.include_controls and "is_control" in selected:
        selected = selected[~selected["is_control"].map(bool_value)]

    for col in EVENT_SORT_COLS:
        if col not in selected:
            selected[col] = 0.0
    captured = pd.DataFrame()
    if not windows.empty and all(col in windows.columns for col in KEY_COLS):
        capture_cols = KEY_COLS + ["at_branch"]
        captured = (
            windows[capture_cols]
            .assign(_captured_branch=windows["at_branch"].map(bool_value).astype(int))
            .groupby(KEY_COLS, dropna=False)["_captured_branch"]
            .max()
            .reset_index()
        )
    if not captured.empty:
        selected = selected.merge(captured, on=KEY_COLS, how="left")
        selected["_captured_branch"] = selected["_captured_branch"].fillna(0).astype(int)
    else:
        selected["_captured_branch"] = 0
    selected["_persistent_sort"] = selected["persistent_branch"].map(bool_value).astype(int)
    selected["_silent_sort"] = selected["event_kind"].eq("silent_logit_divergence").astype(int)
    selected = selected.sort_values(
        [
            "_captured_branch",
            "_persistent_sort",
            "_silent_sort",
            "silent_logit_lead",
            "semantic_cosine_distance",
            "branch_js",
            "max_pre_branch_js",
        ],
        ascending=[False, False, False, False, False, False, False],
        na_position="last",
    )
    return selected.head(args.limit).drop(
        columns=["_captured_branch", "_persistent_sort", "_silent_sort"], errors="ignore"
    )


def metric_cards(event: pd.Series) -> str:
    fields = [
        ("kind", "event_kind"),
        ("branch", "branch_t"),
        ("warning", "warning_t"),
        ("lead", "silent_logit_lead"),
        ("semantic", "semantic_cosine_distance"),
        ("branch JS", "branch_js"),
        ("pre JS", "max_pre_branch_js"),
        ("margin", "branch_min_margin_logit"),
    ]
    cards = []
    for label, field in fields:
        cards.append(
            '<div class="metric">'
            f"<span>{html.escape(label)}</span>"
            f"<strong>{fmt(event.get(field))}</strong>"
            "</div>"
        )
    return '<div class="metrics">' + "".join(cards) + "</div>"


def sparkline(rows: pd.DataFrame, cols: list[tuple[str, str]]) -> str:
    if rows.empty:
        return ""
    width = 720
    height = 132
    pad = 18
    x_values = rows["t"].astype(float).tolist()
    series = []
    values: list[float] = []
    for col, color in cols:
        if col not in rows:
            continue
        points = rows[["t", col]].dropna()
        if points.empty:
            continue
        data = [(float(t), float(v)) for t, v in points.itertuples(index=False)]
        series.append((col, color, data))
        values.extend(v for _, v in data)
    if not series or not values:
        return ""
    min_x, max_x = min(x_values), max(x_values)
    min_y, max_y = min(values), max(values)
    if max_x == min_x:
        max_x += 1.0
    if max_y == min_y:
        max_y += 1.0

    def px(x: float) -> float:
        return pad + (x - min_x) / (max_x - min_x) * (width - 2 * pad)

    def py(y: float) -> float:
        return height - pad - (y - min_y) / (max_y - min_y) * (height - 2 * pad)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="timeline sparkline">',
        f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" class="axis" />',
    ]
    for label, color, data in series:
        pts = " ".join(f"{px(x):.2f},{py(y):.2f}" for x, y in data)
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.4" />')
        last_x, last_y = data[-1]
        parts.append(
            f'<text x="{px(last_x) + 5:.2f}" y="{py(last_y):.2f}" fill="{color}" class="legend">'
            f"{html.escape(label)}</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


def timeline_table(rows: pd.DataFrame) -> str:
    if rows.empty:
        return '<p class="muted">No timeline rows found for this case.</p>'
    available = [col for col in TIMELINE_COLS if col in rows.columns]
    head = "".join(f"<th>{html.escape(col)}</th>" for col in available)
    body = []
    for _, row in rows.sort_values("t").iterrows():
        classes = []
        if bool_value(row.get("at_branch")):
            classes.append("at-branch")
        elif bool_value(row.get("pre_branch_within_1")):
            classes.append("pre-branch")
        cells = "".join(f"<td>{fmt(row.get(col))}</td>" for col in available)
        body.append(f'<tr class="{" ".join(classes)}">{cells}</tr>')
    return f'<table class="timeline"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def silent_divergence_table(summary: pd.DataFrame | None, pair_id: str) -> str:
    if summary is None or summary.empty or "pair_id" not in summary:
        return ""
    rows = summary[summary["pair_id"] == pair_id].copy()
    if rows.empty:
        return ""
    cols = [
        "model_name",
        "branch_t",
        "rows",
        "max_js",
        "max_final_hidden",
        "max_any_hidden",
        "runtime_git_sha",
        "runtime_resolved_device",
        "runtime_resolved_dtype",
    ]
    available = [col for col in cols if col in rows.columns]
    head = "".join(f"<th>{html.escape(col)}</th>" for col in available)
    body = []
    for _, row in rows.sort_values([c for c in ["model_name", "branch_t"] if c in rows]).iterrows():
        cells = "".join(f"<td>{fmt(row.get(col))}</td>" for col in available)
        body.append(f"<tr>{cells}</tr>")
    return (
        "<h4>E10 hidden/logit capture</h4>"
        f'<table class="compact"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'
    )


def render_case(
    event: pd.Series,
    windows: pd.DataFrame,
    prompt_pairs: dict[str, dict[str, Any]],
    silent_summary: pd.DataFrame | None,
) -> str:
    pair_id = clean_token(event.get("pair_id"))
    model_name = clean_token(event.get("model_name"))
    repeat = clean_token(event.get("repeat"))
    case_id = slug(f"{model_name}-{pair_id}-{repeat}")
    pair = prompt_pairs.get(pair_id, {})
    a_prompt = clean_token(pair.get("prompt_a", ""))
    b_prompt = clean_token(pair.get("prompt_b", ""))
    a_diff, b_diff = inline_diff(a_prompt, b_prompt)

    mask = (windows["model_name"] == event.get("model_name")) & (windows["pair_id"] == pair_id)
    if "repeat" in windows.columns and not pd.isna(event.get("repeat")):
        mask = mask & (windows["repeat"] == event.get("repeat"))
    case_windows = windows[mask].copy()

    title = f"{model_name} · {pair_id}"
    subtitle = f"{event.get('category', '')} · repeat {repeat}"
    return f"""
<section class="case" id="{case_id}">
  <h2>{html.escape(title)}</h2>
  <p class="subtitle">{html.escape(subtitle)}</p>
  {metric_cards(event)}
  <div class="prompt-grid">
    <div><h4>Prompt A</h4><pre>{a_diff}</pre></div>
    <div><h4>Prompt B</h4><pre>{b_diff}</pre></div>
  </div>
  <h4>Trajectory window</h4>
  {sparkline(case_windows, [("js_divergence", "#2068c8"), ("centered_logit_normalized_l2", "#c65414"), ("min_margin_logit", "#1f7a4d")])}
  {timeline_table(case_windows)}
  {silent_divergence_table(silent_summary, pair_id)}
</section>
"""


def render_index(events: pd.DataFrame) -> str:
    rows = []
    for _, event in events.iterrows():
        case_id = case_id_for_event(event)
        rows.append(
            "<tr>"
            f'<td><a href="#{case_id}">{html.escape(clean_token(event.get("model_name")))}</a></td>'
            f"<td>{html.escape(clean_token(event.get('pair_id')))}</td>"
            f"<td>{html.escape(clean_token(event.get('event_kind')))}</td>"
            f"<td>{fmt(event.get('branch_t'))}</td>"
            f"<td>{fmt(event.get('silent_logit_lead'))}</td>"
            f"<td>{fmt(event.get('semantic_cosine_distance'))}</td>"
            "</tr>"
        )
    return (
        '<table class="compact"><thead><tr>'
        "<th>model</th><th>pair</th><th>kind</th><th>branch</th><th>lead</th><th>semantic</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def css() -> str:
    return """
body { margin: 0; color: #202124; background: #f6f7f9; font: 15px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding: 32px 24px 64px; }
h1 { font-size: 28px; margin: 0 0 8px; }
h2 { margin: 0; font-size: 22px; }
h4 { margin: 18px 0 8px; font-size: 13px; text-transform: uppercase; letter-spacing: .04em; color: #565f6b; }
a { color: #175cbd; text-decoration: none; }
.muted, .subtitle { color: #667085; }
.case { background: #fff; border: 1px solid #d9dee7; border-radius: 8px; padding: 22px; margin-top: 22px; }
.metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(118px, 1fr)); gap: 10px; margin: 16px 0; }
.metric { border: 1px solid #e0e5ed; border-radius: 6px; padding: 9px 10px; background: #fbfcfd; }
.metric span { display: block; font-size: 11px; color: #667085; text-transform: uppercase; }
.metric strong { display: block; margin-top: 2px; font-size: 15px; }
.prompt-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; margin: 0; padding: 12px; background: #f8fafc; border: 1px solid #e0e5ed; border-radius: 6px; min-height: 84px; }
mark { border-radius: 3px; padding: 0 2px; }
mark.add { background: #dbf5e6; }
mark.del { background: #ffe1e1; }
mark.chg { background: #fff0b8; }
table { border-collapse: collapse; width: 100%; }
th, td { border-bottom: 1px solid #e6eaf0; padding: 7px 8px; text-align: left; vertical-align: top; }
th { font-size: 12px; color: #667085; background: #f8fafc; position: sticky; top: 0; }
.compact { font-size: 13px; }
.timeline { font-size: 12px; display: block; overflow-x: auto; }
.at-branch td { background: #fff0b8; }
.pre-branch td { background: #eaf3ff; }
svg { width: 100%; height: 132px; background: #fbfcfd; border: 1px solid #e0e5ed; border-radius: 6px; }
.axis { stroke: #d0d5dd; stroke-width: 1; }
.legend { font-size: 11px; dominant-baseline: middle; }
@media (max-width: 820px) { .prompt-grid { grid-template-columns: 1fr; } main { padding: 20px 12px 48px; } }
body.figure { background: #fff; }
body.figure main { max-width: 1040px; padding: 18px; }
body.figure .case { border: 0; border-radius: 0; margin-top: 0; padding: 0; }
body.figure h2 { font-size: 24px; }
body.figure .metrics { grid-template-columns: repeat(4, 1fr); }
"""


def html_document(title: str, body: str, body_class: str = "") -> str:
    class_attr = f' class="{html.escape(body_class)}"' if body_class else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>{css()}</style>
</head>
<body{class_attr}>
<main>
  {body}
</main>
</body>
</html>
"""


def case_manifest_row(event: pd.Series, filename: str) -> dict[str, Any]:
    return {
        "file": filename,
        "case_id": case_id_for_event(event),
        "model_name": event.get("model_name"),
        "pair_id": event.get("pair_id"),
        "category": event.get("category"),
        "repeat": event.get("repeat"),
        "event_kind": event.get("event_kind"),
        "branch_t": event.get("branch_t"),
        "warning_t": event.get("warning_t"),
        "silent_logit_lead": event.get("silent_logit_lead"),
        "semantic_cosine_distance": event.get("semantic_cosine_distance"),
        "branch_js": event.get("branch_js"),
    }


def render_casebook_document(events_path: Path, selected: pd.DataFrame, cases: str) -> str:
    body = f"""
  <h1>Trajectory Casebook</h1>
  <p class="muted">Generated from {html.escape(str(events_path))}. Rows are selected branch cases, not statistical summaries.</p>
  {render_index(selected)}
  {cases}
"""
    return html_document("Trajectory Casebook", body)


def write_figure_panels(
    figure_dir: Path,
    selected: pd.DataFrame,
    windows: pd.DataFrame,
    prompt_pairs: dict[str, dict[str, Any]],
    silent_summary: pd.DataFrame | None,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for _, event in selected.iterrows():
        case_id = case_id_for_event(event)
        filename = f"{case_id}.html"
        panel = render_case(event, windows, prompt_pairs, silent_summary)
        title = f"{event.get('model_name')} {event.get('pair_id')}"
        (figure_dir / filename).write_text(html_document(title, panel, "figure"), encoding="utf-8")
        manifest_rows.append(case_manifest_row(event, filename))
    pd.DataFrame(manifest_rows).to_csv(figure_dir / "manifest.csv", index=False)
    print(f"Wrote {len(manifest_rows)} figure panels to {figure_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--prompt-pairs", type=Path, action="append", default=[])
    parser.add_argument("--silent-summary", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/casebooks/trajectory_events"))
    parser.add_argument("--figure-dir", type=Path, default=None)
    parser.add_argument("--no-index", action="store_true")
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--pair-id", action="append", default=[])
    parser.add_argument("--include-controls", action="store_true")
    args = parser.parse_args()

    events = pd.read_csv(args.events)
    windows = pd.read_csv(args.windows)
    selected = select_events(events, windows, args)
    prompt_paths = args.prompt_pairs or default_prompt_pair_paths()
    prompt_pairs = load_prompt_pairs(prompt_paths)
    silent_summary = pd.read_csv(args.silent_summary) if args.silent_summary and args.silent_summary.exists() else None

    cases = "\n".join(render_case(event, windows, prompt_pairs, silent_summary) for _, event in selected.iterrows())
    if not args.no_index:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.out_dir / "index.html"
        out_path.write_text(render_casebook_document(args.events, selected, cases), encoding="utf-8")
        print(f"Wrote {out_path}")
    if args.figure_dir:
        write_figure_panels(args.figure_dir, selected, windows, prompt_pairs, silent_summary)
    if args.no_index and not args.figure_dir:
        raise SystemExit("--no-index requires --figure-dir")


if __name__ == "__main__":
    main()
