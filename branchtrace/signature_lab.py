"""Build and render the interactive signature lab from raw artifacts.

The lab is intentionally data-first: raw E07/E09/E08 artifacts are read into a
compact JSON bundle, and the HTML renderer only visualizes that bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .loaders import stability_run

LAB_SCHEMA_VERSION = "signature-lab/0.1"

FORWARD_WAVES = {
    "activation_patch_v1",
    "activation_patch_v2",
    "activation_patch_v3",
    "activation_patch_v5_replication",
}

CASE_SPECS = [
    {
        "id": "qwen35_2b__token_cert_line_wrap_0573",
        "title": "Boundary rescue with visible runway",
        "short_label": "Edit-boundary example",
        "headline": (
            "Visible branch at t=9. Patching the last shared prompt token "
            "recovers the clean branch."
        ),
        "is_default": True,
        "model_name": "qwen35_2b",
        "pair_id": "token_cert_line_wrap_0573",
        "archetype": "prompt-LCP full rescue, but branch_t=9 so the shared prefix is visible",
        "why": (
            "The branch happens after nine generated tokens. Prompt-LCP patching fully rescues, "
            "so this is the non-token-zero boundary case to read first."
        ),
    },
    {
        "id": "gemma4_e2b_base__token_cert_blank_line_wrap_0212",
        "title": "Generated-prefix rescue after silent divergence",
        "short_label": "Silent-divergence example",
        "headline": (
            "The output stays identical for 45 tokens, but the logits drift. "
            "The strongest fix is inside the generated prefix."
        ),
        "is_default": False,
        "model_name": "gemma4_e2b_base",
        "pair_id": "token_cert_blank_line_wrap_0212",
        "archetype": "silent logit shift migrates into a generated-prefix handle",
        "why": (
            "The visible output stays shared for forty-five generated tokens. "
            "The best causal handle is generated-prefix position 44."
        ),
    },
    {
        "id": "qwen35_2b__token_cert_line_wrap_0378",
        "title": "Tokenization-shift immediate rescue",
        "short_label": "Immediate edge case",
        "headline": (
            "The first generated token changes at t=0. There is no shared "
            "generated runway to inspect."
        ),
        "is_default": False,
        "model_name": "qwen35_2b",
        "pair_id": "token_cert_line_wrap_0378",
        "archetype": "token-0 visible branch; last prompt-side state is effectively final context",
        "why": (
            "This is the edge case class: the first generated token changes immediately, "
            "so there is no generated runway to inspect."
        ),
    },
]

LOGIT_RUN_DIRS = {
    "qwen35_2b": Path(
        "runs/sagemaker_artifacts/chaos-logit-token-cert-qwen2b-thinkoff-20260430-001/runs/qwen35_2b"
    ),
    "gemma4_e2b_base": Path(
        "runs/sagemaker_artifacts/chaos-logit-token-cert-gemma-e2b-base-20260430-001/runs/gemma4_e2b_base"
    ),
}

SUMMARY_CSV = Path("runs/rankings/activation_patch_comparison/case_level_summary.csv")
TRAJECTORY_CSV = Path("runs/trajectory_events/logit_token_cert_v1/trajectory_events.csv")
SAE_DIR = Path("runs/mechinterp_sae")

_ALIGNED_PROMPT_RE = re.compile(r"aligned_prompt_pos_(\d+)_to_(\d+)$")
_GENERATED_PREFIX_RE = re.compile(r"aligned_generated_prefix_pos_(\d+)$")


def _json_ready(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        return _json_ready(value.item())
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    return value


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _artifact(path: Path, repo_root: Path) -> dict[str, Any] | None:
    full = path if path.is_absolute() else repo_root / path
    if not full.exists():
        return None
    return {
        "path": str(full.relative_to(repo_root)),
        "sha256": _sha256(full),
        "size_bytes": full.stat().st_size,
    }


class _TokenizerCache:
    def __init__(self) -> None:
        self._cache: dict[str, Any | None] = {}

    def decode_many(self, model_id: str | None, token_ids: list[int]) -> tuple[list[str], str]:
        if not token_ids:
            return [], "not_run"
        tok = self._get(model_id)
        if tok is None:
            return [f"#{tid}" for tid in token_ids], "not_run"
        out = []
        for tid in token_ids:
            try:
                out.append(tok.decode([int(tid)], clean_up_tokenization_spaces=False))
            except Exception:
                out.append(f"#{tid}")
        return out, "derived"

    def _get(self, model_id: str | None) -> Any | None:
        if not model_id:
            return None
        if model_id in self._cache:
            return self._cache[model_id]
        try:
            from transformers import AutoTokenizer

            self._cache[model_id] = AutoTokenizer.from_pretrained(
                model_id,
                local_files_only=True,
                trust_remote_code=True,
            )
        except Exception:
            self._cache[model_id] = None
        return self._cache[model_id]


def _position_class(label: str) -> str:
    if label == "prompt_lcp_token":
        return "prompt_lcp"
    if label == "final_context_token":
        return "final_context"
    if label == "generated_prefix_token" or label.startswith("aligned_generated_prefix_pos_"):
        return "generated_prefix"
    if label.startswith("aligned_prompt_pos_"):
        return "aligned_prompt_control"
    return "other"


def _position_index(label: str) -> int | None:
    if label == "prompt_lcp_token":
        return -2
    if label == "final_context_token":
        return 10_000
    m = _GENERATED_PREFIX_RE.match(label)
    if m:
        return int(m.group(1))
    m = _ALIGNED_PROMPT_RE.match(label)
    if m:
        return int(m.group(1))
    return None


def _position_short_label(label: str) -> str:
    if label == "prompt_lcp_token":
        return "prompt LCP"
    if label == "final_context_token":
        return "final ctx"
    if label == "generated_prefix_token":
        return "gen prefix"
    m = _GENERATED_PREFIX_RE.match(label)
    if m:
        return f"gen {m.group(1)}"
    m = _ALIGNED_PROMPT_RE.match(label)
    if m:
        a, b = m.groups()
        return f"prompt {a}" if a == b else f"prompt {a}->{b}"
    return label


def _visible_text_diff(a: str, b: str, radius: int = 70) -> dict[str, Any]:
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    j_a = len(a)
    j_b = len(b)
    while j_a > i and j_b > i and a[j_a - 1] == b[j_b - 1]:
        j_a -= 1
        j_b -= 1
    lo = max(0, i - radius)
    hi_a = min(len(a), j_a + radius)
    hi_b = min(len(b), j_b + radius)
    return {
        "source": "derived",
        "first_char_diff": i,
        "a_before": a[lo:i],
        "a_changed": a[i:j_a],
        "a_after": a[j_a:hi_a],
        "b_before": b[lo:i],
        "b_changed": b[i:j_b],
        "b_after": b[j_b:hi_b],
    }


def _case_summary(summary_df: pd.DataFrame, model_name: str, pair_id: str) -> dict[str, Any]:
    hit = summary_df[
        (summary_df["model_name"] == model_name)
        & (summary_df["pair_id"] == pair_id)
        & (summary_df["wave"].isin(FORWARD_WAVES))
    ]
    if hit.empty:
        raise ValueError(f"No forward patch summary row for {model_name}/{pair_id}")
    return _json_ready(hit.iloc[0].to_dict())


def _trajectory_row(events_df: pd.DataFrame, model_name: str, pair_id: str) -> dict[str, Any]:
    hit = events_df[
        (events_df["model_name"] == model_name)
        & (events_df["pair_id"] == pair_id)
        & (events_df.get("repeat", 0) == 0)
    ]
    if hit.empty:
        raise ValueError(f"No trajectory row for {model_name}/{pair_id}")
    return _json_ready(hit.iloc[0].to_dict())


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _logit_timeline(run_dir: Path, pair_id: str, branch_t: int) -> list[dict[str, Any]]:
    rows = []
    seen_t: set[int] = set()
    for row in _iter_jsonl(run_dir / "logit_probes.jsonl"):
        if row.get("pair_id") != pair_id or row.get("repeat", 0) != 0:
            continue
        if row.get("anchor") != "prompt_a_generation":
            continue
        t = int(row["t"])
        if t in seen_t:
            continue
        seen_t.add(t)
        a_top = row.get("topk_a") or []
        b_top = row.get("topk_b") or []
        rows.append(
            {
                "source": "observed",
                "t": t,
                "phase": "branch" if t == branch_t else ("pre_branch" if t < branch_t else "post_branch"),
                "top1_same": bool(row.get("top1_same")),
                "js": _json_ready(row.get("js_divergence")),
                "centered_l2": _json_ready(row.get("centered_logit_normalized_l2")),
                "min_margin_logit": _json_ready(row.get("branch_min_margin_logit")),
                "a_token": a_top[0].get("token") if a_top else None,
                "a_token_id": a_top[0].get("token_id") if a_top else None,
                "a_prob": a_top[0].get("prob") if a_top else None,
                "b_token": b_top[0].get("token") if b_top else None,
                "b_token_id": b_top[0].get("token_id") if b_top else None,
                "b_prob": b_top[0].get("prob") if b_top else None,
                "topk_a": a_top[:3],
                "topk_b": b_top[:3],
            }
        )
    rows.sort(key=lambda r: r["t"])
    return _json_ready(rows)


def _prompt_token_windows(
    prompt_token_ids_a: list[int],
    prompt_token_ids_b: list[int],
    prompt_lcp: int,
    model_id: str | None,
    tokenizer_cache: _TokenizerCache,
    radius: int = 10,
) -> dict[str, Any]:
    lo = max(0, prompt_lcp - radius)
    hi = min(max(len(prompt_token_ids_a), len(prompt_token_ids_b)), prompt_lcp + radius + 1)

    def one_side(ids: list[int], side: str) -> list[dict[str, Any]]:
        ids_window = ids[lo:hi]
        texts, source = tokenizer_cache.decode_many(model_id, ids_window)
        out = []
        for offset, (tid, text) in enumerate(zip(ids_window, texts, strict=False)):
            idx = lo + offset
            role = "common" if idx < prompt_lcp else ("prompt_lcp" if idx == prompt_lcp else "post_lcp")
            out.append(
                {
                    "source": source,
                    "side": side,
                    "index": idx,
                    "token_id": int(tid),
                    "text": text,
                    "role": role,
                }
            )
        return out

    return {
        "source": "derived",
        "prompt_lcp": prompt_lcp,
        "window_start": lo,
        "window_end": hi,
        "a": one_side(prompt_token_ids_a, "a"),
        "b": one_side(prompt_token_ids_b, "b"),
    }


def _patch_grid(patch_csv: Path) -> dict[str, Any]:
    df = pd.read_csv(patch_csv, low_memory=False)
    cells = []
    for row in df.itertuples(index=False):
        label = str(row.position_label)
        rescue = float(row.rescue_fraction)
        cells.append(
            {
                "source": "observed",
                "layer": int(row.layer),
                "position_label": label,
                "position_short_label": _position_short_label(label),
                "position_class": _position_class(label),
                "position_index": _position_index(label),
                "clean_pos": _json_ready(getattr(row, "clean_pos", None)),
                "corrupt_pos": _json_ready(getattr(row, "corrupt_pos", None)),
                "rescue_fraction": rescue,
                "metric_a_minus_b": _json_ready(getattr(row, "metric_a_minus_b", None)),
                "top1_token": _json_ready(getattr(row, "top1_token", None)),
                "top1_prob": _json_ready(getattr(row, "top1_prob", None)),
            }
        )

    positions = []
    for label, group in df.groupby("position_label", sort=False):
        best = group.loc[group["rescue_fraction"].idxmax()]
        positions.append(
            {
                "source": "derived",
                "label": str(label),
                "short_label": _position_short_label(str(label)),
                "class": _position_class(str(label)),
                "index": _position_index(str(label)),
                "best_layer": int(best["layer"]),
                "best_rescue_fraction": float(best["rescue_fraction"]),
                "best_top1_token": _json_ready(best.get("top1_token")),
            }
        )

    def position_key(pos: dict[str, Any]) -> tuple[int, int, str]:
        class_order = {
            "prompt_lcp": 0,
            "aligned_prompt_control": 1,
            "generated_prefix": 2,
            "final_context": 3,
            "other": 4,
        }
        return (class_order.get(pos["class"], 99), pos["index"] or 0, pos["label"])

    positions.sort(key=position_key)
    best_cell = max(cells, key=lambda c: c["rescue_fraction"])

    class_summaries = []
    for cls in ["prompt_lcp", "aligned_prompt_control", "generated_prefix", "final_context"]:
        sub = [p for p in positions if p["class"] == cls]
        if not sub:
            continue
        best = max(sub, key=lambda p: p["best_rescue_fraction"])
        class_summaries.append(
            {
                "source": "derived",
                "class": cls,
                "position_count": len(sub),
                "best_position_label": best["label"],
                "best_position_short_label": best["short_label"],
                "best_layer": best["best_layer"],
                "best_rescue_fraction": best["best_rescue_fraction"],
            }
        )

    return _json_ready(
        {
            "source": "observed",
            "layers": sorted(int(x) for x in df["layer"].unique()),
            "positions": positions,
            "cells": cells,
            "best_cell": best_cell,
            "class_summaries": class_summaries,
        }
    )


def _regime_from_summary(row: dict[str, Any]) -> str:
    if bool(row.get("prompt_lcp_full")):
        return "edit_boundary"
    if int(row.get("first_diff_token") or 0) == 0:
        return "prompt_accumulation"
    return "trajectory_migration"


def _build_case(
    spec: dict[str, str],
    repo_root: Path,
    summary_df: pd.DataFrame,
    events_df: pd.DataFrame,
    tokenizer_cache: _TokenizerCache,
) -> dict[str, Any]:
    model_name = spec["model_name"]
    pair_id = spec["pair_id"]
    summary = _case_summary(summary_df, model_name, pair_id)
    event = _trajectory_row(events_df, model_name, pair_id)

    run_dir = repo_root / LOGIT_RUN_DIRS[model_name]
    metadata = stability_run.load_metadata(run_dir)
    gen_a, gen_b = stability_run.find_pair(run_dir, pair_id, repeat=0)
    prompt_tokens = stability_run.load_prompt_tokens(run_dir, pair_id) or {}

    patch_csv = repo_root / str(summary["csv_path"])
    patch_json_path = patch_csv.with_suffix(".json")
    patch_json = json.loads(patch_json_path.read_text())

    model = metadata["models"][0]
    prompt_a = patch_json["pair"]["prompt_a"]
    prompt_b = patch_json["pair"]["prompt_b"]
    prompt_token_ids_a = prompt_tokens.get("prompt_token_ids_a") or patch_json["prompt_delta"].get("prompt_input_tokens_a") or []
    prompt_token_ids_b = prompt_tokens.get("prompt_token_ids_b") or patch_json["prompt_delta"].get("prompt_input_tokens_b") or []
    prompt_lcp = int(event.get("prompt_token_lcp") or patch_json["prompt_delta"].get("prompt_token_lcp") or 0)
    branch_t = int(event["branch_t"])
    regime = _regime_from_summary(summary)

    return _json_ready(
        {
            "source": "observed",
            "id": spec["id"],
            "title": spec["title"],
            "short_label": spec["short_label"],
            "headline": spec["headline"],
            "is_default": bool(spec.get("is_default")),
            "model": {
                "source": "observed",
                "name": model_name,
                "model_id": model.get("model_id"),
                "family": model.get("family"),
                "size": model.get("size"),
            },
            "pair_id": pair_id,
            "category": patch_json["pair"]["category"],
            "archetype": spec["archetype"],
            "why": spec["why"],
            "regime": regime,
            "runtime": {
                "source": "observed",
                "run_dir": str(run_dir.relative_to(repo_root)),
                "dtype": metadata.get("dtype"),
                "device": metadata.get("device"),
                "max_new_tokens": metadata.get("max_new_tokens"),
                "sample": metadata.get("sample", False),
                "thinking_mode": metadata.get("thinking_mode"),
            },
            "prompt": {
                "source": "observed",
                "a_text": prompt_a,
                "b_text": prompt_b,
                "diff": _visible_text_diff(prompt_a, prompt_b),
                "token_edit_distance": int(event.get("prompt_token_edit_distance") or 0),
                "token_delta_kind": event.get("prompt_token_delta_kind"),
                "token_lcp": prompt_lcp,
                "token_window": _prompt_token_windows(
                    [int(x) for x in prompt_token_ids_a],
                    [int(x) for x in prompt_token_ids_b],
                    prompt_lcp,
                    model.get("model_id"),
                    tokenizer_cache,
                ),
            },
            "generation": {
                "source": "observed",
                "a_text": gen_a.get("generated_text", ""),
                "b_text": gen_b.get("generated_text", ""),
                "a_token_ids": gen_a.get("generated_tokens", []),
                "b_token_ids": gen_b.get("generated_tokens", []),
                "branch_t": branch_t,
                "common_prefix_tokens": int(event.get("common_prefix_tokens") or branch_t),
                "event_kind": event.get("event_kind"),
                "silent_logit_lead": event.get("silent_logit_lead"),
                "a_branch_token": patch_json.get("a_branch_token"),
                "a_branch_token_id": patch_json.get("a_branch_token_id"),
                "b_branch_token": patch_json.get("b_branch_token"),
                "b_branch_token_id": patch_json.get("b_branch_token_id"),
                "clean_replay_top1_token": patch_json.get("clean_replay_top1_token"),
                "corrupt_replay_top1_token": patch_json.get("corrupt_replay_top1_token"),
            },
            "logit_timeline": _logit_timeline(run_dir, pair_id, branch_t),
            "patch": {
                "source": "observed",
                "summary": {
                    "wave": summary.get("wave"),
                    "best_position_class": summary.get("best_position_class"),
                    "best_position_label": summary.get("best_position_label"),
                    "best_layer": summary.get("best_layer"),
                    "best_rescue_fraction": summary.get("best_rescue_fraction"),
                    "prompt_lcp_best_layer": summary.get("prompt_lcp_token_best_layer"),
                    "prompt_lcp_rescue_fraction": summary.get("prompt_lcp_token_best_rescue_fraction"),
                    "final_context_best_layer": summary.get("final_context_token_best_layer"),
                    "final_context_rescue_fraction": summary.get("final_context_token_best_rescue_fraction"),
                    "best_aligned_prompt_rescue_fraction": summary.get("best_aligned_prompt_rescue_fraction"),
                    "best_generated_prefix_rescue_fraction": summary.get("best_generated_prefix_rescue_fraction"),
                    "prompt_lcp_full": bool(summary.get("prompt_lcp_full")),
                    "strict_late_only_full": bool(summary.get("strict_late_only_full")),
                    "replayable": bool(summary.get("replayable")),
                },
                "grid": _patch_grid(patch_csv),
            },
            "artifacts": {
                "patch_csv": _artifact(patch_csv, repo_root),
                "patch_json": _artifact(patch_json_path, repo_root),
                "logit_generations": _artifact(run_dir / "generations.jsonl", repo_root),
                "logit_probes": _artifact(run_dir / "logit_probes.jsonl", repo_root),
                "trajectory_events": _artifact(TRAJECTORY_CSV, repo_root),
                "case_summary": _artifact(SUMMARY_CSV, repo_root),
            },
        }
    )


def _build_panel_summary(summary_df: pd.DataFrame, repo_root: Path) -> dict[str, Any]:
    forward = summary_df[summary_df["wave"].isin(FORWARD_WAVES)].copy()
    regimes = forward.apply(lambda row: _regime_from_summary(_json_ready(row.to_dict())), axis=1)
    counts = regimes.value_counts().to_dict()
    strict_late_only = int(forward["strict_late_only_full"].fillna(False).astype(bool).sum())
    return _json_ready(
        {
            "source": "derived",
            "case_count": int(len(forward)),
            "regime_counts": {
                "edit_boundary": int(counts.get("edit_boundary", 0)),
                "trajectory_migration": int(counts.get("trajectory_migration", 0)),
                "prompt_accumulation": int(counts.get("prompt_accumulation", 0)),
            },
            "strict_late_only_count": strict_late_only,
            "rule_of_three_upper_95": 3 / len(forward) if len(forward) else None,
            "summary_artifact": _artifact(repo_root / SUMMARY_CSV, repo_root),
        }
    )


def _build_sae_cases(repo_root: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for csv_path in sorted((repo_root / SAE_DIR).glob("*__sae_features.csv")):
        if csv_path.name == "sae_feature_delta_summary.csv":
            continue
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        first = df.iloc[0].to_dict()
        key_cols = ["model_name", "pair_id", "category", "layer", "position_label"]
        groups = []
        for keys, group in df.groupby(key_cols, sort=True):
            model_name, pair_id, category, layer, position_label = keys
            clean = group[group["side"] == "clean"].sort_values("rank").head(10)
            corrupt = group[group["side"] == "corrupt"].sort_values("rank").head(10)
            clean_ids = set(clean["feature_id"].astype(int).tolist())
            corrupt_ids = set(corrupt["feature_id"].astype(int).tolist())
            groups.append(
                {
                    "source": "observed",
                    "layer": int(layer),
                    "position_label": str(position_label),
                    "clean_token": clean.iloc[0]["token_text"] if not clean.empty else None,
                    "corrupt_token": corrupt.iloc[0]["token_text"] if not corrupt.empty else None,
                    "top_clean": [
                        {
                            "feature_id": int(r.feature_id),
                            "rank": int(r.rank),
                            "activation": float(r.activation),
                        }
                        for r in clean.itertuples(index=False)
                    ],
                    "top_corrupt": [
                        {
                            "feature_id": int(r.feature_id),
                            "rank": int(r.rank),
                            "activation": float(r.activation),
                        }
                        for r in corrupt.itertuples(index=False)
                    ],
                    "top10_overlap": len(clean_ids & corrupt_ids),
                }
            )
        groups.sort(
            key=lambda g: (
                int(g["layer"]),
                0 if g["position_label"] == "prompt_lcp_token" else 1,
                str(g["position_label"]),
            )
        )
        cases.append(
            {
                "source": "observed",
                "id": f"{first['model_name']}__{first['pair_id']}",
                "model_name": first["model_name"],
                "pair_id": first["pair_id"],
                "category": first["category"],
                "sae_repo": first["sae_repo"],
                "groups": groups,
                "artifacts": {
                    "csv": _artifact(csv_path, repo_root),
                    "json": _artifact(csv_path.with_suffix(".json"), repo_root),
                },
            }
        )
    return _json_ready(cases)


def build_bundle(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    summary_df = pd.read_csv(repo_root / SUMMARY_CSV, low_memory=False)
    events_df = pd.read_csv(repo_root / TRAJECTORY_CSV, low_memory=False)
    tokenizer_cache = _TokenizerCache()
    cases = [
        _build_case(spec, repo_root, summary_df, events_df, tokenizer_cache)
        for spec in CASE_SPECS
    ]
    return _json_ready(
        {
            "schema_version": LAB_SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "title": "Branch Signatures Interactive Lab",
            "default_case_id": CASE_SPECS[0]["id"],
            "panel_summary": _build_panel_summary(summary_df, repo_root),
            "cases": cases,
            "sae_cases": _build_sae_cases(repo_root),
            "source_ledger": [
                {
                    "source": "observed",
                    "meaning": "Copied from raw or derived repo artifacts without reinterpretation.",
                },
                {
                    "source": "derived",
                    "meaning": "Computed by this builder from observed artifacts, such as grouping patch cells.",
                },
                {
                    "source": "not_run",
                    "meaning": "Unavailable in local artifacts; shown as IDs or omitted rather than invented.",
                },
            ],
        }
    )


def render_html(bundle: dict[str, Any]) -> str:
    template_dir = Path(__file__).resolve().parent / "templates"
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("signature_lab.html.j2")
    bundle_json = json.dumps(bundle, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return template.render(bundle=bundle, bundle_json=bundle_json)


def write_lab(repo_root: Path, out_json: Path, out_html: Path) -> None:
    bundle = build_bundle(repo_root)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(bundle, indent=2, ensure_ascii=False) + "\n")
    out_html.write_text(render_html(bundle))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m branchtrace.signature_lab")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--out-json",
        default="experiments/E11_branchtrace_card/paper/signature_lab_data.json",
    )
    parser.add_argument(
        "--out-html",
        default="experiments/E11_branchtrace_card/paper/signature_explainer.html",
    )
    args = parser.parse_args(argv)
    write_lab(Path(args.repo_root), Path(args.out_json), Path(args.out_html))
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_html}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
