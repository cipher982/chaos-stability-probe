"""Build regime-averaged canonical heatmaps for the branchtrace card.

We have 82 activation-patching "rescue heatmaps" (one per case). Each heatmap is
rescue_fraction(layer, patched_position) — how much does splicing run A's
residual stream into run B at each (layer, position) restore A's branch token.

Raw heatmaps cannot be pixel-overlaid because position and layer indices do not
align across cases/models. We re-index onto canonical axes, then average within
each regime.

Regimes (forward universe, waves v1/v2/v3/v5_replication):
  - edit_boundary        (41): prompt_lcp_full == True
  - prompt_accumulation  (14): not prompt_lcp_full, event_kind == immediate_visible_branch
  - trajectory_migration (27): not prompt_lcp_full, event_kind == silent_logit_divergence
  (the remaining cases in the trajectory table are no_visible_branch / reverse; not here)

Canonical axes:
  Position axis (7 columns per case):
    prompt_early   : mean rescue over aligned_prompt_pos_*_to_* where max window idx < prompt_token_lcp - 1
    pre_lcp        : rescue at aligned_prompt_pos_{lcp-1}_to_{lcp-1} if present else nearest earlier
    prompt_LCP     : rescue at position_label == "prompt_lcp_token"
    post_lcp       : rescue at aligned_prompt_pos_* window whose start == prompt_token_lcp + 1
    prompt_late    : mean rescue over aligned_prompt_pos_* in the last 3 positions of the prompt
                     (excluding final_context_token)
    final_context  : rescue at position_label == "final_context_token"
    gen_prefix     : rescue at position_label == "aligned_generated_prefix_pos_0"
                     (first generated token position; may be NaN for immediate branches)

  Layer axis (24 canonical depth bins 0..23):
    Linearly rescale the case's native layer index (0..L_max) into 24 bins via np.interp.

Result per case: rescue_fraction(depth_bin, canonical_column), shape (24, 7).
Missing cells stay NaN; averaging uses np.nanmean.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# These heatmaps routinely contain all-NaN columns (e.g. gen_prefix for
# prompt_accumulation cases); nanmean spams "Mean of empty slice" warnings we
# deliberately tolerate.
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN slice encountered")

ROOT = Path(__file__).resolve().parents[3]  # /Users/davidrose/git/chaos
OUT_DIR = Path(__file__).resolve().parent / "regime_heatmaps"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WAVES = {
    "activation_patch_v1",
    "activation_patch_v2",
    "activation_patch_v3",
    "activation_patch_v5_replication",
}

# Canonical x-axis: signed integer offsets in tokens relative to the edit
# point, plus two special columns. offset = token_index - prompt_token_lcp.
# offset 0 is the edit token itself. Negative = before edit. Positive = after.
OFFSET_RANGE = list(range(-5, 6))  # -5..+5 inclusive
EDIT_COL_INDEX = OFFSET_RANGE.index(0)
CANON_COLS = [f"off_{o:+d}" for o in OFFSET_RANGE] + ["final", "gen"]
CANON_COLS_SHORT = [str(o) for o in OFFSET_RANGE] + ["final", "gen"]
N_DEPTH = 24
FINAL_COL_INDEX = len(OFFSET_RANGE)  # 11
GEN_COL_INDEX = FINAL_COL_INDEX + 1  # 12

MODEL_ORDER = [
    "qwen35_08b",
    "qwen35_2b",
    "qwen35_4b",
    "qwen35_9b",
    "gemma4_e2b_it",
    "gemma4_e2b_base",
    "gemma4_e4b_it",
    "gemma4_e4b_base",
]
MODEL_PRETTY = {
    "qwen35_08b": "Qwen3.5 0.8B",
    "qwen35_2b": "Qwen3.5 2B",
    "qwen35_4b": "Qwen3.5 4B",
    "qwen35_9b": "Qwen3.5 9B",
    "gemma4_e2b_it": "Gemma E2B IT",
    "gemma4_e2b_base": "Gemma E2B base",
    "gemma4_e4b_it": "Gemma E4B IT",
    "gemma4_e4b_base": "Gemma E4B base",
}

ALIGNED_RE = re.compile(r"^aligned_prompt_pos_(\d+)_to_(\d+)$")

# --- Data plumbing --------------------------------------------------------


def find_case_csv(model_name: str, pair_id: str) -> Path | None:
    """Locate the long-format per-case CSV (local first, then SageMaker)."""
    local = ROOT / "runs" / "mechinterp_patch_aligned" / f"{model_name}__{pair_id}.csv"
    if local.exists():
        return local
    # Fallback: search sagemaker artifacts.
    root_sm = ROOT / "runs" / "sagemaker_artifacts"
    target = f"{model_name}__{pair_id}.csv"
    # maxdepth 5, must contain activation-patch, not reverse, not patch_summary
    if not root_sm.exists():
        return None
    for p in root_sm.rglob(target):
        rel_parts = p.relative_to(root_sm).parts
        if len(rel_parts) > 5:
            continue
        path_str = str(p)
        if "activation-patch" not in path_str:
            continue
        if p.name.startswith("patch_summary"):
            continue
        if "__reverse" in p.name:
            continue
        return p
    return None


def load_trajectory_events() -> pd.DataFrame:
    te = pd.read_csv(ROOT / "runs/trajectory_events/logit_token_cert_v1/trajectory_events.csv")
    te = te[(te["repeat"] == 0) & (te["is_control"] == False)]  # noqa: E712
    return te[["model_name", "pair_id", "prompt_token_lcp", "event_kind"]].copy()


def load_case_universe() -> pd.DataFrame:
    cs = pd.read_csv(ROOT / "runs/rankings/activation_patch_comparison/case_level_summary.csv")
    cs = cs[cs["wave"].isin(WAVES)].copy()
    return cs[["model_name", "pair_id", "wave", "prompt_lcp_full"]]


def prompt_token_lcp_for_case(case_csv: Path, te_lookup: dict) -> int | None:
    """Try JSON sidecar first, then trajectory events lookup."""
    js = case_csv.with_suffix(".json")
    if js.exists():
        try:
            j = json.loads(js.read_text())
            v = j.get("prompt_token_lcp")
            if v is not None:
                return int(v)
        except Exception:
            pass
    # fallback
    model = case_csv.stem.split("__", 1)[0]
    pair = case_csv.stem.split("__", 1)[1]
    v = te_lookup.get((model, pair))
    if v is None or pd.isna(v):
        return None
    return int(v)


# --- Canonicalization -----------------------------------------------------


def canonicalize_case(case_csv: Path, prompt_token_lcp: int | None) -> np.ndarray:
    """Return shape (N_DEPTH, len(CANON_COLS)) with NaN for missing."""
    df = pd.read_csv(case_csv)
    if "layer" not in df.columns or "position_label" not in df.columns:
        return None  # type: ignore[return-value]
    layers = np.sort(df["layer"].unique())
    L_max = int(layers.max())
    native = np.full((len(layers), len(CANON_COLS)), np.nan, dtype=np.float64)

    # Parse aligned prompt positions (each label covers [s, e] inclusive)
    aligned = []
    for lbl in df["position_label"].unique():
        m = ALIGNED_RE.match(lbl)
        if m:
            aligned.append((int(m.group(1)), int(m.group(2)), lbl))

    def rescue_by_layer(label: str) -> np.ndarray | None:
        sub = df[df["position_label"] == label]
        if sub.empty:
            return None
        out = np.full(len(layers), np.nan)
        for _, r in sub.iterrows():
            li = int(np.searchsorted(layers, r["layer"]))
            if li < len(layers) and layers[li] == r["layer"]:
                out[li] = r["rescue_fraction"]
        return out

    lcp = prompt_token_lcp

    # Map each aligned position window to integer offsets relative to LCP.
    # If a window spans multiple token indices, take every covered index.
    # For each (offset, layer) cell, average across windows that hit it.
    if lcp is not None:
        # offset -> list of label rescue arrays
        per_offset: dict[int, list[np.ndarray]] = {o: [] for o in OFFSET_RANGE}
        for s, e, lbl in aligned:
            arr = rescue_by_layer(lbl)
            if arr is None:
                continue
            for tok_idx in range(s, e + 1):
                off = tok_idx - lcp
                if off in per_offset:
                    per_offset[off].append(arr)

        # prompt_LCP override: dedicated label if present
        plcp = rescue_by_layer("prompt_lcp_token")
        if plcp is not None:
            per_offset[0] = [plcp]

        for i, off in enumerate(OFFSET_RANGE):
            stacks = per_offset[off]
            if stacks:
                native[:, i] = np.nanmean(np.vstack(stacks), axis=0)

    # final_context: last prompt position
    v = rescue_by_layer("final_context_token")
    if v is not None:
        native[:, FINAL_COL_INDEX] = v

    # gen: first generated token position (the branch token itself)
    v = rescue_by_layer("aligned_generated_prefix_pos_0")
    if v is None:
        v = rescue_by_layer("generated_prefix_token")
    if v is not None:
        native[:, GEN_COL_INDEX] = v

    # Rescale layers 0..L_max to 24 canonical depth bins
    canonical = np.full((N_DEPTH, len(CANON_COLS)), np.nan)
    if L_max <= 0:
        # degenerate: broadcast single layer
        canonical[:] = native[0:1, :]
        return canonical
    native_x = layers.astype(float)
    target_x = np.linspace(0.0, float(L_max), N_DEPTH)
    for c in range(len(CANON_COLS)):
        col = native[:, c]
        if np.all(np.isnan(col)):
            continue
        # Interp only over finite — but layer grid is dense, mostly every value set or none set
        mask = ~np.isnan(col)
        if mask.sum() < 2:
            canonical[:, c] = col[mask][0] if mask.any() else np.nan
            continue
        xs = native_x[mask]
        ys = col[mask]
        canonical[:, c] = np.interp(target_x, xs, ys)
    return canonical


# --- Build corpus ---------------------------------------------------------


def classify_regime(prompt_lcp_full: bool | None, event_kind: str | None) -> str | None:
    if prompt_lcp_full is True:
        return "edit_boundary"
    if event_kind == "immediate_visible_branch":
        return "prompt_accumulation"
    if event_kind == "silent_logit_divergence":
        return "trajectory_migration"
    return None  # no_visible_branch etc. — not in this taxonomy


def build_corpus():
    cs = load_case_universe()
    te = load_trajectory_events()
    te_lookup = {
        (r.model_name, r.pair_id): (r.prompt_token_lcp, r.event_kind)
        for r in te.itertuples(index=False)
    }
    lcp_only = {k: v[0] for k, v in te_lookup.items()}

    per_case = []  # list of dict(model, pair, regime, grid(24,7))
    missing = 0
    uncat = 0
    for _, row in cs.iterrows():
        key = (row.model_name, row.pair_id)
        te_info = te_lookup.get(key)
        event_kind = te_info[1] if te_info is not None else None
        regime = classify_regime(bool(row.prompt_lcp_full), event_kind)
        if regime is None:
            uncat += 1
            continue
        case_csv = find_case_csv(row.model_name, row.pair_id)
        if case_csv is None:
            print(f"[MISS] {row.model_name}__{row.pair_id} (regime {regime})", file=sys.stderr)
            missing += 1
            continue
        lcp = prompt_token_lcp_for_case(case_csv, lcp_only)
        grid = canonicalize_case(case_csv, lcp)
        if grid is None:
            print(f"[SKIP] {row.model_name}__{row.pair_id} bad csv", file=sys.stderr)
            missing += 1
            continue
        per_case.append(
            dict(model=row.model_name, pair=row.pair_id, regime=regime, grid=grid)
        )
    return per_case, missing, uncat


# --- Plotting -------------------------------------------------------------


def pretty_depth_ticks():
    return [0, 5, 10, 15, 20, 23]


def pretty_col_ticks():
    return list(range(len(CANON_COLS)))


def draw_heatmap(ax, grid: np.ndarray, title: str, *, vmin=0.0, vmax=1.0,
                 show_xlabels=True, show_ylabels=True, dark: bool = False):
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#222222" if dark else "#eeeeee")
    im = ax.imshow(
        grid,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    # Anchor lines: edit point at offset 0, divider between prompt range and
    # special (final, gen) columns.
    edge = "white" if dark else "#000000"
    edge_alpha = 0.70 if dark else 0.85
    ax.axvline(EDIT_COL_INDEX, color=edge, alpha=edge_alpha, lw=1.4)
    ax.axvline(FINAL_COL_INDEX - 0.5, color=edge, alpha=0.35, lw=0.8, ls=":")
    ax.text(
        EDIT_COL_INDEX,
        N_DEPTH - 0.4,
        "edit",
        color=edge,
        fontsize=8,
        ha="center",
        va="bottom",
    )
    ax.set_yticks(pretty_depth_ticks())
    ax.set_yticklabels([str(d) for d in pretty_depth_ticks()])
    if show_ylabels:
        ax.set_ylabel("residual-stream depth", fontsize=11)
    if show_xlabels:
        ax.set_xticks(pretty_col_ticks())
        ax.set_xticklabels(CANON_COLS_SHORT, rotation=0, ha="center", fontsize=9)
    else:
        ax.set_xticks(pretty_col_ticks())
        ax.set_xticklabels([""] * len(CANON_COLS))
    ax.set_title(title, fontsize=12, pad=8)
    return im


REGIME_ORDER = ["edit_boundary", "trajectory_migration", "prompt_accumulation"]
REGIME_COLORS = {
    "edit_boundary": "#f2c14e",
    "trajectory_migration": "#4e9cf2",
    "prompt_accumulation": "#e45a5a",
}
REGIME_SUBTITLES = {
    "edit_boundary": "bright at offset 0, early depth",
    "trajectory_migration": "prompt columns suppressed; late final-context only",
    "prompt_accumulation": "bright at late prompt (+3..+5) + final-context",
}


def regime_mean(per_case: list[dict], regime: str) -> tuple[np.ndarray, int]:
    grids = [c["grid"] for c in per_case if c["regime"] == regime]
    if not grids:
        return np.full((N_DEPTH, len(CANON_COLS)), np.nan), 0
    stack = np.stack(grids, axis=0)  # (n, 24, 7)
    return np.nanmean(stack, axis=0), len(grids)


def per_case_column_full_count(per_case: list[dict], regime: str, col_name: str) -> tuple[int, int]:
    """Count cases in regime where per-case max rescue at col >= 1.0 (any depth)."""
    col = CANON_COLS.index(col_name)
    total = 0
    full = 0
    for c in per_case:
        if c["regime"] != regime:
            continue
        total += 1
        series = c["grid"][:, col]
        if np.all(np.isnan(series)):
            continue
        if np.nanmax(series) >= 1.0:
            full += 1
    return full, total


def annotate_peak(ax, grid: np.ndarray, *, exclude_final_context: bool = True):
    """Put a small marker + text at the non-trivial peak cell.

    The final_context column hits 1.0 at late depth in every regime, so mask it
    to surface the informative peak elsewhere. If the whole map (minus
    final_context) is flat / NaN, skip annotation.
    """
    g = grid.copy()
    if exclude_final_context:
        g[:, FINAL_COL_INDEX] = np.nan
    if np.all(np.isnan(g)):
        return
    flat_idx = np.nanargmax(g)
    d, c = np.unravel_index(flat_idx, g.shape)
    val = g[d, c]
    if val < 0.25:  # too weak to bother marking
        return
    ax.scatter([c], [d], s=90, facecolor="none", edgecolor="white", linewidths=1.4, zorder=5)
    ax.text(
        c + 0.25,
        d + 0.5,
        f"{val:.2f}",
        color="white",
        fontsize=9,
        va="bottom",
        ha="left",
        zorder=5,
        bbox=dict(facecolor="#00000066", edgecolor="none", pad=1.2),
    )


def triptych(per_case: list[dict], out_path: Path, *, dark: bool = True):
    if dark:
        plt.style.use("dark_background")
    else:
        plt.style.use("default")
    fig = plt.figure(figsize=(17.0, 7.2))
    gs = fig.add_gridspec(
        1, 4, width_ratios=[1.0, 1.0, 1.0, 0.05], wspace=0.32
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])
    text_color = "white" if dark else "#111111"
    im = None
    for ax, regime in zip(axes, REGIME_ORDER):
        grid, n = regime_mean(per_case, regime)
        sub = REGIME_SUBTITLES[regime]
        pretty = regime.replace("_", "-")
        title = f"{pretty}\n{sub}   (n={n})"
        im = draw_heatmap(ax, grid, title, dark=dark, show_ylabels=True)
        annotate_peak(ax, grid, exclude_final_context=True)
        # Per-case full-rescue stats as corner annotation, not title wall.
        gp_full, gp_tot = per_case_column_full_count(per_case, regime, "gen")
        plcp_full, plcp_tot = per_case_column_full_count(per_case, regime, "off_+0")
        corner = (
            f"per-case full rescue\n"
            f"  edit: {plcp_full}/{plcp_tot}\n"
            f"   gen: {gp_full}/{gp_tot}"
        )
        ax.text(
            0.02,
            0.98,
            corner,
            transform=ax.transAxes,
            color=text_color,
            fontsize=8,
            va="top",
            ha="left",
            family="monospace",
            bbox=dict(
                facecolor=("#000000aa" if dark else "#ffffffcc"),
                edgecolor="none",
                pad=2.5,
            ),
        )
        ax.title.set_fontsize(12)
        ax.title.set_color(REGIME_COLORS[regime])
    fig.suptitle(
        "Canonical rescue maps separate edit-local and trajectory-state branch control  (n=82)",
        fontsize=15,
        color=text_color,
        y=1.02,
    )
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("rescue fraction (clipped at 1.0)", color=text_color, fontsize=10)
    cbar.ax.yaxis.set_tick_params(color=text_color)
    plt.setp(cbar.ax.get_yticklabels(), color=text_color)
    fig.text(
        0.5,
        -0.04,
        "Columns run left-to-right along the prompt, then into generation. "
        "The edit point is the first token where prompts A and B differ (vertical line). "
        "Brighter = splicing clean activations at that (depth, position) restores A's branch token.",
        ha="center",
        va="top",
        color=text_color,
        fontsize=9,
        wrap=True,
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    plt.style.use("default")


def grand_mean_figure(per_case: list[dict], out_path: Path):
    plt.style.use("dark_background")
    grids = [c["grid"] for c in per_case]
    grand = np.nanmean(np.stack(grids, axis=0), axis=0)
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    im = draw_heatmap(ax, grand, f"All {len(grids)} forward cases, canonical grid", dark=True)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("rescue fraction", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.get_yticklabels(), color="white")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    plt.style.use("default")


def depth_profile_figure(per_case: list[dict], out_path: Path):
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    depth = np.arange(N_DEPTH)
    for regime in REGIME_ORDER:
        grid, n = regime_mean(per_case, regime)
        profile = np.nanmean(grid, axis=1)
        ax.plot(
            depth,
            profile,
            label=f"{regime} (n={n})",
            color=REGIME_COLORS[regime],
            lw=2.3,
        )
    ax.set_xlabel("residual-stream depth bin (0..23)", fontsize=11)
    ax.set_ylabel("mean rescue fraction\n(averaged over canonical positions)", fontsize=11)
    ax.set_xticks(pretty_depth_ticks())
    ax.set_title("Rescue vs residual-stream depth by regime", fontsize=13)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=10, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def column_bars_figure(per_case: list[dict], out_path: Path):
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    x = np.arange(len(CANON_COLS))
    width = 0.26
    for i, regime in enumerate(REGIME_ORDER):
        grid, n = regime_mean(per_case, regime)
        col_mean = np.nanmean(grid, axis=0)
        ax.bar(
            x + (i - 1) * width,
            col_mean,
            width=width,
            label=f"{regime} (n={n})",
            color=REGIME_COLORS[regime],
            edgecolor="black",
            linewidth=0.4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(CANON_COLS_SHORT, rotation=0, ha="center")
    ax.set_ylabel("mean rescue fraction\n(averaged over depth bins)", fontsize=11)
    ax.set_title("Mean rescue by canonical position, per regime", fontsize=13)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=10, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def model_facets_figure(per_case: list[dict], out_path: Path):
    fig, axes = plt.subplots(2, 4, figsize=(18.0, 9.0))
    im = None
    for idx, model in enumerate(MODEL_ORDER):
        ax = axes[idx // 4][idx % 4]
        grids = [c["grid"] for c in per_case if c["model"] == model]
        if grids:
            mean = np.nanmean(np.stack(grids, axis=0), axis=0)
            title = f"{MODEL_PRETTY[model]} (n={len(grids)})"
        else:
            mean = np.full((N_DEPTH, len(CANON_COLS)), np.nan)
            title = f"{MODEL_PRETTY[model]} (n=0)"
        show_x = (idx // 4) == 1
        show_y = (idx % 4) == 0
        im = draw_heatmap(ax, mean, title, show_xlabels=show_x, show_ylabels=show_y, dark=False)
    fig.suptitle(
        "Canonical rescue heatmap per model (all regimes pooled)",
        fontsize=15,
        y=1.01,
    )
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label("rescue fraction")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# --- Summary stats --------------------------------------------------------


def argmax_cell(grid: np.ndarray) -> tuple[int, str, float]:
    if np.all(np.isnan(grid)):
        return -1, "<all-nan>", float("nan")
    flat = np.nanargmax(grid)
    d, c = np.unravel_index(flat, grid.shape)
    return int(d), CANON_COLS[int(c)], float(grid[d, c])


def main():
    per_case, missing, uncat = build_corpus()
    print(f"loaded {len(per_case)} cases; missing={missing}; uncategorized={uncat}")

    counts = {r: sum(1 for c in per_case if c["regime"] == r) for r in REGIME_ORDER}
    print("per-regime counts:", counts)

    # arg-maxes (global and with final_context masked, since the last-layer
    # final-context cell is a trivial ceiling that dominates every regime).
    print("\n--- regime argmax signatures ---")
    for regime in REGIME_ORDER:
        grid, n = regime_mean(per_case, regime)
        d, col, val = argmax_cell(grid)
        masked = grid.copy()
        masked[:, FINAL_COL_INDEX] = np.nan
        d2, col2, val2 = argmax_cell(masked)
        print(
            f"  {regime:>22s} (n={n:3d}): "
            f"argmax d={d:2d} col={col:<15s} val={val:.3f}  |  "
            f"non-final d={d2:2d} col={col2:<15s} val={val2:.3f}"
        )

    # Write figures.
    out_triptych = OUT_DIR / "regime_mean_heatmaps_triptych.png"
    triptych(per_case, out_triptych, dark=True)
    print(f"wrote {out_triptych}")

    out_triptych_light = OUT_DIR / "regime_mean_heatmaps_triptych_light.png"
    triptych(per_case, out_triptych_light, dark=False)
    print(f"wrote {out_triptych_light}")

    out_grand = OUT_DIR / "regime_grand_mean.png"
    grand_mean_figure(per_case, out_grand)
    print(f"wrote {out_grand}")

    out_depth = OUT_DIR / "regime_depth_profile.png"
    depth_profile_figure(per_case, out_depth)
    print(f"wrote {out_depth}")

    out_cols = OUT_DIR / "regime_column_bars.png"
    column_bars_figure(per_case, out_cols)
    print(f"wrote {out_cols}")

    out_facets = OUT_DIR / "regime_model_facets.png"
    model_facets_figure(per_case, out_facets)
    print(f"wrote {out_facets}")

    # Per-model breakdown summary
    print("\n--- per-model regime counts ---")
    for m in MODEL_ORDER:
        ms = [c for c in per_case if c["model"] == m]
        by = {r: sum(1 for c in ms if c["regime"] == r) for r in REGIME_ORDER}
        print(f"  {MODEL_PRETTY[m]:<15s}: total={len(ms):2d}  {by}")


if __name__ == "__main__":
    main()
