#!/usr/bin/env python3
"""Build slide-grade visuals for the micro-perturbation sweep."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs/micro_qwen35_08b_500/qwen35_08b"
RANK_DIR = ROOT / "runs/rankings/micro_qwen35_08b_500"
OUT_DIR = ROOT / "talk/micro_visuals"

RED = "#a43d3d"
GREEN = "#2f6f4e"
INK = "#202124"
MUTED = "#5f6368"
GRID = "#e8eaed"
PAPER = "#fbfbf8"
BLUE = "#2d5f9a"
ORANGE = "#c96c2b"

MICRO_MODEL_SUMMARIES = [
    ("Qwen 2B", ROOT / "runs/rankings/micro_token_audit/qwen35_2b/effective_token_category_summary.csv"),
    ("Qwen 4B", ROOT / "runs/rankings/micro_token_audit/qwen35_4b/effective_token_category_summary.csv"),
    ("Qwen 9B", ROOT / "runs/rankings/micro_token_audit/qwen35_9b/effective_token_category_summary.csv"),
    ("Gemma E2B it", ROOT / "runs/rankings/micro_token_audit/gemma4_e2b_it/effective_token_category_summary.csv"),
    ("Gemma E4B it", ROOT / "runs/rankings/micro_token_audit/gemma4_e4b_it/effective_token_category_summary.csv"),
]


def clean_category(value: str) -> str:
    return value.replace("micro_", "").replace("_", " ")


def short_category(value: str) -> str:
    labels = {
        "micro_blank_line_wrap": "blank\nline",
        "micro_control_identical": "identical\ncontrol",
        "micro_crlf_suffix": "CRLF\nsuffix",
        "micro_double_internal_space": "double\nspace",
        "micro_duplicate_punctuation": "duplicate\npunct.",
        "micro_duplicate_small_word": "duplicate\nword",
        "micro_leading_newline": "leading\nnewline",
        "micro_leading_space": "leading\nspace",
        "micro_line_wrap": "line\nwrap",
        "micro_parenthesize_word": "paren\nword",
        "micro_space_after_punctuation": "space\nafter punct.",
        "micro_space_before_punctuation": "space\nbefore punct.",
        "micro_tab_after_space": "tab after\nspace",
        "micro_tab_indent": "tab\nindent",
        "micro_trailing_newline": "trailing\nnewline",
        "micro_trailing_space": "trailing\nspace",
        "micro_triple_internal_space": "triple\nspace",
    }
    return labels.get(value, clean_category(value).replace(" ", "\n"))


def wrap(text: str, width: int = 68, max_lines: int | None = None) -> str:
    text = text.replace("\r", "\\r").replace("\t", "\\t").replace("\n", "\\n")
    lines = textwrap.wrap(text, width=width, replace_whitespace=False, drop_whitespace=False)
    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip() + "..."
    return "\n".join(lines)


def load_merged() -> pd.DataFrame:
    summary = pd.read_csv(RUN_DIR / "summary_with_semantic.csv")
    prompts = pd.DataFrame(json.loads((ROOT / "configs/prompt_pairs_micro_500.json").read_text()))
    prompts = prompts[["id", "prompt_a", "prompt_b"]].rename(columns={"id": "pair_id"})
    return summary.merge(prompts, on="pair_id", how="left")


def category_bar(compact: bool = False) -> Path:
    df = pd.read_csv(RANK_DIR / "category_summary.csv").copy()
    df["label"] = df["category"].map(clean_category)
    df = df.sort_values("semantic_mean")
    colors = [GREEN if v < 1e-5 else RED for v in df["semantic_mean"]]

    fig_size = (14.8, 5.25) if compact else (12, 7.2)
    fig, ax = plt.subplots(figsize=fig_size, facecolor=PAPER)
    ax.set_facecolor(PAPER)
    ax.barh(df["label"], df["semantic_mean"], color=colors, height=0.72)
    ax.set_xlabel("Mean semantic output distance", fontsize=12)
    if not compact:
        fig.text(0.08, 0.965, "Qwen3.5 0.8B: tiny edits are not equally tiny", fontsize=20, weight="bold", color=INK)
        fig.text(
            0.08,
            0.925,
            "Deterministic decode, 500 human-near-identical prompt edits. Green categories were effectively inert.",
            fontsize=11,
            color=MUTED,
        )
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(df["semantic_mean"]) * 1.18)
    for y, v in enumerate(df["semantic_mean"]):
        label = "0" if v < 1e-5 else f"{v:.3f}"
        ax.text(max(v, 0.001) + 0.0015, y, label, va="center", fontsize=9, color=INK)
    rect = [0.03, 0.02, 1, 0.98] if compact else [0.04, 0.03, 1, 0.88]
    fig.tight_layout(rect=rect, pad=1.15)
    out = OUT_DIR / ("01_micro_category_fingerprint_compact.png" if compact else "01_micro_category_fingerprint.png")
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def model_category_heatmap() -> Path:
    rows: list[pd.DataFrame] = []
    for model, path in MICRO_MODEL_SUMMARIES:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["model"] = model
        rows.append(df[["model", "category", "n", "prompt_token_edit_mean", "semantic_mean", "semantic_p90"]])
    if not rows:
        raise FileNotFoundError("No micro-sweep category summaries found")

    combined = pd.concat(rows, ignore_index=True)
    category_order = (
        combined.groupby("category")["semantic_mean"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    model_order = [model for model, path in MICRO_MODEL_SUMMARIES if path.exists()]
    matrix = (
        combined.pivot(index="model", columns="category", values="semantic_mean")
        .reindex(index=model_order, columns=category_order)
    )

    cmap = LinearSegmentedColormap.from_list("token_effect", ["#f7f0ea", "#d9634b", "#7f1d1d"])
    vmax = max(0.12, float(np.nanpercentile(matrix.values, 96)))

    fig, ax = plt.subplots(figsize=(14.8, 5.0), facecolor=PAPER)
    ax.set_facecolor(PAPER)
    im = ax.imshow(matrix.values, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)

    ax.set_yticks(np.arange(len(matrix.index)), matrix.index, fontsize=12)
    ax.set_xticks(
        np.arange(len(matrix.columns)),
        [short_category(c) for c in matrix.columns],
        fontsize=9,
        rotation=0,
        ha="center",
    )
    ax.tick_params(axis="both", length=0)
    ax.set_xticks(np.arange(-0.5, len(matrix.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(matrix.index), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.spines[:].set_visible(False)

    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            v = matrix.iat[y, x]
            if pd.isna(v):
                continue
            label = "0" if v < 1e-5 else f"{v:.2f}"
            color = "white" if v > vmax * 0.62 else INK
            ax.text(x, y, label, ha="center", va="center", fontsize=9.5, color=color, weight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.012)
    cbar.set_label("Mean semantic output distance", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout(rect=[0.015, 0.035, 0.995, 0.995], pad=0.7)
    out = OUT_DIR / "06_micro_model_category_heatmap.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def heatmap(merged: pd.DataFrame) -> Path:
    small = merged[merged["category"] != "micro_control_identical"].copy()
    pivot = small.pivot_table(
        index="prompt_a",
        columns="category",
        values="semantic_cosine_distance",
        aggfunc="mean",
    )
    col_order = (
        pd.read_csv(RANK_DIR / "category_summary.csv")
        .query("category != 'micro_control_identical'")
        .sort_values("semantic_mean", ascending=False)["category"]
        .tolist()
    )
    pivot = pivot[col_order]
    row_order = pivot.mean(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[row_order]
    labels_y = [textwrap.shorten(x, width=48, placeholder="...") for x in pivot.index]
    labels_x = [clean_category(c).replace(" ", "\n") for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(15.5, 9.5), facecolor=PAPER)
    ax.set_facecolor(PAPER)
    data = pivot.fillna(0).values
    im = ax.imshow(data, aspect="auto", cmap="Reds", vmin=0, vmax=max(0.16, float(np.nanmax(data))))
    ax.set_yticks(np.arange(len(labels_y)), labels_y, fontsize=8)
    ax.set_xticks(np.arange(len(labels_x)), labels_x, rotation=0, ha="center", fontsize=8)
    fig.text(0.07, 0.965, "Where tiny edits branch: prompt x perturbation heatmap", fontsize=20, weight="bold", color=INK)
    fig.text(
        0.07,
        0.93,
        "Each cell is semantic distance between outputs for a single base prompt and tiny edit category.",
        fontsize=11,
        color=MUTED,
    )
    ax.set_xticks(np.arange(-0.5, len(labels_x), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels_y), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.spines[:].set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.018)
    cbar.set_label("Output distance", fontsize=10)
    fig.tight_layout(rect=[0.02, 0.03, 1, 0.90], pad=1.8)
    out = OUT_DIR / "02_prompt_x_edit_heatmap.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def trajectory_curves(merged: pd.DataFrame, compact: bool = False) -> Path:
    chosen = [
        ("micro_trailing_space_0000", "trailing space (inert)", GREEN),
        ("micro_double_internal_space_0284", "double internal space", ORANGE),
        ("micro_parenthesize_word_0319", "parenthesized word", RED),
        ("micro_line_wrap_0473", "line wrap", BLUE),
    ]
    curves = pd.read_json(RUN_DIR / "curves.jsonl", lines=True)

    fig, (ax, ax2) = plt.subplots(
        1,
        2,
        figsize=(16, 5.0) if compact else (15.5, 6.6),
        facecolor=PAPER,
        gridspec_kw={"width_ratios": [2.15, 1]},
    )
    ax.set_facecolor(PAPER)
    for pair_id, label, color in chosen:
        g = curves[curves["pair_id"] == pair_id].sort_values("t")
        if g.empty:
            continue
        ax.plot(g["t"], g["token_edit_distance_norm"], label=label, color=color, linewidth=3)
    t = np.arange(1, 65)
    ax.plot(t, np.minimum(1, 2 / t), color="#8a8f98", linewidth=2, linestyle="--", label="one-token offset artifact")
    if not compact:
        fig.text(0.06, 0.965, "Token-path divergence, with an alignment sanity check", fontsize=20, weight="bold", color=INK)
        fig.text(
            0.06,
            0.925,
            "Levenshtein aligns insertions/deletions. A pure one-token offset would fall toward zero, not stay near one.",
            fontsize=11,
            color=MUTED,
        )
    ax.set_xlabel("Generated token position")
    ax.set_ylabel("Normalized token edit distance")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(1, 64)
    ax.grid(color=GRID)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right", fontsize=9)

    selected = merged[merged["pair_id"].isin([x[0] for x in chosen])].copy()
    selected["label"] = selected["pair_id"].map({pid: label for pid, label, _ in chosen})
    selected["color"] = selected["pair_id"].map({pid: color for pid, _, color in chosen})
    selected = selected.set_index("pair_id").loc[[x[0] for x in chosen]].reset_index()
    ax2.set_facecolor(PAPER)
    ax2.barh(selected["label"], selected["semantic_cosine_distance"], color=selected["color"], height=0.55)
    ax2.invert_yaxis()
    ax2.set_title("Endpoint semantic distance", loc="left", fontsize=12, weight="bold")
    ax2.set_xlabel("Embedding distance")
    ax2.grid(axis="x", color=GRID)
    ax2.spines[["top", "right", "left"]].set_visible(False)
    ax2.tick_params(axis="y", length=0, labelsize=9)
    ax2.set_xlim(0, max(selected["semantic_cosine_distance"].max() * 1.2, 0.05))
    for y, v in enumerate(selected["semantic_cosine_distance"]):
        label = "0" if v < 1e-5 else f"{v:.3f}"
        ax2.text(max(v, 0.002) + 0.003, y, label, va="center", fontsize=9, color=INK)

    rect = [0.03, 0.02, 1, 0.98] if compact else [0.03, 0.03, 1, 0.88]
    fig.tight_layout(rect=rect, pad=1.2)
    out = OUT_DIR / ("03_branching_trajectories_compact.png" if compact else "03_branching_trajectories.png")
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def example_cards() -> Path:
    row = pd.read_csv(RANK_DIR / "worst_examples.csv").iloc[0]
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=PAPER)
    ax.axis("off")
    fig.text(0.04, 0.955, "A single line break changes the answer path", fontsize=24, weight="bold", color=INK)
    fig.text(
        0.04,
        0.915,
        f"{clean_category(row['category'])}; semantic distance={row['semantic_cosine_distance']:.3f}; common prefix={int(row['common_prefix_tokens'])} tokens",
        fontsize=12,
        color=MUTED,
    )

    def box(x: float, y: float, w: float, h: float, title: str, text: str, color: str, mono: bool = False) -> None:
        rect = plt.Rectangle((x, y), w, h, transform=ax.transAxes, facecolor="white", edgecolor="#d6d6d6", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.02, y + h - 0.055, title, transform=ax.transAxes, fontsize=12, weight="bold", color=MUTED)
        ax.text(
            x + 0.02,
            y + h - 0.12,
            wrap(text, 72 if mono else 78, 8),
            transform=ax.transAxes,
            fontsize=12 if mono else 11,
            family="monospace" if mono else "sans-serif",
            color=color,
            va="top",
        )

    box(0.04, 0.63, 0.43, 0.20, "Prompt A", str(row["prompt_a"]), INK, True)
    box(0.53, 0.63, 0.43, 0.20, "Prompt B", str(row["prompt_b"]), RED, True)
    box(0.04, 0.14, 0.43, 0.42, "Continuation A", str(row["generated_text_a"]), INK, False)
    box(0.53, 0.14, 0.43, 0.42, "Continuation B", str(row["generated_text_b"]), BLUE, False)
    ax.annotate("", xy=(0.53, 0.73), xytext=(0.47, 0.73), arrowprops=dict(arrowstyle="->", color=MUTED, lw=2), xycoords=ax.transAxes)
    ax.text(0.50, 0.755, "visually tiny", transform=ax.transAxes, fontsize=10, color=MUTED, ha="center")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    out = OUT_DIR / "04_branching_examples.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def human_model_view() -> Path:
    row = pd.read_csv(RANK_DIR / "worst_examples.csv").iloc[0]
    fig, ax = plt.subplots(figsize=(15, 7.5), facecolor=PAPER)
    ax.axis("off")
    ax.text(0.03, 0.94, "The human sees the same request. The model gets a different seed.", fontsize=22, weight="bold", color=INK)
    ax.text(0.03, 0.88, "One line break changed the deterministic continuation for Qwen3.5 0.8B.", fontsize=12, color=MUTED)

    boxes = [
        (0.04, 0.55, 0.43, 0.25, "Prompt A", str(row["prompt_a"]), INK),
        (0.53, 0.55, 0.43, 0.25, "Prompt B", str(row["prompt_b"]), RED),
        (0.04, 0.13, 0.43, 0.32, "Continuation A", str(row["generated_text_a"]), INK),
        (0.53, 0.13, 0.43, 0.32, "Continuation B", str(row["generated_text_b"]), BLUE),
    ]
    for x, y, w, h, title, text, color in boxes:
        rect = plt.Rectangle((x, y), w, h, transform=ax.transAxes, facecolor="white", edgecolor="#d9d9d9", linewidth=1.4)
        ax.add_patch(rect)
        ax.text(x + 0.02, y + h - 0.06, title, transform=ax.transAxes, fontsize=11, weight="bold", color=MUTED)
        ax.text(x + 0.02, y + h - 0.13, wrap(text, 58, 7), transform=ax.transAxes, fontsize=11, family="monospace" if "Prompt" in title else "sans-serif", color=color, va="top")
    ax.annotate("", xy=(0.53, 0.685), xytext=(0.47, 0.685), arrowprops=dict(arrowstyle="->", color=MUTED, lw=2), xycoords=ax.transAxes)
    ax.text(0.485, 0.71, "tiny edit", transform=ax.transAxes, fontsize=10, color=MUTED, ha="center")
    out = OUT_DIR / "05_human_vs_model_view.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def make_canvas(paths: list[Path]) -> Path:
    imgs = [Image.open(p).convert("RGB") for p in paths]
    thumbs = []
    thumb_w = 900
    for img in imgs:
        ratio = thumb_w / img.width
        thumbs.append(img.resize((thumb_w, int(img.height * ratio))))
    gap = 28
    label_h = 34
    cols = 2
    rows = int(np.ceil(len(thumbs) / cols))
    cell_h = max(t.height for t in thumbs) + label_h
    canvas = Image.new("RGB", (cols * thumb_w + (cols + 1) * gap, rows * cell_h + (rows + 1) * gap), PAPER)
    draw = ImageDraw.Draw(canvas)
    for i, (path, img) in enumerate(zip(paths, thumbs)):
        r, c = divmod(i, cols)
        x = gap + c * (thumb_w + gap)
        y = gap + r * (cell_h + gap)
        draw.text((x, y), path.name, fill=INK)
        canvas.paste(img, (x, y + label_h))
    out = OUT_DIR / "00_micro_visual_gallery.png"
    canvas.save(out)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged = load_merged()
    paths = [
        model_category_heatmap(),
        category_bar(),
        category_bar(compact=True),
        heatmap(merged),
        trajectory_curves(merged),
        trajectory_curves(merged, compact=True),
        example_cards(),
        human_model_view(),
    ]
    canvas = make_canvas(paths)
    print("Wrote:")
    for path in [canvas, *paths]:
        print(path)


if __name__ == "__main__":
    main()
