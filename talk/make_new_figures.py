"""Generate four matplotlib figures for the talk deck.

Outputs into talk/concept_images/generated/.
"""
from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np

OUT = pathlib.Path(__file__).parent / "concept_images" / "generated"
OUT.mkdir(parents=True, exist_ok=True)

ACCENT = "#c8402c"
DARK = "#101114"
BG = "#fafafa"
GREY = "#888"
MUTED = "#c9c9c9"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.edgecolor": "#333",
    "axes.labelcolor": "#1a1a1a",
    "xtick.color": "#333",
    "ytick.color": "#333",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
})


# --------------------------------------------------------------------------
# Slide 3 — hybrid sequential system: continuous flow feeding discrete branch
# --------------------------------------------------------------------------
def slide3_hybrid_system() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 8.2), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def panel(x: float, y: float, w: float, h: float, title: str, subtitle: str) -> None:
        box = plt.Rectangle(
            (x, y), w, h,
            facecolor="white",
            edgecolor="#dedede",
            linewidth=1.6,
            transform=ax.transAxes,
            zorder=0,
        )
        ax.add_patch(box)
        ax.text(x + 0.035, y + h - 0.065, title, transform=ax.transAxes,
                fontsize=16, weight="bold", color="#333", va="top")
        ax.text(x + 0.035, y + h - 0.112, subtitle, transform=ax.transAxes,
                fontsize=10.5, color="#777", va="top")

    # Top: continuous activations/logits flowing through layers.
    panel(0.08, 0.57, 0.84, 0.33, "continuous", "activations, logits, KV cache")
    xs = np.linspace(0.14, 0.86, 220)
    for i, offset in enumerate([0.725, 0.68, 0.635, 0.59]):
        y = offset + 0.011 * np.sin(8 * xs + i * 0.8) + 0.006 * np.cos(17 * xs + i)
        ax.plot(xs, y, color=ACCENT, alpha=0.40, lw=2.1, transform=ax.transAxes)
    ax.text(0.50, 0.765, "smooth, differentiable flow", transform=ax.transAxes,
            fontsize=10, color="#777", ha="center", style="italic")
    ax.text(0.18, 0.595, "layer 1", transform=ax.transAxes, fontsize=9, color="#aaa")
    ax.text(0.76, 0.595, "layer N", transform=ax.transAxes, fontsize=9, color="#aaa")

    # Middle: argmax boundary.
    ax.annotate(
        "", xy=(0.50, 0.49), xytext=(0.50, 0.57),
        xycoords=ax.transAxes,
        arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.8),
    )
    ax.text(
        0.50, 0.525, "argmax",
        transform=ax.transAxes,
        fontsize=11,
        weight="bold",
        color=ACCENT,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.28", fc=BG, ec=ACCENT, lw=1.2),
    )

    # Bottom: discrete branch process.
    panel(0.08, 0.10, 0.84, 0.34, "discrete", "tokens, branch decisions")
    nodes = {
        "prompt": (0.22, 0.24),
        "a1": (0.40, 0.30), "a2": (0.40, 0.19),
        "b1": (0.58, 0.35), "b2": (0.58, 0.26), "b3": (0.58, 0.15),
        "c1": (0.76, 0.39), "c2": (0.76, 0.33), "c3": (0.76, 0.25),
        "c4": (0.76, 0.18), "c5": (0.76, 0.12),
    }
    edges = [
        ("prompt", "a1", "#777"), ("prompt", "a2", ACCENT),
        ("a1", "b1", "#777"), ("a1", "b2", "#777"),
        ("a2", "b2", ACCENT), ("a2", "b3", ACCENT),
        ("b1", "c1", "#777"), ("b1", "c2", "#777"),
        ("b2", "c3", "#777"), ("b2", "c4", ACCENT),
        ("b3", "c4", ACCENT), ("b3", "c5", ACCENT),
    ]
    for left, right, color in edges:
        x0, y0 = nodes[left]
        x1, y1 = nodes[right]
        ax.plot([x0, x1], [y0, y1], color=color, lw=1.6, alpha=0.95, transform=ax.transAxes)
    for name, (x, y) in nodes.items():
        color = ACCENT if name in {"a2", "b2", "b3", "c4", "c5"} else "#333"
        size = 54 if name == "prompt" else 32
        ax.scatter([x], [y], s=size, color=color, zorder=3, transform=ax.transAxes)
    ax.text(0.15, 0.225, "prompt", transform=ax.transAxes, fontsize=9, color="#777")
    ax.text(0.52, 0.375, "trajectory A", transform=ax.transAxes, fontsize=9.5, color="#777", style="italic")
    ax.text(0.50, 0.105, "trajectory B — one flipped argmax", transform=ax.transAxes,
            fontsize=9.5, color=ACCENT, style="italic")

    fig.text(
        0.5,
        0.035,
        "Continuous state crosses discrete token boundaries.",
        ha="center",
        fontsize=11,
        color="#555",
        style="italic",
    )
    fig.savefig(OUT / "slide3_hybrid_system.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Slide 10 — compounding bar chart: 1.32^n growth over 40 layers
# --------------------------------------------------------------------------
def slide10_compounding() -> None:
    fig, ax = plt.subplots(figsize=(11, 5.4), dpi=200)
    layers = np.arange(0, 40)
    growth = 1.32 ** layers

    # Highlight the tiny input at the left and the massive output at the right
    bars = ax.bar(layers, growth, color=ACCENT, width=0.78, edgecolor="none")
    bars[0].set_color("#4a4a4a")

    ax.set_yscale("log")
    ax.set_xlabel("Transformer layer (depth)", fontsize=12)
    ax.set_ylabel("Magnitude of perturbation (log)", fontsize=12)
    ax.set_title(
        "Perturbations compound through depth at ~1.32× per layer  (Li et al. 2025, Qwen2-14B)",
        fontsize=13, color="#1a1a1a", pad=14, loc="left",
    )

    # Input annotation
    ax.annotate(
        "input ≈ 0.0009%\nof the final residual",
        xy=(0, 1.0), xytext=(3.5, 6),
        fontsize=11, color="#333",
        arrowprops=dict(arrowstyle="->", color="#333", lw=1.1),
    )
    # Output annotation
    ax.annotate(
        f"after 40 layers:\n×{growth[-1]:,.0f}",
        xy=(39, growth[-1]), xytext=(26, growth[-1] * 0.05),
        fontsize=12, color=ACCENT, weight="bold",
        arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.2),
    )

    ax.grid(axis="y", linestyle=":", color="#ccc", alpha=0.7)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(OUT / "slide10_compounding.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Slide 12 — Qwen forest plot: mean + 95% CI per size
# --------------------------------------------------------------------------
def slide12_qwen_forest() -> None:
    models = ["Qwen3.5 0.8B", "Qwen3.5 2B", "Qwen3.5 4B", "Qwen3.5 9B"]
    means  = [0.089, 0.073, 0.034, 0.037]
    ci_lo  = [0.048, 0.039, 0.018, 0.016]
    ci_hi  = [0.137, 0.115, 0.053, 0.061]
    pvals  = ["p<0.001", "p=0.012", "—", "p=0.78"]

    fig, ax = plt.subplots(figsize=(11, 4.4), dpi=200)
    y = np.arange(len(models))[::-1]  # 9B at top, 0.8B at bottom

    colors = ["#c8402c", "#c8402c", "#4a4a4a", "#4a4a4a"]
    for yi, m, lo, hi, c in zip(y, means, ci_lo, ci_hi, colors):
        ax.hlines(yi, lo, hi, color=c, lw=2.4)
        ax.plot(m, yi, "o", color=c, markersize=9, zorder=3)

    for yi, p in zip(y, pvals):
        ax.text(0.155, yi, p, fontsize=11, color="#333", va="center")

    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=12)
    ax.set_xlabel("Mean semantic divergence (95% CI)", fontsize=12)
    ax.set_xlim(-0.005, 0.185)
    ax.axvline(0.034, color="#999", lw=0.8, linestyle=":", zorder=0)
    ax.text(0.034, -0.55, "baseline (4B)", fontsize=9, color="#888", ha="center")

    ax.set_title(
        "Within-Qwen sensitivity  (n=24 prompt pairs, same family)",
        fontsize=13, color="#1a1a1a", loc="left", pad=12,
    )
    ax.grid(axis="x", linestyle=":", color="#ccc", alpha=0.7)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(OUT / "slide12_qwen_forest.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Slide 15 — stability ≠ responsiveness scatter
# X: perturbation divergence (low = "looks stable")
# Y: drift from BF16 baseline on identical prompts (high = "collapsed")
# --------------------------------------------------------------------------
def slide15_collapse() -> None:
    # Qwen 0.8B quant sweep, illustrative numbers from the speaker notes.
    points = [
        ("BF16",  0.138, 0.000, "#4a4a4a"),
        ("8-bit", 0.120, 0.041, "#4a4a4a"),
        ("4-bit", 0.091, 0.132, ACCENT),
    ]

    fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=200)

    # Quadrant shading
    xmax, ymax = 0.18, 0.18
    ax.axhspan(0.08, ymax, xmin=0, xmax=0.55 / xmax, color=ACCENT, alpha=0.08, zorder=0)
    ax.text(0.045, 0.162, "COLLAPSED, NOT ROBUST", fontsize=10.5, color=ACCENT, weight="bold")
    ax.text(0.145, 0.005, "responsive &\nconsistent",
            fontsize=10, color="#555", ha="center")
    ax.text(0.025, 0.005, "looks 'stable'\nbut not collapsed",
            fontsize=10, color="#555", ha="left")

    for label, x, y, c in points:
        ax.plot(x, y, "o", color=c, markersize=14, zorder=3)
        ax.annotate(
            f"  {label}", xy=(x, y), fontsize=12, color=c, weight="bold",
            va="center", ha="left",
        )

    # Arrow from BF16 -> 4-bit showing the "improvement" illusion
    ax.annotate(
        "", xy=(0.091, 0.128), xytext=(0.136, 0.003),
        arrowprops=dict(arrowstyle="->", color="#999", lw=1.2,
                        connectionstyle="arc3,rad=-0.25"),
    )
    ax.text(0.108, 0.055, "perturbation score\nimproves...",
            fontsize=10, color="#666", ha="center", style="italic")
    ax.text(0.048, 0.108, "...but model drifts\nfrom its baseline",
            fontsize=10, color=ACCENT, ha="left", style="italic")

    ax.set_xlabel("Perturbation divergence  (lower = 'looks stable')", fontsize=12)
    ax.set_ylabel("Drift from BF16 baseline  (higher = collapsed)", fontsize=12)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_title(
        "Stability ≠ responsiveness  —  Qwen 0.8B quant sweep",
        fontsize=13, color="#1a1a1a", loc="left", pad=12,
    )
    ax.grid(linestyle=":", color="#ccc", alpha=0.7)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(OUT / "slide15_collapse.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Slide 18 — correlation bar chart
# --------------------------------------------------------------------------
def slide18_correlations() -> None:
    labels = [
        "Top-1 probability",
        "Top-1 margin (logit)",
        "Full-vocab JS divergence",
        "Top-1 flip rate",
    ]
    rs = [-0.84, -0.39, -0.10, 0.57]
    colors = [ACCENT, "#e08878", "#b0b0b0", ACCENT]

    # Sort by magnitude
    order = np.argsort(-np.abs(rs))
    labels = [labels[i] for i in order]
    rs = [rs[i] for i in order]
    colors = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=200)
    y = np.arange(len(labels))[::-1]
    bars = ax.barh(y, rs, color=colors, edgecolor="none", height=0.6)

    ax.axvline(0, color="#333", lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Pearson r  with  512-token semantic divergence", fontsize=12)
    ax.set_title(
        "Boundary beats bulk  —  which prompt-end logit signal predicts downstream drift?  (n=20)",
        fontsize=12.5, color="#1a1a1a", loc="left", pad=12,
    )

    # Value labels
    for yi, r in zip(y, rs):
        offset = 0.02 if r >= 0 else -0.02
        ha = "left" if r >= 0 else "right"
        ax.text(r + offset, yi, f"{r:+.2f}", fontsize=12, color="#1a1a1a",
                va="center", ha=ha, weight="bold")

    # Annotate the top row as the headline
    ax.annotate(
        "more confident =\nless downstream drift",
        xy=(-0.84, 3), xytext=(-0.95, 3.4),
        fontsize=10, color=ACCENT, style="italic",
    )
    ax.annotate(
        "bulk distribution shift\npredicts ~nothing",
        xy=(-0.10, 1), xytext=(0.10, 1.25),
        fontsize=10, color="#666", style="italic",
    )

    ax.grid(axis="x", linestyle=":", color="#ccc", alpha=0.7)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(OUT / "slide18_correlations.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    slide3_hybrid_system()
    slide10_compounding()
    slide12_qwen_forest()
    slide15_collapse()
    slide18_correlations()
    print("wrote figures to", OUT)
