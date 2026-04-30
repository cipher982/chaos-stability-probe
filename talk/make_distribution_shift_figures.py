from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch


OUT_DIR = Path(__file__).parent / "concept_images"
BG = "#fafafa"
DARK = "#1a1a1a"
MUTED = "#6f6f6f"
GRID = "#d8d8d8"
RED = "#c8402c"
TEAL = "#287b7f"


def normal_density(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def gaussian_2d(x, y, mux, muy, sx, sy, rho=0.0):
    xn = (x - mux) / sx
    yn = (y - muy) / sy
    z = xn**2 - 2 * rho * xn * yn + yn**2
    return np.exp(-z / (2 * (1 - rho**2)))


def save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / name, dpi=220, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def style_ax(ax):
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


def density_shift():
    fig, ax = plt.subplots(figsize=(7.0, 4.1), facecolor=BG)
    style_ax(ax)

    x = np.linspace(-3.0, 3.0, 700)
    a = normal_density(x, -1.15, 0.33)
    b = normal_density(x, 1.15, 0.33)

    ax.fill_between(x, 0, a, color=RED, alpha=0.24)
    ax.plot(x, a, color=RED, lw=3)
    ax.fill_between(x, 0, b, color=TEAL, alpha=0.24)
    ax.plot(x, b, color=TEAL, lw=3)

    ax.axhline(0, color="#8d8d8d", lw=1.2)
    ax.vlines([-1.15, 1.15], 0, 1.35, colors=[RED, TEAL], linestyles="dotted", lw=1.4)

    arrow = FancyArrowPatch(
        (-0.78, 1.16),
        (0.78, 1.16),
        arrowstyle="-|>",
        mutation_scale=18,
        lw=2.2,
        color=DARK,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arrow)
    ax.text(
        0,
        1.28,
        "tiny input change shifts the response distribution",
        ha="center",
        va="bottom",
        color=DARK,
        fontsize=13,
        weight="semibold",
    )

    ax.text(-1.15, 1.55, "Prompt A", ha="center", color=RED, fontsize=15, weight="bold")
    ax.text(1.15, 1.55, "Prompt A + space", ha="center", color=TEAL, fontsize=15, weight="bold")
    ax.text(-1.15, -0.08, "Foundation basin", ha="center", va="top", color=DARK, fontsize=13)
    ax.text(1.15, -0.08, "Hyperion basin", ha="center", va="top", color=DARK, fontsize=13)

    ax.text(
        -2.65,
        0.42,
        "sampling\nstays local",
        ha="left",
        va="center",
        color=MUTED,
        fontsize=12,
    )
    for dx, dy in [(-0.10, 0.16), (0.03, 0.21), (0.12, 0.14), (-0.02, 0.09)]:
        ax.plot(-1.15 + dx, dy, "o", color=RED, alpha=0.55, ms=5)

    ax.set_xlim(-2.85, 2.85)
    ax.set_ylim(-0.22, 1.72)
    save(fig, "distribution-shift-density.png")


def density_shift_slide():
    fig, ax = plt.subplots(figsize=(6.0, 3.75), facecolor=BG)
    style_ax(ax)

    x = np.linspace(-3.0, 3.0, 700)
    a = normal_density(x, -1.15, 0.34)
    b = normal_density(x, 1.15, 0.34)

    ax.fill_between(x, 0, a, color=RED, alpha=0.22)
    ax.plot(x, a, color=RED, lw=3.2)
    ax.fill_between(x, 0, b, color=TEAL, alpha=0.22)
    ax.plot(x, b, color=TEAL, lw=3.2)
    ax.axhline(0, color="#8d8d8d", lw=1.2)
    ax.vlines([-1.15, 1.15], 0, 1.0, colors=[RED, TEAL], linestyles="dotted", lw=1.4)

    ax.annotate(
        "",
        xy=(0.69, 1.06),
        xytext=(-0.69, 1.06),
        arrowprops=dict(arrowstyle="-|>", color=DARK, lw=2.3, mutation_scale=18),
    )
    ax.text(0, 1.17, "tiny input change", ha="center", va="bottom", color=DARK, fontsize=14, weight="bold")

    ax.text(-1.15, 1.42, "Prompt A", ha="center", color=RED, fontsize=16, weight="bold")
    ax.text(1.15, 1.42, "A + space", ha="center", color=TEAL, fontsize=16, weight="bold")
    ax.text(-1.15, -0.08, "Foundation\nbasin", ha="center", va="top", color=DARK, fontsize=13)
    ax.text(1.15, -0.08, "Hyperion\nbasin", ha="center", va="top", color=DARK, fontsize=13)

    ax.text(-2.55, 0.38, "sampling\nstays local", ha="left", va="center", color=MUTED, fontsize=11)
    for dx, dy in [(-0.10, 0.15), (0.02, 0.21), (0.13, 0.14), (-0.02, 0.09)]:
        ax.plot(-1.15 + dx, dy, "o", color=RED, alpha=0.55, ms=4.8)

    ax.set_xlim(-2.75, 2.75)
    ax.set_ylim(-0.27, 1.58)
    save(fig, "distribution-shift-basins.png")


def contour_shift():
    fig, ax = plt.subplots(figsize=(6.7, 4.4), facecolor=BG)
    style_ax(ax)

    gx = np.linspace(-3, 3, 260)
    gy = np.linspace(-2.0, 2.0, 220)
    xx, yy = np.meshgrid(gx, gy)
    za = gaussian_2d(xx, yy, -1.05, -0.15, 0.52, 0.34, rho=-0.35)
    zb = gaussian_2d(xx, yy, 1.05, 0.28, 0.50, 0.36, rho=0.30)

    ax.contourf(xx, yy, za, levels=[0.08, 0.20, 0.38, 0.62, 1.01], colors=[RED], alpha=0.10)
    ax.contour(xx, yy, za, levels=[0.20, 0.38, 0.62], colors=[RED], linewidths=[1.2, 1.7, 2.4])
    ax.contourf(xx, yy, zb, levels=[0.08, 0.20, 0.38, 0.62, 1.01], colors=[TEAL], alpha=0.11)
    ax.contour(xx, yy, zb, levels=[0.20, 0.38, 0.62], colors=[TEAL], linewidths=[1.2, 1.7, 2.4])

    ax.plot([-2.45, 2.45], [-1.15, 1.25], color=GRID, lw=1.2, ls="--")
    ax.text(-1.95, -1.45, "Foundation\nattractor", ha="center", color=RED, fontsize=13, weight="bold")
    ax.text(1.86, 1.40, "Hyperion\nattractor", ha="center", color=TEAL, fontsize=13, weight="bold")

    ax.annotate(
        "",
        xy=(0.72, 0.16),
        xytext=(-0.72, -0.04),
        arrowprops=dict(arrowstyle="-|>", color=DARK, lw=2.2, mutation_scale=18),
    )
    ax.text(0, 0.42, "same meaning,\nnew basin", ha="center", color=DARK, fontsize=13, weight="semibold")

    ax.text(-1.05, -0.84, "Prompt A", ha="center", color=RED, fontsize=14, weight="bold")
    ax.text(1.05, 0.92, "Prompt A + space", ha="center", color=TEAL, fontsize=14, weight="bold")

    ax.set_xlim(-2.75, 2.75)
    ax.set_ylim(-1.75, 1.75)
    save(fig, "distribution-shift-contours.png")


def recommendation_bars():
    fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor=BG)
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_visible(False)

    labels = ["Foundation", "Hyperion", "Dune Messiah", "Left Hand", "Neuromancer"]
    y = np.arange(len(labels))
    a = np.array([0.55, 0.06, 0.18, 0.12, 0.09])
    b = np.array([0.05, 0.57, 0.13, 0.11, 0.14])

    ax.barh(y + 0.18, a, height=0.28, color=RED, alpha=0.82, label="Prompt A")
    ax.barh(y - 0.18, b, height=0.28, color=TEAL, alpha=0.82, label="Prompt A + space")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12, color=DARK)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.66)
    ax.tick_params(axis="x", labelsize=10, colors=MUTED)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_xlabel("probability mass over defensible recommendations", fontsize=11, color=MUTED)
    ax.legend(frameon=False, loc="lower right", fontsize=11)

    ax.text(
        0.0,
        -0.85,
        "A tiny prompt perturbation can reorder the whole answer distribution",
        color=DARK,
        fontsize=14,
        weight="semibold",
    )
    save(fig, "distribution-shift-bars.png")


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": BG,
            "savefig.facecolor": BG,
        }
    )
    density_shift()
    density_shift_slide()
    contour_shift()
    recommendation_bars()


if __name__ == "__main__":
    main()
