"""MVP synthetic-data sketches for the LLM-stability talk.

Generates 6 static matplotlib PNGs that show the *shape* of each visualization
idea, contrasting a "brittle" (0.8B-like) and "stable" (4B-like) model
responding to a one-word-perturbed prompt pair, 64 tokens each.

Data is entirely synthetic and inline. Seeds fixed for reproducibility.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

OUT = "/Users/davidrose/git/chaos/talk/mvp_plots"
T = 64
X = np.arange(1, T + 1)
FORK = 20  # stable model's reasoning-scaffold length

plt.rcParams.update({
    "figure.facecolor": "#111318",
    "axes.facecolor": "#111318",
    "axes.edgecolor": "#888",
    "axes.labelcolor": "#ddd",
    "xtick.color": "#bbb",
    "ytick.color": "#bbb",
    "text.color": "#eee",
    "axes.titlecolor": "#eee",
    "axes.grid": True,
    "grid.color": "#2a2d35",
    "grid.linewidth": 0.6,
    "font.size": 10,
})


def suptitle(fig, title, sub):
    fig.suptitle(title, fontsize=14, y=0.98)
    fig.text(0.5, 0.915, sub, ha="center", fontsize=9, color="#aaa", style="italic")


def smooth(arr, k=3):
    pad = np.concatenate([np.full(k, arr[0]), arr, np.full(k, arr[-1])])
    return np.convolve(pad, np.ones(2 * k + 1) / (2 * k + 1), mode="same")[k:-k]


# -------------------------------------------------- 1. KL RIVER
def plot_kl_river():
    rng = np.random.default_rng(1)
    # brittle: jumps to 3-5 by t=3-5, stays high with spikes
    brittle = np.zeros(T)
    ramp = np.linspace(0, 4.2, 5)
    brittle[:5] = ramp + rng.normal(0, 0.3, 5)
    brittle[5:] = 3.8 + rng.normal(0, 0.7, T - 5) + 0.9 * np.abs(rng.standard_normal(T - 5))
    brittle = np.clip(smooth(brittle, 1), 0, None)

    # stable: flat near 0 for t<20, then gentle rise to 1-2
    stable = np.zeros(T)
    stable[:FORK] = np.abs(rng.normal(0, 0.05, FORK))
    rise = np.linspace(0, 1.6, T - FORK) + rng.normal(0, 0.15, T - FORK)
    stable[FORK:] = np.clip(rise, 0, None)
    stable = smooth(stable, 2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, y, name in zip(axes, [brittle, stable], ["Brittle (0.8B)", "Stable (4B)"]):
        ax.fill_between(X, 0, y, color="#ff6b6b" if "Brittle" in name else "#4dd2a0", alpha=0.55)
        ax.plot(X, y, color="#ff9e9e" if "Brittle" in name else "#8ef0c4", lw=1.2)
        ax.set_title(name)
        ax.set_xlabel("token index")
        ax.set_ylim(0, 6)
    axes[0].set_ylabel("divergence between the two probability distributions (nats)")
    suptitle(fig, "KL River: how far apart are the two next-token distributions?",
             "0.8B decouples within 3-5 tokens; 4B stays coupled for ~20, then softly rises.")
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(f"{OUT}/01_kl_river.png", dpi=130)
    plt.close()


# -------------------------------------------------- 2. RANK OF SIBLING
def plot_rank_of_sibling():
    rng = np.random.default_rng(2)
    # brittle: rank=1 for t=1, then scatter 1..50000 from t=2+
    brittle = np.ones(T)
    for t in range(1, T):
        # heavy-tailed: log-uniform between 1 and 50000
        brittle[t] = 10 ** rng.uniform(0, 4.7)
    # stable: rank=1 for t<20, then mostly 2-50 with occasional bigger
    stable = np.ones(T)
    for t in range(FORK, T):
        base = 10 ** rng.uniform(0.1, 1.7)
        if rng.random() < 0.12:
            base = 10 ** rng.uniform(1.5, 3)
        stable[t] = base

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, y, name in zip(axes, [brittle, stable], ["Brittle (0.8B)", "Stable (4B)"]):
        c = "#ff6b6b" if "Brittle" in name else "#4dd2a0"
        ax.scatter(X, y, s=24, color=c, edgecolor="white", linewidth=0.3, alpha=0.9)
        ax.axhline(1, color="#888", ls=":", lw=0.8)
        ax.set_yscale("log")
        ax.set_ylim(0.8, 1e5)
        ax.set_title(name)
        ax.set_xlabel("token index")
    axes[0].set_ylabel("rank of B's chosen token in A's distribution (log)")
    suptitle(fig, "Rank-of-sibling: how surprising is B's token under A's model?",
             "Rank=1 means identical prediction. 4B holds rank 1 for ~20 tokens; 0.8B scatters instantly.")
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(f"{OUT}/02_rank_of_sibling.png", dpi=130)
    plt.close()


# -------------------------------------------------- 3. LOG-PROB PATHS
def plot_logprob_paths():
    rng = np.random.default_rng(3)

    def run(per_tok_mean, per_tok_sd, seed):
        r = np.random.default_rng(seed)
        inc = r.normal(per_tok_mean, per_tok_sd, T)
        return np.cumsum(inc)

    # brittle: -3 to -6 / tok, very jittery (wandering a plain)
    b_A = run(-4.2, 2.4, 10)
    b_B = run(-4.5, 2.6, 11)
    # stable: -0.5 to -2 / tok, smooth (confident ridge)
    s_A = run(-1.1, 0.25, 12)
    s_B = run(-1.2, 0.28, 13)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    # per-token increments as faint stems behind the cumulative lines
    def per_tok(cum):
        return np.diff(np.concatenate([[0], cum]))
    for ax, (A, B), name in zip(
        axes, [(b_A, b_B), (s_A, s_B)], ["Brittle (0.8B)", "Stable (4B)"]
    ):
        # scale per-tok increments (typ -1..-6) up to be visible on -300..0 canvas
        ptA = per_tok(A) * 12
        ptB = per_tok(B) * 12
        ax.vlines(X - 0.15, 0, ptA, color="#ffd166", alpha=0.35, lw=1.2)
        ax.vlines(X + 0.15, 0, ptB, color="#7aa2ff", alpha=0.35, lw=1.2)
        ax.plot(X, A, color="#ffd166", lw=2.2, label="path A (cumulative)")
        ax.plot(X, B, color="#7aa2ff", lw=2.2, label="path B (cumulative)")
        ax.axhline(0, color="#555", lw=0.6)
        ax.set_title(name)
        ax.set_xlabel("token index")
        ax.legend(loc="lower left", facecolor="#1a1d24", edgecolor="#444", fontsize=8)
    axes[0].set_ylabel("cumulative log-probability (nats)\nfaint stems = per-token log-prob × 12")
    axes[0].set_ylim(-320, 20)
    suptitle(fig, "Running log-prob: is the model walking a ridge or wandering a plain?",
             "0.8B generates from a flat landscape (low per-token confidence); 4B walks a confident ridge.")
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(f"{OUT}/03_logprob_paths.png", dpi=130)
    plt.close()


# -------------------------------------------------- 4. HIDDEN UMAP
def plot_hidden_umap():
    def walk(seed, step=0.6):
        r = np.random.default_rng(seed)
        steps = r.normal(0, step, (T, 2))
        return np.cumsum(steps, axis=0)

    # brittle: diverge immediately
    bA = walk(21, 0.7)
    bB = walk(22, 0.7) + np.array([2.0, -1.5])

    # stable: overlap for FORK tokens, then soft separation
    # use a gentler base walk so scaffold & split are visible on the same axes
    base = walk(31, 0.25)
    sA = base.copy()
    sB = base.copy()
    rng = np.random.default_rng(33)
    # identical-ish for t<FORK (tiny jitter)
    sB[:FORK] += rng.normal(0, 0.06, (FORK, 2))
    # after fork, both drift in gentle but distinct directions
    post = T - FORK
    driftA = np.cumsum(rng.normal([0.15, 0.12], 0.18, (post, 2)), axis=0)
    driftB = np.cumsum(rng.normal([0.18, -0.14], 0.18, (post, 2)), axis=0)
    sA[FORK:] += driftA
    sB[FORK:] += driftB

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    cmap = plt.cm.viridis
    for ax, (A, B), name in zip(
        axes, [(bA, bB), (sA, sB)], ["Brittle (0.8B)", "Stable (4B)"]
    ):
        ax.plot(A[:, 0], A[:, 1], color="#888", lw=0.8, alpha=0.5)
        ax.plot(B[:, 0], B[:, 1], color="#888", lw=0.8, alpha=0.5, ls="--")
        ax.scatter(A[:, 0], A[:, 1], c=X, cmap=cmap, s=28, edgecolor="white",
                   linewidth=0.4, marker="o", label="A")
        ax.scatter(B[:, 0], B[:, 1], c=X, cmap=cmap, s=34, edgecolor="white",
                   linewidth=0.4, marker="^", label="B")
        ax.set_title(name)
        ax.set_xlabel("hidden-state dim 1 (arbitrary)")
        ax.legend(loc="upper left", facecolor="#1a1d24", edgecolor="#444")
    axes[0].set_ylabel("hidden-state dim 2 (arbitrary)")
    suptitle(fig, "Hidden-state trajectory: do the two runs share an internal path?",
             "Color = token index. 0.8B trajectories diverge instantly; 4B shares a scaffold then softly splits.")
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(f"{OUT}/04_hidden_umap.png", dpi=130)
    plt.close()


# -------------------------------------------------- 5. ENTROPY TAPE
def plot_entropy_tape():
    rng = np.random.default_rng(5)
    # brittle: speckled red throughout (high entropy + spikes)
    bA = 2.5 + rng.normal(0, 0.6, T) + 0.8 * (rng.random(T) > 0.6)
    bB = 2.5 + rng.normal(0, 0.6, T) + 0.8 * (rng.random(T) > 0.6)
    # stable: low (blue) for t<FORK, spike at fork, then mixed
    sA = np.concatenate([
        0.3 + np.abs(rng.normal(0, 0.15, FORK)),
        np.concatenate([[3.2, 3.5, 2.9], 1.2 + rng.normal(0, 0.6, T - FORK - 3)])
    ])
    sB = np.concatenate([
        0.3 + np.abs(rng.normal(0, 0.15, FORK)),
        np.concatenate([[3.1, 3.6, 3.0], 1.3 + rng.normal(0, 0.6, T - FORK - 3)])
    ])
    for a in (bA, bB, sA, sB):
        np.clip(a, 0, 4, out=a)

    cmap = LinearSegmentedColormap.from_list("bcr", ["#2c5fbf", "#c9c2a0", "#d94a4a"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (A, B), name in zip(
        axes, [(bA, bB), (sA, sB)], ["Brittle (0.8B)", "Stable (4B)"]
    ):
        strip = np.stack([A, B])
        im = ax.imshow(strip, aspect="auto", cmap=cmap, vmin=0, vmax=4,
                       extent=[0.5, T + 0.5, 0, 2])
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(["B", "A"])
        ax.set_title(name)
        ax.set_xlabel("token index")
        ax.grid(False)
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("entropy (nats) — blue=confident, red=at-a-fork")
    suptitle(fig, "Entropy tape: where is each run uncertain about its next token?",
             "0.8B is uncertain throughout; 4B is confident early, spikes at the fork, then mixed.")
    plt.savefig(f"{OUT}/05_entropy_tape.png", dpi=130)
    plt.close()


# -------------------------------------------------- 6. BUTTERFLY LOG-ODDS
def plot_butterfly():
    rng = np.random.default_rng(6)
    # brittle: |gap| grows to ~5 by t=5 on both sides, stays bulgy
    b_up = np.zeros(T)
    b_dn = np.zeros(T)
    ramp = np.linspace(0, 4.5, 6)
    b_up[:6] = ramp + rng.normal(0, 0.3, 6)
    b_up[6:] = 4.2 + rng.normal(0, 0.8, T - 6)
    b_dn[:6] = ramp + rng.normal(0, 0.3, 6)
    b_dn[6:] = 4.0 + rng.normal(0, 0.8, T - 6)
    b_up = np.clip(smooth(b_up, 1), 0, None)
    b_dn = np.clip(smooth(b_dn, 1), 0, None)

    # stable: near zero for t<FORK, then slight divergence
    s_up = np.zeros(T)
    s_dn = np.zeros(T)
    s_up[:FORK] = np.abs(rng.normal(0, 0.08, FORK))
    s_dn[:FORK] = np.abs(rng.normal(0, 0.08, FORK))
    s_up[FORK:] = np.clip(np.linspace(0, 1.6, T - FORK) + rng.normal(0, 0.25, T - FORK), 0, None)
    s_dn[FORK:] = np.clip(np.linspace(0, 1.4, T - FORK) + rng.normal(0, 0.25, T - FORK), 0, None)
    s_up = smooth(s_up, 2)
    s_dn = smooth(s_dn, 2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, (up, dn), name in zip(
        axes, [(b_up, b_dn), (s_up, s_dn)], ["Brittle (0.8B)", "Stable (4B)"]
    ):
        ax.fill_between(X, 0, up, color="#ffd166", alpha=0.7, label="A vs B on A's token")
        ax.fill_between(X, 0, -dn, color="#7aa2ff", alpha=0.7, label="B vs A on B's token")
        ax.axhline(0, color="#888", lw=0.8)
        ax.set_title(name)
        ax.set_xlabel("token index")
        ax.set_ylim(-6, 6)
        ax.legend(loc="upper right", facecolor="#1a1d24", edgecolor="#444", fontsize=8)
    axes[0].set_ylabel("log-odds gap between the two models (nats)")
    suptitle(fig, "Butterfly log-odds: probability-landscape decoupling, before the token fork",
             "0.8B's two distributions disagree by ~5 nats almost immediately; 4B stays tight for ~20 tokens.")
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(f"{OUT}/06_butterfly_logodds.png", dpi=130)
    plt.close()


if __name__ == "__main__":
    plot_kl_river()
    plot_rank_of_sibling()
    plot_logprob_paths()
    plot_hidden_umap()
    plot_entropy_tape()
    plot_butterfly()
    print("done")
