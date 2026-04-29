"""Probe for genuine time-varying spatial structure in ripple data."""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("talk/mvp_plots/ripple_probe")
OUT.mkdir(parents=True, exist_ok=True)

# =========================================================================
# 1) Activation ripple: [downstream_layer x token] per injection
# =========================================================================
act = json.load(open("runs/activation_ripple_qwen35_2b/activation_ripple.json"))
injs = act["injections"]
token_strs = act["token_strs"]
inject_token = act["injection_token"]  # 14
n_layers = act["n_layers"]  # 24

print("=== ACTIVATION RIPPLE ===")
print(f"seq_len={act['seq_len']}, n_layers={n_layers}, inject_token={inject_token}")

# Plot the ripple grid heatmap per injection
fig, axes = plt.subplots(1, len(injs), figsize=(4 * len(injs), 4), sharey=True)
for ax, inj in zip(axes, injs):
    r = np.array(inj["ripple"])  # [downstream_layers, seq_len]
    im = ax.imshow(r, aspect="auto", cmap="magma", origin="lower")
    ax.set_title(f"inject L{inj['inject_layer']}")
    ax.axvline(inject_token, color="cyan", lw=0.5, alpha=0.6)
    ax.axhline(inj["inject_layer"], color="cyan", lw=0.5, alpha=0.6)
    ax.set_xlabel("token")
axes[0].set_ylabel("downstream layer")
plt.tight_layout()
plt.savefig(OUT / "01_activation_ripple_heatmaps.png", dpi=130)
plt.close()

# Per-token trajectory across downstream layers (treating layer as "time")
fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=False)
for ax, inj in zip(axes.flat, injs):
    r = np.array(inj["ripple"])  # [L_down, seq]
    inj_L = inj["inject_layer"]
    # X = downstream layer index (relative to start of captured range)
    n_down = r.shape[0]
    xs = np.arange(n_down) + inj_L  # roughly absolute layer indices
    for t in range(r.shape[1]):
        if t == inject_token:
            ax.plot(xs, r[:, t], color="red", lw=2.0, alpha=0.9, label="inject tok" if t == inject_token else None)
        else:
            ax.plot(xs, r[:, t], color="k", lw=0.6, alpha=0.35)
    ax.set_title(f"inject L{inj_L}")
    ax.set_xlabel("layer")
    ax.set_ylabel("ripple mag")
    ax.legend(loc="upper right", fontsize=8)
axes.flat[-1].axis("off")
plt.tight_layout()
plt.savefig(OUT / "02_activation_ripple_per_token_trajectories.png", dpi=130)
plt.close()

# For inject_layer=12 specifically: zoom in, look for non-monotonic tokens
inj_12 = next(inj for inj in injs if inj["inject_layer"] == 12)
r12 = np.array(inj_12["ripple"])
print(f"\ninject L12 ripple shape: {r12.shape}")
# For each token, fit monotonicity: count sign changes in diff
mono_counts = []
for t in range(r12.shape[1]):
    d = np.diff(r12[:, t])
    sign_changes = np.sum(np.diff(np.sign(d + 1e-12)) != 0)
    mono_counts.append(sign_changes)
print("sign-change counts per token (inject L12):")
for t, sc in enumerate(mono_counts):
    print(f"  tok {t:2d} ({token_strs[t]!r:20s}): sign_changes={sc}  max={r12[:,t].max():.4f}  argmax_layer_offset={r12[:,t].argmax()}")

# Argmax layer offset per token: is there a "wave arriving later at different tokens"?
argmax_off = r12.argmax(axis=0)
print("\nargmax layer offset per token (inject L12):")
print(argmax_off.tolist())

# plot argmax-layer-offset vs token: does the "peak" arrive at different depths?
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(range(r12.shape[1]), argmax_off, color="steelblue")
ax.axvline(inject_token, color="red", ls="--", label="inject tok")
ax.set_xlabel("token position")
ax.set_ylabel("downstream-layer offset of peak ripple")
ax.set_title("inject L12: does the peak arrive at different depths per token?")
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "03_inject_L12_peak_depth_per_token.png", dpi=130)
plt.close()

# Small multiples: show the full ripple propagation as layer increases, as a 1D
# strip over tokens, one line per layer step.
fig, ax = plt.subplots(figsize=(12, 5))
cmap = plt.cm.viridis
for i in range(r12.shape[0]):
    ax.plot(range(r12.shape[1]), r12[i], color=cmap(i / max(1, r12.shape[0] - 1)),
            lw=1.2, alpha=0.85)
ax.axvline(inject_token, color="red", ls="--", alpha=0.6, label="inject tok")
ax.set_xlabel("token position")
ax.set_ylabel("ripple magnitude")
ax.set_title("inject L12: token-space ripple at each downstream layer (color = layer depth)")
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "04_inject_L12_1D_strip_over_layers.png", dpi=130)
plt.close()

# =========================================================================
# 2) Prompt-perturbation ripple (ripple.jsonl)
# Build [token_idx x layer] grid of resid_cos for each pair/category.
# =========================================================================
print("\n=== PROMPT PERTURBATION RIPPLE ===")
recs = [json.loads(l) for l in open("runs/ripple_qwen35_2b/ripple.jsonl")]

# cosine distance = 1 - cos sim
from collections import defaultdict
grids = defaultdict(dict)  # pair_id -> {token_idx: [layer_cos...]}
for r in recs:
    pid = r["pair_id"]
    t = r["token_idx"]
    layer_cos = [1.0 - L["resid_cos"] for L in r["layers"]]
    grids[pid][t] = layer_cos

# Assemble to [token x layer]
pair_grids = {}
for pid, d in grids.items():
    T = max(d.keys()) + 1
    L = len(next(iter(d.values())))
    g = np.full((T, L), np.nan)
    for t, arr in d.items():
        g[t] = arr
    pair_grids[pid] = g
    print(f"  {pid}: grid shape {g.shape}, min={np.nanmin(g):.3f} max={np.nanmax(g):.3f}")

# Heatmaps of each pair's [token x layer] grid
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
for ax, (pid, g) in zip(axes.flat, pair_grids.items()):
    im = ax.imshow(g, aspect="auto", cmap="magma", origin="lower")
    ax.set_title(pid, fontsize=9)
    ax.set_xlabel("layer")
    ax.set_ylabel("token_idx (decoding step)")
    plt.colorbar(im, ax=ax, fraction=0.04)
for ax in axes.flat[len(pair_grids):]:
    ax.axis("off")
plt.tight_layout()
plt.savefig(OUT / "05_prompt_ripple_token_by_layer_heatmaps.png", dpi=130)
plt.close()

# Semantic_small: per-layer trajectory across tokens (token is time here)
g = pair_grids["semantic_small_change"]
fig, ax = plt.subplots(figsize=(12, 5))
cmap = plt.cm.viridis
for L in range(g.shape[1]):
    ax.plot(range(g.shape[0]), g[:, L], color=cmap(L / (g.shape[1] - 1)), lw=1.0, alpha=0.8)
ax.set_xlabel("decoding step (token_idx)")
ax.set_ylabel("1 - cos sim")
ax.set_title("semantic_small: per-layer cos-dist vs decoding step (color = layer)")
plt.tight_layout()
plt.savefig(OUT / "06_semantic_small_per_layer_vs_step.png", dpi=130)
plt.close()

# Per-token trajectory across layers (layer = time, token = space)
fig, ax = plt.subplots(figsize=(12, 5))
for t in range(g.shape[0]):
    ax.plot(range(g.shape[1]), g[t], color=cmap(t / (g.shape[0] - 1)), lw=0.9, alpha=0.7)
ax.set_xlabel("layer (depth)")
ax.set_ylabel("1 - cos sim")
ax.set_title("semantic_small: per-token (color) cos-dist growth across layers")
plt.tight_layout()
plt.savefig(OUT / "07_semantic_small_per_token_vs_layer.png", dpi=130)
plt.close()

# Check: is per-token trajectory across layers non-monotonic? (key ripple question)
sign_change_counts = []
for t in range(g.shape[0]):
    row = g[t]
    if np.any(np.isnan(row)):
        continue
    d = np.diff(row)
    sc = int(np.sum(np.diff(np.sign(d + 1e-12)) != 0))
    sign_change_counts.append((t, sc, float(row.max()), int(row.argmax())))
print("\n semantic_small: token -> (sign_changes, max_cosdist, argmax_layer)")
for t, sc, mx, am in sign_change_counts[:20]:
    print(f"  tok {t:2d}: sc={sc}  max={mx:.3f}  argmax_layer={am}")

# Is argmax_layer roughly the same for every token? If so, no wave-arrival structure.
argmax_layers = np.array([r[3] for r in sign_change_counts])
print(f"\n argmax_layer stats: mean={argmax_layers.mean():.1f} std={argmax_layers.std():.2f} "
      f"min={argmax_layers.min()} max={argmax_layers.max()}")

# Compare tiers: average cosdist vs layer per tier (does "wave" shape change?)
fig, ax = plt.subplots(figsize=(10, 5))
for pid, g in pair_grids.items():
    mean_by_layer = np.nanmean(g, axis=0)
    ax.plot(mean_by_layer, lw=2, label=pid)
ax.set_xlabel("layer")
ax.set_ylabel("mean 1 - cos sim (avg over tokens)")
ax.set_title("How divergence grows with depth, per perturbation tier")
ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(OUT / "08_tier_mean_cosdist_vs_layer.png", dpi=130)
plt.close()

# Does the peak-layer shift across tokens for semantic_small (wave-arrival)?
# Plot argmax_layer per token and the last-layer cosdist per token
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4))
g = pair_grids["semantic_small_change"]
argm = np.nanargmax(g, axis=1)
a1.bar(range(g.shape[0]), argm, color="steelblue")
a1.set_title("semantic_small: argmax_layer per decoding step")
a1.set_xlabel("token_idx"); a1.set_ylabel("argmax layer")
a2.plot(range(g.shape[0]), g[:, -1], "-o", lw=1.5)
a2.set_title("semantic_small: final-layer cos-dist per token")
a2.set_xlabel("token_idx"); a2.set_ylabel("cos dist at last layer")
plt.tight_layout()
plt.savefig(OUT / "09_semantic_small_peak_layer_and_final_layer.png", dpi=130)
plt.close()

# =========================================================================
# 3) Build the "framing B" movie candidate: for inject_L12, animate frames
#    frame i = downstream layer offset i, show 1D strip of ripple over tokens.
#    Saved as an image grid so we can see all frames at once.
# =========================================================================
fig, axes = plt.subplots(5, 5, figsize=(20, 12), sharex=True, sharey=True)
for i, ax in enumerate(axes.flat):
    if i >= r12.shape[0]:
        ax.axis("off"); continue
    ax.bar(range(r12.shape[1]), r12[i], color="teal")
    ax.axvline(inject_token, color="red", ls="--", alpha=0.5)
    ax.set_title(f"layer offset {i}", fontsize=8)
    ax.set_ylim(0, r12.max() * 1.05)
plt.suptitle("inject L12: ripple-over-tokens at each downstream layer (frames = time)")
plt.tight_layout()
plt.savefig(OUT / "10_inject_L12_frame_grid.png", dpi=130)
plt.close()

print("\nDone. Plots in:", OUT)
