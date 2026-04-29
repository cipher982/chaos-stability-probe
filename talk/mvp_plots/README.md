# MVP talk visualizations

Six static sketches contrasting a **brittle** (Qwen 0.8B-like) vs **stable**
(Qwen 4B-like) model generating 64 tokens from a one-word-perturbed prompt
pair. **All data is synthetic** — these are shape sketches to decide which
ideas are worth building against real captures.

Regenerate with: `.venv/bin/python make_plots.py`

---

### `01_kl_river.png` — KL river
**What this shows:** Per-token KL divergence between the two runs' next-token
distributions, as a filled area chart.
**Why it matters:** The single clearest "when did they decouple?" visual. The
brittle model's river fills up instantly; the stable model's stays a dry bed
for ~20 tokens before gently rising.

### `02_rank_of_sibling.png` — Rank-of-sibling trace
**What this shows:** For each token, where does B's chosen token rank inside
A's probability distribution? Log-scale scatter, rank=1 = identical.
**Why it matters:** More legible than KL for a general audience — "rank 1"
reads as "they agreed." The stable panel's clean floor of 1s for the first
~20 tokens is the point.

### `03_logprob_paths.png` — Running log-prob of generated sequence
**What this shows:** Cumulative sequence log-prob for both runs, with faint
per-token stems behind each cumulative line.
**Why it matters:** Visualizes "walking a confident ridge" (stable, shallow
slope) vs "wandering a flat plain" (brittle, steep slope with jittery stems).
Shared y-axis makes the slope contrast pop.

### `04_hidden_umap.png` — Hidden-state trajectory in 2D
**What this shows:** Each run's hidden state plotted as a 2D trajectory
(fake embedding), points colored by token index (viridis).
**Why it matters:** Makes the "shared scaffold then soft fork" intuition
geometric. Stable panel: tight cluster of early-token dots from both runs,
then two gentle drifts. Brittle panel: two separated clouds from the start.

### `05_entropy_tape.png` — Entropy tape
**What this shows:** Two heatstrips per subplot (top = A's entropy, bottom =
B's) across the 64 tokens. Blue = confident, red = at-a-fork.
**Why it matters:** Lets you eyeball "where is the model uncertain?" at a
glance. Stable: blue scaffold, a red spike at the fork, then mixed. Brittle:
speckled red throughout.

### `06_butterfly_logodds.png` — Butterfly log-odds
**What this shows:** Back-to-back area chart of the signed log-odds gap —
`log P_A / P_B` evaluated on A's token (up) and on B's token (down, mirrored).
**Why it matters:** Shows probability-landscape decoupling *before* the
actual token fork. The key asymmetry: distributions can diverge while the
sampled token still matches.
