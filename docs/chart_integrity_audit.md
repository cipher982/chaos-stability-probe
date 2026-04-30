# Chart Integrity Audit — talk/ visualizations

Date: 2026-04-29

## The confound

Reasoning-tuned models (Qwen3.5 4B/9B reasoning-on, DeepSeek-R1, Phi-4 reasoning+, SmolLM3)
emit a long deterministic preamble scaffold (`Thinking Process:\n1. Analyze the Request:\n…`).
The first ~100–150 generated tokens are near-identical across prompt perturbations, so any
token-index-aligned divergence curve compared against non-reasoning models artificially shows
reasoning models as "stable" in the early window. See
`docs/answer_alignment_detection.md` — Qwen 4b synonym tier: raw = 0.03 vs aligned = 0.97 (18× confound).

## Status per file

| File | Status | Reason |
|---|---|---|
| `talk/family_fork.html` | **COMPROMISED** | 11-model reasoning vs non-reasoning divergence curves on raw generated-token index. Entire chart thesis is confounded by scaffold. |
| `talk/scrubber.html` | **COMPROMISED** | Qwen ladder mixes non-reasoning (0.8B, 2B) with reasoning-tuned (4B, 9B). The headline "2B→4B cliff" is scaffold, not capability. |
| `talk/branching.html` | **COMPROMISED** | 0.8B (non-reasoning) vs 4B (reasoning) side-by-side token streams, explicitly labeled "brittle" vs "stable". Labels encode the confound. |
| `talk/butterfly.html` | **COMPROMISED** (synthetic) | Data is synthetic, but panels are labeled "Qwen 0.8B brittle" vs "Qwen 4B stable" — reinforces the misleading narrative even without real data. |
| `talk/ripple.html` | OK | Single-model (Qwen3.5-2B). Layer × token internal activation dashboard. Not a cross-family comparison. |
| `talk/activation_ripple.html` | OK | Single-model (Qwen3.5-2B). Activation perturbation propagation. Not affected. |
| `talk/pond.html` | OK | Single-model (`data/ripple.json` → Qwen3.5-2B). Layer-wise hidden-state cosine distance across perturbation tiers. Not affected. |
| `talk/family_aligned.html` | **WIP** | Corrected view, but current alignment uses per-model median scaffold length (smears per-pair answer boundary). Per-pair token-exact alignment pending. |
| `talk/index.html` | updated | Tab labels marked with ⚠ for compromised charts. |
| `talk/slides.md` / `slides.html` / `browser.html` | **Needs separate review** | Large deck with ~31 references to family_fork/scrubber/branching/butterfly and stable/brittle/reasoning narrative. Not auto-watermarked in this pass — the deck's rhetoric needs rewriting, not just a banner. |
| `talk/concept_canvas.html` | not audited (static concept art) | Appears to be a concept canvas, not a data visualization. Skim suggests OK but flag for review if used as evidence. |

## What was added

1. Top-of-page yellow/amber banner on the 4 compromised charts with a direct link to `family_aligned.html`.
2. Title-line inline `⚠ scaffold confound` link on each compromised chart.
3. Blue WIP banner on `family_aligned.html` explaining the median-shift alignment caveat.
4. `⚠` / `⚠wip` suffix on compromised tabs in `index.html`.

## Recommended next steps

1. **Rebuild `family_fork.html` with per-pair token-exact answer alignment.** Don't ship the median-shift version (`family_aligned.html`) as the final view.
2. **Reframe `scrubber.html`.** Either (a) drop the 4B/9B reasoning members from the Qwen ladder, leaving a clean non-reasoning cliff story, or (b) separate reasoning-on/off plots and compare at aligned answer-token index.
3. **Rebuild `branching.html`.** Compare 0.8B vs 2B (both non-reasoning) for the size-cliff narrative, OR compare reasoning-on vs reasoning-off for the *same* 4B model to isolate the scaffold effect cleanly.
4. **Replace `butterfly.html` framing.** If kept as synthetic illustration, relabel panels so they don't lean on the "0.8B brittle / 4B stable" trope.
5. **Rewrite slide deck (`talk/slides.md`) sections that cite the compromised charts.** Banners do not fix the presentation deck; audit the rendered browser slides directly.
6. Keep `ripple.html`, `activation_ripple.html`, `pond.html` as-is — they're single-family and the ripple/pond visualizations aren't affected by the scaffold confound.
