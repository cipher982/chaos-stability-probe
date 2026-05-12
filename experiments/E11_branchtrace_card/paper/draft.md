# Three Regimes of Causal Rescue for Token-Certified Prompt Edits

**Status:** skeleton draft, 2026-05-12. Target: NeurIPS interpretability
workshop. 6-8 pages. Figures are caption-only placeholders; see
`figures/` for sources.

---

## Abstract

Tiny token-level prompt edits — a stray newline, an extra parenthesis —
can cause a deterministic, greedy LLM to produce meaningfully different
outputs. Using an 82-case panel across eight open models (Qwen 3.5
0.8B-9B; Gemma 4 E2B / E4B, instruct and base), we apply aligned
residual-stream activation patching at four position classes
(prompt-LCP, aligned-prompt controls, generated-prefix, final-context)
to ask where the branch event is causally localized. We find three
regimes with distinct patching signatures: **edit-boundary** (41/82,
prompt-LCP fully rescues), **trajectory-migration** (27/82 silent-
divergence cases, generated-prefix rescues 25/27 while prompt-LCP is
degraded), and **prompt-accumulation** (14/82 immediate-branch cases
with tokenization shift at token 0; last-prompt-position rescues). A
pre-registered positive test on silent cases predicted generated-prefix
full rescue and observed 50/52. Strict "late-only" rescue — a branch
causally handled only after the prompt boundary and nowhere else — is
empirically **0/82**, so these are not late overwrites but a continuum
of increasingly downstream causal handles as the edit integrates into
generated state. A held-out randomized replication (V5, 40 cases) and
reverse-direction controls (21/21) support the taxonomy. We release
**Branch Cards**, a self-contained JSON schema + HTML view that pairs
two runs with their divergence and patch evidence.

---

## 1. Introduction

Modern LLMs are deployed as if they were deterministic functions of the
prompt: given an input, reviewers expect a stable output. They are not.
Under greedy decoding, tiny token-level prompt perturbations —
whitespace variants, parenthesization, tokenizer-visible edits that do
not alter the apparent meaning — reliably cause generation trajectories
to diverge at some token position, after which downstream text can be
substantively different.

We study this phenomenon as a *trajectory branching* problem and ask a
specific mechanistic question: **when a tiny prompt edit causes a branch
event at token t\*, where in the model's forward pass does the branch
become causally committed?**

We probe four candidate positions per case using aligned residual-stream
activation patching: the prompt token immediately after the edit
(**prompt-LCP**), any other prompt position (**aligned-prompt
controls**), any generated token before the branch (**generated-
prefix**), and the final context position just before the branch
(**final-context**). For each position, a "full rescue" means patching
A's activation into A′ causes A′'s next-token prediction to flip to
A's branch token.

Across the 82-case primary panel, we find three regimes that differ in
*which* position class rescues:

1. **Edit-boundary rescue** (41/82): the prompt-LCP handle alone is
   sufficient.
2. **Trajectory-migration rescue** (27/82): prompt-LCP is degraded
   (median 0.54); the rescue migrates forward into the already-
   generated context, where generated-prefix fully rescues 25/27.
3. **Prompt-accumulation rescue** (14/82): immediate-branch cases
   where tokenization shifts at token 0; the last-prompt-position
   — equivalently final-context at branch_t=0 — rescues while the
   token-0 edit representation is only partial.

Critically, a strict "late-only" regime — no prompt-side handle but a
late-context one — does **not** occur in our panel (0/82). This licenses
the continuum reading: the branch is always handled somewhere, but how
early that handle works depends on how much of the edit has been
integrated into generated state at the branch time.

**Contributions.**
- A three-regime taxonomy of where tiny-edit branches are causally
  localized, derived from aligned residual-stream patching on 82 cases
  across eight models.
- A pre-registered positive prediction (silent branches → generated-
  prefix rescue; observed 50/52) that distinguishes
  trajectory-migration from a null "broad-context overwrite"
  hypothesis.
- A negative control (0/82 strict late-only) that rules out late
  commitment.
- **Branch Cards**: a JSON schema + HTML renderer pairing two runs
  with divergence and patch evidence, with two example cards released
  at submission.

**What we do not claim.** We do not claim universal mechanism, that
edit-boundary rescue is the primary causal handle, or that regime
assignment is a basin-switch/cliff-slip dichotomy. We flag backend /
dtype shifts in branch timing (E10 mean absolute delta 4.25 tokens on
Qwen3.5-2B) and scaffold confounds (E03) as live limitations.

---

## 2. Method

**Prompt pairs.** We construct *token-certified* pairs in which A and
A′ differ by a small, reviewer-visible tokenization edit (inserted
blank line, doubled space, parenthesization, tab after space, etc.)
under each model's own tokenizer. The apparent semantic meaning is
preserved; only tokenization changes. Pairs are pre-filtered to have
small token edit distance (typically 1-3) against the full formatted
prompt, including chat template.

**Decoding.** Greedy (no sampling), `max_new_tokens=256`, chat template
applied, system prompt fixed. Models are loaded in bfloat16 on CUDA
where applicable and in float16 on MPS for local replay experiments.
See `configs/models.json` for exact revisions and trust_remote_code
settings.

**Branch event.** For each pair we record both generations token-for-
token and identify the branch time `t*` as the first position at which
A and A′ emit different token IDs. We also record a *silent
divergence* event when logit distributions over candidate tokens have
diverged materially before top-1 picks do; `silent_logit_lead` is the
distance in tokens between that onset and the visible branch.

**Aligned residual-stream activation patching.** Given a pair with
defined `t*`, we select four position classes at which to patch A's
residual stream into A′'s forward pass while measuring rescue at the
branch time:

- **prompt-LCP**: the prompt token immediately after the first token
  where A and A′'s prompt token IDs differ.
- **aligned-prompt controls**: every other prompt position, aligned
  across the shared prompt region.
- **generated-prefix**: any token A and A′ both generated before
  `t*`.
- **final-context**: the last position before the branch prediction.

Rescue fraction is `1.0` when A′'s post-patch top-1 at the branch
position matches A's branch token. We patch per-layer across all
transformer blocks and report best-layer results.

**Panels.** Our primary panel is 82 cases across 8 models (Qwen 3.5:
0.8B, 2B, 4B, 9B; Gemma 4: E2B base/instruct, E4B base/instruct). The
hand-selected V1-V3 waves (42 cases) were curated to span edit kinds
and models; the held-out V5 wave (40 cases) is a randomized sample
drawn independently from the token-certified pair set. Reverse-direction
V4 (21 cases) repeats a V1 subset with A and A′ swapped. All waves run
on SageMaker against pinned model revisions.

**Branch Cards.** Each card is a Pydantic-validated JSON document
(`branchcard/0.1`) that encapsulates runtime, prompts, token IDs,
branch token tops, patch evidence, suspected controlling span, and
artifact SHA-256s. A 250-line Jinja2 template emits a self-contained
HTML view. Two reference cards are released: an edit-boundary hero
(Qwen 3.5-2B / parenthesize_word_0434) and a silent trajectory-
migration case (Gemma 4 E2B base / blank_line_wrap_0212).

---

## 3. Results

### 3.1 Three regimes of causal rescue

Applying the regime assignment rule to the 82-case panel yields
**41 edit-boundary**, **27 trajectory-migration**, and **14 prompt-
accumulation** cases. Regime assignment is deterministic given
`prompt_lcp_full` and `event_kind`; see §2.

**Headline numbers:**

| Regime | Count | Defining signature |
|---|---|---|
| edit-boundary | 41/82 | prompt-LCP fully rescues |
| trajectory-migration | 27/82 | silent divergence; generated-prefix rescues |
| prompt-accumulation | 14/82 | immediate branch; last-prompt-position rescues |

**Figure 1** (placeholder: `figures/regime_rescue_panel.pdf`).
Per-case rescue fraction at each of the four position classes,
faceted by regime. Edit-boundary cases cluster near 1.0 at prompt-
LCP with low aligned-prompt controls. Trajectory-migration cases
show a clear shift: prompt-LCP partial (median 0.54), generated-
prefix peaks near or above 1.0. Prompt-accumulation cases show
partial prompt-LCP and full last-prompt-position.

### 3.2 Pre-registered positive test

Prior to running the V5 replication wave, we pre-committed a
positive prediction of the trajectory-migration signature: **in
silent-divergence cases, generated-prefix position will fully
rescue**. Observed rate: **50/52** (edit-boundary silent subset 25/25;
trajectory-migration silent subset 25/27). This is tight enough to
distinguish trajectory-migration from a null "anything sufficiently
downstream rescues" hypothesis.

### 3.3 Strict late-only rescue is empirically nonexistent

We define strict late-only rescue as a case in which *no* prompt-side
position (prompt-LCP or any aligned-prompt control) fully rescues,
and *some* late position does. Across 82 cases, **0** pass this
filter. This negates the "late overwrite" framing and supports the
continuum reading: the branch is always handled somewhere, and the
locus shifts forward as the edit integrates into generated state.

### 3.4 Held-out randomized replication (V5)

To rule out hand-selection as the driver of the taxonomy, we ran 40
randomized held-out cases under the same protocol. Best-position full
rescue: **39/40**. Best-position replay (deterministic re-derivation
of A's branch top-1 from patched A′): **36/40**. The regime
distribution on V5 is broadly consistent with V1-V3 and is not
distinguishable from chance resampling.

### 3.5 Reverse-direction controls

On 21 matched cases we ran both directions (A→B and B→A). **21/21**
have full-or-overshoot rescue in at least one direction; **19/21**
are replayable in both directions. Best-position class agrees
between directions in 16/21. This rules out directional artefacts of
the patching metric and demonstrates the branch is a causally
symmetric event under our protocol.

### 3.6 Branch prediction from trajectory signals

Using the E09 trajectory-event panel, at-branch AUROC for predicting
branch vs non-branch from low-margin and JS-divergence features is
**0.947 / 0.883** respectively. Strict pre-branch (k=1 token before)
AUROC drops to **supporting values** (see figure). We present this
as a contextual result, not a central claim: it establishes that
branches are *detectable* before they commit, but does not establish
mechanism.

**Figure 2** (placeholder: `figures/e07_position_classes.png`).
Existing bar chart of best-rescue position classes across the full
panel, used as supporting material.

### 3.7 Forced-prefix replay (support)

As a black-box sibling to activation patching, we force A′ through
A's pre-branch tokens plus the branch token itself and free-decode
10 more tokens, measuring token-LCP vs A's continuation. On 11
Qwen3.5-2B cases, mean rejoin by regime was:

| Regime | Mean rejoin (tokens, of 10) | N |
|---|---|---|
| edit-boundary | 8.0 | 8 |
| prompt-accumulation | 10 | 1 |
| trajectory-migration | 5.0 | 2 |

Qualitatively consistent with the patching story but with N too small
to treat as confirmation. Token-LCP under-counts "same topic, different
surface wording" cases. We present this as a cheap, API-compatible
sibling method whose scaling and metric design we leave to future work.

---

## 4. Discussion

**Integration-time view.** The three regimes differ not in *whether*
the branch is causally handled — it always is — but in how early in
the forward pass the handle is available. When the edit perturbs the
prompt's token representation locally, the handle is at prompt-LCP.
When the edit's effect has migrated into intermediate generated
context by the branch time, the handle has migrated with it. When the
edit is at token 0 (tokenization shift), only the accumulated final-
context position rescues. This is a continuum along the forward pass,
not a set of disjoint mechanisms.

**What this is not.** We do not observe a basin-switch / cliff-slip
split; margin distributions overlap across regimes. We do not claim
that edit-boundary is the universal or canonical handle — 41/82 is not
a majority of "interesting" behavior. And despite the elegance of the
0/82 figure, we resist any claim that late-only rescue is
*impossible*; we claim only that it is absent in this panel.

**Relation to scaffolded reasoning.** Gemma / Qwen models with visible
reasoning scaffolds (E03) show branch behavior modulated by scaffold
timing. We flag this as a confound for cross-recipe comparisons. The
Gemma base / instruct contrast is secondary evidence that training
recipe affects branch behavior, but we do not elevate it to a main
result.

**Why this matters.** If tiny edits can cause branches and those
branches are consistently causally localizable, an artifact that
records the branch location, the edit, and the patch evidence becomes
a practical debugging primitive. Branch Cards are our first attempt.

---

## 5. Branch Cards

A Branch Card is a JSON document that pairs two token-certified runs
with their divergence event and patch evidence. Schema sketch:

```
run_a, run_b        — prompt, tokens, generated text, runtime metadata
edit                — token edit distance, LCP, visible diff span
branch              — branch_t, event_kind, top-k on both sides
replay              — deterministic reproducibility, forced-prefix results
patch_evidence      — regime, best position class, rescue by class
suspected_span      — heuristic token span implicated
selection_provenance — wave, pool, archetype
artifacts           — path + sha256 for every upstream CSV/JSONL
```

Every field carries a `source: observed | derived | not_run` tag so a
reader can distinguish what came from an artifact from what was
re-derived. A small CLI builds a card from existing artifacts and
renders it as self-contained HTML.

**Figure 3** (placeholder: `figures/branch_card_hero.png`).
The Qwen 3.5-2B parenthesize_word_0434 hero card, rendered.

Two example cards ship with the paper: an edit-boundary case (Phase 1
hero) and a trajectory-migration case (Gemma E2B base /
blank_line_wrap_0212, branch_t=45, 45-token shared prefix,
generated-prefix rescue 1.33×).

---

## 6. Limitations

- **Panel size.** 82 primary cases across 8 models is a careful but
  not large panel. V5's 40 held-out cases argue against selection bias
  as the sole driver but do not establish universality.
- **Strict late-only (0/82).** A negative control, not a theorem.
  Larger / different-distribution panels could reveal cases.
- **Backend / dtype.** E10 shows mean absolute branch-time shift of
  4.25 tokens between `ml.g6e.2xlarge` bfloat16 and `ml.g5.2xlarge`,
  and up to 8.80 on Qwen 3.5-4B. Cross-backend comparisons require
  matched runtime metadata. Local MPS float16 is *not* interchangeable
  with SageMaker bfloat16 for branch-timing claims.
- **Scaffold confound (E03).** Models with visible reasoning scaffolds
  produce different branch statistics. Our panel is mixed; regime
  counts are not stratified by scaffold status.
- **Selection provenance.** Of 82 primary cases, 42 are hand-selected
  (V1-V3) and 40 are held-out randomized (V5). Prompt pair is the
  statistical unit.
- **Prompt-accumulation regime definition.** All 14 immediate-branch
  cases have `first_diff_token=0` (tokenization shift at prompt
  start). The regime may be partially a definitional artefact of LCP
  at token 0; see §3 audit in supplementary material.
- **Forced-prefix replay** is a token-LCP sanity check, not a
  mechanistic result. The metric under-counts semantic rejoins with
  different surface forms.

---

## 7. Related Work

- **Activation patching.** Meng et al. (ROME); Zhang & Nanda (tuned
  lens); Heimersheim & Nanda (how to use / interpret activation
  patching). We use their interpretive cautions and adopt position-
  class aligned patching to isolate where the branch is handled.
- **Paraphrase / format sensitivity.** NeurIPS 2024 format-bias work,
  attention-to-format findings, and prompt sensitivity benchmarks
  (cite current work through 2026). Our contribution is at the
  token-level and causal, not correlational.
- **Quasi-Lyapunov / trajectory divergence.** Li et al. on LLM
  dynamical sensitivity. We do not claim a Lyapunov regime; we
  localize the divergence event instead.
- **Thought Anchors / prefix-prefill analyses.** Recent work in
  resampling partial reasoning traces is adjacent to our forced-
  prefix replay primitive.

---

## 8. Conclusion

Tiny token-certified prompt edits produce causally localizable branch
events in modern LLMs. Across an 82-case panel, branches fall into
three regimes defined by which residual-stream position fully rescues:
edit-boundary (41), trajectory-migration (27), and prompt-accumulation
(14). Strict late-only rescue is empirically absent (0/82). A pre-
registered silent-case prediction landed at 50/52. These regimes form
a continuum of increasingly downstream causal handles as the edit
integrates into generated state. Branch Cards package divergence plus
patch evidence as a replayable artifact. All code, configs, and two
example cards are released with the paper.

---

## Appendix: Figure sources (build notes)

- **Fig 1 (regime rescue panel):** rebuild from
  `runs/rankings/activation_patch_comparison/case_level_summary.csv`,
  faceted by regime (edit-boundary / trajectory-migration /
  prompt-accumulation). Per-case bars for `prompt_lcp_token`,
  `best_aligned_prompt_rescue_fraction`,
  `best_generated_prefix_rescue_fraction`,
  `final_context_token_best_rescue_fraction`. Script TBD:
  `experiments/E11_branchtrace_card/figures/build_regime_rescue_panel.py`.
- **Fig 2 (best position classes):** use existing
  `runs/rankings/activation_patch_comparison/e07_best_rescue_position_classes.png`;
  re-emit for paper sizing.
- **Fig 3 (branch card hero):** screenshot of rendered
  `cards/qwen35_2b__parenthesize_word_0434.html` at fixed width.
- **Regime heatmap triptych (supplementary):** existing
  `experiments/E11_branchtrace_card/figures/regime_heatmaps/regime_mean_heatmaps_triptych.png`.
- **Forced-prefix replay table:** from
  `runs/forced_prefix_replay/phase3_qwen2b/summary.csv`.
- **Supplementary at-branch AUROC bars:** from
  `runs/trajectory_events/logit_token_cert_v1/branch_prediction_auc.csv`.
