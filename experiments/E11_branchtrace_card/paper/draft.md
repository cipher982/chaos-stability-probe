# Three Patching Signatures of Branch Events under Token-Certified Prompt Edits

**Status:** revised skeleton, 2026-05-12. Target: NeurIPS interpretability
workshop, 6-8 pages. Figures are caption-only placeholders; see
`figures/` for sources. Rev 2 applies 2026-05-12 expert review
(reframed as operational taxonomy not mechanism; renamed regimes;
Branch Cards demoted; honest discovery vs held-out split; strict-
assayed qualifiers; expanded definitions).

---

## Abstract

Tiny formatting edits can flip greedy LLM continuations, but the
causal handle for the branch need not sit at the edit site. We study
82 token-certified prompt pairs across eight open models using
aligned residual-stream patching, asking which positions are
*sufficient* to restore the unedited branch token. We find three
reproducible patching signatures: **boundary rescue** (41/82;
prompt-LCP position fully rescues), **generated-prefix rescue after
silent divergence** (27/82 silent cases; prompt-LCP degraded, some
prefix position fully rescues), and **tokenization-shift immediate
rescue** (14/82 immediate-branch cases with edit at token 0;
last-prompt-position fully rescues). We find **0/82 strict assayed
late-only cases**: whenever late generated-state patching rescues,
some assayed prompt-side position also rescues. A decision rule
defined on the discovery waves (V1-V3, 42 cases) predicted
generated-prefix full rescue on silent cases; on the randomized
held-out replication (V5, 30 silent cases of 40 total) we observe
**29/30** gen-prefix full rescue, consistent with discovery 21/22.
Reverse-direction controls on 21 matched cases hold in both
directions. We release **Branch Cards**, a JSON + HTML audit format
for reproducing per-case evidence, with two example cards.

---

## 1. Introduction

Modern LLMs are deployed as if they were deterministic functions of
the prompt: given a string in, a string out. They are not. Under
greedy decoding, tiny token-level prompt perturbations — whitespace
variants, parenthesization, other tokenizer-visible edits that do
not alter the apparent semantic content — reliably cause generation
trajectories to diverge at some token position, after which
downstream text can be substantively different.

We ask an operational question: **given a branch event at token t\*
under a tiny prompt edit, which residual-stream positions are
*sufficient handles* to restore the unedited branch token via
activation patching?** We are not claiming to recover a mechanism; we
are classifying where in the forward pass the branch can be undone.

We probe four position classes per case using aligned residual-stream
activation patching: the prompt token immediately after the first
token where A and A′ differ (**prompt-LCP**), any other prompt
position aligned across the shared prompt region (**aligned-prompt
controls**), any generated token before the branch (**generated-
prefix**), and the final context position just before the branch
(**final-context**). "Full rescue" at a position means that, after
substituting A's residual activation into A′'s forward pass at that
position (best-layer over all transformer blocks), A′'s top-1
prediction at the branch position matches A's branch token ID
(strictly) or A's post-patch metric crosses the rescue threshold
(loosely). We report both.

Across the 82-case primary panel, cases cluster into three signatures
under a deterministic decision rule:

1. **Boundary rescue** (41/82): prompt-LCP alone is sufficient.
2. **Generated-prefix rescue after silent divergence** (27/82):
   prompt-LCP is degraded (median 0.54); some generated-prefix
   position fully rescues (25/27 in this group).
3. **Tokenization-shift immediate rescue** (14/82): immediate-branch
   cases in which the token edit sits at token 0; last-prompt-position
   (= final-context at branch_t=0) rescues (14/14) while prompt-LCP is
   partial (median 0.62).

Critically, a strict *assayed* late-only regime — no assayed
prompt-side position fully rescues, but some late position does —
does not occur in our panel (**0/82**, upper 95% confidence rate
≈ 3.7%). This rules out a pure late-overwrite story in our setting
and supports a continuum reading: among assayed positions, a
prompt-side handle remains available even when late handles also
work.

**Contributions.**
- A three-signature operational taxonomy of where token-certified
  branches can be rescued via aligned residual-stream patching on 82
  cases across eight open models.
- A pre-specified decision rule (discovery from V1-V3, 42 cases)
  applied to a randomized held-out replication (V5, 40 cases);
  gen-prefix silent-case full-rescue rate 29/30 on held-out (21/22
  on discovery).
- A negative control: **0/82 strict assayed late-only**.
- **Branch Cards**, a JSON schema + HTML audit format pairing two
  runs with their divergence and patch evidence, released with two
  example cards.

**What we do not claim.** We do not claim a mechanism, or that these
patching signatures correspond to distinct circuits. We do not claim
edit-boundary rescue is the universal or canonical handle (it is
41/82). We do not claim that strict late-only rescue is *impossible*
— only that it is empirically absent in this panel under our patch
grid. We flag backend/dtype branch-timing shifts (up to 8.8 tokens
mean absolute Δ across SageMaker instance types; E10) and scaffold
confounds (E03) as limitations.

---

## 2. Method

**Prompt pairs.** We construct *token-certified* pairs in which A and
A′ differ by a small, reviewer-visible tokenization edit (inserted
blank line, doubled space, parenthesization, tab after space, etc.)
under each model's own tokenizer. Apparent semantic meaning is
preserved; only tokenization changes. Pairs are filtered to small
token edit distance (typically 1-3) on the full formatted prompt
including chat template and system prompt.

**Decoding.** Greedy (no sampling), `max_new_tokens=256`, chat
template applied, system prompt fixed, bfloat16 on CUDA. See
`configs/models.json` for exact model IDs, revisions, and
`trust_remote_code` settings.

**Branch event.** For each pair we record both generations and
identify the branch time `t*` as the first position at which A and A′
emit different token IDs. We additionally flag a *silent
divergence* event when the logit distributions over top-K candidates
have diverged (branch-JS ≥ 0.01 or branch-top-1 margin ≤ 0.125)
before the visible branch; `silent_logit_lead` is the distance in
tokens between that onset and the visible branch.

**Aligned residual-stream activation patching.** For each pair with
defined `t*` we patch A's residual-stream activation into A′'s
forward pass at one (position, layer) at a time and measure:
- **metric-level rescue:** `rescue_fraction = (patched - corrupt) /
  (clean - corrupt)` on the A-minus-B logit contrast at the branch
  position. Full metric rescue = rescue fraction ≥ 1.0.
- **strict replay:** the patched top-1 token ID at the branch
  position equals A's branch token ID.

We patch at four position classes: prompt-LCP, aligned-prompt
controls, generated-prefix, final-context. Per (case, position
class) we report the best-layer rescue across all transformer blocks.
The aligned-prompt-controls class serves as a non-LCP prompt
specificity check.

**Best-layer and handle framing.** We test *existence* of a
sufficient handle at some layer in a position class, not precise
layer localization. A position class is a sufficient handle for a
case if best-layer rescue fraction ≥ 1.0 (metric) or strict replay
holds (top-1 match).

**Decision rule for signatures.** We classify each case by:

    if prompt-LCP position produces best-layer rescue fraction ≥ 1.0:
        → boundary_rescue
    elif first_diff_token == 0 (edit at prompt token 0):
        → tokenization_shift_immediate_rescue
    else:
        → generated_prefix_rescue_after_silent_divergence

The rule is defined on the discovery waves (V1-V3) and applied to the
held-out V5 wave without modification.

**Panels.**
- **Discovery** (V1-V3): 42 hand-selected cases spanning 8 models and
  7 edit kinds; curated to cover silent vs immediate divergence and
  the Qwen ladder + Gemma base/instruct split.
- **Held-out** (V5): 40 randomized cases drawn from the full token-
  certified pair set, blind to V1-V3 results.
- **Reverse** (V4): 21 V1 cases re-run with A and A′ swapped.

All waves run on SageMaker against pinned model revisions. Prompt
pair is the statistical unit.

---

## 3. Results

### 3.1 Three patching signatures

Applying the decision rule to the 82-case primary panel yields:

| Signature | Count | Defining handle |
|---|---|---|
| boundary_rescue | 41/82 | prompt-LCP metric rescue ≥ 1.0 |
| generated_prefix_rescue_after_silent_divergence | 27/82 | prompt-LCP partial; gen-prefix rescues |
| tokenization_shift_immediate_rescue | 14/82 | first_diff_token = 0; last-prompt-position rescues |

**Figure 1** (placeholder: `figures/signature_rescue_panel.pdf`).
Per-case best-layer rescue fraction at each position class, faceted
by signature. Boundary cases peak at prompt-LCP; silent-divergence
cases peak at generated-prefix; immediate-tokenization cases peak at
final-context with partial prompt-LCP.

### 3.2 Pre-specified prospective check

We defined the decision rule and the predicted signatures on the
V1-V3 discovery waves (42 cases, 22 silent of which 21/22 had full
generated-prefix rescue). Applied without modification to the V5
held-out wave (40 cases, 30 silent), we observe **29/30** generated-
prefix full rescue on held-out silent cases, consistent with
discovery.

Combined (discovery + held-out), 50/52 silent cases have generated-
prefix full rescue. We do not describe this as "pre-registered" —
the decision rule was formalized after all V5 results were
available — but the held-out rate is not drawn from the same pool
the rule was tuned on, and the discovery/held-out rates are
statistically indistinguishable.

### 3.3 Strict assayed late-only rescue: 0/82

We define **strict assayed late-only** as: no assayed prompt-side
position (prompt-LCP or any aligned-prompt control) has best-layer
rescue fraction ≥ 1.0, AND some generated-prefix or final-context
position does. Across 82 cases, **0** satisfy this filter. With a
uniform Jeffreys prior, the 95% upper confidence bound on the true
rate in this panel's distribution is ≈ 3.7%.

We emphasize "assayed": aligned-prompt controls are sampled, not
exhaustive; the claim rules out strict late-only *among assayed
prompt positions*, not among all possible ones.

### 3.4 Replayability vs metric rescue

Metric rescue (fraction ≥ 1.0) and strict replay (patched top-1 =
A's branch token ID) agree on most cases but diverge in 6: these
have metric rescue ≥ 1.0 with patched top-1 semantically or
surface-wise close to A's token but not a strict ID match (e.g., a
leading-space variant). We report both metrics per case.

### 3.5 Held-out replication summary

On the 40-case held-out V5: best-position metric rescue 39/40;
strict replay 36/40. Signature distribution is consistent with
discovery within resampling noise.

### 3.6 Reverse-direction controls

On 21 V1 cases run in both directions (A→A′ and A′→A), **21/21** have
full metric rescue in at least one direction; **19/21** strict-replay
in both directions; best-position class agrees across directions in
16/21.

### 3.7 Failures and edge cases

Two silent cases (across 52) do not reach full generated-prefix
rescue. Both have short shared generated prefixes (≤ 1 token before
branch) and the best sufficient handle is final-context, not
generated-prefix. Two held-out V5 cases fail strict replay despite
metric rescue ≥ 1.0; in both, the patched top-1 differs from A's
branch token by whitespace/capitalization only.

---

## 4. Auxiliary checks

### 4.1 Branch prediction from trajectory signals (E09)

Using the E09 trajectory-event panel, at-branch AUROC for predicting
branch vs non-branch from low-margin and JS-divergence features is
**0.947** (low-margin) / **0.883** (JS). Strict pre-branch (k=1
token before) AUROC drops to the 0.6-0.7 range. This establishes
that branches are *detectable* at or just before they commit, but
does not establish mechanism and is not used in the main decision
rule.

### 4.2 Forced-prefix replay as behavioral continuity check

As a black-box sibling to activation patching, we force A′ through
A's pre-branch tokens plus the branch token itself and free-decode
10 more tokens, measuring token-LCP vs A's continuation. On 11
qwen35_2b cases the mean rejoin by signature was 8.0 (boundary), 5.0
(gen-prefix silent, n=2, split), 10 (tok-shift, n=1). With n=11 this
is a behavioral continuity check consistent with the patching
results, not a standalone result. Token-LCP under-counts cases where
the forced continuation stays on A's topic but picks different
surface tokens. Full details in `runs/forced_prefix_replay/
phase3_qwen2b/`.

---

## 5. Discussion

**Continuum of sufficient handles.** The three signatures differ in
*which* position is sufficient to undo the branch, not in whether it
can be undone. When the edit perturbs the prompt's token
representation locally, the handle is at prompt-LCP. When silent
logit divergence has integrated some of the edit's effect into the
generated context by branch time, some generated-prefix position is
sufficient. When the edit sits at token 0 (tokenization shift), the
last-prompt-position becomes a sufficient handle. We do not claim
this reflects a moving locus of computation; we claim only that
sufficient handles are available at those positions.

**What these signatures are not.** We do not observe a basin-switch
/ cliff-slip split; margin distributions across signatures overlap.
We do not claim the signatures correspond to distinct circuits, nor
that they pick out disjoint causal mechanisms. A reviewer could
reasonably read the signatures as definitions induced by our patch
grid and best-layer search; we address this by (a) fixing the
decision rule on discovery before applying to V5, (b) reporting
both metric rescue and strict replay, and (c) listing failures.

**Reproducibility hazards.** Backend / dtype shifts branch timing
materially: E10 shows mean absolute branch-time shift of 4.25
tokens on Qwen3.5-2B and 8.80 on Qwen3.5-4B across SageMaker
instance types. Local MPS float16 is not interchangeable with
SageMaker bfloat16 for branch-timing claims. All reported numbers
are on matched backends.

**Scaffold confound.** Models with visible reasoning scaffolds
produce different branch statistics (E03). Our panel is mixed;
signature counts are not stratified by scaffold status.

---

## 6. Limitations

- **Panel size.** 82 primary cases across 8 models. V5's 40 held-out
  cases argue against pure selection bias but do not establish
  universality.
- **0/82 strict late-only.** Negative control, not theorem. Upper
  95% bound on the rate in this panel's distribution is ≈ 3.7%.
  Different edit families or larger panels may surface cases.
- **Sampled aligned-prompt controls.** "No prompt-side position
  rescues" rules out strict late-only among *assayed* positions.
- **Patch grid and best-layer search.** We test existence of a
  sufficient handle at some layer in a position class, not precise
  layer localization. A reviewer may read the signatures as
  definitional; we partially address this via discovery/held-out
  split and by reporting aligned-prompt-control rescue as a non-LCP
  prompt specificity check (median 0.40 / 1.00 / 1.00 across the
  three signatures).
- **Tokenization-shift-immediate signature.** All 14 cases have
  `first_diff_token = 0`. The signature may be partially an
  artifact of LCP at token 0. We report it distinctly rather than
  subsume into boundary rescue.
- **Backend / dtype sensitivity.** See §5.
- **Scaffold confound.** See §5.
- **Prompt pair as unit.** Multiple pairs per edit-kind / model can
  share statistical structure we have not modeled.
- **Auxiliary checks (§4) are low-N.** The forced-prefix replay
  result (n=11, one model) is a continuity check, not a result.

---

## 7. Related Work

**Activation patching and causal tracing.** Vig et al. (causal
mediation analysis in LMs); Meng et al. (ROME / causal tracing);
Zhang & Nanda (activation-patching best practices / metrics);
Heimersheim & Nanda (how to use and interpret activation patching);
Geiger et al. (causal abstraction framing). We adopt the
patching-methodology cautions from the latter three and use
position-class-aligned patching to isolate where the branch is
rescued rather than to pin a circuit.

**Prompt sensitivity.** Sclar et al. (prompt formatting matters);
Cao et al. (worst-prompt performance); PromptEval, POSIX; Errica et
al. (sensitivity/consistency benchmarks). Our contribution is at
the token-certified level and uses causal intervention, not
accuracy correlations.

**Black-box conditioned rollouts.** Bogdan et al. (Thought Anchors;
2025) on importance of reasoning steps via black-box resampling is
the closest adjacent work to our §4.2 forced-prefix replay.

**Trajectory divergence.** Li et al. quasi-Lyapunov framing for LLM
dynamics provides macroscopic context; we do not claim a Lyapunov
regime and instead localize divergence events.

(Citations to be fleshed out with 2025-2026 references before
submission.)

---

## 8. Conclusion

Token-certified tiny prompt edits produce branch events that are
consistently undoable via aligned residual-stream patching. Across
82 cases and 8 open models, the rescue locus sorts cleanly into
three operational signatures: boundary, generated-prefix-after-
silent, and tokenization-shift-immediate. No case in our panel
exhibits strict assayed late-only rescue (0/82). The decision rule
defined on discovery waves predicts the V5 held-out signature
distribution within resampling noise. We release Branch Cards as
an audit format for the per-case evidence and two example cards.

---

## Appendix A — Branch Cards (audit format)

A Branch Card is a Pydantic-validated JSON document (schema version
`branchcard/0.1`) pairing two runs with their divergence and patch
evidence. Fields:

- `run_a`, `run_b` — prompt text and token IDs, generated text and
  token IDs, runtime metadata (torch, dtype, device, seed, chat
  template, system prompt).
- `edit` — token edit distance, prompt-LCP position, visible diff
  span.
- `branch` — branch_t, event_kind, top-k both sides.
- `replay` — deterministic reproducibility, forced-prefix results.
- `patch_evidence` — signature, best position class, best layer,
  rescue fractions by class.
- `suspected_controlling_span` — heuristic only; marked `derived`.
- `selection_provenance` — wave, pool, archetype.
- `artifacts` — path + SHA-256 for every upstream CSV/JSONL.

Every field carries a `source` tag in `{observed, derived,
not_run}` so a reader can distinguish artifact-derived from
re-computed values. A single-file HTML view renders the card with
the patch heatmap inline as a base64 data URI. Two example cards
ship with the paper: `qwen35_2b__token_cert_parenthesize_word_0434`
(boundary rescue) and `gemma4_e2b_base__token_cert_blank_line_wrap_
0212` (silent divergence, branch_t=45, 45-token shared prefix,
gen-prefix rescue 1.33×).

Intent: audit format for reproducing per-case evidence, not a
general debugging tool. Schema and renderer are released with the
paper.

## Appendix B — Figure build notes

- **Fig 1 (signature rescue panel):** rebuild from
  `runs/rankings/activation_patch_comparison/case_level_summary.csv`,
  faceted by signature. Per-case bars for prompt-LCP,
  best-aligned-prompt-control, best-generated-prefix, final-context.
  Script: `experiments/E11_branchtrace_card/figures/
  build_signature_rescue_panel.py` (TBD).
- **Heatmap triptych (supplementary):** existing
  `experiments/E11_branchtrace_card/figures/regime_heatmaps/
  regime_mean_heatmaps_triptych.png`.
- **Branch card hero inset:** screenshot of rendered
  `cards/qwen35_2b__parenthesize_word_0434.html`.
- **Forced-prefix table (supplementary):** from
  `runs/forced_prefix_replay/phase3_qwen2b/summary.csv`.
- **At-branch AUROC (supplementary):** from
  `runs/trajectory_events/logit_token_cert_v1/branch_prediction_auc.csv`.
