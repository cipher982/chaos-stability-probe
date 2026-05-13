# Three Patching Signatures of Branch Events under Token-Certified Prompt Edits

**Status:** rev 4 submission-cleanup, 2026-05-12. Target: NeurIPS
interpretability workshop, 6-8 pages. Rev 4 applies the third expert
review: prompt-LCP definition tightened, aligned-prompt controls
labeled sampled explicitly, §3.2 reframed as "reserved-wave rule
transfer", rule-of-three iid caveat added, abstract/conclusion
softened, held-out summary merged into §3.2, reverse-direction moved
to §4 exploratory with branch prediction and forced-prefix replay,
§3 trimmed to six subsections, denominator-flow table added at §3.1,
all three wrong-donor matches tabulated in §3.5.

---

## Abstract

Tiny formatting edits can change a greedy LLM's continuation. A
space, newline, or bracket change is enough to branch the model
onto a different first word, and sometimes a different answer. The
patch that undoes the branch is not always at the edited token.

We study 82 token-certified prompt pairs across eight open models
(Qwen 3.5 at 0.8B to 9B, Gemma 4 at E2B and E4B, base and instruct).
For each pair we ask a concrete question: which residual-stream
positions, if patched from the unedited run into the edited run, make
the edited model pick the original next token? We find three
recurring patching signatures. **Boundary rescue** (41/82): patching
the first differing prompt token (the "prompt-LCP" position) is
enough. **Generated-prefix rescue
after silent divergence** (27/82): the top-1 pick does not change
until several tokens in, and some position in the shared generated
prefix is enough to restore the original token. **Tokenization-shift
immediate rescue** (14/82): the edit sits at token 0 of the prompt,
and the last prompt position rescues. No case in the panel needs a
late handle alone: every case has at least one prompt-side position
that works too, so strict late-only rescue is 0/82 (upper 95% bound
≈ 3.7% by the rule of three).

A decision rule fixed on the hand-curated waves transferred to a
randomized held-out wave with no changes: on silent cases in the
held-out wave, 29/30 had full generated-prefix rescue, against 21/22
on the curated waves. A donor-specificity control on qwen35_2b (three
random donors per case, same position and layer) matched the target
branch token in 10/10 self-patches and only 3/30 wrong-donor patches,
arguing against a purely generic-perturbation explanation at least
for this one-model control. We release **Branch Cards**, a JSON + HTML
audit format for per-case evidence, with two example cards.

---

## 1. Introduction

Greedy decoding is deterministic. Fix the model, the prompt, the
runtime, and the seed, and you get the same tokens out. But prompts
that look equivalent to a person — the same sentence with or without
an extra blank line, for example — are not equivalent to the
tokenizer. Under greedy decoding, these small formatting edits
reliably push the generation onto a different token, and it stays
different from there.

This paper asks a concrete question: **when a tiny prompt edit makes
the model pick a different next token at position t\*, where in the
forward pass can we patch the unedited model's activation into the
edited model and recover the original token?**

For each pair we patch four kinds of positions, one at a time:

- **prompt-LCP** (for "longest common prefix"): the first prompt
  token at which A and A′ differ. If both prompts share the first
  *k* tokens, this is the token at index *k*.
- **aligned-prompt controls**: sampled non-boundary positions from
  the shared prompt region (positions inside the common prefix, not
  the prompt-LCP position itself). Controls are a sample, not an
  exhaustive sweep.
- **generated-prefix**: any generated token that both runs produced
  the same before the branch.
- **final-context**: the last position before the model picks the
  branch token.

We say a position *rescues* the branch if, after substituting the
unedited run's residual activation at that position (best layer over
all transformer blocks), the edited model's top-1 pick matches the
original branch token. We also report a metric-level variant
(`rescue_fraction ≥ 1.0` on the A-minus-B logit contrast). The two
agree in most cases; we flag the handful where they don't.

The 82 cases cluster into three signatures under a deterministic
rule:

1. **Boundary rescue** (41/82): the prompt-LCP position rescues.
2. **Generated-prefix rescue after silent divergence** (27/82): the
   visible branch comes several tokens after a silent logit shift;
   the prompt-LCP position is partial (median rescue fraction 0.54)
   and some generated-prefix position rescues fully.
3. **Tokenization-shift immediate rescue** (14/82): the edit sits at
   token 0 of the prompt; the last-prompt-position rescues (the
   prompt is short enough that this equals final-context at
   branch_t=0); prompt-LCP is partial (median 0.62).

No case in the panel has the opposite shape — no prompt-side position
works but some late position does. The closest analog — "strict
late-only" — is 0/82. With the rule of three, that puts the upper
95% bound on this pattern's rate in the panel's distribution at
about 3.7%. This is an empirical result on our panel, not a
theorem. Different edit families or larger panels might surface
cases.

We treat this clustering as **operational** rather than
mechanistic. We are classifying *where in the forward pass a branch
can be undone*, not recovering a circuit. A different patch grid,
model set, or edit family would likely surface different
signatures. The signatures are useful descriptions, not discovered
mechanisms.

**Contributions.**

- Three recurring patching signatures on 82 token-certified cases
  across eight open models, with a deterministic decision rule.
- A discovery-to-replication split: the rule fixed on the curated
  waves predicts the held-out wave's signature distribution (silent
  full-rescue rate: discovery 21/22, held-out 29/30, combined 50/52).
- Negative controls. **0/82** cases are strict late-only (§3.3). A
  donor-specificity test on qwen35_2b shows strict replay 3/30 with
  wrong donors against 10/10 with self-patches (§3.5).
- **Branch Cards**: a JSON schema and HTML renderer packaging two
  runs, their divergence, and the patch evidence. Two example cards
  ship with the paper.

**What we do not claim.** We do not claim a mechanism, or that the
three signatures correspond to distinct circuits. We do not claim
boundary rescue is the canonical handle; it is 41 of 82. We do not
claim strict late-only rescue is impossible, only that it is absent
in this panel under this patch grid. Backend and precision settings
shift branch timing by several tokens (up to 8.8 mean absolute
tokens on Qwen 3.5-4B across SageMaker instance types), so our
numbers are on matched backends.

---

## 2. Method

**Prompt pairs.** Each pair differs by a small, reviewer-visible
tokenization edit: an inserted blank line, doubled internal space,
added parentheses, tab after space, and so on. The apparent meaning
is unchanged; only the tokenized form differs. We keep pairs with
small token edit distance (typically 1 to 3) on the full formatted
prompt, chat template and system prompt included.

**Decoding.** Greedy, no sampling, 256 max new tokens, bfloat16 on
CUDA. Model revisions pinned in `configs/models.json`.

**Branch event.** We decode A and A′ and find `t*`, the first
position where the two runs pick different token IDs. We flag a
*silent divergence* event when the logit distributions diverge
meaningfully (branch-JS ≥ 0.01 or top-1 margin ≤ 0.125) before the
visible pick changes.

**Activation patching.** For each (position, layer) we substitute
the A run's residual activation into A′'s forward pass at that
position and measure two things at the branch position:

- **Metric rescue.** `rescue_fraction = (patched − corrupt) /
  (clean − corrupt)` on the A-minus-B logit contrast. Full rescue
  means `rescue_fraction ≥ 1.0`.
- **Strict replay.** The patched top-1 at the branch position
  matches A's branch token ID exactly.

Strict replay is the cleaner criterion for reviewers. Metric rescue
is what the patching literature reports and what our panel CSVs
record; we report both, and where they disagree we note it (§3.4).

We patch every layer. "A position class rescues" means some layer
in that class produces full rescue or strict replay. We are testing
that a sufficient handle exists somewhere in the class, not
localizing a specific layer.

**Decision rule for signatures.** Applied to each case:

    if the prompt-LCP position has metric rescue >= 1.0 at some layer:
        → boundary_rescue
    elif first_diff_token == 0:
        → tokenization_shift_immediate_rescue
    else:
        → generated_prefix_rescue_after_silent_divergence

The rule was specified on the curated waves (V1 through V3, 42
cases). It transferred to the held-out wave (V5, 40 cases) with no
modification.

**Panels.**

- **Discovery**: 42 hand-curated cases across 8 models and 7 edit
  kinds.
- **Held-out**: 40 randomized cases drawn from the same
  token-certified pair set, independent of discovery.
- **Reverse**: 21 discovery cases re-run with A and A′ swapped.

All runs went through SageMaker against pinned model revisions. The
statistical unit is the prompt pair.

---

## 3. Results

### 3.1 Three patching signatures

The decision rule assigns every case in the 82-case panel to one
signature:

| Signature | Count | Defining handle |
|---|---|---|
| boundary_rescue | 41 | prompt-LCP full rescue |
| generated_prefix_rescue_after_silent_divergence | 27 | some generated-prefix position fully rescues |
| tokenization_shift_immediate_rescue | 14 | edit at token 0; last-prompt-position rescues |

**Figure 1** (`figures/signature_rescue_panel.pdf`). Box + jittered
strip of best-layer rescue_fraction per case at the four position
classes, one panel per signature. Each signature's characteristic
shape reads at a glance: boundary has prompt-LCP well above 1.0 (and
other classes also rescuing); generated-prefix-after-silent has
prompt-LCP and aligned-prompt controls clearly below 1.0 with
generated-prefix and final-context glued to 1.0; tokenization-shift-
immediate has prompt-LCP below 1.0 with aligned-prompt / final-
context at 1.0 and generated-prefix empty (immediate branches have
no shared generated prefix). Source:
`runs/rankings/activation_patch_comparison/case_level_summary.csv`.

**Denominator reference.** Several headline numbers use different
denominators; we list them together here to keep the paper's counts
legible.

| Quantity | Number | Denominator |
|---|---|---|
| primary panel | 82 | — |
| silent-divergence cases | 52 | of 82 |
| silent cases with full gen-prefix rescue | 50 | of 52 silent |
| generated-prefix signature (prompt-LCP partial) | 27 | of 82 |
| boundary signature | 41 | of 82 |
| tokenization-shift-immediate signature | 14 | of 82 |
| strict assayed late-only | 0 | of 82 |
| held-out (V5) cases | 40 | of 82 |
| reverse-direction matched cases | 21 | subset of discovery |
| donor-specificity control (qwen35_2b) | 10 cases × 3 donors | 30 trials |

### 3.2 Reserved-wave rule transfer

We chose the signature decision rule by looking at the curated
waves. The held-out wave (V5) had been run through SageMaker before
we wrote the rule down, so the patch CSVs for both waves existed
when we formalized the signatures. What V5 did *not* see was the
rule, the signature labels, or any hand-selection — V5 cases are a
randomized sample from the token-certified pair set.

So this is a rule-transfer check on a reserved wave, not a
preregistered prediction. The question it answers is: does the
rule we chose on the curated cases describe the randomized held-out
cases without adjustment?

One signature has a concrete numeric test: in silent-divergence
cases, some generated-prefix position should fully rescue.

- **Curated (discovery).** 22 silent cases; 21 with full generated-
  prefix rescue.
- **Held-out.** 30 silent cases; 29 with full generated-prefix
  rescue.
- **Combined.** 50 of 52 silent cases.

**Held-out summary.** Across all 40 V5 cases, 39/40 had metric full
rescue at some position and 36/40 had strict replay. The signature
distribution matches discovery within resampling noise.

The caveat on the 50/52 number: the generated-prefix position class
includes several positions within a case. "Some generated-prefix
position rescues" is a weaker claim than "a specific generated-prefix
position rescues." We report the rate at which *any* such position
rescues; we do not claim the same index across cases.

### 3.3 Strict assayed late-only rescue: 0 of 82

We defined this case as: no prompt-side position rescues (neither
prompt-LCP nor any aligned-prompt control has full rescue at any
layer), and some late position does. Zero cases in our panel fit.
Applying the rule-of-three heuristic to 0/82 gives an upper 95%
bound of about 3.7%, under the assumption that cases are
independent draws from the panel's distribution. Our pairs are not
iid — multiple pairs per edit kind and per model share structure —
so the bound is heuristic, not a formal confidence interval.

"Assayed" is load-bearing. Aligned-prompt controls are sampled,
not exhaustive. We rule out strict late-only *among tested*
prompt positions, not among all possible ones.

### 3.4 Metric rescue and strict replay agree in most cases

For 76 of 82 cases, the position-class with full metric rescue
(rescue_fraction ≥ 1.0) also produces strict replay (patched top-1
= A's branch token ID). The six disagreements all have a near-match
top-1: a surface variant of A's branch token, usually differing in
a leading space or in capitalization. We list them in the
supplementary table and report both metrics throughout.

### 3.5 Donor-specificity check

The signatures could be cheap: maybe *any* activation at the
signature position, patched in, pushes the edited model across the
threshold, not specifically the unedited run's activation. To test
this we ran a small local check on qwen35_2b.

For each of 10 cases whose best signature position rescued under
strict replay, we fixed the target case's recipient, position, and
winning layer, and patched from three random donor cases (same
model, different target A-branch token). Self-patch (the unedited
run's own activation) rescued 10 of 10 cases strictly. Wrong-donor
patches rescued 3 of 30. Median rescue_fraction: 1.00 for
self-patch, 0.00 for wrong-donor.

Three donor strict-matches is not zero, so we list all three:

| Target case | Sig | Pos / layer | Target A-token | Donor A-token |
|---|---|---|---|---|
| blank_line_wrap_0521 | boundary | prompt_lcp / L3 | `Retry` | ` ensure` |
| parenthesize_word_0344 | boundary | prompt_lcp / L0 | `Quality` | ` established` |
| line_wrap_0378 | boundary | prompt_lcp / L1 | `Retry` | ` ensure` |

All three are boundary-rescue cases patched at the prompt-LCP
position, and in each the target's A-token (`Retry`, `Quality`) is a
common continuation word that the recipient context already favors.
Two of the three use the same donor case. The pattern is consistent
with prompt-LCP positions being sensitive to a range of donor
activations when the target branch token itself is a predictable
completion, rather than with the signature failing. Strict replay
still fails in 27 of 30 wrong-donor trials.

We ran this check on only one model. Treat it as a rule-out: if the
signatures had been trivially satisfiable by any donor, strict
replay would have been much higher than 3/30.

### 3.6 Failures and edge cases

- Two silent cases (of 52) do not reach full generated-prefix
  rescue. Both have at most one shared generated token before the
  branch, and the best handle is the final-context position.
- Two held-out cases have metric rescue ≥ 1.0 but fail strict
  replay by whitespace or capitalization only.
- One panel case has a corrupt baseline whose A-minus-B metric is
  nearly equal to the clean baseline, so the metric rescue fraction
  is undefined. We exclude it from metric rescue statistics but
  include it in strict replay.

---

## 4. Additional checks

Three exploratory signals line up with the patching story. None is
used to define or validate signatures; we include them because they
point the same direction from different angles.

**Reverse-direction patching.** Twenty-one discovery cases ran with
A and A′ swapped. In at least one direction, all 21 had metric full
rescue. In both directions, 19 had strict replay. The best-position
class agreed between directions in 16. Supportive rather than
decisive; 16 of 21 is a middling agreement rate.

**Branch prediction from trajectory signals.** Across 52 silent
cases, two features — low top-1 margin and high JS divergence with
the other run — detect the branch at or just before it commits
(at-branch AUROC 0.947 and 0.883 respectively). Strict pre-branch
(one token before the visible pick) AUROC drops into the 0.6–0.7
range. Branches are detectable close in time; this says nothing
about mechanism.

**Forced-prefix replay.** A black-box analog on 11 qwen35_2b cases:
force A′ through A's pre-branch tokens and A's branch token,
free-decode 10 more tokens, measure token LCP with A's continuation.
Mean rejoin by signature: 8.0 for boundary (n=8), 5.0 for gen-prefix
silent (n=2, split between 10 and 0), 10 for tok-shift (n=1). With
n = 11 this is a continuity check, not a result. The metric
under-counts cases where the forced continuation stays on A's topic
but picks different surface tokens.

---

## 5. Discussion

**A continuum of sufficient handles.** The signatures differ in
which position rescues, not in whether one does. When the edit
perturbs the prompt locally, the prompt-LCP position is enough.
When the edit's effect has already nudged the logits before the
visible top-1 changes, some position in the generated prefix is
enough. When the edit sits at token 0, last-prompt-position (the
only non-boundary prompt position available with so short a
prompt) is enough. We aren't saying the branch "migrates" or "accumulates." We're
saying a sufficient handle sits at those positions.

**What the signatures are not.** We do not observe a clean
basin-switch / cliff-slip split: margin distributions across
signatures overlap. We do not claim these patching signatures are
distinct circuits. A reviewer could read the signatures as
definitions induced by our patch grid and best-layer search; the
donor-specificity check (§3.5) is our response, not a final answer.

**Reproducibility hazards.**

- *Backend and precision change branch timing.* Across SageMaker
  instance types, mean absolute branch-time drift is 4.25 tokens
  on Qwen 3.5-2B and 8.80 on Qwen 3.5-4B. Local MPS float16 and
  SageMaker bfloat16 are not interchangeable for branch-timing
  claims.
- *Reasoning scaffolds change the statistics.* Models whose chat
  templates produce visible reasoning scaffolds show different
  branch patterns. Our panel mixes scaffolded and non-scaffolded
  models.

---

## 6. Limitations

- **Panel size.** 82 cases across 8 models. Held-out 40 cases
  reduce selection-bias concern but do not establish universality.
- **Strict late-only is empirical, not formal.** Upper 95% bound
  on the rate in this panel's distribution is about 3.7% (rule of
  three). Other edit families or larger panels might surface cases.
- **Aligned-prompt controls are sampled.** The "no prompt-side
  position rescues" half of strict late-only rules out the pattern
  *among the positions we test*, not among all possible prompt
  positions.
- **Best-layer search is an existence claim.** We test whether
  some layer in a position class rescues, not where precisely.
- **Tokenization-shift-immediate has structural confounds.** All
  14 cases have `first_diff_token = 0`. The signature may be
  partly a definitional artifact of the LCP convention at token 0.
  We report it distinctly rather than fold it into boundary
  rescue.
- **Donor-specificity check is one model.** The qwen35_2b check in
  §3.5 does not prove donor-specificity for every panel model.
- **Backend and precision caveats** as above.
- **Prompt pair is the unit.** Multiple pairs per edit kind and
  model share structure we do not model.
- **Auxiliary checks are low-N.** §4 is supportive, not standalone.

---

## 7. Related Work

**Activation patching and causal tracing.** Our patching metric and
aligned-position setup follow the causal-tracing lineage. Vig et al.
(2020) introduced causal mediation analysis for interpreting
language-model components; Meng et al. (2022) extended the technique
to factual association recall with ROME; Geiger et al. (2021) give
the formal causal-abstraction framing that underpins interchange
interventions. Heimersheim & Nanda (2024) document practical
interpretation pitfalls, and Zhang & Nanda (2024) compare metrics
and corruption strategies across common activation-patching setups.
We adopt the "existence of a sufficient handle at some layer" framing
these write-ups recommend and apply position-class-aligned patching
to classify where a branch can be rescued, not to localize a circuit.

**Prompt sensitivity.** Sclar et al. (2024) document large accuracy
swings under meaning-preserving format edits in few-shot settings.
Cao et al. (2024) propose RobustAlpacaEval and show worst-prompt
performance is substantially below average. PromptEval (Polo et al.,
2024) estimates performance distributions across many prompt
templates efficiently; POSIX (Chatterjee et al., 2024) introduces a
prompt-sensitivity index; Errica et al. (2025, NAACL) quantifies
sensitivity and consistency under prompt rephrasing; Pecher et al.
(2026) argue much of this variance reflects prompt
underspecification. These works correlate prompt choice with
accuracy. Ours intervenes at the residual-stream level on
tokenizer-certified pairs to ask where the branch commits
causally, which is a different question.

**Black-box conditioned rollouts.** Bogdan et al. (2025, "Thought
Anchors") introduce a black-box counterfactual-resampling method
for localizing which reasoning steps matter most in a chain of
thought. Our forced-prefix check in §4 is in the same family — fix
a shared prefix, let the model continue, measure whether it sticks
to the original trajectory — but applied to formatting-driven
branches rather than reasoning-step importance.

---

## 8. Ethics, licenses, and data release

**Models.** We use the Qwen 3.5 family (Alibaba, Apache 2.0) at 0.8B,
2B, 4B, and 9B, and the Gemma 4 family (Google, Gemma Terms of Use)
at E2B and E4B, both base and instruct. No model weights are
redistributed; we record pinned Hugging Face model revisions in each
Branch Card's `run_a.runtime.model_revision` field. Users replaying
our results agree to the respective model licenses when they pull
the weights.

**Intended use.** This paper is interpretability research. The
token-certified edit methodology could in principle be used to
search for low-effort prompt perturbations that elicit specific
outputs, including harmful ones. Our focus is understanding where
branches commit; we do not release, benchmark, or optimize
jailbreak prompts, and the 82-case panel is restricted to neutral
instruction-following tasks (definitions, summaries, how-to
explanations).

**Data.** All prompt pairs were generated for this project. No
user data, no scraped web content. Prompt-pair seed templates are
in `configs/prompt_pairs_token_certified/` and the full pair JSON
files are released with the code.

**Released artifacts.**

- Branch Card JSON schema (`branchcard/0.1`) and two reference
  cards — boundary-rescue hero and silent trajectory-migration
  hero — under MIT license.
- Python package `branchtrace/` (schema, loaders, CLI, HTML
  renderer) under MIT.
- Patching, replay, and donor-control scripts under
  `scripts/`, plus per-case artifact CSVs for the 82-case panel
  and the held-out V5 wave.

**Compute.** Patching runs used roughly 60 SageMaker GPU hours total
(ml.g6e.2xlarge and ml.g5.2xlarge mixed). Donor and forced-prefix
replay checks ran in under ten minutes on a 64GB MacBook (MPS
float16). No model training or fine-tuning was required.

---

## 9. Conclusion

Greedy LLMs branch on formatting edits, and the branch is causally
committed at a specific, findable residual-stream position. Across
82 token-certified cases on 8 open models we identify three
distinct patching signatures — boundary, generated-prefix-after-
silent, tokenization-shift-immediate — and find **zero cases** in
the panel where only a late, post-prompt handle rescues. The rule
we derived on the curated waves predicts the randomized held-out
wave's signature rates with no adjustment. The obvious "maybe the
model just overwrites context late" story does not describe these
branches. The obvious "the edit itself is always the handle" story
does not describe them either. Both endpoints are there in the
data, connected by a middle regime where the branch is already
latent in shared generated context before it becomes visible. We
release Branch Cards as an audit format so others can inspect and
reproduce the per-case evidence.

---

## Appendix A — Branch Cards

A Branch Card is a JSON document (`branchcard/0.1`, Pydantic-
validated) that pairs two runs with their divergence and patch
evidence. Top-level fields:

- `run_a`, `run_b` — prompt, token IDs, generation, runtime
  metadata (torch, dtype, device, seed, chat template, system
  prompt).
- `edit` — token edit distance, prompt-LCP position, visible diff
  span.
- `branch` — `branch_t`, event kind, top-k on both sides.
- `replay` — deterministic reproducibility, forced-prefix results.
- `patch_evidence` — signature, best position class, best layer,
  rescue fractions per class.
- `suspected_controlling_span` — heuristic; marked `derived`.
- `selection_provenance` — wave, pool, archetype.
- `artifacts` — path + SHA-256 for every upstream CSV/JSONL.

Every field carries a `source` tag (`observed`, `derived`,
`not_run`). A single-file HTML view renders the card with the
patch heatmap inline as a base64 data URI. Two example cards ship
with the paper: `qwen35_2b__token_cert_parenthesize_word_0434`
(boundary rescue) and
`gemma4_e2b_base__token_cert_blank_line_wrap_0212` (silent
divergence with generated-prefix rescue, branch_t = 45, 45-token
shared prefix).

**Figure 2** (`figures/branch_card_hero.png`). Rendered HTML of the
boundary-rescue hero card, showing the Edit, Runs, Branch, and
Patch-evidence sections end to end with the per-layer rescue
heatmap inline. Remaining card sections (Replay, Suspected
controlling span, Selection provenance, Runtime, Caveats,
Artifacts) are omitted from the figure for space and appear in
the shipped HTML card.

This is an audit format for reproducing per-case evidence, not a
general branch-debugging tool.

## References

- Vig, J., Gehrmann, S., Belinkov, Y., Qian, S., Nevo, D., Singer,
  Y., & Shieber, S. (2020). *Causal Mediation Analysis for
  Interpreting Neural NLP: The Case of Gender Bias.* NeurIPS 2020.
  arXiv:2004.12265.
- Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). *Locating
  and Editing Factual Associations in GPT.* NeurIPS 2022.
  arXiv:2202.05262.
- Geiger, A., Lu, H., Icard, T., & Potts, C. (2021). *Causal
  Abstractions of Neural Networks.* NeurIPS 2021.
  arXiv:2106.02997.
- Heimersheim, S., & Nanda, N. (2024). *How to use and interpret
  activation patching.* arXiv:2404.15255.
- Zhang, F., & Nanda, N. (2024). *Towards Best Practices of
  Activation Patching in Language Models: Metrics and Methods.*
  arXiv:2309.16042.
- Sclar, M., Choi, Y., Tsvetkov, Y., & Suhr, A. (2024). *Quantifying
  Language Models' Sensitivity to Spurious Features in Prompt
  Design.* ICLR 2024. arXiv:2310.11324.
- Cao, B., Yan, Y., Luo, T., Wang, M., Chen, W., & Zhang, M.
  (2024). *On the Worst Prompt Performance of Large Language
  Models.* NeurIPS 2024. arXiv:2406.10248.
- Polo, F. M., Xu, R., Weber, L., Silva, M., Bhardwaj, O., Choshen,
  L., de Oliveira, A. F. M., Sun, Y., & Yurochkin, M. (2024).
  *Efficient multi-prompt evaluation of LLMs (PromptEval).*
  NeurIPS 2024. arXiv:2405.17202.
- Chatterjee, A., Renduchintala, H. S. V. N. S. K., Bhatia, S., &
  Chakraborty, T. (2024). *POSIX: A Prompt Sensitivity Index For
  Large Language Models.* arXiv:2410.02185.
- Errica, F., Siracusano, G., Sanvito, D., & Bifulco, R. (2025).
  *What Did I Do Wrong? Quantifying LLMs' Sensitivity and
  Consistency to Prompt Engineering.* NAACL 2025. arXiv:2406.12334.
- Pecher, B., Srba, I., Bielikova, M., & Vanschoren, J. (2026).
  *Revisiting Prompt Sensitivity in Large Language Models for Text
  Classification: The Role of Prompt Underspecification.*
  arXiv:2602.04297.
- Bogdan, P. C., Macar, U., Nanda, N., & Conmy, A. (2025).
  *Thought Anchors: Which LLM Reasoning Steps Matter?*
  arXiv:2506.19143.

## Appendix B — Figure build notes

- **Fig 1 (signature rescue panel).** Rebuild from
  `runs/rankings/activation_patch_comparison/case_level_summary.csv`,
  faceted by signature. Per-case bars for prompt-LCP,
  best-aligned-prompt-control, best-generated-prefix, and
  final-context. Script: `experiments/E11_branchtrace_card/
  figures/build_signature_rescue_panel.py`.
- **Heatmap triptych (supplementary).** Existing
  `experiments/E11_branchtrace_card/figures/regime_heatmaps/
  regime_mean_heatmaps_triptych.png`.
- **Branch Card hero screenshot (Fig 2).** Rebuild via
  `experiments/E11_branchtrace_card/figures/build_branch_card_hero.sh`.
  Renders `cards/qwen35_2b__parenthesize_word_0434.html` with
  headless Chrome at 1200×3500, then crops at y=2460 to end after
  the Patch-evidence heatmap and before the Replay section.
- **Donor-specificity table (§3.5).** From
  `runs/wrong_donor_control/qwen2b/donor_control.csv`.
- **Forced-prefix replay (§4).** From
  `runs/forced_prefix_replay/phase3_qwen2b/summary.csv`.
- **Branch prediction (§4).** From
  `runs/trajectory_events/logit_token_cert_v1/branch_prediction_auc.csv`.
