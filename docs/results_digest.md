# Results Digest

Last updated: 2026-04-30 after E9 logit-token processing, token-certified v3
partial processing, and the trajectory-branching research pivot.

This is the compact, talk-oriented current-state readout. The talk is a
chaos/dynamical-systems teaching talk first; the stability probe is supporting
evidence, not the subject. If older planning docs disagree with this file,
prefer this file for interpretation and `experiment_index.md` for experiment
state.

## Talk Pitch (Three Paragraphs) — teaching-lens reframe, 2026-04-29

**The subject.** This is a Learning Club talk. I'm teaching a lens — chaos
and dynamical systems — and walking through how it *might* apply to the LLMs
we work with. Not defending a thesis. Not publishing a paper. The pedagogical
goal is to take people who've heard of the butterfly effect past the pop-
culture framing: chaos isn't randomness, it's deterministic amplification of
small differences, and it's a real language from physics for talking about
systems where nearby states don't stay nearby. The LLM question the lens
helps pose: when we perturb the input a little, does the output stay nearby,
branch, or reconverge? The honest claim isn't "LLMs are chaotic." The honest
claim is that at inference time they're **hybrid sequential systems** —
continuous activations and logits feed a discrete branching process — and
that small prompt changes can move distributions or flip argmax branches.
Sensitivity is real, varies by model/prompt/metric, and is worth measuring.

**What I tried.** I ran a perturbation probe across ~18 open-weight models
(Qwen3 0.8B–9B, Gemma 4, Phi-4, DeepSeek-R1-Distill, Mistral, OLMo 2/3,
Granite, Falcon, SmolLM, plus a legacy lane of GPT-2 XL, GPT-J, Pythia, OPT,
LLaMA-1) on a 21–24 prompt-pair ladder spanning identical / no-op formatting
/ punctuation / synonym / paraphrase / small semantic / positive control.
Decoding is deterministic (argmax, `do_sample=False`) to isolate input
sensitivity from sampling variance. Primary metric is sentence-embedding
cosine distance, reported as bootstrap-CI'd clusters rather than a precision-
implying leaderboard, with supporting logit-level (JS divergence, top-1
flip rate, top-1 probability) and hidden-state signals. The experiment isn't
the contribution — it's the concrete thing I show to make the lens tangible.

**The two useful measurement findings, and the honest closing.** The probe
surfaced two confounds that would fool anyone running this kind of
measurement naively. First, the **scaffold confound**: models that emit
`<think>` or "Thinking Process:" preambles score much lower divergence on
short-output sentence-embedding distance simply because the identical
preamble dominates the similarity score. That's a warning about *evaluation*,
not a claim about *model dynamics* — and at 512 tokens the story stops being
clean (Phi-4 is certain at prompt-end and chaotic at 512 tokens). Second,
the **collapse confound**: Qwen3-0.8B at 4-bit looks more stable under
perturbation but drifts far from its BF16 outputs, consistent with
collapse onto a narrower output manifold — a naive single-axis stability
metric can't tell a robust model from a collapsed one. The practitioner
upshot is operational: don't evaluate LLM behavior on a single prompt,
single decode, or single output-distance metric; treat prompting as
operating a high-gain branching system; test neighborhoods, not single
prompts. The chaos lens doesn't prove anything about LLMs. It suggests
questions benchmarks don't ask, and those questions are worth asking.

## Current Best Story

The core talk is not "we proved LLMs are chaotic." It is:

> Chaos gives us a useful language for systems where tiny perturbations can
> either vanish, persist, or amplify. LLMs have exactly the kinds of moving
> states where that question becomes interesting: activations, logits, tokens,
> prefixes, KV caches, and quantized weights.

The experiment is there to make that lens concrete. It shows that nearby
prompts can produce different generation trajectories, and that the amount of
sensitivity varies by model and size even under deterministic decode.

The logit-level layer refines the text-only story. Text distance is a lossy
downstream symptom: an argmax can flip because two next-token logits were nearly
tied, or long text can diverge even when full-vocab prompt-end JS barely moves.
The completed logit probe records full-vocab KL/JS divergence, top-token
margins, winner-rank shifts, and teacher-forced logit divergence along the same
continuation.

The first higher-N E9 logit-token readout now has five completed models:
Qwen3.5 0.8B/2B/4B/9B thinking-off and Gemma4 E2B instruct, each with 525
token-certified pairs. Identical controls are effectively zero. Non-control
semantic means are `0.0873` for Qwen0.8B, `0.0872` for Qwen2B, `0.0827` for
Qwen4B, `0.0787` for Qwen9, and `0.0589` for Gemma E2B IT. Prompt-end JS is
lowest for Qwen2B/Qwen9/Gemma (`0.00606`, `0.00630`, `0.00637`), higher for
Qwen4B (`0.00741`), and highest for Qwen0.8B (`0.00877`). The internal
confidence story differs: Qwen0.8B has the lowest mean margin (`2.09`), Qwen4B
has the highest top-1 flip rate (`0.0533`), Qwen9 is lower-flip/wider-margin,
and Gemma has the lowest flip rate (`0.0114`) with the widest margin (`10.32`).
On common-prefix branch windows, the metric split matters: at the actual branch
timestep, low margin and JS are strong classifiers (`0.953` and `0.891` AUROC,
clustered by prompt pair). The older `branch_within_1` decision-window target
includes the branch timestep, so its `0.766` JS AUROC should not be described as
pure one-token-ahead warning. The stricter pre-branch-within-1 target is weaker
but still above chance for centered logit L2 (`0.649`) and JS (`0.620`). On the
longer shared-prefix subset (`branch_t >= 5`), strict pre-branch-within-1 drops
to centered L2 `0.568` and JS `0.558`, while at-branch remains strong. This
supports the decision-boundary version of the story more than a generic
"formatting changes outputs" version, but genuine pre-branch lead-time warning
is not yet a strong result.

The same E9 readout is a warning against turning this into a parameter-count
story. On the 500 shared non-control Qwen ladder cases, branch timing is only
monotonic earlier-with-size in `10.4%` of cases and monotonic later-with-size in
another `10.4%`. Pairwise timing deltas also change sign: Qwen2B branches later
than Qwen0.8B on average, while Qwen9 branches earlier than Qwen2B and Qwen4B
on average. The stronger claim is mechanism typing: tiny token-certified edits
push different models across different local branch boundaries, and "bigger"
does not determine where those boundaries sit.

The expanded scaffold-long logit pass adds the most promising mechanistic
thread so far. Across the 20-model 512-token panel, prompt-end top-1 probability
is strongly anti-correlated with 512-token semantic divergence (`r ~= -0.84`),
top-1 flip rate is positively correlated (`r ~= +0.57`), top-1 margin is weaker
but directionally aligned (`r ~= -0.39`), and full-vocab JS is basically not the
useful scalar (`r ~= -0.10`). The better story is decision-boundary fragility:
small prompt changes matter when they move the model across a low-margin argmax
boundary into a different response attractor. This still carries the scaffold
caveat: for reasoning/scaffold models, the high-confidence boundary may be
"enter the scaffold" rather than "answer content is robust."

The first local mechanistic-interpretability pass turns that boundary story
into a causal test. On Qwen3.5 0.8B and 2B, token-certified formatting edits
such as adding `(a)`, inserting a tab after a space, or wrapping a line can flip
early branch tokens under deterministic decode. Residual activation patching
then asks whether clean activations can rescue the clean branch token inside the
perturbed run. Across eight local patch cases, every replayable high-signal
case is rescuable at the late final-context residual state; the more
interesting aligned sweep shows that the parenthesized `(a)` case also has a
narrow edit-boundary/LCP rescue band in early layers (`rescue_fraction ~= 0.92`
at layer 1 on Qwen3.5 0.8B). The tab-after-space case is less localized: its
strongest rescue appears at the shared generated-prefix/final context, with
weaker prompt-boundary rescue. That gives a better hypothesis than "there is a
whitespace feature": tiny formatting edits can create local residual-state
differences near the tokenization/edit boundary, and some of those differences
survive or get amplified into low-margin branch-token choices.

The first SAE pilot now exists for Qwen3.5 2B using Qwen-Scope residual-stream
SAEs. This is still feature-ID evidence, not human-labeled feature semantics.
For the parenthesized `(a)` case, the prompt-boundary position has almost
disjoint top-20 SAE features between clean `" a"` and corrupt `" ("` at layer 0
(`overlap=1/20`) and remains mostly disjoint at layer 23 (`overlap=3/20`).
For the tab-after-space case, the final generated-prefix/final-context position
keeps high top-20 overlap (`16-18/20`) while prompt-boundary features are much
less overlapping (`5/20`). The aligned Qwen3.5 2B patch maps agree:
parenthesized `(a)` has strong prompt-boundary rescue at layer 0
(`rescue_fraction ~= 0.86`), while tab-after-space is dominated by the last
shared generated-prefix/final context. This matches the patching story:
parenthesized `(a)` looks like a sharp edit-boundary representation shift;
tab-after-space looks more distributed by the time the branch token is chosen.

The first SageMaker E07 patch wave makes the causal story less anecdotal. On
six selected token-certified branch cases each for Qwen3.5 2B/4B/9B, aligned
residual patching produced finite rescue values in 17 of 18 cases and
replayable full-or-overshoot rescue in 16 of 18. The strongest Qwen2B/4B
rescues are often at the prompt LCP/edit boundary, while Qwen9 in this selected
wave is more often rescued at the final shared context. This does not prove a
universal size law, but it does support the claim that many tiny-edit branch
events are causally movable, not merely post-hoc output differences.

The E10 hidden/logit capture has an early size-contrast hint on the same five
branch cases, but backend/dtype matters. Local Qwen3.5 2B/4B MPS/float16
captures and SageMaker CUDA/bfloat16 recaptures agree on several immediate
branches but can shift branch timing materially: Qwen2B has mean absolute
branch-t delta `4.25` over comparable cases, and Qwen4B has `8.80`, with max
shifts of `17` and `35` tokens. The Qwen9 metadata recapture is now processed:
on these selected CUDA cases, mean visible branch-t is `9.0` for Qwen2B
(excluding one no-visible-branch case), `7.8` for Qwen4B, and `1.8` for Qwen9.
That is useful for choosing intervention cases, but still too selected to call
a scaling law.

## Trajectory-Branching Research Frame

The post-talk paper direction should not become a generic robustness or
accuracy benchmark. Boundary labels such as nuisance edit, task-relevant edit,
tokenizer no-op, scaffold-heavy, and content-bearing are controls and
interpretation metadata. The central research object is the **structured
divergence event**: a localized point where two token-certified, nearby prompt
trajectories that were visibly identical or near-identical begin to separate in
logits, hidden states, generated tokens, or downstream semantics.

The working thesis is:

> LLM prompt sensitivity is not merely diffuse output variance. Under
> token-visible tiny perturbations, paired generations can follow the same
> visible prefix until localized branch events occur. Those events may be
> preceded by silent logit or hidden-state divergence, enriched near low-margin
> decision cliffs or high-confidence basin switches, and amplified into
> downstream semantic differences.

Concrete hypotheses:

1. **H1: branch localization.** A large share of high-divergence prompt pairs
   can be traced to localized branch windows rather than smooth global drift.
2. **H2: silent warning.** Logit or hidden-state divergence often appears in
   the common-prefix window before visible token divergence.
3. **H3: margin cliffs and basin switches.** Branch events are enriched near
   low top-1/top-2 margins or high entropy, but the more surprising cases are
   high-confidence basin switches where both continuations look locally
   confident.
4. **H4: scaffold masking.** Reasoning or template scaffolds can keep visible
   text identical while logits or hidden states have already separated.
5. **H5: local causality.** Forced-prefix or residual activation patching can
   delay, suppress, or flip selected branch events, turning the story from
   descriptive sensitivity into an intervention test.

Compelling evidence would be a paired case where a tiny token-visible edit
leaves the generated text identical for many tokens, hidden states separate
first, logit divergence rises next, margin collapses at the branch token, the
visible continuation then splits, and a forced-prefix or activation-patching
intervention moves the branch. The broader panel should then show that these
events are not cherry-picked: event location, warning lead time, margin
profile, scaffold-mask rate, and intervention response differ by model family
or recipe.

The kill condition is also clear. If most divergence happens immediately at
token 0, if hidden/logit separation has no predictive value beyond trivial
entropy, if branches do not persist, or if interventions cannot move the
branch, then the trajectory frame becomes weaker and the project should retreat
to a more modest prompt-regression / boundary-calibration tool.

Quantization and compression are now supporting material, not the thesis. The
main thesis is dynamical sensitivity: when the input changes a little, does the
model's response trajectory stay nearby, branch, or reconverge? Output quality
is a separate axis. A degenerate model can be stable, and a strong model can be
sensitive; this probe measures sensitivity, not goodness.

The strongest slide pair is the Qwen3.5 size comparison:

- `Qwen3.5-4B` and `Qwen3.5-9B` are effectively tied on the 24-pair robust
  ladder.
- `Qwen3.5-4B` separates cleanly from `Qwen3.5-0.8B` and `Qwen3.5-2B`.
- `Qwen3.5-0.8B` remains the most sensitive Qwen point in the robust ladder,
  but the `2B` vs `0.8B` contrast is not clean enough to overclaim.

That gives a sober claim: stability appears to be a measurable model property,
but it is not a simple monotonic function of parameter count from this tiny
probe.

Post-talk token-certified micro runs strengthen the measurement hygiene story.
The earlier micro sweep found that many raw character edits never survived
tokenization or chat-template formatting as real prompt-token deltas. The v3
reinforcement wave now uses model-specific certified prompt files: 25 identical
controls plus 500 effective prompt-token perturbations per selected model. As
of the 2026-04-30 19:35 -0300 processing pass, processed v3 means are:

| Model | Effective pairs | Mean 512-token semantic distance | P90 |
| --- | ---: | ---: | ---: |
| Gemma4 E4B base | 500 | 0.129 | 0.303 |
| Qwen3.5 0.8B thinking-off | 500 | 0.093 | 0.165 |
| Qwen3.5 2B thinking-off | 500 | 0.091 | 0.165 |
| OLMo3 7B instruct | 152 | 0.086 | 0.137 |
| Qwen3.5 4B thinking-off | 500 | 0.086 | 0.160 |
| Qwen3.5 9B thinking-off | 500 | 0.079 | 0.141 |
| Gemma4 E4B instruct | 500 | 0.068 | 0.165 |
| Gemma4 E2B instruct | 500 | 0.059 | 0.112 |

This is still not the final v3 panel: Gemma4 E2B base `-003` completed but
still produced partial raw rows with no `summary.csv`; repair job `-004` is in
progress. OLMo3 `-004` was recoverable from raw generations, but it has only
`152` effective non-control rows after CUDA failures near the end, so treat it
as a partial sanity check rather than a full panel point. The safe claim is
already useful: token-aware filtering is not a pedantic detail; it changes
which examples are admissible evidence, and the base-vs-instruct Gemma split
remains a high-signal recipe contrast. The token-certified Qwen ladder is
ordered 0.8B/2B/4B/9B by decreasing mean sensitivity, but 0.8B and 2B are not
cleanly separated by paired permutation.

The next important contrast is training era/post-training recipe, but it should
not be reduced to either "modern equals stable" or "older equals stable." A
follow-up check found a metric split: at token edit distance around `t=60`,
modern/instruct models look slightly more surface-divergent than legacy/base
models, but at 512-token semantic distance the sign flips and modern/instruct
models are more semantically contractive. The first legacy lane says something
more nuanced: `GPT-2 XL`, `GPT-J`, `Pythia`, and `OPT` often look brittle or
template-echo-y on this probe, while the best-effort `LLaMA-1 7B` conversion is
relatively stable. Within Gemma, base models are more sensitive than
instruction-tuned variants. The cleaner talk claim is that training and
post-training recipes change response attractors; era alone does not predict
the signature.

## Reasoning/Scaffold Confound

A later visual inspection found a major confound: some "stable" models are not
necessarily preserving answer content at first; they are preserving a response
scaffold. `Qwen3.5 4B` and `Qwen3.5 9B` emit a deterministic
`Thinking Process:` preamble, `Phi-4 reasoning plus` and `SmolLM3` emit
`<think>`, and `DeepSeek R1 Qwen 7B` emits visible chain-of-thought style
deliberation. These scaffolds can inflate common-prefix length and reduce text
distance even when the content eventually branches.

This variable is now encoded in
`configs/models.json` under each model's `observed_behavior` block. The
derived analysis artifacts remain in
`runs/rankings/scaffold_analysis/model_scaffold_annotations.csv`. On the final
21-model readout, observed reasoning/scaffold models are strongly associated
with lower small-perturbation semantic distance:

| Group | Models | Mean semantic distance | Mean common prefix |
| --- | ---: | ---: | ---: |
| Observed reasoning scaffold | 5 | 0.033 | 38.3 tokens |
| No observed reasoning scaffold | 16 | 0.141 | 20.5 tokens |

The bootstrap difference for scaffold minus non-scaffold semantic distance is
`-0.107` with a 95% interval of `[-0.140, -0.076]`. This is not proof that
reasoning improves content robustness; it may be measuring scaffold adherence.
Raw prefix inspection now backs this caveat directly:

- `runs/inspection/generation_prefixes_final21.csv`
- `runs/inspection/generation_prefix_summary_final21.csv`

The scan shows `Qwen3.5 4B` and `Qwen3.5 9B` start every checked output with
`Thinking Process:`, while `Phi-4 reasoning plus` and `SmolLM3` start every
checked output with `<think>`. The current short outputs are often still inside
the scaffold when generation stops, so content-only claims need longer
continuations.

For the talk, the honest claim is:

> Some apparent stability is format stability. Post-training can create a
> strong attractor into a reasoning scaffold, delaying visible divergence. The
> Qwen thinking-off controls show the scaffold effect is real but small and
> mixed, not a monotonic "reasoning makes models stable" result.

Important correction after inspecting raw 512-token generations: for Qwen3.5
4B/9B with default thinking enabled, the model usually does not produce a clean
answer stream within the 512-token budget. It emits a long draft/critique/polish
process and every checked Qwen 4B/9B reasoning-on row hits the token cap. Phi-4,
DeepSeek R1, and SmolLM3 show the same broad problem to varying degrees:
the artifact is a deliberation stream, not an answer-first response. Therefore
the reasoning-on full-output metrics should not be compared directly to
non-reasoning answer-first models as "answer stability." They are still useful,
but the label is different: they measure stability of the visible
deliberation/scaffold process.

The Qwen thinking-mode control has now landed. The harness supports
`--thinking-mode disabled`, which passes `enable_thinking=False` into the chat
template. In the currently installed Qwen3.5 tokenizer this still renders an
empty `<think>...</think>` block in the assistant prefix, so label it as
"thinking-off / empty-think prefix" rather than "no scaffold." It removes the
long generated reasoning scaffold and gives a same-weights, same-prompts
comparison against the default Qwen runs.

512-token semantic distance, default thinking -> `enable_thinking=False`:

| Model | Default | Thinking disabled | Readout |
| --- | ---: | ---: | --- |
| Qwen3.5 4B | 0.050 | 0.067 | scaffold helps slightly |
| Qwen3.5 9B | 0.057 | 0.072 | scaffold helps slightly |
| Qwen3.5 2B | 0.075 | 0.072 | wash |
| Qwen3.5 0.8B | 0.103 | 0.079 | scaffold hurts / noisy |

Therefore the current answer-first comparison should:

- Show Qwen thinking-on and thinking-off as a paired control, not as a simple
  proof that scaffolds cause stability.
- Show non-reasoning or direct-answer models separately from deliberation-stream
  models.
- Treat reasoning-on runs as a "deliberation attractor" example, not merged
  into answer trajectory charts without caveats.

## Talk Spine

Audience goal: teach chaos and dynamical systems clearly enough that LLM
behavior becomes a vivid example, not a prerequisite.

1. **What chaos means:** not randomness, but sensitive dependence on initial
   conditions.
2. **Exponential divergence:** two nearby trajectories can separate roughly
   like `e^(lambda t)` before saturating.
3. **State space matters:** for weather the state is physical variables; for
   LLMs the "state" might be hidden activations, logits, full token prefix, KV
   cache, or quantized weights.
4. **LLM translation:** generation is not a smooth physical trajectory; it is a
   continuous hidden-state system repeatedly crossing discrete token decision
   boundaries.
5. **Original probe:** a small measurement showing that this lens produces
   visible, model-dependent behavior.
6. **Training-recipe contrast:** base, instruction-tuned, and older models can
   have different response attractors.
7. **Quantization as another system:** treat BF16, 8-bit, and 4-bit variants
   as separate dynamical systems. The main comparison is within-system prompt
   sensitivity; distance-from-BF16 is a caveat about quality/fidelity, not the
   central experiment.

The experiments should take up the middle of the talk, not replace the
conceptual spine.

## Prior Art Anchors

Use [prior_art.md](prior_art.md) as the source list. In the live talk, keep this
to a few anchors:

- Lorenz / logistic map for classical chaos and Lyapunov intuition.
- Li et al.'s quasi-Lyapunov analysis for LLM-specific chaos framing.
- Successive paraphrasing attractor cycles for an intuitive LLM dynamical
  example.
- RNN Lyapunov-spectrum work for pre-LLM precedent in neural sequence models.
- Edge-of-stability training work for the broader neural-network stability
  story.
- TurboQuant / KIVI for the static compression and quantization contrast.

## Statistical Readout

Robust-wave readout on 24 small-perturbation prompt pairs:

| Model | Mean | 95% bootstrap interval |
| --- | ---: | ---: |
| Qwen3.5 4B | 0.0345 | 0.0182-0.0528 |
| Qwen3.5 9B | 0.0368 | 0.0165-0.0606 |
| Qwen3.5 2B | 0.0728 | 0.0389-0.1150 |
| Gemma4 E4B it | 0.0778 | 0.0406-0.1221 |
| Qwen3.5 0.8B | 0.0887 | 0.0484-0.1368 |

Paired permutation tests:

| Contrast | p-value | Readout |
| --- | ---: | --- |
| Qwen3.5 4B vs Qwen3.5 0.8B | 0.0004 | clean separation |
| Qwen3.5 4B vs Qwen3.5 2B | 0.0123 | clean separation |
| Qwen3.5 4B vs Qwen3.5 9B | 0.7781 | indistinguishable |
| Gemma4 E4B it vs Qwen3.5 4B | 0.0143 | Gemma more sensitive here |

Use this robust five-model result for the Qwen ladder slide. Use the broader
21-model panel as a lower-budget cluster demo, not as a precise leaderboard.

The model ranking is useful as a first pass, but exact positions should not be
treated as paper-grade. The small-perturbation score uses only 9 rows per
model: 3 no-op, 3 punctuation, and 3 synonym prompt pairs.

Historical short-output readout including legacy/base controls. This is useful
for provenance, but the current deck uses the 512-token semantic panel for the
era/recipe slide:

| Model | Mean | 95% bootstrap interval |
| --- | ---: | ---: |
| Qwen3.5 4B | 0.013 | 0.001-0.030 |
| Phi-4 reasoning plus | 0.024 | 0.010-0.038 |
| Qwen3.5 9B | 0.026 | 0.003-0.052 |
| DeepSeek R1 Qwen 7B | 0.030 | 0.007-0.055 |
| Mistral 7B v0.3 | 0.055 | 0.020-0.093 |
| LLaMA1 7B | 0.058 | 0.004-0.131 |
| Gemma4 E4B it | 0.069 | 0.017-0.153 |
| Qwen3.5 2B | 0.097 | 0.022-0.204 |
| OLMo/Gemma base/Qwen 0.8B band | about 0.13-0.14 | overlapping |
| GPT-J / Gemma E4B base / Pythia / OPT / GPT-2 XL | about 0.19-0.28 | high but overlapping |

Artifact:

- `runs/rankings/final_21model_readout/small_perturbation_bootstrap.csv`

Robust from bootstrap over prompt pairs:

- `Qwen3.5-4B` is much more stable than `Qwen3.5-0.8B`.
- `Qwen3.5-4B`, `Phi-4-reasoning-plus`, `Qwen3.5-9B`, and
  `DeepSeek-R1-Distill-Qwen-7B` form the stable top cluster.
- Older/base models often land toward the brittle semantic end, especially
  `GPT-2 XL`, `OPT 6.7B`, `Pythia 6.9B`, and `GPT-J 6B`, but `LLaMA1 7B` is an
  important counterexample. Token edit distance tells a different surface-form
  story, so era alone is not the causal story.

Not robust enough to claim:

- Exact ordering inside the top cluster.
- Exact ordering across the middle pack.
- A monotonic "bigger is always more stable" size law.
- `Qwen3.5-4B` beating `Qwen3.5-9B` as a general claim; it leads in the point
  estimate, but the uncertainty overlaps.

Preferred presentation: show buckets and confidence intervals, then emphasize
large-effect contrasts rather than a precise leaderboard.

Artifact:

- `runs/rankings/wave2_13model_bootstrap/small_perturbation_bootstrap_buckets.png`

## Deterministic Expanded Ladder

Mean normalized token edit distance:

| Category | Qwen3.5 0.8B | Qwen3.5 4B | Qwen3.5 9B | Gemma 4 E2B | Gemma 4 E4B | OLMo 3 7B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `control_identical` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `noop_format` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.516 |
| `punctuation` | 0.620 | 0.036 | 0.240 | 0.552 | 0.406 | 0.625 |
| `synonym` | 0.641 | 0.057 | 0.062 | 0.474 | 0.531 | 0.625 |
| `paraphrase` | 0.891 | 0.495 | 0.490 | 0.807 | 0.859 | 0.901 |
| `semantic_small` | 0.805 | 0.260 | 0.292 | 0.809 | 0.808 | 0.953 |
| `positive_control` | 0.984 | 0.526 | 0.505 | 0.969 | 0.974 | 0.995 |

Mean sentence-embedding cosine distance:

| Category | Qwen3.5 0.8B | Qwen3.5 4B | Qwen3.5 9B | Gemma 4 E2B | Gemma 4 E4B | OLMo 3 7B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `control_identical` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `noop_format` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.089 |
| `punctuation` | 0.154 | 0.005 | 0.047 | 0.146 | 0.035 | 0.112 |
| `synonym` | 0.261 | 0.035 | 0.030 | 0.111 | 0.173 | 0.203 |
| `paraphrase` | 0.196 | 0.100 | 0.091 | 0.128 | 0.162 | 0.131 |
| `semantic_small` | 0.596 | 0.414 | 0.374 | 0.606 | 0.614 | 0.619 |
| `positive_control` | 1.022 | 0.722 | 0.708 | 1.000 | 1.004 | 1.032 |

Artifacts:

- `runs/talk_figures/qwen_semantic_divergence.png`
- `runs/talk_figures/qwen_hidden_state_divergence.png`
- `runs/talk_figures/cross_lab_semantic_divergence.png`
- `runs/comparisons/qwen35_panel_expanded_size_ladder/compare_semantic_divergence.png`
- `runs/comparisons/panel_cross_lab_expanded/compare_semantic_divergence.png`
- `runs/comparisons/qwen35_expanded_size_ladder/compare_output_divergence.png`
- `runs/comparisons/qwen35_expanded_size_ladder/compare_semantic_divergence.png`
- `runs/comparisons/qwen35_expanded_size_ladder/compare_final_layer_hidden_divergence.png`
- `runs/comparisons/cross_lab_expanded/compare_output_divergence.png`
- `runs/comparisons/cross_lab_expanded/compare_semantic_divergence.png`

## Controls

The sampling controls are important because they prevent an easy objection.

- Same prompt, same seed, sampled decode: zero divergence in the control run.
- Same prompt, different seeds, sampled decode: high divergence.

Talk track:

> If we do not control sampling, we mostly measure randomness. The main
> perturbation ladder uses deterministic decode so the comparison is about
> prompt sensitivity, not stochastic sampling variance.

## Notable Model-Specific Behavior

Several models showed no-op formatting fragility. The most slide-worthy example
is OLMo 3 7B on a trailing-space-only code prompt:

- Prompt A generated code directly with a docstring and normalization logic.
- Prompt B generated an explanatory answer with a fenced code block and a much
  simpler `s == s[::-1]` implementation.

Other high-divergence no-op examples:

- `SmolLM3-3B`: a leading newline changed the stability prompt from a long
  `<think>` planning answer to a direct numbered list.
- `OLMo 2 7B`: a trailing newline on the weather prompt changed the phrasing
  enough to produce a semantic distance of about `0.18`.
- `Falcon3-10B`: a leading newline on the stability prompt diverged after a
  shared first reason and changed the second reason.

This is a good example, but should be labeled carefully:

> This is not proof that OLMo is unstable in general. It is evidence that the
> probe can surface prompt-template or formatting sensitivity that average
> benchmark scores would hide.

## Failures Worth Mentioning Only If Asked

- `NVIDIA-Nemotron-Nano-9B-v2` did not load because the container lacked
  `mamba-ssm`.
- `gpt-oss-20b` remained a tooling miss after several retries. MXFP4 wants
  Triton `>=3.4.0`; installing that normally upgrades Torch past the SageMaker
  driver, while dequantized bf16 falls back into CPU/CUDA tensor splits.
- Local Gemma 4 E2B stalled on Mac MPS, but the SageMaker CUDA path succeeded.

These are useful engineering notes, not central talk material.

## Current Caveats

- The expanded ladder is 21 prompt pairs, not a benchmark.
- The semantic metric uses `sentence-transformers/all-MiniLM-L6-v2`; it is a
  pragmatic distance proxy, not ground truth.
- Hidden-state distances are prompt-state probes, not formal Lyapunov
  exponents.
- Model-specific chat templates and instruction tuning may dominate some
  effects.
- Trajectory metrics require the observable generated token stream. They do not
  apply cleanly to API models with hidden reasoning tokens unless the analysis
  is explicitly changed to final-answer-only behavior.
- Quantization results should be framed as sensitivity profiles of separate
  systems. Distance-from-BF16 is useful only to prevent confusing stability with
  fidelity or quality.

## Quantization Readout

Within-system perturbation sensitivity:

| Model | BF16 | 8-bit | 4-bit |
| --- | ---: | ---: | ---: |
| Qwen3.5 0.8B | 0.138 | 0.110 | 0.091 |
| Qwen3.5 4B | 0.013 | 0.025 | 0.026 |

Read this as: for each fixed precision, how much do nearby prompts diverge
inside that system? Do not read it as a quality score.

Distance-from-BF16 is a caveat, not the main axis:

| Model | 8-bit | 4-bit |
| --- | ---: | ---: |
| Qwen3.5 0.8B | 0.098 | 0.132 |
| Qwen3.5 4B | 0.019 | 0.056 |

Talk framing:

> Each quantization is its own dynamical system. The question is whether nearby
> prompts stay nearby within that system. Fidelity to BF16 is a separate caveat,
> not the thing this talk is trying to measure.

Artifact:

- `runs/quantization_fidelity/qwen_quantized_vs_bf16_small_semantic.png`

## Wave 2: 13-Model Ranking (Historical Short-Output Panel)

Wave 2 more than doubled the successful model set from 6 to 13 completed stability
profiles.

Added successful profiles:

- `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
- `mistralai/Mistral-7B-Instruct-v0.3`
- `HuggingFaceTB/SmolLM3-3B`
- `ibm-granite/granite-3.3-8b-instruct`
- `allenai/OLMo-2-1124-7B-Instruct`
- `microsoft/Phi-4-reasoning-plus`
- `tiiuae/Falcon3-10B-Instruct`

Small-perturbation ranking uses mean semantic distance over:

- `noop_format`
- `punctuation`
- `synonym`

Lower is more stable.

Bucketed view:

- **Stable cluster:** `qwen35_4b`, `phi4_reasoning_plus`, `qwen35_9b`,
  `deepseek_r1_qwen7b`.
- **Middle cluster:** `mistral7b_v03`, `gemma4_e4b`, `smollm3_3b`,
  `granite33_8b`, `falcon3_10b`, `gemma4_e2b`.
- **Brittle cluster:** `olmo3_7b`, `qwen35_08b`, `olmo2_7b`.

Point-estimate ranking:

| Rank | Model | Mean small-perturbation semantic distance |
| ---: | --- | ---: |
| 1 | `qwen35_4b` | 0.013 |
| 2 | `phi4_reasoning_plus` | 0.024 |
| 3 | `qwen35_9b` | 0.026 |
| 4 | `deepseek_r1_qwen7b` | 0.030 |
| 5 | `mistral7b_v03` | 0.055 |
| 6 | `gemma4_e4b` | 0.069 |
| 7 | `smollm3_3b` | 0.074 |
| 8 | `granite33_8b` | 0.079 |
| 9 | `falcon3_10b` | 0.082 |
| 10 | `gemma4_e2b` | 0.086 |
| 11 | `olmo3_7b` | 0.135 |
| 12 | `qwen35_08b` | 0.138 |
| 13 | `olmo2_7b` | 0.144 |

New artifacts:

- `runs/comparisons/wave2_13model_expanded`
- `runs/rankings/wave2_13model/stability_rankings.csv`
- `runs/rankings/wave2_13model/small_perturbation_ranking.png`

Interpretation:

- The top four are all quite stable on tiny prompt perturbations.
- `Phi-4-reasoning-plus` is a strong new result: it is near the top on small
  perturbations but less stable than Qwen 9B / DeepSeek on broader meaningful
  perturbations.
- `DeepSeek-R1-Distill-Qwen-7B` is the strongest model on the broader
  meaningful-perturbation mean, even though it ranks fourth on the small
  perturbation mean.
- The OLMo family remains fragile on no-op formatting in this prompt ladder.
- `Qwen3.5-0.8B` remains a useful fragile baseline.

## Current Next Work

The talk is ready enough for the Learning Club framing. Further work should
move from point-estimate stability rankings to event-level trajectory
cartography.

Going forward, every new thread should have one compact row in
`docs/experiment_index.md`, with restart details in the relevant
`experiments/E##_*/README.md`. The main research direction is now
mechanism-typed replication, not broad leaderboard expansion.

1. Mine structured divergence events from every completed logit run: visible
   branch token, first silent logit warning, margin/entropy at branch,
   persistence, scaffold/content label, and final semantic divergence.
2. Run the margin-cliff prediction analysis: while outputs are still identical,
   predict whether a branch occurs within 1, 2, 5, or 10 tokens.
3. Run a focused hidden-state silent-divergence pilot on Qwen3.5 2B/4B/9B and
   selected Gemma contrasts. Save selected layers if full hidden capture is too
   expensive.
4. Expand residual-patching/SAE analysis by mechanism type:
   edit-boundary shocks, accumulated branch bias, inert token deltas, and
   replay-unstable false positives.
5. Add negative controls: prompt-token-effective edits that do not branch, so
   branch-localized features are not confused with generic tokenization changes.
6. Make scaffold/content boundary extraction auditable: preserve raw text,
   boundary span, confidence label, and score-before/after for every generation.
7. Expand prompt-pair count before ranking more models. Treat prompt pair as the
   statistical unit.
8. Separate sampling controls from input-sensitivity controls. They answer
   different questions, even when the distances are similar in magnitude.
9. Pair perturbation divergence with responsiveness/baseline drift so collapse
   cannot masquerade as robustness.
10. Use matched recipe comparisons where possible: base vs instruct within the
   same family, scaffold on vs off for the same weights, quantized vs BF16 for
   the same model.
11. Finish the token-certified v3 panel and rerun any timed-out partials before
   treating it as the publishable micro-perturbation table.
12. Keep older short-output tables as provenance, but use the 512-token
   semantic panel, token-certified micro runs, event-level logit analysis, and
   intervention results for new claims.

Historical pre-result hypotheses have been superseded by the completed
scaffold-long, Qwen thinking-off, quantization, and logit-correlation readouts.
Use `docs/experiment_index.md` for experiment state, this digest for current
interpretation, and `talk/companion_notes.md` for delivery notes.
