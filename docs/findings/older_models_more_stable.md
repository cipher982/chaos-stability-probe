# Are older language models more stable under prompt perturbation?

**Status:** mostly superseded by follow-up analysis. The original token-edit
trend is real but weak and metric-dependent; semantic-distance analysis points
the other way. Treat this as a measurement caution, not an era claim.

**2026-04-29 follow-up:** after re-running the same 14-model panel against the
512-token semantic artifacts, the story split by metric. At token edit distance
around `t=60`, modern/instruct models are a little more divergent than legacy
base models (`0.580` vs `0.517`, `r_year = +0.286`), but the effect nearly
vanishes if `LLaMA-1 7B` is removed (`0.580` vs `0.567`). On 512-token semantic
distance over the same perturbation tiers, modern/instruct models are *less*
divergent than legacy/base models (`0.190` vs `0.288`, `r_year = -0.717`).
So the current readout is not "older models are more stable." It is:
**surface token trajectories and semantic answer content tell different
stories; era is a weak proxy for recipe, tokenizer/template behavior, and
response attractors.**

## Question

Do older (pre-2024) language model architectures produce more output-stable generations under small prompt perturbations than more recent (2024-2025) instruction-tuned models? "Stable" here means: when you perturb the prompt by a single word swap, a punctuation change, or a rewording, the generated token sequence diverges less from its unperturbed counterpart.

## Claim we can currently defend

Across a panel of 14 non-reasoning/direct-answer models spanning 2019-2025,
the token-level trajectory metric and the semantic-content metric disagree.
Token edit distance at `t=60` weakly suggests newer/instruct models diverge
more at the surface-token level. Sentence-embedding distance over 512-token
outputs suggests the opposite: newer/instruct models are semantically more
contractive, while several legacy/base models echo templates or loop.

The defensible claim is therefore:

> Era alone does not predict stability. Training recipe, template behavior,
> tokenizer surface, and response attractors all change the sensitivity
> signature. `LLaMA-1 7B` is the stable legacy outlier; Gemma base-vs-instruct
> is the cleaner evidence that recipe matters.

## Evidence

Panel: 14 models, 5 perturbation tiers (noop_format, punctuation, synonym, paraphrase, semantic_small), mean token edit distance across tiers at decoding step t=60 (greedy, 64-token horizon). Higher = more divergent output.

### Mean edit-distance at t=60, ordered by release year

| Year | Model                     | t=20  | t=60  | Instruct? |
|------|---------------------------|-------|-------|-----------|
| 2019 | GPT-2 XL                  | 0.472 | 0.513 | no        |
| 2021 | GPT-J 6B                  | 0.589 | 0.589 | no        |
| 2022 | OPT 6.7B                  | 0.523 | 0.538 | no        |
| 2023 | LLaMA-1 7B                | 0.249 | 0.317 | no        |
| 2023 | Pythia 6.9B               | 0.603 | 0.626 | no        |
| 2024 | Mistral 7B v0.3           | 0.438 | 0.516 | yes       |
| 2025 | Qwen3.5 0.8B              | 0.449 | 0.535 | yes       |
| 2025 | Qwen3.5 2B                | 0.448 | 0.567 | yes       |
| 2025 | Qwen3.5 4B (thinkoff)     | 0.490 | 0.593 | yes       |
| 2025 | Qwen3.5 9B (thinkoff)     | 0.431 | 0.543 | yes       |
| 2025 | Falcon 3 10B              | 0.446 | 0.594 | yes       |
| 2025 | Gemma 4 E4B it            | 0.416 | 0.576 | yes       |
| 2025 | Granite 3.3 8B            | 0.477 | 0.627 | yes       |
| 2025 | OLMo-3 7B                 | 0.565 | 0.672 | yes       |

### Aggregate

- **Legacy (≤2023, n=5):** mean t=60 divergence = **0.517**
- **Modern (≥2024, n=9):** mean t=60 divergence = **0.580**
- Difference: +0.063 (modern is ~12% more divergent)
- Pearson correlation (year × t=60 divergence): **r = +0.286**, n=14

### Per-cohort variance

- **Base (non-instruct, all legacy, n=5):** mean = 0.517, std = 0.107 (range 0.317–0.626)
- **Instruct (all modern, n=9):** mean = 0.580, std = 0.046 (range 0.516–0.672)

The modern cohort has *tighter* variance — every modern instruct-tuned model lands in a narrow 0.52–0.67 band. The legacy cohort is bimodal: Pythia-6.9B is actually the most divergent model in the entire panel (0.626), while LLaMA-1 7B is the most stable (0.317). The "legacy is more stable" story is carried substantially by LLaMA-1.

## Confounds that prevent a clean claim

1. **Era is collinear with instruction tuning.** All legacy models in our panel are base models; all modern models are instruct-tuned. We cannot separate "older → more stable" from "base → more stable." To do so we would need at least one modern base model and one legacy instruct model in comparable configurations.

2. **Era is collinear with RLHF.** Same issue — all legacy are pre-RLHF, all modern are post-RLHF. Fine-tuning for preference may systematically reshape output distributions in ways that amplify divergence.

3. **Era is collinear with tokenizer and chat-template conventions.** Older models have fewer tokens to spend on structure, and no chat template. Divergence measured in token-edit-distance space is sensitive to tokenizer vocabulary, which has grown substantially.

4. **Small N, large variance in legacy cohort.** LLaMA-1 single-handedly pulls the legacy mean down. Remove it and legacy mean jumps to 0.567, erasing the effect.

5. **"Stability" was originally defined only as sequence-level edit-distance.**
   A model that answers every prompt with the same short refusal would look
   maximally stable by this metric. The semantic follow-up directly checks this
   concern and flips the era trend, consistent with several legacy/base models
   being more templated or loopy rather than more semantically robust.

6. **Greedy decoding only.** All numbers above are from greedy (temperature-0) runs. Sampling could reveal a different picture.

## Stronger version of the claim, if we can support it

> Modern instruction-tuned LLMs may exhibit more sensitive *surface token*
> trajectories under small prompt perturbations while also being more
> semantically contractive at the answer-content level. Across 14 models
> spanning 2019-2025, token edit distance at `t=60` has a weak positive
> correlation with release year (`r = +0.29`), but 512-token semantic distance
> has a stronger negative correlation (`r = -0.72`). That split is consistent
> with a hypothesis that post-training changes response style and answer basins
> without necessarily making answer content more fragile.

Note this is not a clean negative result against newer models. It is a warning
that "stability" is not one scalar: token trajectory stability, semantic answer
stability, template adherence, and collapse/responsiveness are separable axes.

## What we should do before publishing

1. **Expand within-family base-vs-instruct controls.** Gemma 4 base is now in
   the panel and already shows that recipe matters. More families would be
   needed before making a general base-vs-instruct claim.

2. **Run instruction tuning within era.** If we can run one 2019-era model (GPT-2 XL) with an instruct finetune (FLAN, Alpaca, etc.) and compare against vanilla GPT-2 XL, that would directly probe the instruct-vs-base axis separated from era.

3. **Remove LLaMA-1 and re-compute.** Done for the token-edit trend: removing
   `LLaMA-1 7B` nearly erases the legacy-vs-modern gap (`0.567` vs `0.580`).
   The claim is closer to "`LLaMA-1` is unusually stable" than "legacy is
   stable."

4. **Diverge-quality correlation check.** Correlate model divergence with a benchmark quality score (MMLU, HumanEval, whatever's available across this panel). If divergence tracks *inversely* with quality, the claim risks being reduced to "lower-quality models repeat themselves more."

5. **Beyond edit distance.** Semantic cosine distance is the current tie-breaker
   for the talk framing because it is less sensitive to surface-form shuffling.
   The semantic reanalysis already flips the era trend, so any future chart
   should show token and semantic views side by side rather than treating
   token edit distance as the finding.

## Suggested chart

Two-panel scatter plot: x = release year (2019-2025). Left y-axis = mean
`t=60` token edit divergence. Right y-axis = 512-token semantic distance.
One point per model. Color/marker by instruction-tuned (filled) vs base
(open). Display Pearson `r` on each panel. Annotate `LLaMA-1`, `OLMo-3`, and
the Gemma base/instruct pairs.

The point of the chart should be the disagreement between metrics, not a fitted
era trend. A good caption would be: "Surface token paths and semantic content
can move in opposite directions."

## Verdict

- **Safe to include only as a preliminary measurement caution.**
- **Not safe as "older models are more stable."** The semantic reanalysis
  contradicts that headline.
- **Worth investing in** as a token-vs-semantic split: if it holds up, it adds
  a useful dimension to stability evals and gives a concrete reason to report
  multiple metrics instead of a single leaderboard.
