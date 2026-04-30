# Prior Art and Experiment Implications

Last updated: 2026-04-30.

This is a pivot-specific prior-art review for the current trajectory-branching
thread in this repo. It is intentionally more tactical than
`docs/prior_art.md`: the goal is to identify which existing work already covers
the ground, which claims need caveats, and which methods can directly improve
E05-E10.

## Executive Readout

The broad claim "small prompt changes can change LLM behavior" is not new.
There is already a substantial prompt-sensitivity literature covering
formatting changes, prompt order, adversarial prompt perturbations,
intent-preserving rewrites, and worst-case prompt performance.

The project still has a differentiated angle if it stays focused on:

- token-certified prompt perturbations rather than raw character edits;
- localized branch events rather than aggregate output instability;
- separating at-branch diagnosis from true pre-branch warning;
- logit, margin, entropy, hidden-state, and text signals on the same paired
  trajectories;
- forced-prefix or activation-patching interventions at selected branch
  points;
- scaffold/reasoning streams as response-attractor confounds rather than as
  simple robustness wins.

The safest framing is:

> Prior work shows prompt sensitivity exists. This project asks where, when,
> and how a tiny token-visible prompt perturbation becomes a different
> autoregressive trajectory.

Avoid pitching this as "we discovered LLM prompt chaos" or "we measured a
Lyapunov exponent." The stronger technical story is finite-horizon branching
in a hybrid sequential system: continuous hidden/logit states repeatedly cross
discrete token decision boundaries.

## Current Repo Frame

Relevant current experiments:

- E05: token-certified micro perturbations.
- E06: prompt-end and teacher-forced logit boundary probes.
- E07: residual activation patching of branch tokens.
- E08: SAE feature pilot at causal branch positions.
- E09: trajectory-event mining and branch-window AUROC.
- E10: hidden/logit silent divergence along shared generated prefixes.

Current repo thesis:

> Under token-visible tiny perturbations, paired generations can follow the
> same visible prefix until localized branch events occur. Those events may be
> preceded by silent logit or hidden-state divergence, enriched near low-margin
> decision cliffs or high-confidence basin switches, and amplified into
> downstream semantic differences.

That thesis is not fully proven yet. The current evidence supports at-branch
diagnosis more strongly than pre-branch warning. Keep the distinction explicit.

## Literature Clusters

### 1. Prompt Sensitivity and Formatting Brittleness

This is the crowded baseline literature. Cite it early so the project does not
look like it is rediscovering prompt brittleness.

#### Salinas and Morstatter 2024: The Butterfly Effect of Altering Prompts

Reference:

- Abel Salinas and Fred Morstatter, "The Butterfly Effect of Altering Prompts:
  How Small Changes and Jailbreaks Affect Large Language Model Performance."
- arXiv: https://arxiv.org/abs/2401.03729
- ACL Anthology: https://aclanthology.org/2024.findings-acl.275/

Main relevance:

- Very close to the repo's original intuitive framing.
- Shows small prompt construction changes, including trailing spaces, can flip
  classification answers.
- Good citation for "this phenomenon exists."

Experiment implications:

- Do not claim novelty for whitespace or formatting sensitivity itself.
- Use this as a foil: their unit is usually task answer change; this repo's
  unit is the branch event inside the generated trajectory.
- Add a comparison paragraph in any writeup: "Unlike prompt-label robustness
  work, we inspect where the paired token trajectories diverge and whether
  logit/hidden signals forecast or cause that divergence."

#### Sclar et al. 2023/2024: Prompt Formatting Sensitivity

Reference:

- Melanie Sclar, Yejin Choi, Yulia Tsvetkov, Alane Suhr, "Quantifying
  Language Models' Sensitivity to Spurious Features in Prompt Design or: How I
  learned to start worrying about prompt formatting."
- arXiv: https://arxiv.org/abs/2310.11324
- OpenReview: https://openreview.net/forum?id=RIu5lyNXjT

Main relevance:

- Shows large accuracy swings from prompt formatting in few-shot settings.
- Reports that sensitivity can persist under increased model size, more
  demonstrations, and instruction tuning.

Experiment implications:

- The repo's "bigger is not monotonic" result has precedent. Do not overframe
  it as surprising by itself.
- Their few-shot format sensitivity suggests adding a controlled few-shot
  branch-event lane could be useful, but only if it serves E09/E10. Avoid a
  broad few-shot benchmark detour.
- If extending prompts, explicitly separate:
  - zero-shot task prompt perturbations;
  - few-shot ordering/formatting perturbations;
  - chat-template/tokenizer perturbations.

#### Zhao et al. 2021: Calibrate Before Use

Reference:

- Tony Z. Zhao, Eric Wallace, Shi Feng, Dan Klein, Sameer Singh, "Calibrate
  Before Use: Improving Few-Shot Performance of Language Models."
- arXiv: https://arxiv.org/abs/2102.09690

Main relevance:

- Early GPT-3/GPT-2 prompt instability paper.
- Shows prompt format, examples, and example order can move accuracy from near
  chance to near state-of-the-art.
- Introduces contextual calibration using content-free inputs to estimate
  label bias.

Experiment implications:

- For classification-style prompts, logit bias toward answer tokens may be a
  trivial explanation. The repo should prefer open generation branch cases or
  include label-bias controls.
- A useful adaptation: add content-free or "N/A" prompt controls for
  classification-like branches to distinguish semantic perturbation sensitivity
  from answer-token prior bias.

#### Lu et al. 2021/2022: Fantastically Ordered Prompts

Reference:

- Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, Pontus Stenetorp,
  "Fantastically Ordered Prompts and Where to Find Them: Overcoming Few-Shot
  Prompt Order Sensitivity."
- arXiv: https://arxiv.org/abs/2104.08786
- ACL Anthology: https://aclanthology.org/2022.acl-long.556/

Main relevance:

- Demonstrates that few-shot example order can dominate performance.
- Good permutations are not reliably transferable across models.

Experiment implications:

- If the project adds few-shot examples, model-specific branch boundaries are
  expected. Non-transfer across models is prior art, not a new result.
- Use entropy or margin statistics of candidate prompt variants as a cheap
  screen for branch-risky prompt neighborhoods.

#### PromptRobust 2023/2024

Reference:

- Kaijie Zhu et al., "PromptRobust: Towards Evaluating the Robustness of Large
  Language Models on Adversarial Prompts."
- arXiv: https://arxiv.org/abs/2306.04528
- GitHub: https://github.com/amazon-science/PromptRobust

Main relevance:

- Evaluates adversarial prompt perturbations across character, word, sentence,
  and semantic levels.
- Large benchmark orientation.

Experiment implications:

- Avoid rebuilding a broad adversarial-prompt benchmark unless there is a very
  specific branch-event question.
- Useful source of perturbation operators if E05 needs a larger, less
  hand-authored perturbation pool.
- The repo's token-certified filter is a good improvement over raw perturbation
  categories; keep it central.

#### POSIX 2024

Reference:

- "POSIX: A Prompt Sensitivity Index For Large Language Models."
- arXiv: https://arxiv.org/abs/2410.02185
- ACL Anthology: https://aclanthology.org/2024.findings-emnlp.852/

Main relevance:

- Defines a prompt-sensitivity index using relative change in log-likelihood
  of a fixed response under intent-preserving prompt replacements.

Experiment implications:

- This is close to the repo's teacher-forced logit lane.
- Consider adding a POSIX-like fixed-response likelihood delta as a scalar
  baseline in E06/E09:
  - likelihood of clean continuation under clean prompt;
  - likelihood of clean continuation under perturbed prompt;
  - likelihood of perturbed continuation under clean prompt;
  - likelihood of perturbed continuation under perturbed prompt.
- This could help distinguish "the branch continuation became globally less
  likely" from "argmax crossed a narrow local boundary."

#### RobustAlpacaEval / Worst Prompt Performance 2024

Reference:

- "On the Worst Prompt Performance of Large Language Models."
- arXiv: https://arxiv.org/abs/2406.10248
- GitHub: https://github.com/cbwbuaa/On-the-Worst-Prompt-Performance-of-LLMs

Main relevance:

- Argues for evaluating lower-bound performance over semantically equivalent
  prompts, not just average performance.

Experiment implications:

- If this repo becomes a tool, worst-neighborhood behavior is a practical
  product metric.
- For the current research frame, avoid turning E09 into "worst prompt score."
  Worst-case aggregation hides branch anatomy.
- A useful later metric: "worst branch lead-time / worst semantic branch under
  an equivalence class."

#### ProSA 2024

Reference:

- Jingming Zhuo et al., "ProSA: Assessing and Understanding the Prompt
  Sensitivity of LLMs."
- ACL Anthology: https://aclanthology.org/2024.findings-emnlp.108/
- arXiv: https://arxiv.org/abs/2410.12405
- GitHub: https://github.com/open-compass/ProSA

Main relevance:

- Instance-level prompt sensitivity framework.
- Introduces PromptSensiScore.
- Uses decoding confidence to analyze mechanisms.
- Finds higher model confidence correlates with prompt robustness.

Experiment implications:

- This strongly supports the repo's margin/confidence direction.
- Add ProSA/PromptSensiScore as a baseline aggregate if a paper-grade
  comparison is needed.
- Do not stop at aggregate confidence. The repo's advantage is locating where
  confidence/margin changes along paired trajectories.
- Candidate feature to add to E09 windows: normalized confidence drop relative
  to model's own local confidence distribution, not only raw margin or JS.

#### Benchmarking Prompt Sensitivity / PromptSET 2025

Reference:

- Amirhossein Razavi et al., "Benchmarking Prompt Sensitivity in Large
  Language Models."
- arXiv: https://arxiv.org/abs/2502.06065

Main relevance:

- Treats prompt sensitivity prediction itself as a task.
- Benchmarks LLM self-evaluation, text classification, and query-performance
  prediction methods.

Experiment implications:

- If E09 wants to predict branch risk before generation, this is the closest
  benchmark framing.
- Potentially useful baseline: train a simple prompt-pair classifier from
  prompt-level features, then compare against logit/hidden trajectory features.
  If prompt-only features already predict most branch risk, the silent
  divergence story weakens.
- Do this as a falsification check, not as a new benchmark project.

#### Prompt Underspecification Work 2025/2026

References:

- Branislav Pecher et al., "Revisiting Prompt Sensitivity in Large Language
  Models for Text Classification: The Role of Prompt Underspecification."
  https://arxiv.org/abs/2602.04297
- "What Prompts Don't Say: Understanding and Managing Underspecification in
  LLM Prompts." https://arxiv.org/abs/2505.13360

Main relevance:

- Some measured sensitivity may come from underspecified prompts rather than
  fragility.
- The classification paper reports that underspecified prompts have higher
  variance and lower relevant-token logits, while specific instruction prompts
  reduce the issue.
- It also reports that underspecification effects may emerge mainly in final
  layers, with only marginal impact on internal representations.

Experiment implications:

- This is a major control requirement. Label each prompt pair by
  underspecification level.
- Add "fully specified" versions of high-branch prompts:
  - explicit output format;
  - explicit task objective;
  - explicit treatment of whitespace/punctuation;
  - explicit instruction not to infer hidden requirements.
- Hypothesis to test:
  - If branch events vanish under full specification, the effect is mostly
    ambiguity/underspecification.
  - If token-certified formatting edits still branch under fully specified
    prompts, the boundary story is stronger.
- For E10 hidden-state claims, the final-layer emergence result is a warning:
  branch-relevant differences may be late-logit phenomena, not deep semantic
  representation shifts. That makes late-layer patching less surprising and
  early-layer rescue more valuable.

### 2. Dynamical Systems, Attractors, and Chaos Framing

This literature is useful for the talk lens. It is less useful as proof of
novelty because many works use dynamical language loosely.

#### Li et al. 2025: Quasi-Lyapunov Analysis

Reference:

- Xiaojian Li et al., "Cognitive Activation and Chaotic Dynamics in Large
  Language Models: A Quasi-Lyapunov Analysis of Reasoning Mechanisms."
- arXiv: https://arxiv.org/abs/2503.13530

Main relevance:

- Direct LLM chaos/Lyapunov framing.
- Claims minor perturbations substantially affect reasoning.

Experiment implications:

- Use as a citation that LLM dynamics/chaos framing exists.
- Do not adopt "LLMs are chaotic systems" as the repo claim.
- If discussing Lyapunov, call repo metrics finite-horizon proxies or
  branch-event diagnostics, not Lyapunov exponents.

#### Wang et al. 2025: Attractor Cycles in Successive Paraphrasing

Reference:

- Zhilin Wang et al., "Unveiling Attractor Cycles in Large Language Models: A
  Dynamical Systems View of Successive Paraphrasing."
- ACL Anthology PDF: https://aclanthology.org/2025.acl-long.624.pdf

Main relevance:

- Treats iterative paraphrasing as a dynamical system.
- Shows convergence to stable periodic states such as 2-cycles.

Experiment implications:

- Good support for "attractor" language, but their map is repeated model calls
  under paraphrase transformation, not one autoregressive run.
- A possible future extension: after a branch event, test whether continuations
  reconverge semantically under paraphrase or summarization transforms.
- Do not let this drag E09 into iterative paraphrasing unless reconvergence
  becomes the explicit question.

#### Recurrent Self-Attention Dynamics / Neural Dynamics Precedent

References:

- Sompolinsky, Crisanti, Sommers 1988, "Chaos in Random Neural Networks."
- Poole et al. 2016, "Exponential Expressivity in Deep Neural Networks Through
  Transient Chaos." https://arxiv.org/abs/1606.05340
- Schoenholz et al. 2017, "Deep Information Propagation."
  https://arxiv.org/abs/1611.01232
- Tomihari and Karakida 2025, "Recurrent Self-Attention Dynamics: An
  Energy-Agnostic Perspective from Jacobians." https://arxiv.org/abs/2505.19458

Main relevance:

- Establishes that neural networks and attention can be studied with
  dynamical-systems tools.

Experiment implications:

- Useful talk background, not a direct methods path for the current repo.
- Avoid computing formal Jacobian/Lyapunov spectra unless the project pivots
  into theory. It is likely a time sink relative to E09/E10.
- If a single theoretical metric is needed, finite-time local expansion of
  hidden/logit distances along teacher-forced shared prefixes is more aligned
  with current artifacts than full Lyapunov estimation.

### 3. Probability Concentration, Branching Factor, and Alignment

#### Yang and Holtzman 2025/2026: LLM Probability Concentration

Reference:

- Chenghao Yang and Ari Holtzman, "LLM Probability Concentration: How Alignment
  Shrinks the Generative Horizon."
- arXiv: https://arxiv.org/abs/2506.17871
- OpenReview: https://openreview.net/forum?id=oRnOH9N3Bl

Main relevance:

- Introduces Branching Factor (BF), a token-invariant measure of the effective
  number of plausible next steps.
- Finds BF often decreases as generation progresses.
- Finds alignment tuning sharply reduces BF from the outset.
- Argues aligned/CoT models can appear more stable because they move into
  lower-entropy trajectories, often via stylistic tokens.

Experiment implications:

- This is one of the most relevant finds for the current pivot.
- Add BF or effective support size to E06/E09 alongside entropy, margin, JS,
  and top-1 probability.
- Use BF to separate:
  - low-margin cliff: multiple plausible next tokens with a narrow winner;
  - high-confidence basin switch: both prompts have concentrated but different
    next-token distributions;
  - scaffold attractor: both prompts enter the same low-BF stylistic path.
- This paper gives a principled version of the repo's scaffold confound.
  "Reasoning scaffold stability" can be reframed as entry into a low-entropy
  response attractor.
- Suggested metric:
  - `branching_factor = exp(entropy(logits))` over full vocab or top-k
    renormalized mass.
  - Track `bf_clean`, `bf_corrupt`, `bf_delta`, and `bf_min`.
  - Compute BF at prompt end, pre-branch windows, branch token, and post-branch
    persistence windows.

Applied in repo:

- Added full-vocab effective branching factor (`exp(entropy)`) to logit capture
  rows and downstream E09/E10 summaries.
- Added E09 branch/window BF fields so cases can be separated into low-margin
  cliffs versus high-confidence/low-BF basin switches.
- Existing logit artifacts can be reprocessed because E09 derives BF from
  entropy when explicit BF columns are absent.

Potential dead end:

- Do not make BF the whole thesis. It may duplicate entropy/top-k metrics
  unless it cleanly explains scaffold/base-vs-instruct differences.

### 4. Determinism and Numerical Confounds

This literature is not the main project, but it protects the current claims.
The repo intentionally uses deterministic decode to isolate prompt
perturbation. That only works if runtime nondeterminism is controlled or
logged.

#### BF16 / Hardware / Batch Nondeterminism

References:

- "Why Your LLM's 'Deterministic' Output Isn't-And How to Fix It."
  https://deep-paper.org/en/paper/2506.09501/
- Horace He / Thinking Machines Lab, "Defeating Nondeterminism in LLM
  Inference." https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
- Raja Gond et al., "LLM-42: Enabling Determinism in LLM Inference with
  Verified Speculation." https://arxiv.org/abs/2601.17768
- "Deterministic Inference across Tensor Parallel Sizes That Eliminates
  Training-Inference Mismatch." https://arxiv.org/abs/2511.17826

Main relevance:

- Greedy decoding is theoretically deterministic, but deployed inference can
  vary due to precision, reduction order, dynamic batching, tensor parallelism,
  and kernel behavior.
- This matters especially near low-margin argmax boundaries.

Experiment implications:

- Keep logging:
  - model ID and revision;
  - tokenizer revision;
  - torch/transformers versions;
  - device;
  - dtype;
  - backend;
  - batch size;
  - tensor parallelism if any;
  - git SHA and dirty state.
- The repo's E10 metadata recapture is exactly the right move.
- Add identical-prompt repeat controls near selected branch cases, not only
  generic smoke controls:
  - same prompt, same runtime, repeated N times;
  - same prompt, different batch size if feasible;
  - same prompt, FP32 vs FP16/BF16 for small models if feasible.
- If a branch token flips under identical-prompt repeats, it is a numerical
  nondeterminism case, not prompt sensitivity.

Potential dead end:

- Full deterministic kernel engineering is out of scope. The practical move is
  controls and metadata, not rebuilding inference.

### 5. Hidden-State, Logit, and Confidence-Based Risk Signals

This cluster is directly useful for E06/E09/E10.

#### CCPS 2025: Perturbed Representation Stability

Reference:

- Reza Khanmohammadi et al., "Calibrating LLM Confidence by Probing Perturbed
  Representation Stability."
- arXiv: https://arxiv.org/abs/2505.21772
- ACL Anthology: https://aclanthology.org/2025.emnlp-main.530/

Main relevance:

- Applies targeted perturbations to final hidden states.
- Extracts stability features from the model's response to perturbation.
- Uses those features to predict confidence/correctness.

Experiment implications:

- Strong support for using internal perturbation sensitivity as a confidence
  signal.
- For E10, consider a small "epsilon-to-flip" analogue:
  - At branch windows, add controlled perturbations in the direction of
    clean-corrupt hidden-state delta, random orthogonal directions, and
    top-token contrast directions.
  - Measure how much perturbation is needed to flip top-1.
- This could distinguish a genuinely fragile decision boundary from a broad
  high-confidence basin switch.
- Keep it small. A full CCPS-style classifier is probably a detour unless the
  project becomes branch-risk prediction.

#### Semantic Entropy and Hallucination Uncertainty

References:

- Kuhn et al., "Detecting hallucinations in large language models using
  semantic entropy." Nature 2024. https://www.nature.com/articles/s41586-024-07421-0
- "Do LLMs Know about Hallucination? An Empirical Investigation of LLM's Hidden
  States." https://arxiv.org/abs/2402.09733
- "HalluCana: Fixing LLM Hallucination with A Canary Lookahead."
  https://arxiv.org/abs/2412.07965
- Amazon Science page: https://www.amazon.science/publications/hallucana-fixing-llm-hallucination-with-a-canary-lookahead

Main relevance:

- Uses entropy, semantic uncertainty, hidden states, and lookahead branches to
  detect/correct generation risk.
- HalluCana's lookahead branch idea is adjacent to branch-event analysis.

Experiment implications:

- Add semantic-entropy-style multi-sample controls only if sampling enters the
  story. For deterministic E09, token/logit entropy is more direct.
- HalluCana suggests a useful intervention process:
  - at a candidate risky branch window, create canary lookaheads for top-k
    candidate tokens;
  - measure whether branches quickly reconverge, diverge semantically, or hit
    scaffold loops;
  - use this to type branch events, not just detect them.
- A branch event that immediately reconverges is less interesting than one
  whose top-k lookaheads enter distinct semantic basins.

Potential dead end:

- Hallucination detection itself is not the repo thesis. Use uncertainty
  methods only as branch-risk tools.

### 6. Mechanistic Interpretability, Activation Patching, SAEs, and Circuits

This literature validates the E07/E08 direction but also raises the bar:
feature labels are not enough; interventions matter.

#### Activation Patching / Causal Tracing

References:

- TransformerLens activation patching docs:
  https://transformerlensorg.github.io/TransformerLens/generated/code/transformer_lens.patching.html
- nnsight activation patching tutorial:
  https://nnsight.net/tutorials/tutorials/causal_mediation_analysis/activation_patching/
- LessWrong explainer, "How to use and interpret activation patching":
  https://www.lesswrong.com/posts/FhryNAFknqKAdDcYy/how-to-use-and-interpret-activation-patching

Main relevance:

- Activation patching asks which clean activations restore clean behavior in a
  corrupted run.
- This is exactly E07's causal shape.

Experiment implications:

- Keep patching metrics behavior-specific:
  - rescue clean branch token;
  - suppress corrupt branch token;
  - delay branch;
  - preserve common prefix longer;
  - change downstream semantic basin.
- Add negative controls:
  - patch unrelated prompt positions;
  - patch random same-norm vectors;
  - patch from a third prompt with similar length;
  - patch layers/positions outside the selected band.
- Report patching as local causality, not "the model has a whitespace feature."

#### Anthropic Circuit Tracing 2025

Reference:

- Anthropic, "Circuit Tracing: Revealing Computational Graphs in Language
  Models." https://transformer-circuits.pub/2025/attribution-graphs/methods.html
- Open-source tools announcement:
  https://www.anthropic.com/research/open-source-circuit-tracing

Main relevance:

- Treats attribution graphs as hypotheses that require validation via
  perturbation/intervention.
- Uses feature-level computational graphs, transcoders, and interventions.

Experiment implications:

- Good north star for E08, but too heavy to replicate wholesale.
- Do not spend time building full attribution graphs unless E07/E08 first
  yields a small set of highly replayable, high-effect branch cases.
- Borrow the discipline:
  - feature observation -> hypothesis -> intervention -> output effect.
- The casebook should tell one mechanistic story per branch, not dump feature
  IDs.

#### Anthropic Scaling Monosemanticity / Mapping the Mind

Reference:

- Anthropic, "Mapping the Mind of a Large Language Model."
  https://www.anthropic.com/research/mapping-mind-language-model
- Transformer Circuits, "Scaling Monosemanticity."
  https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html

Main relevance:

- SAE features can be human-interpretable and causally manipulable.

Experiment implications:

- Use as an interpretability anchor, but avoid overclaiming from top-k feature
  overlap alone.
- For E08, the minimum upgrade from feature-ID evidence is:
  - show top activating dataset examples for a branch-relevant feature;
  - ablate or boost the feature if tooling supports it;
  - show effect on branch-token logits or rescue fraction.

#### Gemma Scope / Gemma Scope 2

References:

- Gemma Scope blog:
  https://deepmind.google/blog/gemma-scope-helping-the-safety-community-shed-light-on-the-inner-workings-of-language-models/
- Gemma Scope page:
  https://deepmind.google/models/gemma/gemma-scope/
- Gemma Scope paper: https://arxiv.org/abs/2408.05147

Main relevance:

- Provides open SAEs across Gemma layers/sublayers.
- Gemma Scope 2 includes SAEs and transcoders for Gemma 3.

Experiment implications:

- If Qwen-Scope feature semantics are hard to label, Gemma may be the better
  model organism for feature-level stories.
- Candidate path:
  - Use Gemma E2B/E4B branch cases from E09.
  - Find a branch event with strong logit/margin signal and enough shared
    prefix.
  - Run Gemma Scope SAE feature extraction at prompt-boundary and branch
    positions.
  - Prefer cases where the feature has Neuronpedia examples or existing labels.
- Avoid switching the whole project to Gemma just because the SAE ecosystem is
  nicer. Use it for one interpretable case study.

#### Qwen-Scope

References:

- Qwen SAE example model card:
  https://huggingface.co/Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_100
- Larger Qwen3.5 SAE card:
  https://huggingface.co/Qwen/SAE-Res-Qwen3.5-35B-A3B-Base-W128K-L0_100

Main relevance:

- Provides residual-stream SAEs for Qwen3/Qwen3.5 models.
- Directly compatible with the current Qwen3.5 branch cases.

Experiment implications:

- E08 is well aligned with this resource.
- Next useful step is not more feature IDs; it is intervention or labeling:
  - find feature exemplars;
  - compare clean/corrupt activations at causal patch positions;
  - ablate/boost candidate features;
  - verify branch-logit movement.
- If using base-model SAEs on instruct/thinking-off checkpoints, label that as
  an approximation.

### 7. Reasoning, Scaffolds, and Low-Entropy Response Attractors

This thread is central to the repo's scaffold confound.

Relevant references:

- Yang and Holtzman, probability concentration / branching factor:
  https://arxiv.org/abs/2506.17871
- Sclar et al., instruction tuning does not eliminate formatting sensitivity:
  https://arxiv.org/abs/2310.11324
- Controlled prompt variation reasoning robustness:
  https://arxiv.org/abs/2504.02111

Main relevance:

- Reasoning or instruction scaffolds can create stable visible prefixes without
  proving content robustness.
- Some perturbations can accidentally trigger chain-of-thought-like behavior.
- Alignment may narrow the generation horizon by steering into stylistic
  low-entropy paths.

Experiment implications:

- Treat scaffold entry as a branch event type:
  - both prompts enter same scaffold;
  - one prompt enters scaffold and the other direct-answers;
  - both enter scaffold but diverge internally before final answer;
  - hidden/logit divergence occurs while visible scaffold tokens remain equal.
- Add scaffold mask rate to E09:
  - fraction of branches occurring before scaffold boundary;
  - inside scaffold;
  - at answer boundary;
  - after answer starts.
- For reasoning-on models, do not compare answer stability unless answer
  extraction is reliable. Compare deliberation-stream stability separately.

## Concrete Algorithmic Ideas To Try

### A. Add Branching Factor to E06/E09

Implement per-position:

- full-vocab entropy;
- effective branching factor `exp(entropy)`;
- top-k effective branching factor after top-k renormalization;
- top-1/top-2 margin;
- top-1 probability;
- top-k cumulative mass;
- JS/KL between clean and corrupt distributions.

Use these to classify branch windows:

- low-margin cliff: low margin, high or medium BF;
- basin switch: high confidence on both sides, different top tokens;
- diffuse uncertainty: high BF and high entropy;
- scaffold attractor: low BF and shared scaffold-token prefix;
- numeric-risk case: near-tie plus inconsistent identical-prompt repeats.

### B. POSIX-Style Cross-Likelihood Matrix

For each paired branch case, compute:

| Prompt | Continuation scored |
| --- | --- |
| clean prompt | clean continuation |
| clean prompt | corrupt continuation |
| corrupt prompt | clean continuation |
| corrupt prompt | corrupt continuation |

This helps answer:

- Is each continuation only likely under its own prompt?
- Is one continuation globally dominant?
- Did the perturbation mainly flip a local argmax while both paths remain
  plausible?
- Does branch persistence show up as likelihood separation after the branch?

### C. Prompt-Only Falsification Baseline

Before claiming silent hidden/logit warning, train or compute cheap prompt-only
features:

- prompt token edit distance;
- edit category;
- edit position relative to message/template boundaries;
- changed token IDs;
- token frequency/rank;
- tokenizer boundary indicators;
- prompt length;
- model family;
- scaffold label.

Then compare branch-risk prediction:

- prompt-only baseline;
- prompt-end logits;
- pre-branch logits;
- hidden-state deltas;
- combined features.

If prompt-only features explain most branch risk, the mechanism story should
become "tokenizer/edit-boundary risk typing" rather than silent trajectory
warning.

### D. Underspecification Control Pair

For selected high-branch prompts, generate a fully specified variant:

- explicit output contract;
- explicit handling of punctuation/whitespace;
- explicit no hidden assumptions;
- explicit answer format.

Run the same token-certified perturbations.

Interpretation:

- Branch disappears: original was underspecification-sensitive.
- Branch remains but moves later: specification delays basin choice.
- Branch remains at same edit boundary: stronger evidence for local
  tokenization/logit boundary sensitivity.

### E. Branch Lookahead Typing

At a risky branch position:

1. Force top-k candidate next tokens for both clean and corrupt prompts.
2. Greedily continue for a short horizon.
3. Measure semantic distance, reconvergence, scaffold entry, and final answer
   differences.

This answers whether the branch is:

- cosmetic and quickly reconvergent;
- scaffold-routing;
- semantic basin switch;
- pathological loop;
- task-answer change.

This is a practical middle ground between pure observation and full activation
patching.

### F. Epsilon-to-Flip / Directional Perturbation

For a selected branch window:

- perturb final hidden state in random directions;
- perturb in clean-minus-corrupt direction;
- perturb in unembedding contrast direction for clean-vs-corrupt branch token;
- measure minimum epsilon needed to flip top-1.

This gives a boundary-distance estimate that is more mechanistic than raw
margin.

Use sparingly. This can become a rabbit hole.

### G. Activation Patching Controls

For each high-signal E07 case, add:

- random same-norm patch;
- unrelated prompt patch;
- non-causal position patch;
- layer-local patch outside expected band;
- clean-to-corrupt and corrupt-to-clean directions;
- rescue-token logit delta, not only token outcome.

This makes the causal claim more defensible.

### H. Feature-Level Upgrade Path

For E08:

1. Choose one branch case with strong E07 rescue and clean E09 event metadata.
2. Extract SAE features at causal layer/position.
3. Inspect top activating examples or Neuronpedia labels if available.
4. Ablate/boost candidate features if tooling supports it.
5. Report effect on branch-token logits and rescue fraction.

Stop if feature labels are vague or intervention has no effect. Feature IDs
without causal movement are backup material, not the core story.

## Paths That Look Less Worth It

### Broad Prompt Robustness Benchmark

PromptRobust, ProSA, POSIX, RobustAlpacaEval, and PromptSET already cover broad
evaluation. A new broad benchmark would be expensive and less distinctive than
the event-level branch analysis.

### Formal Lyapunov Exponent Claims

LLM generation is discrete-token, finite-horizon, template-mediated, and
runtime-dependent. A formal Lyapunov claim would require a much stronger state
definition and careful dynamical-system setup. It is not needed for the talk or
the current experiments.

### Full Circuit Tracing Pipeline

Anthropic-style attribution graphs are a north star, but building that stack is
too much unless a small number of branch cases become clearly worth tracing.
Activation patching plus SAE feature inspection is the pragmatic local path.

### Hallucination Detection as Main Target

Uncertainty and hallucination work provides useful methods, but hallucination
is not the repo thesis. Keep those methods as branch-risk diagnostics.

### Parameter-Count Scaling Story

Prior work already shows prompt sensitivity can persist across model sizes, and
the repo's own branch timing is not monotonic. Treat size as a model-family
contrast, not the causal axis.

## Promising Open Angles

### 1. Branch-Event Taxonomy

Prior work measures prompt sensitivity, confidence, or robustness. A reusable
taxonomy of branch event types may be genuinely useful:

- immediate branch;
- delayed visible branch;
- silent hidden/logit branch;
- scaffold-masked branch;
- reconvergent branch;
- persistent semantic basin split;
- high-confidence basin switch;
- low-margin cliff;
- numeric nondeterminism risk.

### 2. Token-Certified Perturbation Hygiene

Many prompt perturbation papers operate at character/string level. The repo's
token-certified filtering is a practical contribution, especially when chat
templates normalize away raw edits.

### 3. Branch Windows Instead of Whole Outputs

Most robustness work aggregates task accuracy or output distance. Branch-window
analysis can explain why aggregate metrics disagree.

### 4. Causal Branch Movement

The strongest paper-grade differentiator is showing that branch events are
not only observed but movable:

- forced-prefix intervention;
- activation patch rescue;
- feature ablation/boost;
- branch lookahead reconvergence.

### 5. Scaffold as Attractor, Not Noise

The scaffold confound can become a useful finding if framed as response
attractor selection:

- alignment and reasoning templates can compress the generation horizon;
- stable scaffold tokens can mask unstable hidden/logit states;
- answer-content robustness must be evaluated after scaffold/answer boundary.

## Suggested Next Experiments

### Near-Term, High Value

1. Add BF/effective-support metrics to E06/E09.
2. Add an underspecification-control variant for a handful of high-branch cases.
3. Add POSIX-style cross-likelihood matrix for recommended E09 cases.
4. Add identical-prompt repeat controls for selected low-margin branch cases.
5. Add branch lookahead typing for the most slide-worthy cases.

### Medium-Term

1. Build a small branch-event taxonomy table and label 50-100 cases.
2. Add prompt-only branch-risk baseline.
3. Expand E07 with negative patch controls and logit-delta outcomes.
4. Pick one Qwen-Scope or Gemma Scope case for a feature-level causal story.

### Only If This Becomes Paper-Grade

1. Larger randomized prompt neighborhoods stratified by underspecification.
2. Model-family matched base/instruct/scaffold comparisons.
3. Cross-runtime reproducibility lane for branch-sensitive cases.
4. Public casebook with raw generations, token diffs, branch windows, logits,
   and intervention outcomes.

## Reference List

### Prompt Sensitivity / Robustness

- Salinas and Morstatter, "The Butterfly Effect of Altering Prompts."
  https://arxiv.org/abs/2401.03729
- Sclar et al., "Quantifying Language Models' Sensitivity to Spurious Features
  in Prompt Design." https://arxiv.org/abs/2310.11324
- Zhao et al., "Calibrate Before Use." https://arxiv.org/abs/2102.09690
- Lu et al., "Fantastically Ordered Prompts."
  https://arxiv.org/abs/2104.08786
- Zhu et al., "PromptRobust." https://arxiv.org/abs/2306.04528
- POSIX. https://arxiv.org/abs/2410.02185
- "On the Worst Prompt Performance of Large Language Models."
  https://arxiv.org/abs/2406.10248
- ProSA. https://aclanthology.org/2024.findings-emnlp.108/
- PromptSET / Benchmarking Prompt Sensitivity.
  https://arxiv.org/abs/2502.06065
- Prompt underspecification in classification.
  https://arxiv.org/abs/2602.04297
- What Prompts Don't Say. https://arxiv.org/abs/2505.13360

### Dynamics / Attractors

- Li et al., "Cognitive Activation and Chaotic Dynamics in Large Language
  Models." https://arxiv.org/abs/2503.13530
- Wang et al., "Unveiling Attractor Cycles in Large Language Models."
  https://aclanthology.org/2025.acl-long.624/
- Poole et al., "Exponential Expressivity in Deep Neural Networks Through
  Transient Chaos." https://arxiv.org/abs/1606.05340
- Schoenholz et al., "Deep Information Propagation."
  https://arxiv.org/abs/1611.01232
- Tomihari and Karakida, "Recurrent Self-Attention Dynamics."
  https://arxiv.org/abs/2505.19458

### Probability Concentration / Branching Factor

- Yang and Holtzman, "LLM Probability Concentration: How Alignment Shrinks the
  Generative Horizon." https://arxiv.org/abs/2506.17871
- OpenReview page: https://openreview.net/forum?id=oRnOH9N3Bl

### Determinism / Numerical Confounds

- "Why Your LLM's 'Deterministic' Output Isn't-And How to Fix It."
  https://deep-paper.org/en/paper/2506.09501/
- Thinking Machines Lab, "Defeating Nondeterminism in LLM Inference."
  https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
- LLM-42. https://arxiv.org/abs/2601.17768
- Deterministic inference across tensor parallel sizes.
  https://arxiv.org/abs/2511.17826

### Confidence / Hidden-State / Hallucination Signals

- CCPS. https://arxiv.org/abs/2505.21772
- CCPS ACL page. https://aclanthology.org/2025.emnlp-main.530/
- Semantic entropy. https://www.nature.com/articles/s41586-024-07421-0
- Hidden states and hallucination. https://arxiv.org/abs/2402.09733
- HalluCana. https://arxiv.org/abs/2412.07965

### Mechanistic Interpretability

- TransformerLens activation patching docs.
  https://transformerlensorg.github.io/TransformerLens/generated/code/transformer_lens.patching.html
- nnsight activation patching tutorial.
  https://nnsight.net/tutorials/tutorials/causal_mediation_analysis/activation_patching/
- LessWrong activation patching explainer.
  https://www.lesswrong.com/posts/FhryNAFknqKAdDcYy/how-to-use-and-interpret-activation-patching
- Anthropic circuit tracing.
  https://transformer-circuits.pub/2025/attribution-graphs/methods.html
- Anthropic open-source circuit tracing.
  https://www.anthropic.com/research/open-source-circuit-tracing
- Anthropic, "Mapping the Mind of a Large Language Model."
  https://www.anthropic.com/research/mapping-mind-language-model
- Scaling Monosemanticity.
  https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html
- Gemma Scope. https://arxiv.org/abs/2408.05147
- Gemma Scope blog.
  https://deepmind.google/blog/gemma-scope-helping-the-safety-community-shed-light-on-the-inner-workings-of-language-models/
- Gemma Scope page. https://deepmind.google/models/gemma/gemma-scope/
- Qwen-Scope 2B SAE.
  https://huggingface.co/Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_100
- Qwen-Scope larger SAE example.
  https://huggingface.co/Qwen/SAE-Res-Qwen3.5-35B-A3B-Base-W128K-L0_100
