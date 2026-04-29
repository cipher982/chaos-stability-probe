# Talk Rebuttals and Steel Man Objections

This is a prep artifact, not the canonical current-state readout. For current
claims, use [results_digest.md](results_digest.md). This file is for likely
questions, objections, and sober answers.

## Core Position

This is a Learning Club talk. Teaching a lens, not defending a thesis.

Honest claim:

> LLMs aren't textbook chaotic systems — but at inference time they're hybrid
> sequential systems: continuous activations feed a discrete branching
> process. Small semantic or formatting changes can move distributions or flip
> argmax branches. Sensitivity is real, varies by model/prompt/metric, and is
> worth measuring. The chaos lens helps organize the phenomenon; it doesn't
> prove anything about LLMs.

## What I'm Not Claiming (Say Out Loud Early)

Disarms the obvious Q&A objections up front:

- **Not** "LLMs are chaotic." Classical chaos needs things LLMs don't have
  (infinitesimal perturbations, fixed iterated map, asymptotic limit).
- **Not** "I measured a Lyapunov exponent." Token space is discrete.
- **Not** "bigger = more stable" or "reasoning = stable." Neither holds up
  in the data (4B ≈ 9B; Phi-4 reasoning+ is in the brittle band at 512
  tokens).
- **Not** "sentence-embedding distance is ground truth." It's a proxy.
- **Not** "lower divergence = better." Stability is a property, not a score.

If any of those would have been someone's objection — good, we agree.

## Objections

### This is not chaos.

**Yes, but:** Correct. We measured prompt/output sensitivity and trajectory
branching, not formal Lyapunov exponents.

**Response:** Chaos theory motivated the question, but the experiment is a
controlled proxy. A formal chaos claim would require a stronger trajectory
divergence analysis and careful state-space definition.

**Talk version:** I am using chaos as the lens, not claiming a theorem. The
point is to teach the audience how to think about perturbation, amplification,
and stability margins in LLMs.

### You don't need chaos — this is logit-margin brittleness plus decoding discontinuity.

**Yes, but:** This is the strongest technical objection. Probably right.

**Response:** Agreed — logit margins and argmax boundaries are the **proximate
mechanism**. Small prompt change → argmax crosses a low-margin boundary →
different first token → autoregressive feedback → different trajectory. The
cross-model data supports it: top-1 probability at prompt-end correlates with
512-token semantic divergence at r = −0.84, while full-vocab JS divergence
correlates at r = −0.10. The bulk distribution often isn't moving; the
decision boundary is fragile.

**Talk version:** The dynamical-systems lens doesn't replace that mechanism,
it *organizes* it — as finite-time branching rather than as magic chaos. Own
the mechanism, keep the lens as the teaching frame.

### The prompt set is too small.

**Yes, but:** Correct. The expanded ladder has 21 prompt pairs, and the
small-perturbation ranking uses only 9 rows per model.

**Response:** This is not a benchmark. The robust claims are the large-effect
contrasts and the measurement method. Exact model ordering is not paper-grade.

### The prompts were hand-written, so the results could be prompt-selection bias.

**Yes, but:** Correct. A different prompt distribution could change the
ordering.

**Response:** The next step is a larger randomized prompt set stratified by
category. For this talk, the hand-written ladder is a controlled probe, not a
representative benchmark.

### The semantic metric is not ground truth.

**Yes, but:** Correct. `all-MiniLM-L6-v2` cosine distance is a pragmatic proxy,
not an oracle.

**Response:** We also log token edit distance, common prefix length, raw
generations, hidden-state distances, and failure rows. The embedding score is
useful for aggregation, but raw examples and multiple metrics matter.

### You are ranking models unfairly.

**Yes, but:** Correct if interpreted as a leaderboard.

**Response:** The safer presentation is bucketed: stable cluster, middle
cluster, brittle cluster. Cross-family comparisons are descriptive signatures,
not quality judgments.

### OLMo is a research model; you are punishing it for not being over-instructed.

**Yes, but:** That is plausible.

**Response:** This should be framed as a training-recipe signature, not "OLMo is
bad." The probe may be detecting less aggressive instruction tuning or different
template behavior.

### Chat templates could be driving the no-op formatting results.

**Yes, but:** Correct, and that may be part of the practical finding.

**Response:** The deployment-relevant system is model plus tokenizer plus chat
template. Still, a raw-prompt control is needed before attributing the effect to
the model internals alone.

### Deterministic decoding is artificial; real systems sample.

**Yes, but:** Real systems often sample, but sampling variance would swamp the
prompt-perturbation effect.

**Response:** Deterministic decode isolates prompt sensitivity. Separate
sampling controls show different-seed sampling produces high divergence even
for identical prompts.

### Same prompt repeated twenty times would be a better test.

**Yes, but:** It tests a different thing.

**Response:** Same-prompt deterministic repeats mainly test hardware/framework
nondeterminism. Prompt sensitivity needs variation across prompt pairs, not just
reruns of the same prompt.

### The one-forward-pass hidden-state comparison is not the same as output stability.

**Yes, but:** Correct.

**Response:** Hidden-state distance measures prompt representation divergence.
Autoregressive generation measures trajectory branching. They are related but
not interchangeable.

### Looking at next-token changes collapses too much information.

**Yes, but:** Correct, which is why the hidden-state probe compares layer-wise
vectors rather than only the next token.

**Response:** A sharper version should also compare final-layer hidden vectors,
LM-head logits, top-k distributions, and argmax margins.

### LLM generation is not a smooth dynamical system.

**Yes, but:** Correct. It mixes continuous hidden states with discrete token
selection.

**Response:** The better phrase is "branching dynamics under autoregressive
decoding." Token choice boundaries can turn small hidden/logit differences into
large prefix differences.

### This does not apply to hidden-reasoning API models.

**Yes, but:** Correct for the trajectory-token version of the experiment. If a
model has hidden reasoning tokens that are not exposed, we cannot measure the
full generated trajectory.

**Response:** Our runs use open-weight Hugging Face models where
`model.generate()` returns the generated token stream we analyze. If reasoning
text is visible, it is included in the metric. Closed API models with hidden
reasoning would need a different final-answer-only analysis.

### The next token depends on the whole context, not just the previous token.

**Yes, but:** Correct.

**Response:** The state is the whole prefix, represented operationally by the
KV cache and current hidden activations. It is still a stateful recurrence, but
not a first-order scalar Markov process over tokens.

### Stable output might mean degraded output.

**Yes, but:** Especially under quantization.

**Response:** Quantization analysis must track both perturbation stability and
distance from the BF16 baseline / answer quality. Stable garbage is not a win.

### 8-bit quantization probably will not change behavior.

**Yes, but:** That is expected.

**Response:** 8-bit is a sanity/control point. The interesting signal is more
likely at 4-bit and below, and in the interaction between capacity and bit
budget.

### Why not test 3-bit, 2-bit, or 1-bit directly?

**Yes, but:** That is a good stress test, but it may require a different backend
such as HQQ, GPTQ, AWQ, or GGUF/llama.cpp.

**Response:** Those results need backend labels and may not be directly
comparable to the Transformers/bitsandbytes path. They are useful as a
destructive-low-bit lane, not as clean replacements.

### Quantization backend differences could explain everything.

**Yes, but:** Correct for low-bit experiments.

**Response:** The clean grid should first use one backend where possible. Any
HQQ/GPTQ/GGUF results should be labeled by backend and treated as exploratory
unless we can control the runtime path.

### You are comparing instruction-tuned and base models.

**Yes, but:** Cross-family, yes.

**Response:** That is why within-family comparisons are cleaner. Base-vs-instruct
Gemma is explicitly a separate question about whether instruction tuning is
semantically contractive.

### Model size is not the only variable even inside a family.

**Yes, but:** Correct.

**Response:** "Same family" is not "perfectly controlled." The Qwen ladder is
cleaner than cross-family comparisons, but still not a randomized controlled
trial of parameter count alone.

### The 4B-beats-9B result may be noise.

**Yes, but:** Correct.

**Response:** The robust claim is that both 4B and 9B are much more stable than
0.8B in this probe. The exact 4B-vs-9B ordering is not robust enough to claim as
general.

### The middle-pack ordering is not meaningful.

**Yes, but:** Correct.

**Response:** Present buckets and confidence intervals, not a precise 13-model
leaderboard.

### Common-prefix length can be misleading.

**Yes, but:** Correct. A model can share boilerplate prefixes while semantically
diverging later.

**Response:** Common-prefix length is a diagnostic, not the main score. Pair it
with semantic distance, token edit distance, and raw examples.

### Token edit distance can overstate harmless wording differences.

**Yes, but:** Correct.

**Response:** That is why semantic distance is the primary slide metric, while
token edit distance remains useful for trajectory diagnostics.

### Embedding distance can understate important factual or code differences.

**Yes, but:** Correct.

**Response:** For code and factual prompts, raw examples matter. A future
version should include task-specific validators or an LLM judge, but not as the
sole metric.

### The positive controls are not calibrated.

**Yes, but:** Correct.

**Response:** They are meant to verify the metric can see obviously different
outputs. They are not a formal upper bound on semantic distance.

### Hardware or CUDA nondeterminism could be the cause.

**Yes, but:** It is always a possible source.

**Response:** Identical-prompt deterministic controls are near zero, suggesting
framework nondeterminism is not dominating. End-to-end reruns are still useful
as a reproducibility check.

### Different model load paths could affect results.

**Yes, but:** Especially with quantization and custom code.

**Response:** Record model IDs, dtype, quantization mode, backend, and failures.
Do not compare tooling misses as stability results.

### Failed models bias the sample.

**Yes, but:** Correct.

**Response:** Failed models are reported as tooling/dependency misses, not
removed silently or interpreted as unstable.

### The result may not generalize to production prompts.

**Yes, but:** Correct.

**Response:** Production teams should run this style of probe on their own
prompt distributions. The talk demonstrates a method and early signatures.

### Why should anyone care about no-op formatting?

**Yes, but:** Because no-op changes happen constantly in real systems: template
edits, whitespace, serialization, Markdown, chat wrappers.

**Response:** If a newline flips response style, prompt changes that look safe
in code review may not be behaviorally safe.

### Does lower divergence always mean better?

**Yes, but:** No.

**Response:** Stability is task-dependent. Creative work may benefit from
diversity. Operational systems often want consistency. The metric is a property,
not a universal score.

### Are you measuring robustness or sameness?

**Yes, but:** Good distinction.

**Response:** We measure output stability under small perturbations. Robustness
would additionally require correctness or task success under perturbation.

### What would make this publishable?

**Yes, but:** Larger prompt set, pre-registered metrics, repeated runs,
task-specific validators, backend controls, and statistical analysis over prompt
distributions.

**Response:** The current work is a talk-quality probe and a measurement
proposal, not a final benchmark.

### What is the single safest claim?

**Response:** Nearby prompts can produce meaningfully different generation
trajectories under deterministic decode, and the size of that effect varies by
model in this controlled probe.
