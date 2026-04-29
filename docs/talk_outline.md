# Draft Talk Outline

> **⚠ Superseded (2026-04-29).** This outline is an earlier 17-slide draft
> from before the teaching-lens pivot. The **live deck** is
> [`../talk/slides.md`](../talk/slides.md) (27 slides including new
> "What I'm not / am claiming" framing and a practitioner-upshot close).
> Kept here for git history only.

## Title

**LLMs as Systems in Motion**

Alternative:

- **Chaos, Compression, and LLMs**
- **Nearby Prompts, Distant Trajectories**
- **How Stable Is an LLM?**

## Thesis

Chaos is not just "randomness" or "butterfly effect" folklore. It is a way of
thinking about systems where nearby states can separate under iteration.

LLMs are not weather systems, but they are systems in motion: hidden states move
through layers, logits create token decisions, generated prefixes become the
next state, and quantization perturbs the machinery. The talk uses chaos and
Lyapunov-style thinking as a lens, then uses our experiment as a concrete probe.

Quantization is no longer the center of the talk. It is one perturbation source.
The main question is dynamical sensitivity: when we make a tiny input change,
does the model absorb it, branch, or reconverge?

## Slide 1: The Butterfly Everyone Knows

Start from the familiar idea:

- A tiny weather perturbation can eventually change the forecast.
- The deep idea is not "randomness"; it is sensitive dependence on initial
  conditions.

Line:

> Chaos is what happens when "almost the same state" does not stay almost the
> same under repeated updates.

## Slide 2: Exponential Divergence

Teach the core picture:

- two nearby starting points
- distance over time
- slow early separation
- rapid growth
- eventual saturation

Introduce the loose Lyapunov idea:

> A Lyapunov exponent asks: on average, how quickly do nearby trajectories
> separate?

Do not bury the audience in formalism. One equation is enough:

```text
distance(t) ~ distance(0) * e^(lambda t)
```

## Slide 3: Not All Instability Is Chaos

Clarify:

- randomness is not chaos
- noise can swamp a signal
- a discontinuous branch can look like a jump rather than smooth divergence
- stable systems can still be wrong or boring

This slide prevents later overclaiming without making the talk defensive.

## Slide 3: Do Not Mix the Axes

Retitle in slides as: **What Is Moving?**

Five possible LLM "systems":

1. weights/static model
2. depth dynamics through layers
3. generation dynamics token by token
4. reasoning dynamics step by step
5. training dynamics

Say explicitly:

> "The model is chaotic" is not precise enough. We need to say which system is moving.

## Slide 5: Why LLMs Are Weird Dynamical Systems

Autoregressive generation:

```text
p(token_t | prompt, token_1, ..., token_{t-1})
```

State is not just the previous token. It is the whole prefix, represented in
practice by hidden states and KV cache.

Important twist:

- hidden activations/logits are continuous-ish
- token selection is discrete
- tiny logit changes can cross an argmax boundary
- after a token branch, future prefixes differ

Phrase:

> LLM generation is continuous machinery repeatedly making discrete decisions.

## Slide 6: Prompt Perturbation as a Toy Weather Experiment

Show two near-identical prompts:

```text
Explain how small changes can affect a complex system over time.
Explain how small changes can affect a complex system, over time.
```

Then show:

- Qwen3.5 0.8B: diverges early and strongly.
- Qwen3.5 4B: stays much closer.

Point:

> Same family. Same decode settings. Tiny prompt change. Different trajectory
> behavior.

## Slide 7: Controls

Show sampled-control result:

- same prompt + same seed: zero divergence
- same prompt + different seed: high divergence

Point:

> If we don't control sampling, we mostly measure randomness, not prompt sensitivity.

## Slide 8: Experiment Design

Prompt perturbation ladder:

- identical
- no-op formatting
- punctuation
- synonym
- paraphrase
- small semantic change
- positive control

Metrics:

- common generated-token prefix
- normalized token edit distance
- sentence-embedding cosine distance
- final-layer hidden-state cosine distance

Frame:

> This is not a proof of chaos. It is a Lyapunov-inspired way to ask whether
> nearby states stay nearby.

## Slide 9: Qwen3.5 Size Ladder

Use:

- `runs/comparisons/qwen35_expanded_size_ladder/compare_output_divergence.png`
- `runs/comparisons/qwen35_expanded_size_ladder/compare_semantic_divergence.png`

Talk track:

> In this small probe, the 4B model is much more stable than the 0.8B model under small prompt perturbations.

Caveat:

> This is one model family and 21 prompt pairs. It is not a universal law.

## Slide 10: No-Op Formatting Failure

Show a concrete side-by-side example:

- OLMo 3 direct code answer.
- Same prompt with spacing-only change yields explanatory fenced code answer.

Point:

> Some perturbations are not semantically meaningful to us, but they can still
> move the model into a different response basin.

## Slide 11: Hidden State vs Output

Use:

- `runs/comparisons/qwen35_expanded_size_ladder/compare_final_layer_hidden_divergence.png`

Point:

> Prompt-state distances can be tiny while generated outputs diverge. That suggests generation can amplify small representation differences across decision boundaries.

This is the bridge to dynamical systems.

## Slide 12: Long-Generation Curves

Use the Wave 4 long-probe results if they are ready:

- stable model
- brittle model
- reasoning model

Ask:

> Does divergence grow gradually, jump at branch points, reconverge
> semantically, or saturate?

This is where the Lyapunov analogy becomes visual.

## Slide 13: Prior Art Anchors

Keep this light: it is not a lit-review slide. The point is permission, not
exhaustiveness.

- Lorenz / logistic map: chaos means deterministic sensitive dependence.
- Li et al. QLE paper: LLM internals have been analyzed with quasi-Lyapunov
  tools.
- Attractor cycles paper: repeated LLM transformations can converge to stable
  cycles.
- Edge of stability: neural-network training itself sits near a stability
  boundary.
- RNN Lyapunov work: ML has used this language before LLMs.

Point:

> I am not inventing the dynamics lens. I am applying it to the parts of LLM
> behavior AI engineers touch every day.

## Slide 14: Older Base Models vs Modern Chat Models

Use this if the legacy lane finishes cleanly:

- GPT-2 / GPT-J / OPT / Pythia are older base-model recipes and look brittle
  in this probe.
- LLaMA-1 is older but relatively stable, so era alone is not the explanation.
- Gemma base vs Gemma instruction is the cleaner within-family contrast.
- The question is not "which is better?"
- The question is whether training/post-training changes response attractors.

Point:

> Post-training can change the shape of the response basin, but it is not a
> one-variable story.

## Slide 15: Compression and Quantization

Connect back to the broader idea:

- quantization is a perturbation to the model's machinery
- apparent stability can improve if the model collapses into a narrower
  response manifold
- stability is cheap; fidelity is not
- pair perturbation stability with distance-from-BF16

Show current 2x3 grid:

| Model | BF16 | 8-bit | 4-bit |
| --- | --- | --- | --- |
| Qwen3.5 0.8B | 0.138 | 0.110 | 0.091 |
| Qwen3.5 4B | 0.013 | 0.025 | 0.026 |

Next plot:

- `runs/quantization_fidelity/qwen_quantized_vs_bf16_small_semantic.png`
- this distinguishes robustness from stable degeneration

Line:

> A quantized model can be less sensitive to prompt perturbations and still be
> less faithful to the original model. That is not free stability; it is a
> changed system.

## Slide 16: Why This Matters

Practical reasons:

- regression testing prompts
- choosing models for deterministic workflows
- understanding instruction tuning and attractors
- quantization sensitivity hypotheses
- debugging long-form generation drift

## Slide 17: Open Questions

- Are instruction-tuned models more stable than base models?
- Does size generally increase stability within a family?
- Are reasoning models more stable on constrained tasks?
- Do high-divergence prompts correlate with quantization sensitivity?
- Can "stability profile" become a standard model card field?

## Closing Line

> I am not claiming we proved LLMs are chaotic. I am claiming chaos gives us a
> useful vocabulary for a real engineering question: when we perturb these
> systems, what gets absorbed, what gets amplified, and where does the stability
> margin run out?
