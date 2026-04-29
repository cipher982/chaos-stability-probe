# Presenter Notes

> **⚠ Superseded (2026-04-29).** This file is an earlier draft from before
> the teaching-lens pivot. It references slides that no longer exist and a
> thesis I'm no longer defending. The **live companion** is
> [`../talk/speaker_notes.md`](../talk/speaker_notes.md), and the live deck
> is [`../talk/slides.md`](../talk/slides.md). Kept here for git history only.

## One-Sentence Thesis

Chaos gives us a vocabulary for a real LLM engineering question: when a system
is perturbed, what gets absorbed, what gets amplified, and where does the
stability margin run out?

## What Not To Claim

Avoid these:

- "LLMs are chaotic" as a blanket statement.
- "This is a Lyapunov exponent for language models."
- "Bigger models are always more stable."
- "Semantic embedding distance is ground truth."
- "A prompt comma causing different output proves deep chaos."

Cleaner claim:

> This is a chaos-inspired exploration of LLM dynamics. The experiment is a
> concrete probe, not a proof of formal chaos.

## Talk Flow

1. Start with the idea.

   Most people know chaos as "the butterfly effect." Teach the real point:
   nearby states can separate under repeated updates, sometimes approximately
   exponentially before saturating.

2. Translate it to LLMs.

   LLMs are not smooth physical systems, but they do have moving states:
   hidden activations through layers, logits, token choices, prefixes, KV cache,
   and quantized weights. The question is where perturbations get absorbed or
   amplified.

3. Then introduce the engineering problem.

   We already measure quality, cost, latency, context length, and benchmark
   score. But production systems also care whether small changes in input or
   prompt formatting cause large behavioral changes.

4. Define the axes.

   "The model" is not one dynamical system. There are weights, hidden states
   through depth, autoregressive generation, reasoning traces, and training.
   This experiment only probes prompt-state and output-generation stability.

5. Show controls.

   Same prompt plus same seed stays stable. Same prompt plus different sampling
   seed diverges. Therefore uncontrolled sampling mostly measures randomness.

6. Show the Qwen result.

   On the expanded ladder, `Qwen3.5-4B` is dramatically more stable than
   `Qwen3.5-0.8B` on punctuation, synonym, paraphrase, and small semantic
   changes.

7. Show no-op formatting examples.

   This keeps the experiment practical and memorable: a whitespace-only change
   can move a model into a different response style.

8. Show hidden-state vs output divergence.

   Some hidden-state distances are tiny while output edit distance is large.
   The interpretation is not magic: generation can cross decision boundaries
   where small prompt-state differences lead to different token paths.

9. Show cross-lab caveat.

   `Gemma 4 E2B` and `Qwen3.5-0.8B` look similarly fragile on token edit
   distance. `OLMo 3 7B` has a surprising no-op formatting sensitivity. This
   argues for measuring stability directly rather than assuming it from model
   family or parameter count.

10. End with the bigger conjecture.

   Compression and quantization are perturbations too. The open question is
   whether the compression floor and the dynamical stability margin interact.

## Best Charts To Use

Use these first:

- `runs/comparisons/qwen35_expanded_size_ladder/compare_semantic_divergence.png`
- `runs/comparisons/qwen35_expanded_size_ladder/compare_final_layer_hidden_divergence.png`
- `runs/comparisons/cross_lab_expanded/compare_semantic_divergence.png`

Use output edit-distance charts if the audience is comfortable with token-level
metrics:

- `runs/comparisons/qwen35_expanded_size_ladder/compare_output_divergence.png`
- `runs/comparisons/cross_lab_expanded/compare_output_divergence.png`

## Strongest Verbal Example

OLMo no-op formatting example:

- One prompt asks for a palindrome function.
- A spacing-only variant flips from direct code to a more explanatory fenced
  code answer with a simpler implementation.

Use it as an intuition pump, then immediately caveat:

> This is one prompt pair. The point is not to dunk on a model; the point is
> that a stability probe catches behavior that aggregate benchmark scores hide.

## Likely Pushback

Question:

> Isn't this just arbitrary prompt engineering?

Answer:

> Yes, individual examples are arbitrary. That is why the harness uses a ladder
> of controlled perturbation classes and reports aggregate distance by class.
> The next step would be hundreds or thousands of prompt pairs.

Question:

> Isn't deterministic decode unrealistic?

Answer:

> Deterministic decode is the right first control. Once the deterministic floor
> is understood, sampling can be added as a separate variance axis.

Question:

> Why not use semantic similarity only?

Answer:

> Semantic similarity is useful, but it can hide trajectory differences that
> matter in code, structured output, and agent workflows. Token edit distance,
> semantic distance, and hidden-state distance answer different questions.

Question:

> Does this prove larger models are more stable?

Answer:

> No. It shows a strong difference inside one Qwen comparison and mixed evidence
> elsewhere. Stability is probably shaped by training, tuning, templates, and
> scale.

## Good Closing

I am not claiming we proved LLMs are chaotic. I am claiming chaos gives us a
useful vocabulary for behavior we already see: tiny prompt changes, sampling
changes, and numeric perturbations sometimes vanish and sometimes amplify. If we
build software out of these systems, we should learn where that stability
margin is.
