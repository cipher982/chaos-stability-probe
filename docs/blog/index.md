---
title: "Nearby Prompts, Distant Trajectories"
layout: default
---

<!--
AGENT NOTE: Every slide starts with a marker of the form
  SLIDE N / slide_images/slide.0NN.png / "Title"
Slides are 1-indexed. Slide N corresponds to slide_images/slide.0NN.png.
When referencing a slide, cite both the number AND the title
(e.g. "slide 14 / Within-Qwen"). Do not count from slide 2.
Run scripts/check_slide_numbering.py after reordering slides.
-->

<!-- _class: title -->
<!-- _paginate: false -->
<!-- _footer: "" -->

<!-- SLIDE 1 / slide_images/slide.001.png / "Nearby Prompts, Distant Trajectories" -->

![bg cover](https://raw.githubusercontent.com/cipher982/chaos-stability-probe/main/talk/concept_images/generated/title_background_v2.png)

<div class="title-layout">
  <div class="title-copy">
    <h1>Nearby Prompts,<br>Distant Trajectories</h1>
    <h2>Teaching a lens: chaos, dynamical systems,<br>and how they <em>might</em> apply to LLMs.</h2>
  </div>
</div>

<!--
Set the table. ~60 seconds.

"Most of you have heard of the butterfly effect. I want to teach you what's
actually under it, compounding, nonlinear divergence, and then walk through
how that lens MIGHT apply to the LLMs we work with. This is exploratory:
the goal is useful vocabulary, with a few experiments to make it concrete.
Some of what I tried clicked. Some of it didn't. I'll show you both."

Tone: curious and exploratory. Keep it away from a paper-defense posture.
-->

---


<!-- SLIDE 2 / slide_images/slide.002.png / "What I'm **not** claiming" -->

## What I'm **not** claiming.

<div style="font-size:0.95em">

- **Not** "LLMs are chaotic." Classical chaos needs things LLMs don't have.
- **Not** "I measured a Lyapunov exponent." Token space is discrete.
- **Not** "bigger = more stable" or "reasoning = stable." Neither holds up.
- **Not** "sentence-embedding distance is ground truth." It's a proxy.
- **Not** "lower divergence = better." Stability is a property, not a score.

</div>

> Setting expectations: chaos as vocabulary,
> experiments as illustrations.

<!--
~45 seconds. Lift the Q&A risk off the rest of the talk by disarming the
obvious objections up front. Everything I'm not claiming is something
someone in the room might otherwise object to later.

Say out loud: "If any of these would have been your objection, great, we
agree. I'm going to use chaos as a lens, show you where it seems to apply,
and tell you where the data got messy."
-->

---


<!-- SLIDE 3 / slide_images/slide.003.png / "What I **am** claiming" -->

## What I **am** claiming.

<div class="twocol" style="grid-template-columns: 1fr 1.05fr; gap: 28px; align-items: center;">
<div style="font-size:0.82em">

- Inference time: **hybrid sequential system**, continuous activations feed a **discrete branching process.**
- Small changes can **move distributions** or **flip argmax branches.** Varies a lot by model, prompt, metric.
- Naive measurement has **specific failure modes** worth naming.
- Chaos vocabulary **organizes** the phenomenon. Doesn't *prove* anything.

> **Upshot:** test neighborhoods, not single prompts.

</div>
<div>

![h:500](https://raw.githubusercontent.com/cipher982/chaos-stability-probe/main/talk/concept_images/generated/slide3_hybrid_system.png)

</div>
</div>

<!--
~60 seconds. This is the honest thesis. It's modest enough to defend and
interesting enough to teach. The "hybrid sequential system" framing is the
cleanest one a dynamicist won't fight, continuous activations, discrete
branching, finite-time sensitivity.

Operational line to repeat at the end: "test neighborhoods, not single
prompts."
-->

---


<!-- SLIDE 4 / slide_images/slide.004.png / "Chaos starts with sensitivity" -->

## Chaos starts with sensitivity.

![bg right:50% fit](https://raw.githubusercontent.com/cipher982/chaos-stability-probe/main/talk/concept_images/double-pendulum-divergence.png)

<div style="font-size:0.9em">

**Chaos is deterministic amplification of small differences.**

- Same equations. Starting angles differ by **half a degree**.
- A few seconds later, totally different places.
- Think less dice roll, more amplifier.

> Forecasts go wrong because tiny measurement errors grow.

</div>

<!--
~60 seconds. Kill the pop-culture "chaos = chaotic = random" intuition
before any LLM content. Everyone in the room understands a pendulum.

Key line out loud: "Same input gives same output. The trick is that
NEIGHBORING inputs can land far apart after enough time. The system is
deterministic, and it magnifies differences faster than we can track."

Don't mention LLMs yet. Let the physics breathe.
-->

---


<!-- SLIDE 5 / slide_images/slide.005.png / "Small differences grow. Measurably" -->

## Small differences grow. Measurably.

![bg right:50% fit](https://raw.githubusercontent.com/cipher982/chaos-stability-probe/main/talk/concept_images/logistic-bifurcation.png)

<div style="font-size:0.58em; line-height:1.22">

**Logistic map:** $x_{n+1} = r \cdot x_n (1 - x_n)$

- **Low r:** one value. **Mid r:** 2, 4, 8 cycles. **High r:** never repeats.

> **Lyapunov λ**, how fast nearby trajectories separate. λ > 0: chaotic.

Trained nets **sit near this boundary** (Langton 1990; Zhang 2024).

*Which side is any LLM on?*

</div>

<!--
~90 seconds. This is the bridge slide, it earns you the rest of the talk.

The logistic map is the cleanest "deterministic iteration can produce any
regime" demonstration. Point at the bifurcation diagram and say: "same
equation, one knob. Turn it up, you get a phase transition into chaos."

Lyapunov introduced GENTLY, no formula on slide, just "measures how fast
trajectories separate." The actual formula |δ(t)| ≈ |δ(0)| · e^(λt) is in
speaker notes.

Edge-of-chaos citation is load-bearing, it's what makes "is this LLM near
the boundary?" a legitimate research question rather than a metaphor.
-->

---


<!-- SLIDE 6 / slide_images/slide.006.png / "Same input. Same weights. Different output" -->

## Same input. Same weights. Different output.

<div style="font-size:0.76em; line-height:1.2">

**Prompt A:** `Write a concise Python function that checks whether a string is a palindrome.`
**Prompt B:** same prompt, **trailing space added**. *(argmax decode, no sampling.)*

</div>

<div style="font-size:0.86em"><div class="twocol" style="gap:24px"><div><p><strong style="color:#c8402c">Output A, OLMo-3 7B</strong></p><pre><code>def is_palindrome(s: str) -&gt; bool:
    """
    Check if the given string
    is a palindrome, ignoring
    case and non-alphanumeric
    characters.
    ...
    """
    cleaned = ''.
</code></pre></div><div><p><strong style="color:#c8402c">Output B, OLMo-3 7B</strong></p><pre><code>Certainly! Here's a concise
Python function to check if a
string is a palindrome:

    def is_palindrome(s: str):
        return s == s[::-1]

How it works: ...
</code></pre></div></div></div>

<!--
Real data. Same weights, argmax decode, only the trailing space changed.

"This shouldn't produce two different essays. But it does. This is the puzzle
the rest of the talk explains, and before anyone asks 'is this just
temperature,' the next slide separates those two ideas."

Pre-empt: argmax decoding has no sampling step, so seed is inert. That
means what you see here is not 'the model got unlucky', it's the model's
most confident response under one input vs. its most confident response
under the other. The distribution itself moved.

Don't dunk on OLMo. Other models do this too. This is an existence proof.
-->

---


<!-- SLIDE 7 / slide_images/slide.007.png / "Temperature is a separate axis" -->

## Temperature is a separate axis.

|  | **Same prompt** | **Tiny prompt change** |
|---|---|---|
| **Temp = 0** (argmax) | Byte-identical. Boring. | **★ this is what the talk measures** |
| **Temp > 0** (sampling) | Different draws, same vibe. | Different draws **and** different vibe. Confounded. |

> **Temperature:** from a fixed distribution, what token do we sample?
> **Sensitivity:** how far did the distribution itself move?

Starred cell = our probe: zero sampling noise, output still moves. The model's response *function* shifted.

<span style="font-size:0.8em">*(At T=0.7 on OLMo-3, within-prompt and between-prompt sampling distances can match in magnitude, so deterministic decode gives the clean probe.)*</span>

<!--
This slide is the pedagogical linchpin. ~2 minutes.

The audience has a ChatGPT-shaped intuition: "LLMs are random, that's
temperature." That intuition collapses two different phenomena. Break them
apart here or they'll re-collapse them during Q&A.

Weather analogy: change the initial conditions by 0.1 degrees and a week
later the forecast can land somewhere else because the system amplifies small
input changes. Same claim, for LLMs.

One-line hook to repeat twice during the talk:
"Temperature samples from a distribution. Sensitivity asks how far the
distribution moved."
-->

---


<!-- SLIDE 8 / slide_images/slide.008.png / "So: is an LLM a dynamical system?" -->

## So: is an LLM a dynamical system?

- It has **state** (hidden activations, logits, KV cache, prefix).
- It has **iteration** (each token feeds into the next).
- It's **deterministic** under argmax.
- Small input perturbations can produce large output changes.

That's the checklist. The remaining question is whether the *magnitude* of
amplification is interesting, and whether we can measure it.

> But there's a catch: classical chaos needs perturbations going to zero.
> Token space is discrete. We'll come back to this.

<!--
~60 seconds. This slide is much shorter than before, the chaos background
already happened on the last two slides. This one's job is just to bridge:
"LLMs check every box on the dynamical-systems checklist."

The catch line matters: discrete token space means |δ| → 0 doesn't cleanly
work. Flag it now, resolve it later on the "meaning-preserving perturbation"
slide.
-->

---


<!-- SLIDE 9 / slide_images/slide.009.png / "Both outputs can be correct" -->

## Both outputs can be correct.

<div class="twocol" style="grid-template-columns: 0.9fr 1.1fr; gap: 30px; align-items: start;">
<div style="font-size:0.60em; line-height:1.28">

> A double pendulum isn't "wrong"; it obeys physics and lands elsewhere.
> Same bar for LLMs.

- "Book like *Dune*" → Foundation. Add a trailing space → Hyperion.
- Both recommendations are defensible; neither is a hallucination.
- Sampling stays local; sensitivity can **move the distribution**.
- **Measure:** output divergence per *meaning-preserving* input change.

</div>
<div style="padding-top: 8px;">

![w:500](https://raw.githubusercontent.com/cipher982/chaos-stability-probe/main/talk/concept_images/distribution-shift-basins.png)

</div>
</div>

<!--
This is the slide that answers "is this just regeneration noise?" without
having to argue it. You show two equally valid outputs, not one correct
and one broken, and the audience gets it.

The "put NOT at the front" trap: big semantic tokens move the output a lot
because they SHOULD, you changed the meaning. That's the model working,
not the model being sensitive. The interesting quantity is output-move /
input-move, conditioned on input-move being small.

Li et al. hedge: classical Lyapunov needs |δ| → 0. In continuous activation
space you can take that limit (they do). In discrete token space you can't.
So we either (a) restrict to meaning-preserving perturbations, or (b) move
the probe into activation space. Both are open extensions with compute.
-->

---


<!-- SLIDE 10 / slide_images/slide.010.png / "State, and prior work, short version" -->

## State, and prior work, short version

<div class="twocol" style="grid-template-columns: 0.78fr 1.22fr; gap: 24px; align-items: start;">
<div style="font-size:0.47em; line-height:1.18">

An LLM's state: hidden activations + logits + prefix + KV cache.

- **Li et al. 2025**, QLE on Qwen2-14B. ~1.32× per layer. *Quasi*-Lyapunov (finite depth).
- **Geshkovski 2023**, attention as particle dynamics.
- **Poole / Schoenholz**, edge-of-chaos signal prop.

> Chaos math is cleanest in **activation space**. These probes observe its **output-text shadow**.

</div>
<div>

![w:610](https://raw.githubusercontent.com/cipher982/chaos-stability-probe/main/talk/concept_images/generated/slide10_compounding.png)

</div>
</div>

<!--
Combined state + prior art. ~90 seconds.

Li et al. key numbers to land: 1.32× per-layer magnitude growth in layers 0–9;
MLP contributes 55.8% of final residual, attention 44.2%, initial input 0.0009%.
That last number is the chaos-in-one-line: we're perturbing the 0.0009% and
watching the 100% move.

Honest framing: Li et al. ran ONE model (Qwen2-14B), defined an iterative
(token-level) QLE but never computed it, and never compared models. I run 18
models on the axis they didn't: token-level. Complementary, not redundant.
-->

---


<!-- SLIDE 11 / slide_images/slide.011.png / "The experiment" -->

## The experiment

<div style="font-size:0.9em">

- **~21 models:** Qwen (0.8B → 9B), Gemma 4, Phi-4, DeepSeek-R1, Mistral,
  Granite, Falcon, SmolLM, OLMo 2 & 3; legacy: GPT-2 XL, GPT-J, Pythia, OPT, LLaMA-1.
- **Prompt ladder:** identical / no-op formatting / punctuation / synonym /
  paraphrase / small semantic / positive control.
- **Deterministic decode** (`do_sample=False`, argmax), divergence is a shift in the model's most confident response.
- **Metrics:** sentence-embedding cosine distance (primary) + token edit +
  hidden-state distance + logit JS/KL. All proxies; no ground truth.
- **Analysis:** bootstrap CIs + paired permutation tests. Present clusters, not ranks.
- **Reproducibility:** deterministic decode, prompt-token deltas logged, model/config metadata published with artifacts.

</div>

<!--
Lay out methodology cleanly. Anticipate methods questions.

Critical control: same prompt + deterministic decode = 0.000 divergence.
Same prompt + sampling = high divergence. Sampling controls are why
deterministic decode is the right first probe.

Cap n honestly: 21 prompt pairs in the panel; a hardened Qwen wave went to 42.
Save the punchlines for next slide.
-->

---


<!-- SLIDE 12 / slide_images/slide.012.png / "What actually matters?" -->

## What actually matters?

<div class="micro-sweep-visual">
  <img src="https://raw.githubusercontent.com/cipher982/chaos-stability-probe/main/talk/micro_visuals/token_micro_v2_category_heatmap.png" alt="">
</div>

<div class="takeaway">
Columns are sorted by average effect, but the stronger pattern is row-wise: some models are much more sensitive than others.
</div>

<!--
This is the new, more intuitive experiment. It is stronger than synonyms or
paraphrases because the perturbations are almost invisible to a human:
line breaks, spaces, punctuation placement, parenthesizing one word.

The key point is the contrast, not the exact numbers:
- the raw character-edit slide was invalid because many edits were token-identical
- this chart drops those pairs and keeps only effective post-template token deltas
- internal layout/syntax edits branch across model families once they survive

Say: "This is not 'any byte flips the model.' It's more structured than
that. Some edits never reach the model as distinct input. Some survive the
template/tokenizer and move the model into a different basin."
-->

---


<!-- SLIDE 13 / slide_images/slide.013.png / "Same-looking prompt. Different trajectory" -->

## Same-looking prompt. Different trajectory.

![w:1120](https://raw.githubusercontent.com/cipher982/chaos-stability-probe/main/talk/micro_visuals/03_branching_trajectories_compact.png)

<div class="takeaway">
Read this as token-path divergence. Quality and endpoint meaning are separate checks.
</div>

<!--
This slide makes the dynamical-systems framing concrete.

Y=0 means the generated prefixes are token-for-token identical. A rising line
means more edits are needed to align the generated token prefixes. Treat it as
a token-path diagnostic; quality and semantic distance are separate axes.

Useful narration:
"This uses Levenshtein, so simple insertion/deletion offsets get aligned. If
the problem were only 'same output shifted by one token,' the dashed gray line
is what you'd expect. The blue/red lines staying high means the path itself
changed. The right panel checks semantic endpoint distance separately."
-->

---


<!-- SLIDE 14 / slide_images/slide.014.png / "Within-Qwen: one clean contrast" -->

## Within-Qwen: one clean contrast.

![w:780](https://raw.githubusercontent.com/cipher982/chaos-stability-probe/main/talk/concept_images/generated/slide12_qwen_forest.png)

<div style="font-size:0.82em">

- 0.8B is **meaningfully more sensitive** than 4B (p<0.001). 2B also separates (p=0.012).
- 4B vs 9B: indistinguishable at this n. No size law from this panel.
- Caveat: 4B/9B emit `Thinking Process:` preambles. Scaffold confound, next slide.

</div>

<!--
This was my single cleanest finding yesterday. Then I looked at the raw
generations. Qwen 4B/9B start every answer with an identical reasoning
scaffold. That scaffold inflates common prefix and suppresses early
semantic distance. The capacity-vs-sensitivity story in this slide is
real but partly confounded, I'll own that in the next two slides instead
of burying it.
-->

---


<!-- SLIDE 15 / slide_images/slide.015.png / "Scaffold 'stability' is mostly metric artifact" -->

## Scaffold "stability" is mostly metric artifact.

<div style="font-size:0.82em">

**Short outputs (64 tokens):** scaffolded models look ~4× more stable.
Identical `<think>` preambles dominate sentence-embedding similarity.
**Evaluation warning:** this mostly exposes a metric trap.

**Long outputs (512 tokens) expose the mixed bag:**

| Scaffolded model | 512-tok semantic | Prompt-end top-1 prob |
|---|---:|---:|
| DeepSeek-R1 7B | 0.027 (stable) | 0.99976 |
| Qwen 4B / 9B | 0.050 / 0.057 | 0.970 / 0.988 |
| SmolLM3 3B | 0.080 (middle) | 0.99983 |
| **Phi-4 reasoning+** | **0.160 (brittle)** | **0.99999996** |

</div>

<p style="font-size:0.72em; margin-top: 0.3em"><strong>Phi-4:</strong> 0.160 divergence at 512 tokens, <code>&lt;think&gt;</code> never closes (repetition loop). Confident logits can still branch. Thinking-off is mixed: helps big Qwens, <em>hurts</em> 0.8B.</p>

<!--
This is the scaffold slide after the 512-token rerun. Keep the short-output
finding, it's what a naive probe would report, but land the fact that
longer outputs expose scaffolded models as a mixed bag.

Phi-4 is the crown-jewel counterexample:
- visible <think> scaffold
- top-1 probability 0.99999996 at prompt end (most confident in panel)
- JS divergence 1.4e-9 (bulk distribution didn't move)
- 512-token semantic 0.160 (more brittle than GPT-2 XL)

That is the single cleanest dissociation between "confident logits" and
"stable trajectory" in the whole dataset.

Thinking-off numbers (Qwen default vs enable_thinking=False):
  4B: 0.050 → 0.067 (scaffold helps ~25%)
  9B: 0.057 → 0.072 (scaffold helps ~20%)
  2B: 0.075 → 0.072 (wash)
  0.8B: 0.103 → 0.079 (scaffold HURTS, scaffold in small model is noisy)

So the scaffold effect is real and mixed. Don't overclaim either direction.
-->

---


<!-- SLIDE 16 / slide_images/slide.016.png / "Era, recipe, and the LLaMA-1 surprise" -->

## Era, recipe, and the LLaMA-1 surprise

<div style="font-size:0.76em">

**512-token semantic distance, non-scaffold models only:**

| Model | Semantic | Era |
|---|---:|---|
| **LLaMA-1 7B** | **0.053** | 2023 base, stable outlier |
| Gemma E2B **instruct** | 0.056 | modern chat |
| Mistral 7B v0.3 | 0.068 | modern chat |
| Gemma E4B **instruct** | 0.072 | modern chat |
| Gemma E4B **base** | 0.119 | modern base |
| Gemma E2B **base** | 0.199 | modern base |
| GPT-2 XL / OPT / Pythia / GPT-J | 0.14 – 0.22 | pre-chat base |

**LLaMA-1 is a stable outlier in this probe, not a law.** Within Gemma, **instruct ≫ base**, recipe over calendar. Era is a weak predictor; token-path and semantic metrics diverge.

</div>

<!--
512-token numbers from runs/rankings/scaffold_long_wave/small_perturbation_bootstrap.csv.

Key updates vs earlier slide:
- Gemma E2B and E4B base actually *swap* order between the 64-token panel
  and the 512-token panel. Good talking point if asked: "stability is a
  scale-dependent measurement, which is why we report clusters rather than ranks."
- LLaMA-1 actually beats Gemma E2B it by a hair on 512 tokens. Still in the
  stable band. Don't overclaim against one community conversion.
- Qwen 4B/9B removed from this slide because they're scaffolded; they live
  on the scaffold slide now.

If pushed on "is LLaMA-1 real": possible explanations, pretraining corpus,
tokenizer, or just a community-conversion artifact. One data point, treat as
a flag rather than a law.

Follow-up on the "older models more stable?" hunch:
at token edit distance around t=60, modern/instruct models look slightly more
surface-divergent than legacy/base models. At 512-token semantic distance,
the sign flips: modern/instruct models are more semantically contractive.
So this slide is about recipe and metric choice more than calendar year.
-->

---


<!-- SLIDE 17 / slide_images/slide.017.png / "Stability and responsiveness split" -->

## Stability and responsiveness split.

<div class="twocol" style="grid-template-columns: 1.05fr 1fr; gap: 24px; align-items: center;">
<div style="font-size:0.78em">

> `"the the the"` forever is extremely stable. So is a model collapsed on one fixed answer. **Neither is what we want.**

Qwen 0.8B quant sweep, 4-bit scored *lower* perturbation divergence (0.138 → 0.091). Sounds "more stable", until you check drift from BF16 on identical prompts: **0.132, huge.**

That looks more like collapse onto a narrower manifold than robustness.

> **Fix:** pair perturbation distance with drift-from-baseline. Both axes.

</div>
<div>

![w:560](https://raw.githubusercontent.com/cipher982/chaos-stability-probe/main/talk/concept_images/generated/slide15_collapse.png)

</div>
</div>

<!--
Collapsed the old two quant slides into one conceptual point. The numbers
are exploratory (n=9), so don't oversell them, the POINT is the principle:
a one-axis stability metric can confuse collapse with robustness. That's
the scientifically useful takeaway regardless of whether the Qwen 0.8B data
is itself definitive.

If pushed on whether the quant finding is solid: "this was n=9 per cell, and
the within-system flip is p=0.19. Treat it as an existence example of the
collapse confound rather than a quantization conclusion."
-->

---


<!-- SLIDE 18 / slide_images/slide.018.png / "Measuring is the hard part" -->

## Measuring is the hard part.

**Three ways a naive stability probe will mislead you:**

<div style="font-size:0.85em">

| Confound | What happens | Caught by |
|---|---|---|
| **Collapse** | Degenerate model scores "stable" because outputs stop responding to input. (Qwen 0.8B 4-bit.) | Distance from baseline on identical prompts. |
| **Scaffold** | Short-output score dominated by deterministic preamble. (Qwen 4B/9B, SmolLM3.) | Longer continuations; scaffold stripping. |
| **Confidence can still branch** | Low prompt-end JS + sharp top-1 argmax still allowed a divergent trajectory. **Phi-4: top-1 prob 0.99999996 at prompt-end; `<think>` never closes; 0.160 at 512 tokens.** | Multi-scale measurement: short vs long, logit vs text. |

</div>

> The most useful contribution here is naming the failure modes
> **before the field starts quoting the numbers.**

<!--
Third row is new. Phi-4 is the cleanest single counterexample in the data:
top-1 probability 0.99999996 at prompt end, visible <think> scaffold, and
the second-most brittle model in the panel at 512 tokens (0.160, above
GPT-2 XL at 0.144). Sharp-logit ≠ stable-trajectory.

If someone asks "what's the fix": the mature version is multi-scale. Short
outputs can be dominated by scaffolds. Prompt-end logits can miss
decision-boundary fragility. No single measurement is safe alone.
-->

---


<!-- SLIDE 19 / slide_images/slide.019.png / "Long-generation trajectories" -->

## Long-generation trajectories

![h:390](https://raw.githubusercontent.com/cipher982/chaos-stability-probe/main/runs/trajectory_figures/qwen_thinkoff_trajectory_and_semantic.png)

<span style="font-size:0.78em">Updated direct-answer control: token paths can diverge quickly even when 512-token semantic distance stays in a tight band. Use trajectory shape as a diagnostic rather than a leaderboard.</span>

<!--
Updated from the older DeepSeek/Qwen reasoning-on longprobe chart. This is the
current Qwen thinking-off control, using 24 small perturbation prompt pairs
from the scaffold-long wave.

Key visual: lexical token trajectories saturate faster than the semantic
metric. The semantic bars are close: 4B/9B/2B/0.8B are not cleanly separable
under this direct-answer control.

"This is why I don't want to sell one curve as the answer. Token paths,
semantic distance, scaffold behavior, and logit boundaries all see different
parts of the same system."
-->

---


<!-- SLIDE 20 / slide_images/slide.020.png / "Mechanism: boundary beats bulk" -->

## Mechanism: boundary beats bulk

![w:780](https://raw.githubusercontent.com/cipher982/chaos-stability-probe/main/talk/concept_images/generated/slide18_correlations.png)

<div style="font-size:0.78em">

> Small prompt change → argmax crosses a low-margin boundary → different first token → autoregressive feedback → different trajectory.

**The distribution often barely moves in bulk.** A low-margin next-token decision is fragile; one flipped argmax steers generation into a different basin.

</div>

<!--
This is the mechanism slide. Say out loud:

"If you took one thing from the measurement side of this talk: the whole
next-token distribution does not have to shift much. Some next-token decisions
are made at very low margin. A tiny input perturbation crosses that boundary,
flips the argmax, and the model is now generating from a different starting
token. Autoregression does the rest."

Phi-4 is the extreme case in the next slide: prompt-end top-1 probability
0.99999996 (the model is *certain*), JS ~1.4e-9 (distribution hasn't moved
at all), yet 512-token semantic divergence is 0.160, higher than GPT-2 XL.
Prompt-end confidence says nothing about trajectory stability on its own.
-->

---


<!-- SLIDE 21 / slide_images/slide.021.png / "A question the lens suggests" -->

## A question the lens suggests.

<div class="twocol"><div>

### Static floor
How few bits to store the model?
- TurboQuant, KIVI, KV quantization.
- Rate-distortion bounds.
- Well-characterized.

</div><div>

### Dynamical floor?
How few bits before *behavior* drifts?
- Might depend on model sensitivity.
- Stable models *might* tolerate more compression.
- **Open. My data doesn't settle it.**

</div></div>

> Compression has a static floor. Does it have a *dynamical* one too?
> The chaos lens suggests the question. I don't have the answer.

<!--
Softened from "two floors" conjecture to "a question the lens suggests."
This matches the teaching-lens framing, we're raising interesting questions
and being clear about what the data can support.

If someone asks "do you believe it?": "I lean yes, but my own data has the
Qwen 0.8B collapse case that would naively falsify the claim. So: open
question I'd love someone else to chase."
-->

---


<!-- SLIDE 22 / slide_images/slide.022.png / "The practitioner upshot" -->

## The practitioner upshot.

<div style="font-size:0.78em">

> **Don't evaluate on a single prompt, single decode, or single metric.
> Prompting is operating a high-gain branching system.**

**Operational:**
- **Reliability:** test prompt *neighborhoods* around the canonical prompt.
- **Model comparison:** report sensitivity *ranges* over equivalent prompts.
- **Output metrics:** strip boilerplate, compare answer spans, watch prefixes.
- **Decoding:** deterministic for sensitivity; sampling separately for deployment.
- **Quantization:** lower divergence ≠ robustness. Also check baseline drift.

</div>

> **The chaos lens gives us questions
> that standard benchmarks rarely ask, and those questions are worth asking.**

<!--
Land and stop. This is the honest closing.

"If you remember one thing: prompting is operating a high-gain branching
system. Test neighborhoods, not single prompts. Watch for the confounds
when you score stability. That's it."
-->

---


<!-- _class: big-quote -->

<!-- SLIDE 23 / slide_images/slide.023.png / "Questions?" -->

# Questions?

<!-- Go to backup slides if asked about methods, specific models, or failures. -->

---


<!-- SLIDE 24 / slide_images/slide.024.png / "Backup, 'Would I get the same answer if I ran it?'" -->

## Backup, "Would I get the same answer if I ran it?"

<span style="font-size:0.85em">

**Argmax decode has no sampling step.** `do_sample=False` → highest-logit token wins each step. Seed is inert.

- Same prompt twice → byte-identical output.
- Prompt A vs B → top-token at some position **flipped**. Most confident response moved.

**Temperature > 0?** 30-sample cluster test (OLMo-3, palindrome pair, T=0.1):

- Prompt A: 30 samples cluster tightly. Prompt B: same.
- A-cluster and B-cluster are visibly separate.

The samples form two different attractors. Sampling noise is smaller than the shift between them.

</span>

---


<!-- SLIDE 25 / slide_images/slide.025.png / "Backup, 'Is this chaos?' defense" -->

## Backup, "Is this chaos?" defense

- Formal chaos needs exponential divergence under iteration. **Not proven.**
- What was measured: small input perturbations producing different outputs,
  varying by model, reproducible under deterministic decode.
- Consistent with behavior near a chaos boundary.
- The frame is the contribution; the experiment is a probe.

---


<!-- SLIDE 26 / slide_images/slide.026.png / "Backup, Related work I came across late" -->

## Backup, Related work I came across late

<div style="font-size:0.68em">

**Prompt sensitivity / brittleness:**
- **Salinas & Morstatter 2024**, "Butterfly Effect of Altering Prompts." My whitespace example, already published.
- **Sclar et al. 2023**, formatting sensitivity; up to 76-point swings on LLaMA-2-13B.
- **Lu et al. 2021**, example ordering alone moves few-shot near-random to near-SOTA.
- **PromptRobust / POSIX / RobustAlpacaEval**, published sensitivity benchmarks.

**Dynamical systems in NNs:**
- **Poole 2016 / Schoenholz 2017**, edge-of-chaos signal propagation.
- **Geshkovski et al. 2023**, attention as interacting-particle dynamics.
- **Tomihari & Karakida 2025**, Jacobian/Lyapunov analysis of self-attention.

</div>

---


<!-- SLIDE 27 / slide_images/slide.027.png / "Backup, Statistical honesty" -->

## Backup, Statistical honesty

<div style="font-size:0.88em">

- **n = 9** prompt pairs per model; **n = 24** in hardened Qwen wave. Small.
- **Robust at n = 24:** Qwen 4B vs 0.8B p<0.001; Qwen 4B vs 2B p=0.012; cluster membership.
- **Weak at this n:** Qwen 4B vs 9B (p=0.78); middle-pack ordering; standalone quant flip.
- Scaffold vs non-scaffold is **confounded with post-training recipe**, needs different *models*, not more prompts.

</div>

> Disagreeing with a specific model ordering is fair; those are intentionally
> underclaimed. Disagreeing with the broad clusters needs a much larger prompt set.

---


<!-- SLIDE 28 / slide_images/slide.028.png / "Backup, Failed experiments" -->

## Backup, Failed experiments

- **gpt-oss-20b:** MXFP4 / Triton driver mismatch on SageMaker image.
- **Nemotron Nano 9B v2:** container lacked `mamba-ssm`.
- **Phi-4 mini:** Transformers version / custom-code import failure.

Reported as tooling misses rather than stability findings.

---


<!-- SLIDE 29 / slide_images/slide.029.png / "Backup, Full bootstrap readout (512 tokens)" -->

## Backup, Full bootstrap readout (512 tokens)

<div style="font-size:0.88em"><div class="twocol"><div>

**Stable / mid**

| Model | Mean | 95% CI |
|---|---:|---:|
| DeepSeek-R1 Qwen 7B | 0.027 | 0.018 – 0.036 |
| Qwen3.5 4B | 0.050 | 0.033 – 0.066 |
| LLaMA-1 7B | 0.053 | 0.017 – 0.100 |
| Gemma 4 E2B it | 0.056 | 0.033 – 0.080 |
| Qwen3.5 9B | 0.057 | 0.037 – 0.075 |
| Mistral 7B v0.3 | 0.068 | 0.047 – 0.089 |
| Gemma 4 E4B it | 0.072 | 0.038 – 0.110 |
| Qwen3.5 2B | 0.075 | 0.050 – 0.103 |

</div><div>

**Higher sensitivity / caveats**

| Model | Mean | 95% CI |
|---|---:|---:|
| OLMo 2 7B | 0.088 | 0.055 – 0.127 |
| Qwen3.5 0.8B | 0.103 | 0.061 – 0.153 |
| OLMo 3 7B | 0.104 | 0.077 – 0.135 |
| Gemma 4 E4B base | 0.119 | 0.071 – 0.173 |
| GPT-2 XL | 0.144 | 0.082 – 0.208 |
| **Phi-4 reasoning+** | **0.161** | 0.072 – 0.255 |
| Gemma 4 E2B base | 0.199 | 0.152 – 0.248 |

</div></div></div>

<span style="font-size:0.62em">n=24 pairs, 512 tokens. Cluster view. **Phi-4: scaffolded yet brittle**, scaffold does not guarantee stability.</span>
