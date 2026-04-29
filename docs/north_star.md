# Chaos, Compression, and LLM Stability: North Star

> **⚠ Partially superseded (2026-04-29).** The "Claims to avoid" and
> "Measurement Principles" sections still apply. The **Talk Shape** section
> (Act 1–5) and the **Initial Model Panel** list are historical — the live
> deck is [`../talk/slides.md`](../talk/slides.md), and the thesis has
> shifted from "compression has a floor, chaos is why" to a teaching-lens
> framing: *"LLMs aren't textbook chaotic, but they're hybrid sequential
> systems — teach the lens, name the measurement pitfalls."* See
> [`results_digest.md`](results_digest.md) for the current pitch.

## Working Goal

Build a vivid 30-45 minute learning-club talk that teaches chaos and dynamical
systems through LLMs.

The concrete question:

> When an LLM is perturbed, what gets absorbed, what gets amplified, and where
> does the stability margin run out?

The talk should not become a methods paper. The experiment supports the lens;
it is not the whole talk.

## Core Framing

Compression asks: how few bits can represent a model or state without losing the behavior we care about?

Chaos asks: how quickly do nearby states stop being nearby?

LLMs expose several different "states" and therefore several different stability questions. The talk should keep these axes separate:

1. **Weights / static model object**  
   The trained parameters sitting on disk. Quantization and rate-distortion mostly live here, though KV cache quantization is about runtime state.

2. **Depth dynamics / one forward pass**  
   The residual stream and hidden activations as they move through transformer layers. This is closest to the quasi-Lyapunov framing in "Cognitive Activation and Chaotic Dynamics in Large Language Models."

3. **Generation dynamics / autoregressive iteration**  
   Token by token generation. This is the intuitive user-facing behavior: do near-identical prompts lead to nearby outputs, or do they diverge?

4. **Reasoning dynamics / multi-step traces**  
   Chain-of-thought or explicit reasoning steps. Treat this separately from raw token dynamics.

5. **Training dynamics**  
   SGD and edge-of-stability work. This is relevant background, but likely too much for the main thread unless needed.

The talk should avoid saying "the model is chaotic" without specifying which axis.

## What We Can Claim

Reasonable claims:

- LLM inference can be analyzed with dynamical-systems tools.
- Different axes of the system can have different stability behavior.
- We can measure prompt-output divergence and hidden-state divergence empirically.
- Compression/quantization and dynamical stability rhyme because both are about preserving behavior under information loss or perturbation.
- The idea that a model's dynamical regime affects its compression floor is a plausible conjecture worth testing.

Claims to avoid:

- "Chaos explains the quantization floor."
- "We measured the Lyapunov exponent of GPT-4."
- "Temperature 0.7 is a universal critical point."
- "Golden Gate Claude is literally zero chaos."
- "Language entropy directly equals LLM divergence rate."

## Talk Shape

### Hook

Start grounded:

> We talk about model quality, latency, cost, and context length. But there's another property we rarely measure directly: stability. If I make a tiny change to the prompt, or a tiny change to the internal state, how fast does the model stop being the same system?

Possible title:

- **How Stable Is an LLM?**
- **Nearby Prompts, Distant Trajectories**
- **LLMs as Systems in Motion**

### Act 1: Why Stability Matters

Use familiar examples:

- Similar prompts sometimes produce very different long outputs.
- Some models collapse into strong attractors: refusal boilerplate, repetitive loops, feature-steered behavior like Golden Gate Claude.
- Too much stability is boring or collapsed; too little stability is noise.

### Act 2: The Axes

Make the static/dynamic distinction explicit. Put the five axes on one slide.

The talk's credibility depends on not mixing these together.

### Act 3: What Literature Says

Use papers as anchors, not as unquestionable foundations:

- **Cognitive Activation and Chaotic Dynamics in Large Language Models**: quasi-Lyapunov exponents; intra-network, iterative, and reasoning dynamics.
- **Lyapunov-style hallucination work**: stability framing around factual regions, if useful.
- **Speech/embedding trajectory work**: use carefully as language/semantic trajectory evidence, not as direct LLM generation evidence.
- **TurboQuant / rate-distortion**: a concrete contrast for static/runtime state compression floors.

### Act 4: Our Small Probe

Present the overnight experiment:

- identical prompt repeated: baseline implementation/API variance
- near-identical prompt pairs: input sensitivity
- hidden-state distances by layer: depth dynamics
- optional base vs instruct/model-family comparisons

Do not oversell. The value is showing a concrete measurement lens.

### Act 5: Open Question

End with:

> Stability might become a model property we measure alongside latency, cost, context length, and benchmark score.

Open questions:

- Are instruction-tuned models more contractive than base models?
- Do reasoning models preserve nearby trajectories longer on constrained tasks?
- Do high-divergence prompts become more sensitive to quantization?
- Can we build a practical "stability profile" for a model?

## Experiment North Star

Build a harness that can run the same perturbation protocol across a panel of open-weight models and produce reproducible artifacts:

- raw generations as JSONL
- hidden-state distance metrics as JSONL/CSV
- summary CSVs
- plots suitable for slides

The first deliverable is not a perfect Lyapunov estimator. It is a defensible stability probe.

## Initial Model Panel

Core panel:

- `google/gemma-4-E2B-it` or `google/gemma-4-E4B-it`
- `Qwen/Qwen3.5-4B`
- `Qwen/Qwen3.5-9B`
- `Qwen/Qwen3.5-27B`
- `openai/gpt-oss-20b`
- `mistralai/Mistral-Small-3.2-24B-Instruct-2506`
- `allenai/Olmo-3-7B-Instruct`
- `nvidia/NVIDIA-Nemotron-Nano-9B-v2`
- `meta-llama/Llama-4-Scout-17B-16E-Instruct`

Treat Llama 4 and Gemma 4 as "nice if access/tooling works," not blockers.

## Measurement Principles

- Separate noise floor from perturbation response.
- Prefer deterministic decode for the main comparison.
- Include same-prompt repeats as a baseline.
- Use perturbation ladders instead of one-off examples.
- Save raw outputs so every plot can be audited.
- Report distributions and confidence intervals when possible.
- Keep model-specific quirks in the registry, not scattered in experiment code.

## Success Criteria

Minimum useful result:

- Harness runs one small open model end-to-end locally.
- Outputs include JSONL generations, a summary CSV, and at least one plot.
- Docs explain what was measured and what cannot be concluded.

Strong overnight result:

- Harness runs 4-8 models on SageMaker.
- Produces baseline variance and prompt-pair divergence plots.
- Produces hidden-state layer divergence for models that support it.
- Flags models/templates that failed without breaking the whole run.

Talk-worthy result:

- A chart shows clear differences across model families, sizes, or tuning regimes.
- At least one plot illustrates why "stability profile" is a useful lens.
- The narrative stays honest about exploratory scope.
