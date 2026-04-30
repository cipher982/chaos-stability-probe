# Prior Art: Chaos, Dynamical Systems, and LLMs

This is a supporting reference doc, not the current-state narrative. Use
[results_digest.md](results_digest.md) for current claims and
[`../talk/slides.md`](../talk/slides.md) for the current slide flow.

## How To Use This In The Talk

The goal is not to survey every paper. The goal is to show the audience that
the chaos/dynamics lens has roots:

1. Classical chaos gives the vocabulary.
2. Neural-network theory already uses dynamical-systems language.
3. Recent LLM work applies Lyapunov-style ideas to layers, generation,
   paraphrasing, and hallucination.
4. Quantization/rate-distortion gives the compression side of the analogy.
5. Prompt-sensitivity work already documents the phenomenon empirically —
   cite it so I don't look like I'm inventing a well-studied field.

## Prompt Sensitivity / Brittleness (directly adjacent empirical work)

This is the field my experiment sits inside. These are the papers to know
about so the talk doesn't look uninformed. Add them to the backup-slide
citations.

- **Salinas & Morstatter 2024, "The Butterfly Effect of Altering Prompts."**
  Small prompt alterations — including whitespace-level changes — flip
  classification answers. Very close in spirit to my OLMo trailing-space
  demo. Cite as the empirical precedent for the whole phenomenon.
  https://aclanthology.org/2024.findings-acl.275/
- **Sclar et al. 2023, prompt formatting sensitivity** (arXiv:2310.11324).
  Up to 76 accuracy-point swings on LLaMA-2-13B from format-only prompt
  changes. Shows the scale of the effect.
- **Lu et al. 2021, "Fantastically Ordered Prompts"** (arXiv:2104.08786).
  Few-shot example order alone moves performance from near-random to
  near-SOTA. Best orders don't transfer across models.
- **Zhao et al. 2021, "Calibrate Before Use"** (arXiv:2102.09690).
  Prompt-format instability in GPT-3-era few-shot; early documentation of
  the problem.
- **PromptRobust** (arXiv:2306.04528). Adversarial prompt perturbation
  benchmark across character / word / sentence / semantic levels.
- **POSIX** (arXiv:2410.02185). Published prompt-sensitivity index based
  on log-likelihood changes under intent-preserving rewrites.
- **Worst Prompt Performance / RobustAlpacaEval** (arXiv:2406.10248).
  Argues for evaluating *lower-bound* performance over semantically
  equivalent prompts, not just best/average.
- **Prompt underspecification work** (arXiv:2602.04297). Argues some
  measured sensitivity comes from underspecified prompts rather than
  model fragility.

## Dynamical Systems in NNs (pre-LLM precedent)

If anyone says "this is just an LLM metaphor," these are the receipts that
dynamical-systems tools have been applied to neural networks for decades.

- **Sompolinsky, Crisanti, Sommers 1988**, "Chaos in Random Neural Networks,"
  Phys. Rev. Lett. 61, 259. The foundational chaos-in-RNN paper.
- **Poole et al. 2016**, "Exponential Expressivity in Deep Neural Networks
  Through Transient Chaos" (arXiv:1606.05340). Signal propagation through
  deep feedforward nets has order/chaos phase transitions.
- **Schoenholz et al. 2017**, "Deep Information Propagation." Same
  edge-of-chaos framework applied to trainability in deep nets. Predates
  Zhang et al. 2024's edge-of-chaos-and-LLMs work.
- **Haber & Ruthotto 2017**, "Stable Architectures for Deep Neural Networks"
  (arXiv:1705.03341). Deep learning framed as parameter estimation for
  nonlinear dynamical systems.
- **Geshkovski et al. 2023**, "The Emergence of Clusters in Self-Attention
  Dynamics" (arXiv:2305.05465). Attention as interacting-particle dynamics
  with clustering behavior.
- **Tomihari & Karakida 2025**, "Recurrent Self-Attention Dynamics"
  (arXiv:2505.19458). Jacobian-based Lyapunov analysis of self-attention;
  normalization suppresses oscillatory modes toward criticality.

## Must-Use Anchors

### Classical chaos: Lorenz and the logistic map

Use this to teach the audience the core idea before LLMs enter.

- Lorenz 1963, "Deterministic Nonperiodic Flow": foundation of modern chaos
  and weather predictability framing.
- Logistic map: one-line iterated equation with bifurcation and Lyapunov
  exponent visuals.

Talk use:

- Start from "butterfly effect," then correct it: chaos is deterministic
  sensitive dependence, not ordinary randomness.
- Show `distance(t) ~= distance(0) * e^(lambda t)`.
- Use the logistic map as the clean "iteration can create complexity" toy.

Sources:

- https://journals.sagepub.com/doi/10.1177/0309133315623099
- https://web.physics.rutgers.edu/grad/509/Logistic%20map.html

### Cognitive Activation and Chaotic Dynamics in LLMs

Paper:

- Xiaojian Li et al., "Cognitive Activation and Chaotic Dynamics in Large
  Language Models: A Quasi-Lyapunov Analysis of Reasoning Mechanisms"
  (`arXiv:2503.13530`).

What it claims:

- Introduces a Quasi-Lyapunov Exponent (QLE).
- Analyzes chaotic characteristics at different layers.
- Claims information accumulation follows a nonlinear exponential law.
- Claims small initial perturbations can substantially affect reasoning.

Talk use:

- Primary LLM-specific literature anchor.
- Use it to justify that "LLM dynamics" is not just our metaphor.
- Be careful: this is quasi-Lyapunov, not a universally accepted formal
  Lyapunov exponent for all LLM behavior.

Source:

- https://arxiv.org/abs/2503.13530

### Recurrent self-attention dynamics

Paper:

- Akiyoshi Tomihari and Ryo Karakida, "Recurrent Self-Attention Dynamics: An
  Energy-Agnostic Perspective from Jacobians" (`arXiv:2505.19458`, NeurIPS
  2025 poster).

What it contributes:

- Analyzes self-attention inference dynamics with Jacobians.
- Finds normalization suppresses Lipschitzness and oscillatory components.
- Uses Lyapunov exponents from Jacobians and reports normalized dynamics close
  to a critical state.

Talk use:

- Support for "attention itself can be studied as a dynamical system."
- Useful bridge from toy chaos to transformer internals.

Source:

- https://arxiv.org/abs/2505.19458

### Attractor cycles in LLM paraphrasing

Paper:

- Zhilin Wang et al., "Unveiling Attractor Cycles in Large Language Models: A
  Dynamical Systems View of Successive Paraphrasing" (`arXiv:2502.15208`,
  ACL 2025).

What it contributes:

- Treats iterative paraphrasing as a dynamical system.
- Shows repeated paraphrasing can converge to stable periodic states such as
  2-cycles.

Talk use:

- Concrete, intuitive "LLMs have attractors" example.
- Good for non-research audience because successive paraphrasing is easy to
  understand.

Source:

- https://arxiv.org/abs/2502.15208

### Lyapunov probes for hallucination

Paper:

- Bozhi Luan et al., "Lyapunov Probes for Hallucination Detection in Large
  Foundation Models" (`arXiv:2603.06081`).

What it contributes:

- Frames hallucination detection as a dynamical-systems stability problem.
- Models factual knowledge as stable regions and hallucination-prone cases as
  boundary/transition regions.
- Uses derivative-based stability constraints under perturbation.

Talk use:

- Practical payoff: instability is not just philosophical; it may correlate
  with hallucination-prone regions.
- Caveat: "Lyapunov probe" is stability-inspired; it is not necessarily
  computing classic Lyapunov exponents.

Source:

- https://arxiv.org/abs/2603.06081

### Edge of stability in neural-network training

Paper:

- Jeremy Cohen et al., "Gradient Descent on Neural Networks Typically Occurs at
  the Edge of Stability" (`arXiv:2103.00065`, ICLR 2021).

What it contributes:

- Shows full-batch gradient descent often operates where the top Hessian
  eigenvalue hovers just above the classical `2 / step size` stability
  threshold.
- Loss can behave non-monotonically in the short term but still decrease over
  longer timescales.

Talk use:

- Optional background: training itself also lives near a stability boundary.
- Do not let this swallow the talk; it is a separate axis from inference.

Source:

- https://arxiv.org/abs/2103.00065

## Useful But Secondary

### Lyapunov spectra in recurrent neural networks

Paper:

- "Lyapunov spectra of chaotic recurrent neural networks" (`arXiv:2006.02427`).

Use:

- Useful historical bridge: before transformer/LLM papers, people already used
  Lyapunov spectra to reason about neural sequence models.
- Good Q&A answer if someone says "this is just an LLM metaphor."

Source:

- https://arxiv.org/abs/2006.02427

### Lyapunov exponents for RNN information propagation

Paper:

- "On Lyapunov Exponents for RNNs: Understanding Information Propagation Using
  Dynamical Systems Tools" (Frontiers, 2022).

Use:

- Shows the established ML version of the idea: recurrent computation,
  gradient/error propagation, and stability can be studied with Lyapunov
  spectra.
- Mention only if asked for pre-LLM precedent.

Source:

- https://www.frontiersin.org/journals/applied-mathematics-and-statistics/articles/10.3389/fams.2022.818799/full

### Finite-time Lyapunov exponents of deep neural networks

Paper:

- "Finite-time Lyapunov exponents of deep neural networks"
  (`arXiv:2306.12548`).

Use:

- Shows that input perturbation growth/decay through a deep feed-forward
  network can be analyzed with finite-time Lyapunov exponents.
- Good support for "depth as time" intuition.

Source:

- https://arxiv.org/abs/2306.12548

### Speech embedding Lyapunov spectra

Paper:

- "Lyapunov Spectral Analysis of Speech Embedding Trajectories in Psychosis"
  (`arXiv:2602.16273`).

Use:

- Shows language trajectories in embedding space can have multi-scale dynamical
  signatures.
- Be precise: this measures human speech trajectories embedded by LLMs, not the
  LLM's own generation dynamics.

Source:

- https://arxiv.org/abs/2602.16273

### Edge of chaos and deep nets

Paper:

- "The edge of chaos: quantum field theory and deep neural networks"
  (`arXiv:2109.13247`).

Use:

- Optional theoretical color: criticality via largest Lyapunov exponent, and
  depth/correlation length as a trainability scale.
- Probably too technical for the main talk unless simplified heavily.

Source:

- https://arxiv.org/abs/2109.13247

### Transformers for chaotic systems

Paper:

- "Chaos Meets Attention: Transformers for Large-Scale Dynamical Prediction"
  (ICML 2025).

Use:

- Nice reversal: transformers are also used to model physical chaotic systems.
- Do not confuse this with "LLMs are chaotic"; it is about transformers applied
  to chaotic data.

Source:

- https://openreview.net/forum?id=Rxg8vCZSee

### Concept attractors in LLMs

Paper:

- "Concept Attractors in LLMs and their Applications" (OpenReview, 2025).

Use:

- Related but lower-priority bridge from internal representation geometry to
  attractor language.
- Useful if the conversation turns from chaotic divergence to semantic basins
  and guardrails.

Source:

- https://openreview.net/forum?id=ZqwyrPXbV9

## Compression / Quantization Anchors

### TurboQuant

Use:

- Static/runtime compression side of the talk.
- Good contrast: rate-distortion gives a compression floor for representing KV
  vectors; chaos/Lyapunov thinking gives a dynamical floor for tracking
  trajectories.

Important caution:

- Do not say "the Lyapunov exponent is the quantization floor."
- Safer: "These are two floors that rhyme: one static, one dynamical. The open
  question is whether the model's dynamical regime affects how far compression
  can go before behavior changes."

Source:

- https://www.tomshardware.com/tech-industry/artificial-intelligence/googles-turboquant-compresses-llm-kv-caches-to-3-bits-with-no-accuracy-loss

### KIVI

Use:

- Established 2-bit KV cache quantization baseline.
- Shows that low-bit KV compression can preserve quality surprisingly well, but
  the bit budget / quality relationship is subtle.

Source:

- https://arxiv.org/abs/2402.02750

## Non-Academic Explainers / Talk Inspiration

These can help with language, but should not carry technical claims.

- Veritasium, "The Science of the Butterfly Effect": good model for turning
  "butterfly effect" folk knowledge into deterministic predictability limits.
- Steven Strogatz / MIT OCW nonlinear dynamics materials: useful source of
  simple examples and classroom pacing.
- "Why Large Language Models Drift: A Dynamical Systems Perspective" (Medium).
  Useful framing around drift, internal state, and stability language.
- Logistic-map tutorials for visuals: bifurcation diagram plus Lyapunov curve.

Sources:

- https://www.veritasium.com/videos/2019/12/6/the-science-of-the-butterfly-effect-
- https://ocw.mit.edu/courses/18-353j-nonlinear-dynamics-i-chaos-fall-2012/pages/related-resources/
- https://medium.com/@tmineard/why-large-language-models-drift-a-dynamical-systems-perspective-7e39040de462
- https://sysidentpy.org/user-guide/tutorials/chaotic-systems/logistic-map/

## Slide-Level Literature Plan

Use only 4-5 named citations in the live talk:

1. Lorenz / logistic map: teach chaos.
2. Li et al. QLE paper: LLM-specific chaos framing.
3. Attractor cycles in paraphrasing: intuitive LLM dynamical example.
4. Edge of stability: training lives near a stability boundary.
5. TurboQuant/KIVI: compression floor / quantization anchor.

Everything else is backup for Q&A.
