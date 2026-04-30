# Companion Notes, second monitor cheat sheet

**Format.** Each slide has two blocks:

- **SAY**, short sentences you can literally read or start speaking. Use as a life raft.
- *Remember*, italic, in brackets. Internal reminders. **Never say these out loud.**

Glance, pick a SAY line, start talking, ad-lib from there.

---

## Presenting over Teams (remote, no video, screen share)

- Nobody sees your face. Your **voice** is the whole performance.
- Slow down ~20%. Over Teams, people can't read your body, so pace carries it.
- Vary pitch and pace on purpose, flat voice = they tab away.
- Pauses feel weirder than in a room, but still work. Count to 2 after a big line.
- **Mute when not speaking** if someone interrupts, keyboard/sniff noise is brutal on Teams.
- Have a glass of water. Dry mouth is the #1 remote-talk failure.
- Watch the chat peripherally but don't let it derail, address in Q&A.
- If someone unmutes to interject, let them. Don't talk over, Teams audio will cut one of you.
- Since there's no video, you can literally read from this doc on your second monitor. Just don't sound like you're reading.
- Name yourself at the start: "okay, I'll kick off."

**One line to land twice**
> Temperature samples from a distribution. Sensitivity asks how far the distribution moved.

**Closing line to memorize**
> Prompting is operating a high-gain branching system. Test neighborhoods, not single prompts.

**If you stumble**
- Don't apologize. Pause, sip water, restart the sentence.
- "Let me put that differently" is free.
- Blank on a number? "Roughly X, exact figure's in the backup." Move on.

---

## Slide 1, Nearby Prompts, Distant Trajectories

**SAY:**
- "Okay, this talk is called Nearby Prompts, Distant Trajectories."
- "Most of you have heard of the butterfly effect. I want to teach you what's actually under it, compounding, nonlinear divergence."
- "Then I want to walk through how that lens *might* apply to the LLMs we work with."
- "This is exploratory. I'm using chaos as vocabulary, then testing how far it gets us."
- "Some of what I tried clicked. Some of it didn't. I'll show you both."

*[Remember: curious tone, exploratory, not a paper defense. ~60 seconds. Lorenz attractor behind you is just mood.]*

---

## Slide 2, What I'm not claiming

**SAY:**
- "Before I get into anything, I want to lift some obvious objections off the table."
- "I'm *not* claiming LLMs are chaotic, classical chaos needs things LLMs don't have."
- "I'm not claiming I measured a Lyapunov exponent. Token space is discrete."
- "I'm not claiming bigger equals more stable, or that reasoning models are more stable. Neither holds up."
- "I'm not claiming sentence-embedding distance is ground truth, it's a proxy."
- "And I'm not claiming lower divergence is better. Stability is a property, not a score."
- "If any of these would have been your objection, great, we agree."

*[Remember: ~45 seconds. This buys you permission for everything later. Don't rush.]*

---

## Slide 3, What I am claiming

**SAY:**
- "Here's the honest thesis."
- "At inference time, an LLM is a **hybrid sequential system**, continuous activations feeding a discrete branching process."
- "Small changes can move distributions, or flip argmax branches. It varies a lot by model, prompt, and metric."
- "Naive measurement has specific failure modes worth naming."
- "Chaos vocabulary *organizes* the phenomenon. It doesn't *prove* anything."
- "The upshot, and I'll repeat it at the end: test neighborhoods, not single prompts."

*[Remember: this is your spine. "Hybrid sequential system" is the framing a dynamicist won't fight. Return here when you lose the thread.]*

---

## Slide 4, Chaos isn't randomness

**SAY:**
- "Quick detour into the physics, because the word 'chaos' is doing a lot of work."
- "Chaos is not randomness. Chaos is **deterministic amplification of small differences**."
- "Same equations. Two pendulums started at angles that differ by **half a degree**."
- "A few seconds later, they're in totally different places."
- "It's not rolling dice. It's magnifying what's already there."
- "Forecast errors compound because tiny measurement errors grow."

*[Remember: do NOT mention LLMs yet. Let the physics breathe. Everyone knows pendulums and weather.]*

---

## Slide 5, Small differences grow. Measurably.

**SAY — short (life raft):**
- "This is the logistic map. Simplest equation that shows deterministic chaos."
- "One equation, one knob called r. Low r: single value. Middle: 2, 4, 8 cycles. High r: never repeats."
- "That's a phase transition into chaos."
- "The rate at which nearby trajectories pull apart is called the **Lyapunov exponent**. Positive lambda means chaotic."
- "Langton 1990, Zhang 2024 — trained neural networks tend to sit *near* this boundary."
- "So a legitimate question is: which side of the boundary is any given LLM on?"

**SAY — long (verbatim, ~90 seconds):**

> "This is the logistic map. One equation: x-next equals r times x times one minus x. One knob, r. You iterate — feed the output back in as the input. That's the whole system.
>
> The chart on the right is a bifurcation diagram. The x-axis is the knob r, **not time**. Each vertical slice is a different setting of r, and the dots in that slice are all the values the system ends up visiting forever.
>
> Low r: one dot. The system settles on a single value. Turn r up, the system splits — now it bounces between two values forever. Turn it up more, four values. Then eight. Then sixteen. The splits accelerate, pile up, and by r around 3.57 the dots fill the column — chaos. The system never repeats.
>
> The rate at which nearby trajectories pull apart has a name: the **Lyapunov exponent**, lambda. Positive lambda means chaotic. Zero means edge. Negative means stable.
>
> Here's why this matters for us. Langton, 1990 — cellular automata. Poole and Schoenholz, 2016. Zhang, 2024. Trained neural networks are known to sit near this stable–chaotic boundary. Not by accident — that's where they train, and that's where signal propagates cleanly.
>
> So the honest question isn't 'are LLMs chaotic.' It's 'which side of the boundary is any given LLM on?' That question is what the rest of the talk probes."

---

**Background — what the chart actually is**

- **Equation:** x_{n+1} = r · x · (1 - x). Variable x lives in [0,1]. Knob r lives roughly in [0, 4].
- **Bifurcation diagram:** for each r on x-axis, run the iteration for thousands of steps (throw away transient), plot where it settles on the y-axis. Every column is an independent experiment.
- **What the "lines" are:** NOT lines. Thousands of adjacent scatter columns that look connected because neighboring r values give similar answers. No time is passing left-to-right.
- **Regimes by r:**
  - r < 3: one dot. Fixed point.
  - 3 < r < ~3.45: two dots. Period-2.
  - ~3.45 < r < ~3.54: four dots. Period-4.
  - …doublings accelerate…
  - r > ~3.57: chaos. Smeared column.
  - Windows of order inside chaos (famous period-3 window near r=3.83).

**Background — the Lyapunov exponent**

The formula (don't put on slide): |δ(t)| ≈ |δ(0)| · e^(λt). Gap between two nearby trajectories grows exponentially at rate λ.
- λ > 0: chaotic (exponential divergence).
- λ = 0: edge.
- λ < 0: stable (gap shrinks).
- For logistic map: λ is the long-run average of log|r(1-2x)| along a trajectory.
- **Plain-English gloss:** "time-averaged local stretching rate."

**Background — what "universal" means here**

- The logistic map is one equation. The chart covers ONE system.
- **Feigenbaum universality (1978):** an entire class of 1D "folding" maps follows the same period-doubling route to chaos with the same Feigenbaum constant, δ ≈ 4.669.
- Real systems in this class: dripping faucets, some laser intensities, certain electronic circuits, some chemistry.
- **Does NOT cover:** higher-dim systems (Lorenz, Navier-Stokes), NNs as geometric objects, arbitrary "chaotic" things. Those can hit chaos by different routes (quasiperiodic, intermittency, crisis).

**Background — the edge-of-chaos citation chain**

This is LOAD-BEARING. Without it, the slide is pop science.

- **Langton 1990 — "Computation at the Edge of Chaos."** Cellular automata. Interesting computation (memory, signal propagation) happens in a narrow band between ordered and chaotic regimes.
- **Poole 2016 / Schoenholz 2017.** Deep NN signal propagation at initialization. Ordered regime: signals shrink to zero. Chaotic regime: signals decorrelate. Networks only train if initialized near the edge. This is concrete math, not metaphor.
- **Zhang 2024.** Modern reprise for transformers. Same edge-of-chaos story tracks training stability.

The line "trained nets sit near this boundary" is shorthand for: gradient descent selects for edge-of-chaos behavior because the other regimes have vanishing or exploding gradients.

**Background — what the knob is**

- For the logistic map: r is literally the growth rate. How aggressively the feedback kicks.
- **No universal "the knob."** Each system has its own.
  - Dripping faucet: flow rate.
  - Driven pendulum: driving amplitude/frequency.
  - Rayleigh-Bénard convection: temperature gradient.
  - Laser: pump power.
  - Heart: pacing frequency.
  - Neural net (Poole/Schoenholz): weight variance + nonlinearity gain.
- **Pattern:** the knob is usually "how hard the system is being driven."

**Background — why the "middle value" isn't an attractor**

- In the period-2 region, there IS a fixed point in the middle of the two branches. But it's **unstable** — a repeller.
- Mechanic: any nudge above the midpoint overshoots below it next step, and *further* from the middle. Each iteration flips + amplifies.
- System gets bounded by [0,1] walls so the overshoot-amplify stabilizes into a period-2 cycle.
- Population analogy: big year → crowding → small year → abundance → big year. The feedback rule forbids the average.

---

**If asked: "Is this chart covering every system?"**
> "No. One equation, one knob. What makes it iconic is that a whole class of folding systems follow the same period-doubling route — that's Feigenbaum universality. Dripping faucets, some electronics, some chemistry. It's the canonical teaching example, not a universal diagram."

**If asked: "Is this supposed to be animal populations?"**
> "It started as a population model in the 1800s — Verhulst. Got famous in the 1970s when Robert May and Feigenbaum realized it's the simplest system that shows the full route to chaos. Today the population framing is just a way to build intuition. The equation is pure math. Think of it like the harmonic oscillator of physics — not a specific physical thing, but representative of a broad class."

**If asked: "Is this the x-axis time?"**
> "No — that's the trap. X-axis is the knob, r. Each vertical slice is a different, independent system. The dots in that slice are all the values that system ends up visiting forever. No time passes as you move left to right."

**If asked: "Then why does it look like a line chart?"**
> "It isn't. Adjacent columns of r give similar settling values, so the dots *happen* to land near each other and your eye connects them. But nothing is flowing along those curves. They're just the locus of fates as you sweep the parameter."

**If asked: "Why doesn't the system just settle at the average of those two branches?"**
> "Because the middle point is unstable — it's a ridge, not a valley. Any tiny nudge gets amplified and flipped each iteration. The dynamics actively push the system off the middle. The 'average' is a fixed point of the equation, but the system can't sit there."

**If asked: "What's the knob? What's r in general?"**
> "For the logistic map, r is the growth rate — how aggressively the feedback kicks. There's no universal 'the knob.' Each system has its own — flow rate for a faucet, driving amplitude for a pendulum, pump power for a laser. In high-dim systems the boundary is a *surface* in parameter space, not a point on a number line."

**If asked: "What's the knob for an LLM?"**
> "Honestly — not settled. Signal-propagation work treats things like weight initialization variance and nonlinearity gain as knobs during training. For a trained LLM at inference, the relevant quantities are probably things like depth, attention temperature, layer norm scale. I don't have a clean scalar r for you and I don't want to pretend I do. The rest of the talk is me probing the question without claiming I've answered it."

**If asked: "Is this a model of LLMs?"**
> "No. LLMs are not a 1D folding map. I'm borrowing *vocabulary* — fixed points, bifurcations, Lyapunov exponents, edge of chaos — not geometry. The bridge is: trained neural networks are known to sit near the chaos boundary. That's what buys me permission to use this vocabulary for the rest of the talk."

**If asked: "Have people actually measured LLMs at the edge of chaos?"**
> "For initialization, yes — Poole, Schoenholz, plus transformer follow-ups. For trained autoregressive inference, much less settled. Li et al. 2025, which I'll show on slide 10, computed a quasi-Lyapunov exponent on Qwen2-14B in activation space — about 1.32× per layer. That's the closest thing to a direct measurement."

**If asked: "Do all chaotic systems reach chaos by period-doubling?"**
> "No. Period-doubling is one route. Others: quasiperiodic (Ruelle-Takens), intermittency (Pomeau-Manneville), crisis. Lorenz — weather — gets to chaos a different way entirely. This chart shows one specific route."

**If asked: "Period 3 implies chaos?"**
> "Yes — Li-Yorke theorem, 1975. Any 1D map with a period-3 cycle has cycles of every period and is chaotic in the topological sense. The period-3 window in the diagram near r=3.83 is a real phenomenon, not a glitch. That's a rabbit hole, though — I'd only go there if pressed."

---

**One-liners to land:**

> "The x-axis is the knob, not time."

> "Lambda measures the time-averaged local stretching rate. Positive means nearby trajectories pull apart exponentially."

> "The reason chaos isn't metaphor here is that trained neural networks are already known to sit near the chaos boundary. The question isn't whether the boundary is relevant — it's where on the boundary any given model lives."

> "I'm borrowing vocabulary, not geometry. The logistic map is the harmonic oscillator of chaos."

*[Remember: ~90 seconds target. The edge-of-chaos citation is what turns metaphor into a research question. Don't write the Lyapunov formula on screen. If someone tries to collapse "specific chart / general phenomenon / conceptual LLM connection" into one claim, separate them verbally — that's the single most likely Q&A trap.]*

---

## Slide 6, Same input. Same weights. Different output.

**SAY:**
- "Here's the visceral version. OLMo-3, 7B."
- "Prompt A asks for a concise palindrome function. Prompt B is the exact same prompt, with a **trailing space** added."
- "Argmax decode. No sampling. Same weights."
- "Output A is a utility-style function with a docstring. Output B is a chatty conversational answer with a one-liner implementation."
- "This shouldn't produce two different essays. But it does."
- "Before anyone asks, this isn't temperature. Argmax has no sampling step. The seed is inert."
- "The model's most confident response under A is different from its most confident response under B. The distribution itself moved."

*[Remember: don't dunk on OLMo. Many models do this. It's the clearest single example, not the whole story.]*

---

## Slide 7, Temperature is a separate axis

**SAY — short (life raft):**
- "I want to pull temperature and sensitivity apart, because people collapse them."
- "Temperature is: given a fixed distribution, which token do we sample?"
- "Sensitivity is: how far did the distribution itself move when the prompt changed?"
- "The 2x2: same prompt + temp zero is byte-identical. Same prompt + temp positive is sampling noise. Tiny prompt change + temp positive is confounded."
- "Tiny prompt change + temp zero, that's the starred cell. That's what this talk measures."
- "**Temperature samples from a distribution. Sensitivity asks how far the distribution moved.**"

**SAY — long (verbatim, ~75 seconds):**

> "The most common pushback on this kind of work is 'isn't that just temperature?' I want to kill that conflation right now, because it's two different phenomena and people collapse them.
>
> Temperature is about sampling. You have a fixed next-token distribution, and temperature controls how peaky or flat it is when you draw. Sensitivity is about the distribution itself — when I change the prompt by a trailing space, how far does the whole distribution move before we even sample?
>
> Look at the 2x2. Same prompt, temperature zero: byte-identical output. Boring — that's just determinism. Same prompt, temperature above zero: different draws, same vibe — that's sampling noise. Now, tiny prompt change with temperature on: different draws *and* different vibe, and you can't tell the two apart. Confounded.
>
> The starred cell, tiny prompt change at temperature zero, is the clean probe. Sampling noise is literally removed because argmax has no sampling step. Anything that moves in that cell is the model's response function shifting.
>
> One line, and I'll repeat this at the end: temperature samples from a distribution. Sensitivity asks how far the distribution moved."

---

**Background — temperature, precisely**

- Temperature T rescales logits before softmax: p_i ∝ exp(logit_i / T).
- T → 0: distribution collapses to argmax (all mass on max-logit token).
- T = 1: raw distribution.
- T → ∞: uniform over vocab.
- `do_sample=False` in HF transformers bypasses sampling entirely. Seed is inert. Determinism is exact up to floating-point kernel nondeterminism (negligible vs. the effects measured).

**Background — why argmax is the right probe**

- Deterministic decode removes *stochastic* variance. What's left is the model's *functional* response to the input.
- At T=0.7 on OLMo-3, within-prompt sampling distance and between-prompt sampling distance can match in magnitude — the two effects fall into the same band. That's why T>0 is not a clean experimental surface for this question.
- Argmax isolates the "did the distribution move?" signal from the "did we draw differently?" signal.

**Background — the common misread**

- "ChatGPT-shaped intuition": users associate variance with temperature, so they reflexively explain *any* output variance as sampling. This is the confusion to disarm.
- Sensitivity and temperature are orthogonal axes. You can have zero sampling noise and still see the output move. That's what the starred cell demonstrates.

---

**If asked: "Isn't this just temperature?"**
> "No. Temperature is sampling from a fixed distribution. What I'm measuring is the distribution itself moving when the prompt changes. The probe is argmax decode, do_sample=False. There is no sampling step. Temperature is not in the experiment."

**If asked: "What if I set T=0?"**
> "T=0 is exactly what I use. That's the point. With T=0 the same prompt gives byte-identical output. So when a trailing-space prompt change produces a different output, it's not sampling — it's the model's response function shifting."

**If asked: "What's the difference between temperature and sensitivity?"**
> "Temperature asks: given a fixed distribution, which token do we draw? Sensitivity asks: when I perturb the prompt, how far does the distribution itself move? Orthogonal axes. Temperature lives inside a single forward pass. Sensitivity compares two forward passes on different inputs."

**If asked: "If temp=0 gives the same output twice, how can the output 'move'?"**
> "Same prompt twice at T=0 gives the same output — that's determinism. *Different* prompt at T=0 can give a different output, because the distribution is a function of the input. Change the input, the distribution changes, the argmax can flip. That's the whole phenomenon."

**If asked: "Does this mean temperature is bad?"**
> "No. Temperature is a deployment choice. It's just not the right knob for measuring sensitivity — it adds variance that masks the thing I'm trying to see. For deployment you might want T>0. For a sensitivity probe, you want T=0."

**If asked: "What about T=0.1, doesn't sampling noise stay small?"**
> "I ran that control on OLMo-3, 30 samples per prompt at T=0.1. Prompt A's samples cluster tightly, Prompt B's cluster tightly, and the two clusters are *visibly separate*. The shift between clusters is bigger than the within-cluster sampling noise. So even with a little temperature, the sensitivity signal dominates."

---

**One-liners to land:**

> "Temperature samples from a distribution. Sensitivity asks how far the distribution moved."

> "Argmax has no sampling step. The seed is inert. Anything moving in the starred cell is the model's response function."

> "These are orthogonal axes. Don't collapse them."

*[Remember: pedagogical linchpin. ~2 minutes. Say the one-liner slowly and consider repeating it. Footnote: at T=0.7 on OLMo-3, within-prompt and between-prompt sampling distances match, that's why deterministic decode is the only clean probe. If Q&A keeps coming back to temperature, it means you lost them here — slow down and redo the 2x2.]*

---

## Slide 8, So: is an LLM a dynamical system?

**SAY:**
- "Let's run the checklist."
- "It has **state**, hidden activations, logits, KV cache, prefix."
- "It has **iteration**, each token feeds into the next."
- "It's **deterministic** under argmax."
- "And small input perturbations can produce large output changes."
- "So on the checklist, it looks like a dynamical system. The real question is whether the *magnitude* of amplification is interesting, and whether we can measure it."
- "There's a catch. Classical chaos wants perturbations going to zero. Token space is discrete. I'll come back to this."

*[Remember: short bridge slide, ~60 seconds. The catch line sets up the meaning-preserving slide.]*

---

## Slide 9, Both outputs can be correct

**SAY:**
- "Important framing. When a prompt perturbation produces a different output, that's not the model making a mistake."
- "A double pendulum isn't *wrong* when it lands somewhere different. Same bar for LLMs."
- "'Book like Dune', the model recommends Foundation. Add a trailing space, it recommends Hyperion."
- "Both defensible. Neither a hallucination. The model picked a different basin."
- "So the thing to measure is divergence per unit of **meaning-preserving** input change."
- "If you stick 'NOT' at the front of the prompt, of course the output moves, you changed the meaning. That's the model working."
- "The interesting quantity is: output move divided by input move, when the input move is small."

*[Remember: Li et al. hedge, |δ|→0 works in activation space, not token space. Two open extensions: restrict to meaning-preserving, or move probe into activation space.]*

---

## Slide 10, State, and prior work, short version

**SAY — short (life raft):**
- "Quick detour into prior work. An LLM's state is hidden activations, logits, prefix, KV cache."
- "Li et al. 2025, quasi-Lyapunov on Qwen2-14B. Magnitudes grow **1.32× per layer**."
- "*Quasi*-Lyapunov because it's finite depth, not an infinite-time iterated map."
- "MLP contributes **55.8%** of the final residual. Attention **44.2%**. Initial input: **0.0009%**."
- "Chaos in one line: we perturb the 0.0009% and watch the 100% move."
- "They ran one model. I run about eighteen, on the axis they didn't — the output-text level. Complementary, not redundant."

**SAY — long (verbatim, ~80 seconds):**

> "Quick detour into prior work so you know I'm not the first person here.
>
> An LLM's state, the thing that evolves through a forward pass, is hidden activations plus logits plus prefix plus KV cache. That's what we'd need to track for a real dynamical analysis.
>
> Li et al. in 2025 did the closest thing we have. They did a quasi-Lyapunov-style analysis on Qwen2-14B in activation space. They call it *quasi*-Lyapunov because classical Lyapunov assumes a fixed map iterated to infinity — LLMs have finite depth. So it's the finite-depth analog.
>
> The number that matters: magnitudes grow about **1.32× per layer** in the first ten layers. Compounded, that's the model amplifying a tiny input perturbation by orders of magnitude by the time it hits the output.
>
> The one-line version is even sharper. They decompose the final residual stream: MLP contributes **55.8%**, attention **44.2%**, and the initial input is **0.0009%**. We are perturbing that 0.0009% and watching the 100% move. That's chaos in one line.
>
> Important honesty: they ran one model, Qwen2-14B. They defined a token-level iterative exponent but never computed it, and never compared models. I run eighteen-ish models on exactly that axis — the output-text level. Complementary, not redundant.
>
> Also worth naming: Geshkovski 2023, attention as interacting-particle dynamics. And Poole / Schoenholz on edge-of-chaos signal propagation. The field has been pointing at this for years."

---

**Background — Li et al. 2025 in one page**

- Paper: quasi-Lyapunov exponent (QLE) analysis of Qwen2-14B, in activation space.
- **Why "quasi":** classical Lyapunov λ requires a fixed map T iterated t → ∞. LLMs have finite depth L. Li et al. use the per-layer Jacobian norm as the analog — it measures local stretching but can't be iterated indefinitely.
- **Headline:** per-layer magnitude growth ≈ **1.32×** in layers 0–9.
- **Residual-stream decomposition at the final layer:** MLP 55.8%, attention 44.2%, initial input 0.0009%.
- The 0.0009% figure is the clearest framing: the output is dominated by in-network transformations, and the input is a vanishingly small fraction of what ends up in the final residual. Small input perturbations ride a big amplification chain.
- **Gap they left:** they defined an iterative (token-level) QLE but never computed it, and never compared models. Token-level across many models is exactly what my experiments probe.

**Background — "state" of an LLM**

- Hidden activations: per-layer residual stream.
- Logits: output of final layer before softmax.
- Prefix: previously generated tokens (autoregressive history).
- KV cache: cached keys/values for attention over the prefix.
- A full dynamical description would track all four. My probe only observes the downstream shadow at the *output-text level*.

**Background — activation space vs. token space**

- Activation space is continuous. You can take |δ| → 0 cleanly. Classical Lyapunov works there.
- Token space is discrete. You can't take δ → 0 — smallest perturbation is one token. So chaos math gets cleanest in activation space; text-level probes observe the shadow.
- Li et al. chose activation space. I chose text-level. Different probes of the same system.

**Background — other adjacent citations**

- **Geshkovski et al. 2023:** attention layers as interacting-particle dynamics. Tokens as particles, attention as the interaction force.
- **Poole 2016 / Schoenholz 2017:** edge-of-chaos signal propagation in deep nets. The original "trained nets sit near the chaos boundary" result.
- **Tomihari & Karakida 2025:** Jacobian/Lyapunov analysis specifically for self-attention.

---

**If asked: "What's a Lyapunov exponent in an LLM?"**
> "Classical Lyapunov is the time-averaged rate at which nearby trajectories separate under iteration of a fixed map. In an LLM, the cleanest analog is per-layer: measure the Jacobian norm — how much a small perturbation in activations gets stretched as it passes through a layer. That's Li et al.'s quasi-Lyapunov. It's a finite-depth analog because we don't have infinite iteration."

**If asked: "What does 1.32× per layer mean?"**
> "It means a small perturbation in the hidden activations grows by about 32% each layer, on average, in the first ten layers of Qwen2-14B. Compounded over depth, that's orders of magnitude of amplification by the time the signal reaches the output."

**If asked: "What does 0.0009% mean?"**
> "At the final layer, they decomposed the residual stream into contributions. MLP was 55.8%, attention 44.2%, and the initial input embedding was 0.0009%. So the output is almost entirely transformations done inside the network. The input is a tiny seed, and the network amplifies it massively. That's chaos in one line: perturb the 0.0009% and watch the 100% move."

**If asked: "Is this peer reviewed?"**
> "Li et al. 2025 is a recent arXiv paper. It's where the 1.32× and the 0.0009% come from. The broader edge-of-chaos NN literature — Poole, Schoenholz — is peer-reviewed and established. I'd treat Li et al. as the most directly analogous prior work, not as final word."

**If asked: "Why only Qwen2-14B?"**
> "That's the limitation of their paper, not mine. They ran one model in activation space. They didn't compare across models, and they didn't compute their own token-level exponent. That gap is exactly what my experiments sit in — text-level divergence across eighteen models."

**If asked: "What's 'quasi'-Lyapunov?"**
> "Classical Lyapunov needs a fixed map iterated to infinite time. LLMs have finite depth — they stop at some layer. 'Quasi' is the finite-depth honest analog: per-layer Jacobian stretching, not an infinite-horizon rate. It captures the spirit without overclaiming the math."

**If asked: "Why activation space not token space?"**
> "Activation space is continuous — you can take perturbations to zero, which is what classical chaos math needs. Token space is discrete — smallest perturbation is one token. So activation-space probes get the cleaner math. Text-level probes, like mine, observe the downstream shadow and trade rigor for cross-model comparability."

**If asked: "So is this prior work or your work?"**
> "Both. Li et al. is the rigorous single-model activation-space probe. Mine is the text-level cross-model survey. They probe the same system from different angles. I'd never claim Li et al.'s math; they'd never claim my breadth."

---

**One-liners to land:**

> "We're perturbing the 0.0009% and watching the 100% move."

> "1.32× per layer, compounded, is a lot of amplification."

> "Chaos math is cleanest in activation space. Text-level probes see the downstream shadow."

*[Remember: don't oversell Li et al. as proving my thesis — frame as complementary. Key numbers to land: 1.32×, 55.8% / 44.2% / 0.0009%. If you blank, say "per-layer amplification, roughly a third per layer, and the input is less than a thousandth of the final residual." ~90 seconds.]*

---

## Slide 11, The experiment

**SAY:**
- "Here's the setup."
- "About 21 models. Qwen from 0.8B to 9B, Gemma 4, Phi-4, DeepSeek-R1, Mistral, Granite, Falcon, SmolLM, OLMo 2 and 3, plus legacy, GPT-2 XL, GPT-J, Pythia, OPT, LLaMA-1."
- "Prompt ladder: identical, no-op formatting, punctuation, synonym, paraphrase, small semantic, positive control."
- "Deterministic decode. do_sample=False. Divergence is a shift in the model's *most confident* response."
- "Primary metric: sentence-embedding cosine distance. Supporting: token edit, hidden-state distance, logit JS and KL. All proxies, no ground truth."
- "Stats: bootstrap CIs and paired permutation tests. I present **clusters, not ranks**."
- "Pinned HF revisions, bf16, chat templates disabled, config published with artifacts."

*[Remember: n=21 prompt pairs in the panel, n=42 in the hardened Qwen wave. Key control: same prompt + deterministic decode = 0.000 divergence. Don't oversell.]*

---

## Slide 12, What actually matters?

**SAY — short (life raft):**
- "This is the cleanest answer to: what actually matters?"
- "Rows are models. Columns are edit categories — the kinds of tiny changes I tested."
- "Correction up front: I threw away the character-level version. A lot of its green cells weren't model behavior, they were token-identical *after* the chat template."
- "This is the token-audited subset: pairs where the effective prompt tokens actually changed."
- "The edits that survive tokenization and move outputs are **internal structure** — line breaks, parenthesized words, duplicated punctuation, internal spacing."
- "The stronger pattern is row-wise. Some models are much more sensitive than others."
- "Not 'any byte flips the model.' More structured than that."

**SAY — long (verbatim, ~75 seconds):**

> "This is the chart I'd point to if you asked me: of all the tiny edits I tried, what actually matters? Rows are models. Columns are categories of edit. Brightness is how much the output moves on that category.
>
> I want to own a correction before getting into it. The original version of this chart was character-level, and honestly it wasn't a clean experiment. A lot of the bright cells were not the model responding — they were cases where two different-looking strings tokenized identically after the chat template, so the model literally saw the same input. I threw that version out.
>
> What's here is the token-audited subset. Every pair shown actually changed the effective prompt tokens the model saw. So the signal is real.
>
> The column sort goes by average effect, but the stronger pattern is row-wise. Some models are much more sensitive than others to the same edits. That's the real finding.
>
> On the column side, what survives tokenization and moves outputs is mostly *internal structure*: line breaks inside the prompt, parenthesizing one word, duplicated punctuation, internal spacing. Not random bytes. Not leading whitespace. Structured edits that actually produce a different token sequence.
>
> The headline is: this is not 'any byte flips the model.' It's more structured than that. Some edits never reach the model as distinct input. Some survive the template, change tokens, and move the model into a different basin."

---

**Background — what the heatmap shows**

- **Rows:** models in the panel (Qwen family, Gemma, Phi-4, DeepSeek-R1, OLMo, etc.).
- **Columns:** edit categories — line breaks, internal spacing, punctuation duplication, parenthesization of single words, case flips, Unicode near-equivalents, etc.
- **Cell value:** average output divergence for that (model, category) pair, on pairs that survived the tokenizer audit.
- **Sort:** columns sorted by average effect across models. Row order approximately matches overall sensitivity.

**Background — the character-level correction**

- The old version swept character-level edits without checking tokens. Many edits produced strings that, once wrapped in the chat template and tokenized, were *byte-identical at the token level*. The model literally saw the same input.
- "Green cells" (low divergence) were often just this tokenization artifact, not model behavior.
- Current version drops those pairs and keeps only pairs with a real post-template token delta.
- This is the slide where you show methodological honesty. Name the bug.

**Background — what "tiny edit" means here**

- Operationally: a single-character or single-whitespace change to the user prompt that preserves meaning to a human reader.
- Examples that survive: internal line break inside a sentence, adding parentheses around a word, duplicating a period, flipping "," to ";".
- Examples that often *don't* survive (get normalized): leading/trailing whitespace, some punctuation at prompt boundaries — the chat template can strip or merge these.

**Background — tokenizer vs. model**

- Tokenizer is deterministic preprocessing. If the tokenizer produces the same token IDs for two different-looking strings, the model has no way to distinguish them.
- Chat templates (HF `apply_chat_template`) wrap user content in special tokens and can normalize whitespace. Effects that look like "the model ignored the edit" are often "the template erased the edit."
- Correct framing: the template/tokenizer *normalized* the input, not the model *ignored* the input.

---

**If asked: "What are the rows? What are the columns?"**
> "Rows are models. Columns are categories of edit — the kinds of tiny perturbations I tested. Line breaks, internal spacing, parenthesizing a word, duplicated punctuation, and so on. Each cell is the average output divergence for that model on that category, using pairs that actually produced different tokens after the template."

**If asked: "What's a 'tiny edit'?"**
> "A single-character or single-whitespace change to the user prompt that a human reader wouldn't flag as changing the meaning. Things like adding a line break inside a sentence, or wrapping one word in parentheses. Not synonyms, not paraphrases. Almost invisible at a glance."

**If asked: "Why did the character-level version get thrown out?"**
> "Because a lot of the edits I tested produced strings that, after the chat template wrapped them and the tokenizer ran, were byte-identical at the token level. The model saw the same input. I was measuring tokenizer behavior, not model behavior. The current chart keeps only pairs with a real post-template token delta."

**If asked: "Aren't some of these just tokenizer differences?"**
> "That's exactly why the original chart went in the bin. The current chart is token-audited — every pair shown actually changed the effective tokens. That said, tokenizer differences across *models* are a real limitation. I can't cleanly separate 'model is sensitive' from 'model's tokenizer happens to expose more edits,' and I'd flag that as an open question."

**If asked: "Which models are most sensitive?"**
> "The row-wise pattern is the one to read. I don't want to call out a leaderboard, because the edit set is small and the bars are wide. But the Qwen 0.8B end of the panel is visibly brighter than, say, the larger instruct-tuned models in the middle. Cluster view, not rank."

**If asked: "How is this measured?"**
> "For each prompt pair, I run the original and the edited prompt through the model with deterministic decode, take the two outputs, and score output divergence with sentence-embedding cosine distance. Average across pairs in a category to get a cell. All deterministic, no sampling."

**If asked: "What about the edits that *didn't* move anything?"**
> "Leading and trailing whitespace at the very boundaries of the prompt often get normalized by the chat template, so they never reach the model as distinct tokens. Those produce near-zero divergence for a boring reason — the template swallowed the edit. That's a different story from 'the model is robust.'"

---

**One-liners to land:**

> "This is not 'any byte flips the model.' It's more structured than that."

> "Some edits never reach the model as distinct input. Some survive the template and move the model into a different basin."

> "The stronger pattern is row-wise. Some models are much more sensitive than others to the same edits."

*[Remember: do NOT say the model "ignored" the edit — say the template/tokenizer normalized it. This is the slide where you show methodological honesty by naming the character-level bug. Same slide number as the old version, no deck shift. Land the row-wise finding and move on.]*

---

## Slide 13, Same-looking prompt. Different trajectory.

**SAY:**
- "This one makes the dynamical-systems framing concrete."
- "Y equals zero means the generated prefixes are token-for-token identical. A rising line means more edits are needed to align the token prefixes."
- "This uses Levenshtein distance, so simple insertion/deletion offsets get aligned away."
- "If the problem were just 'same output shifted by one token,' you'd see the dashed gray line decay to zero."
- "The blue and red lines stay high, that means the path itself changed, not just a one-token offset."
- "The right panel checks semantic endpoint distance separately."
- "This is a **token-path diagnostic**. Not quality, not a semantic claim."

*[Remember: this is branching in the dynamical-systems sense. The Levenshtein framing pre-empts "isn't this just a trivial shift?"]*

---

## Slide 14, Within-Qwen: one clean contrast

**SAY:**
- "Here's the cleanest within-family contrast. Qwen 3.5, four sizes."
- "0.8B is meaningfully more sensitive than 4B, p less than 0.001. 2B also separates from 4B, p equals 0.012."
- "But 4B vs 9B is indistinguishable at this n. I am *not* claiming bigger equals stable."
- "And there's a caveat I'll own on the next slide, 4B and 9B emit a 'Thinking Process:' preamble. That's a scaffold confound."

*[Remember: this was the single cleanest finding before the scaffold self-audit. Don't say "4B beats 9B." Say cluster membership. ~45 seconds.]*

---

## Slide 15, Scaffold "stability" is mostly metric artifact

**SAY — short (life raft):**
- "Self-audit time. At **64 tokens** of output, scaffolded models look ~**4× more stable**. That's because identical `<think>` preambles dominate the sentence-embedding similarity."
- "This is a warning about **evaluation**, not a property of the models."
- "Push output to **512 tokens** and it's a mixed bag."
- "DeepSeek-R1 7B: **0.027**, genuinely stable. Qwen 4B / 9B: **0.050 / 0.057**. SmolLM3 3B: **0.080**, middle."
- "**Phi-4 reasoning+: 0.160, more brittle than GPT-2 XL.** And its prompt-end top-1 probability is **0.99999996** — the most confident in the panel."
- "Confident logits do not mean stable trajectory."
- "Thinking-off isn't monotonic: scaffold helps the big Qwens, **hurts** the 0.8B."

**SAY — long (verbatim, ~85 seconds):**

> "Self-audit time. This is the slide where I walk back something from two slides ago.
>
> At 64 tokens of output, scaffolded models — the ones that start every answer with a `<think>` block or a 'Thinking Process:' preamble — look about four times more stable than non-scaffolded models. That's the naive finding. It is also a metric artifact. Sentence-embedding similarity is dominated by the identical preamble; the actual answer hasn't started yet. So this is a warning about *evaluation*, not a claim about the models.
>
> Push output to 512 tokens and the scaffold cover blows. It becomes a mixed bag.
>
> DeepSeek-R1 7B: 0.027. Genuinely stable. Qwen 4B and 9B: 0.050 and 0.057, still decent. SmolLM3 3B: 0.080, middle of the pack.
>
> And then Phi-4 reasoning-plus. **0.160**, more brittle than GPT-2 XL at 512 tokens. *And* its prompt-end top-1 probability is **0.99999996** — the most confident model in the panel. The bulk distribution, scored by JS, barely moves — 1.4 times 10 to the minus 9. And yet the 512-token output diverges at 0.160. That is the single cleanest dissociation between 'confident logits' and 'stable trajectory' in the whole dataset.
>
> On top of that, the Qwen thinking-off control isn't monotonic. For 4B and 9B the scaffold helps — turn it off and divergence goes up. For 0.8B, the scaffold *hurts* — turn it off and divergence drops. Scaffold behavior in small models is noisy.
>
> The headline: scaffold does not equal stable."

---

**Background — what a scaffold is**

- Scaffold = deterministic preamble the model emits before answering. E.g., `<think>...</think>` blocks (DeepSeek-R1, Phi-4 reasoning+, Qwen3.5 4B/9B), or literal "Thinking Process:" text (SmolLM3).
- Because the preamble is highly deterministic, two runs on different prompts often share a long identical prefix.
- Sentence embeddings of short outputs (64 tokens) will therefore register high similarity even if the *post-scaffold* answer diverges — the scaffold dominates the vector.

**Background — key numbers (512 tokens, n=24 pairs)**

| Model | Semantic | Top-1 at prompt end |
|---|---:|---:|
| DeepSeek-R1 7B | 0.027 | 0.99976 |
| Qwen 4B | 0.050 | 0.970 |
| Qwen 9B | 0.057 | 0.988 |
| SmolLM3 3B | 0.080 | 0.99983 |
| Phi-4 reasoning+ | **0.160** | **0.99999996** |

- Phi-4 also: JS divergence at prompt end ≈ **1.4e-9** (bulk distribution essentially unchanged). `<think>` tag frequently fails to close — repetition loop eats context.

**Background — thinking-off deltas (Qwen, default scaffold vs `enable_thinking=False`)**

- 4B: 0.050 → 0.067 (scaffold helps ~25%).
- 9B: 0.057 → 0.072 (scaffold helps ~20%).
- 2B: 0.075 → 0.072 (wash).
- 0.8B: 0.103 → 0.079 (scaffold *hurts*).

Interpretation: the stabilizing effect of the scaffold is size/recipe dependent, not universal. Small-model scaffolds look noisy — the preamble itself wobbles.

**Background — why Phi-4 is the counterexample you'll remember**

- Visible `<think>` scaffold — should, by the 64-token story, look stable.
- Top-1 at prompt end 0.99999996 — model is maximally certain.
- JS 1.4e-9 — bulk distribution didn't move.
- 512-token semantic 0.160 — second-most brittle in the panel, above GPT-2 XL (0.144).
- Mechanism hypothesis: `<think>` never closes, the model enters a repetition loop, and low-margin boundary decisions inside the loop diverge hard. Confident logits, unstable trajectory.

---

**If asked: "What's a scaffold?"**
> "A deterministic preamble the model emits before actually answering. DeepSeek-R1, Phi-4 reasoning+, and Qwen 4B/9B wrap their answers in a `<think>` block. SmolLM3 literally prints 'Thinking Process:'. Because the preamble is so consistent, two outputs on different prompts share a long identical prefix, and sentence-embedding similarity over the short window mostly scores the preamble."

**If asked: "Why does Phi-4 look brittle if it has the scaffold?"**
> "Because a scaffold doesn't guarantee stability — it just buys you a similar-looking prefix. Phi-4's `<think>` tag frequently never closes, and the model falls into a repetition loop. Inside that loop, low-margin boundary decisions diverge hard. So at 512 tokens its semantic divergence is 0.160, second-most brittle in the panel, even though its prompt-end logits are the most confident I measured."

**If asked: "Doesn't a high top-1 probability mean the model is confident?"**
> "Confident at the *prompt-end*, yes — one token. Phi-4 is at 0.99999996, the most confident model in the panel. But confidence on the very next token says almost nothing about what the *trajectory* does over the next 500 tokens. The bulk distribution hasn't moved — JS is 1.4 times 10 to the minus 9 — and the output still diverges at 0.160. Confidence ≠ trajectory stability."

**If asked: "Is thinking-on good or bad?"**
> "It's mixed. For Qwen 4B and 9B, thinking-on reduces divergence by about 20 to 25%. For Qwen 0.8B, thinking-on actually *increases* divergence — the scaffold wobbles more than the no-scaffold baseline. So 'reasoning models are more stable' doesn't hold up as a law. It's size- and recipe-dependent."

**If asked: "What about Qwen 4B vs 9B?"**
> "They're 0.050 and 0.057 at 512 tokens. At this sample size, indistinguishable — I wouldn't call one more stable than the other. p=0.78 between them. Both in the stable cluster, though."

**If asked: "Is DeepSeek-R1 just winning because of the scaffold?"**
> "Partly, yes. At 64 tokens it looks overwhelmingly stable because the `<think>` preamble is long and near-identical across prompts. At 512 tokens it's still the most stable in the panel at 0.027, so some of it is real — but I wouldn't claim DeepSeek is four times more stable than everything else. A chunk of that gap is scaffold credit."

**If asked: "So should I not use sentence embeddings?"**
> "Use them, but don't use them alone and don't use them on short outputs where a scaffold can dominate. I pair them with longer continuations, logit-level metrics, and token-path diagnostics. No single measurement is safe."

---

**One-liners to land:**

> "Scaffold does not equal stable."

> "Phi-4 is the cleanest dissociation I have: certainty at the prompt end, brittleness over 500 tokens."

> "Short outputs of scaffolded models mostly measure the scaffold."

*[Remember: Phi-4 numbers — top-1 0.99999996, JS 1.4e-9, 512-token 0.160. `<think>` never closes, repetition loop. Thinking-off deltas: 4B 0.050→0.067, 9B 0.057→0.072, 2B wash, 0.8B 0.103→0.079. If you blank on Phi-4's top-1, say "effectively one — the model is maximally certain" and move on.]*

---

## Slide 16, Era, recipe, and the LLaMA-1 surprise

**SAY:**
- "Cross-family view. 512-token semantic, scaffold-free models only."
- "**LLaMA-1 7B: 0.053.** A 2023 base model. It's the stable outlier."
- "Gemma E2B instruct 0.056. Mistral 7B v0.3 0.068. Gemma E4B instruct 0.072."
- "Gemma E4B base jumps to 0.119. Gemma E2B base 0.199."
- "Legacy base, GPT-2 XL, OPT, Pythia, GPT-J, all in the 0.14 to 0.22 range."
- "Two takeaways. One: LLaMA-1 is content-stable *without* a scaffold. Surprising."
- "Two: within Gemma, instruct is much more stable than base. **Recipe over calendar.**"
- "And token-edit distance gives you a *different* ordering than semantic. Metrics disagree."
- "Era doesn't predict sensitivity. Stability isn't one scalar."

*[Remember: LLaMA-1 could be tokenizer, pretraining corpus, or community-conversion artifact. Treat as a flag, not a law. E2B and E4B base swap order between 64-tok and 512-tok panels, this is why I report clusters, not ranks.]*

---

## Slide 17, Stability and responsiveness split

**SAY — short (life raft):**
- "Most important principled point in the talk. Stability and responsiveness are *different things*."
- "'The the the' forever is extremely stable. A model collapsed onto one fixed answer is stable. **Neither is what we want.**"
- "Qwen 0.8B quant sweep: at 4-bit, perturbation divergence dropped **0.138 → 0.091**. Sounds 'more stable.'"
- "But drift from BF16 on **identical prompts**: **0.132**. Huge."
- "The model moved a large distance from its own baseline. That's collapse onto a narrower manifold, not robustness."
- "**Fix: pair perturbation distance with drift from baseline. Both axes.**"

**SAY — long (verbatim, ~80 seconds):**

> "This is the most important principled point in the talk. If you walk away with one idea about how to measure stability, make it this one: stability and responsiveness are different things, and a one-axis metric can't tell them apart.
>
> Thought experiment. Imagine a model that outputs 'the the the' forever, regardless of prompt. It's *extremely* stable under any perturbation probe — same tokens no matter what you give it. A model that's collapsed onto one canned answer is also extremely stable. Neither is what we want. Stability is a property, not a score.
>
> Here's the data. I quant-swept Qwen 0.8B through BF16, 8-bit, and 4-bit. At 4-bit, perturbation divergence dropped from **0.138 to 0.091**. If that's the only number you look at, you conclude 4-bit is more stable. Great, ship it.
>
> Then I checked the second axis — drift from the BF16 baseline, on the *identical* prompts. That drift is **0.132**. Huge. The 4-bit model has moved a large distance from its own BF16 reference output.
>
> Put those together. On identical prompts it's already 0.132 away from baseline. On perturbed prompts it only moves another 0.091. That's not robustness — that's a model that's collapsed onto a narrower output manifold. It's less *responsive*, not more stable.
>
> The fix is operational: whenever you score stability, pair perturbation distance with drift from baseline. Both axes. If a model has low perturbation distance *and* low baseline drift, it's genuinely stable. If it has low perturbation distance but high baseline drift, that's collapse wearing a stability costume."

---

**Background — the two axes**

- **Perturbation distance:** how much the output moves when the *prompt* changes slightly. What the rest of the talk measures.
- **Baseline drift:** how much the output moves when *nothing about the prompt changes* but the *system* (quantization, precision, kernel, framework version) changes. Computed by running the reference system and the candidate system on identical prompts and comparing.
- A healthy model has low on *both*. Collapse has low perturbation + high drift. Responsiveness has higher perturbation + low drift (responding correctly to meaningful input changes).

**Background — the Qwen 0.8B quant numbers**

- BF16 baseline → 8-bit → 4-bit sweep.
- **Perturbation divergence:** 0.138 (BF16) → ... → 0.091 (4-bit). Looks like quantization buys stability.
- **Drift from BF16 on identical prompts at 4-bit:** 0.132. Comparable in magnitude to the perturbation signal itself. The 4-bit model is simply not the same model as BF16 anymore.
- Sample size: n=9 prompt pairs per cell. Within-system perturbation flip is p=0.19. Treat as an existence example, not a quantization conclusion.

**Background — why this isn't a quantization dunk**

- The point is *principled*, not empirical. Quantization isn't necessarily bad — sometimes it barely moves outputs.
- The point is: any one-axis stability metric will confuse collapse with robustness. Need the second axis to tell them apart.
- The Qwen 0.8B case is the clearest *example* of the confound, not a verdict on 4-bit quantization broadly.

**Background — what "collapse" looks like operationally**

- Outputs become more similar across *different* prompts (lower perturbation distance).
- Outputs become more different from the reference system's outputs on the *same* prompts (higher baseline drift).
- In extreme cases, outputs stop responding meaningfully to input at all — canned responses, repetition, generic filler.

---

**If asked: "If 4-bit is 'more stable,' isn't that good?"**
> "Only if it's still responding correctly to the prompt. 'The the the' forever is extremely stable — so is a model that says 'I cannot help with that' to everything. Neither is useful. In the Qwen 0.8B case, 4-bit has lower perturbation distance (0.091) but it's also moved 0.132 from its own BF16 baseline on identical prompts. So it's less *responsive*, not more stable."

**If asked: "What's drift from baseline?"**
> "Run the reference system — say BF16 — and the candidate system — 4-bit — on exactly the same prompts. Compare their outputs. If they diverge, the candidate has drifted from baseline even though the prompt didn't change. That drift tells you how much the model has moved as a function — independent of prompt perturbation."

**If asked: "Is this just quantization bad?"**
> "No, and I want to be careful about that. The sample size here is n=9, and the within-system flip is p=0.19 — not significant. I'm using this as an existence example of the *confound*, not as a conclusion about 4-bit. The point is that any one-axis stability metric can confuse collapse with robustness. That's the principle. Qwen 0.8B is just where I caught it."

**If asked: "How do I know a stable model isn't collapsed?"**
> "Check the second axis. Run the model on a reference set of prompts. Does it give sensible, differentiated answers? Are outputs meaningfully different across meaningfully different prompts? Does it drift from a baseline you trust? A genuinely stable model is stable *and* responsive. A collapsed one is stable *because* it's not responsive."

**If asked: "What's the fix?"**
> "Operational version: pair perturbation distance with drift from baseline. Report both. A model with low perturbation *and* low baseline drift is genuinely stable. Low perturbation with high baseline drift is the collapse signature. And sanity-check that the model is still meaningfully differentiating across prompts — that's the responsiveness check the collapse signature catches."

**If asked: "So we shouldn't quantize?"**
> "Not my claim. Quantize freely — but measure what you're buying. Lower divergence under perturbation is not automatically a win. Pair it with drift from the unquantized baseline and with a responsiveness check. Some quants will come through clean; some will collapse. You need the second axis to tell."

---

**One-liners to land:**

> "Stability is a property, not a score."

> "'The the the' forever is extremely stable. That's not what we want."

> "Low perturbation distance with high baseline drift is collapse wearing a stability costume."

*[Remember: n=9, p=0.19 on the within-system flip. Existence example of the confound, not a quantization conclusion. Sell the *principle*, not the numbers. If pressed on the data, retreat to: "Small n, exploratory — use as an example of the trap, not a verdict on 4-bit."]*

---

## Slide 18, Measuring is the hard part

**SAY:**
- "Three traps a naive stability probe will fall into."
- "**Collapse.** Degenerate model scores stable because outputs stop responding. Qwen 0.8B 4-bit. Caught by distance-from-baseline on identical prompts."
- "**Scaffold.** Short-output score dominated by deterministic preamble. Qwen 4B and 9B, SmolLM3. Caught by longer continuations or stripping the scaffold."
- "**Confident isn't stable.** Phi-4 has top-1 probability 0.99999996 at prompt end, JS divergence 1.4 times 10 to the minus 9. Bulk distribution hasn't moved. But 512-token output is 0.160."
- "Sharp logits do not mean a stable trajectory."
- "Honestly, the useful part here is **naming these failure modes before the field starts quoting numbers**."

*[Remember: mature version is multi-scale. Short vs long, logit vs text. No single measurement is safe alone.]*

---

## Slide 19, Long-generation trajectories

**SAY:**
- "Qwen thinking-off control. 24 small-perturbation prompt pairs."
- "Left panel: token-prefix divergence over time. Right panel: same runs scored by 512-token semantic distance."
- "Token paths split and saturate fast. Semantic distance tracks differently."
- "The semantic bars are close, 4B, 9B, 2B, 0.8B are not cleanly separable under this direct-answer control."
- "So treat trajectory shape as a diagnostic, not a leaderboard."
- "Token paths, semantic distance, scaffold behavior, logit boundaries, all seeing different parts of the same system."

*[Remember: prompt-pair curves, not independent samples. Don't sell as a size law. This is why I want multi-scale measurement.]*

---

## Slide 20, Mechanism: boundary beats bulk

**SAY — short (life raft):**
- "If you take one thing from the measurement side of this talk, take this."
- "Mechanism: small prompt change → argmax crosses a **low-margin boundary** → different first token → autoregressive feedback → different trajectory."
- "The whole next-token distribution often barely moves in bulk. The fragile point is the decision boundary."
- "**Boundary beats bulk.**"
- "Phi-4 case: top-1 prob **0.99999996**, JS **1.4e-9** (distribution hasn't moved at all), yet 512-token divergence **0.160** — higher than GPT-2 XL."
- "Prompt-end confidence says nothing about trajectory stability on its own."

**SAY — long (verbatim, ~85 seconds):**

> "If you take one thing from the measurement side of this talk, take this. This is the single most important conceptual claim I have.
>
> Mechanism: small prompt change causes the argmax to cross a **low-margin decision boundary** at some position. That flips the first token. Because generation is autoregressive, the now-different first token becomes the context for the second, and the second for the third, and so on. The model steers into a different basin. One boundary crossing, amplified by feedback, produces a totally different trajectory.
>
> Here's the subtle part. The whole next-token distribution *often barely moves in bulk*. The vast majority of probability mass stays where it was. What matters isn't the bulk — it's whether the *boundary* between the top two tokens got crossed. Boundary beats bulk.
>
> Phi-4 is the extreme case, and this is the example I want you to remember. At the prompt end, Phi-4's top-1 probability is 0.99999996. The model is effectively certain. JS divergence on the next-token distribution, comparing the original prompt to the perturbed prompt, is 1.4 times 10 to the minus 9. The distribution didn't move at all. And yet the 512-token output diverges at 0.160, higher than GPT-2 XL's 0.144.
>
> So: maximally confident logits, bulk distribution essentially frozen, trajectory still ends up somewhere completely different. That only makes sense if some decision *downstream* of the prompt end happened at low margin, flipped, and the autoregression did the rest.
>
> Practical implication: prompt-end confidence tells you almost nothing about trajectory stability. If you want to catch fragility, you need to look at where the decision boundaries are thin across the *whole* generation, not just what the distribution looks like at position one."

---

**Background — boundary vs. bulk, precisely**

- **Bulk shift:** JS or KL divergence between the full next-token distributions before and after prompt perturbation. Captures how much the whole distribution moved.
- **Boundary margin:** gap between top-1 and top-2 logits at a given decoding step. Small margin = fragile argmax.
- A distribution can have near-zero bulk shift and still have a razor-thin boundary between top-1 and top-2. A small logit perturbation crosses it, the argmax flips, the output diverges.
- Phi-4's prompt end: bulk is frozen (JS 1.4e-9), and the immediate top-1 has enormous margin (prob 0.99999996). So the boundary crossing that causes divergence isn't at the prompt end — it happens somewhere inside the generation, at some later low-margin position.

**Background — the autoregressive amplifier**

- Once one token flips, every subsequent forward pass runs on a different context.
- Even if each individual next-token decision is high-margin, the *compounded* effect of different contexts produces large output divergence.
- This is the "small δ in initial condition, large δ later" pattern from classical sensitive dependence — but discrete, and driven by boundary crossings rather than continuous stretching.

**Background — why prompt-end metrics miss it**

- Naive stability probe: measure JS between distributions at prompt end.
- This catches *bulk shift* at position 1 only. It misses:
  - Low-margin decisions at positions 2, 3, ... N.
  - Boundary crossings that happen deep in the generation.
  - Autoregressive amplification of small early flips.
- Multi-scale measurement (short+long outputs, logit+text metrics) catches the boundary cases the prompt-end JS misses.

**Background — Phi-4's dissociation, fully laid out**

| Metric | Phi-4 value | What it means |
|---|---|---|
| Top-1 prob at prompt end | 0.99999996 | Maximally certain about next token |
| JS divergence at prompt end | 1.4e-9 | Bulk distribution essentially unchanged |
| 512-token semantic divergence | 0.160 | Second-most brittle in the panel |
| Compare: GPT-2 XL at 512 tokens | 0.144 | Phi-4 is *more* brittle than a 2019 base model |

This is a clean counterexample to "sharp logits mean stable output."

---

**If asked: "What does 'boundary beats bulk' mean?"**
> "Bulk is the whole next-token distribution — how much probability mass shifted when the prompt changed. Boundary is the margin between the top-1 and top-2 logits — how close the argmax decision is. The distribution can barely move in bulk, and still have a razor-thin boundary. A small perturbation crosses the boundary, flips the argmax, and the output diverges. The fragile thing is the boundary, not the bulk."

**If asked: "Is the distribution moving or not?"**
> "At the prompt end, often not much. Phi-4's JS is 1.4e-9 — that's the bulk distribution essentially frozen. The distribution *is* moving somewhere downstream in the generation, at some low-margin position, but it doesn't have to move at position one for the output to diverge. Autoregression amplifies a single flip."

**If asked: "What's a low-margin decision?"**
> "Any decoding step where the top-1 and top-2 logits are close. The model is effectively coin-flipping between two tokens. A tiny perturbation anywhere in the prompt or context can nudge that decision either way. Once one flips, the generation is running on a different prefix from there forward."

**If asked: "If the distribution barely moves, how does the output change so much?"**
> "One flipped token changes the entire downstream context. The second forward pass now runs on a different prefix, and so does the third, and the fourth. Even if every individual decision is high-confidence, the compounded effect of different contexts produces a completely different output. That's the autoregressive amplifier."

**If asked: "What's the Phi-4 case here?"**
> "Phi-4 has the most confident prompt-end distribution in the panel — top-1 probability 0.99999996. Its JS divergence at prompt end is 1.4 times 10 to the minus 9, meaning the bulk distribution is frozen under perturbation. And yet at 512 tokens the output diverges at 0.160 — more brittle than GPT-2 XL. That's the cleanest single counterexample to 'sharp logits mean stable trajectory.'"

**If asked: "Does this mean we should look at logits?"**
> "Logits are *part* of the answer but not sufficient alone. Prompt-end logits miss boundary crossings that happen deeper in the generation. You want multi-scale: prompt-end logits, longer-window text divergence, maybe per-step margin tracking. No single metric sees the whole picture. The mature version is multi-scale measurement."

**If asked: "Why not just look at per-step logits during generation?"**
> "You can, and it's a reasonable research direction. The issue is cost — you'd need to trace per-step margins across many prompt pairs and many models. My text-level probe is a cheap downstream shadow. Per-step logit analysis would be the more rigorous follow-up. I'd take it."

**If asked: "Is this a proof of anything?"**
> "No. It's a mechanism story consistent with the data. The Phi-4 dissociation — certain logits, frozen bulk, divergent trajectory — is hard to explain *without* something like low-margin boundary crossings downstream of the prompt end. But I haven't directly traced the boundary crossing. That's the obvious next experiment."

---

**One-liners to land:**

> "Boundary beats bulk."

> "Prompt-end confidence says nothing about trajectory stability."

> "The whole distribution barely moves. A thin boundary gets crossed. Autoregression does the rest."

> "Phi-4: maximally certain, bulk frozen, trajectory diverges. That's the cleanest dissociation I have."

*[Remember: this is the single most important conceptual claim in the talk. Slow down and land it. Phi-4 numbers: top-1 0.99999996, JS 1.4e-9, 512-token 0.160 vs GPT-2 XL 0.144. If pressed on "proof," retreat to "mechanism story consistent with the data, direct per-step margin tracing is the follow-up."]*

---

## Slide 21, A question the lens suggests

**SAY:**
- "A question the lens suggests. I don't have the answer."
- "We know compression has a **static floor**, how few bits to store the model. Well-characterized. TurboQuant, KIVI, rate-distortion bounds."
- "The open question: is there a **dynamical floor**, how few bits before *behavior* drifts?"
- "It might depend on model sensitivity. Stable models might tolerate more compression. Sensitive ones might not."
- "My own data has the Qwen 0.8B collapse case, which would naively falsify that."
- "So it's open. I lean yes, but I can't defend it. Someone else should chase it."

*[Remember: softened from "two floors conjecture" to "question the lens suggests." Don't die on this hill in Q&A.]*

---

## Slide 22, The practitioner upshot

**SAY:**
- "Practical takeaways."
- "**Don't evaluate on a single prompt, single decode, or single metric.**"
- "Reliability: test prompt **neighborhoods** around the canonical prompt."
- "Model comparison: report sensitivity **ranges** over equivalent prompts."
- "Output metrics: strip boilerplate, compare answer spans, watch prefixes."
- "Decoding: deterministic for the sensitivity signal, sampling separately for deployment."
- "Quantization: lower divergence doesn't mean robust. Also check drift from baseline."
- "The chaos lens gives us questions that benchmarks rarely ask, and those questions are worth asking."
- "If you remember one line: **prompting is operating a high-gain branching system. Test neighborhoods, not single prompts.**"

*[Remember: land practical, then stop. Don't add. Take the pause.]*

---

## Slide 23, Questions?

**SAY if asked:**
- "*Is this chaos?*" → "No theorem. Finite-time input sensitivity in a hybrid discrete/continuous system. Chaos is the teaching lens."
- "*Isn't this just temperature?*" → "No. The probe is argmax decode. Sampling noise is removed."
- "*Did templates or tokenizers cause this?*" → "They're part of the system, and sometimes part of the confound. The main ladder controls for templates where possible. Tokenizer differences are a real limitation."
- "*Is sentence-embedding distance valid?*" → "It's a proxy. That's why the deck also shows token trajectories, raw examples, logits, scaffold checks, and baseline drift."
- "*Are older models more stable?*" → "No. LLaMA-1 is an interesting outlier. Not proof of an era law."
- "*Do scaffolds cause stability?*" → "Not cleanly. Thinking-off results are mixed, not monotonic."
- "*What should a team do with this?*" → "Run deterministic neighborhood probes on their real prompt distribution, and check multiple failure modes before changing prompts or models."
- "*Larger equals more stable?*" → "Within Qwen, 0.8B is clearly more sensitive than 4B. But 4B vs 9B is a wash. I don't have a size law."

**Recovery lines:**
- "The existence is obvious. The magnitude differences and the confounds are the measurement problem."
- "This is already temperature zero."
- "Only stable if the model still responds correctly to meaningful input changes."
- "Agreed, that's why the talk is about metric triangulation rather than one score."

*[Remember: repeat every question back. Buys time, and helps everyone on Teams hear it. Then answer.]*

---

## Slide 24, Backup: "Would I get the same answer if I ran it?"

**SAY:**
- "Argmax decode has no sampling step. do_sample=False picks the highest-logit token each step. Seed is inert."
- "Same prompt twice → byte-identical output. Prompt A vs B → a top-token flips somewhere and the outputs diverge."
- "I also ran a temperature 0.1 control, 30 samples per prompt on OLMo-3, palindrome pair."
- "Prompt A's samples cluster tightly. Prompt B's cluster tightly. And the two clusters are visibly separate."
- "Two attractors, not one unlucky draw. Sampling noise is smaller than the shift between clusters."

*[Remember: precision caveat, GPU kernels have tiny nondeterminism, but it's not the source of the big divergences.]*

---

## Slide 25, Backup: "Is this chaos?" defense

**SAY:**
- "Formal chaos needs exponential divergence under iteration, infinitesimal perturbations, asymptotic time."
- "LLM generation is finite depth, finite length, tokenized."
- "So this is *not* a proof of chaos."
- "What I measured: small input perturbations producing different outputs, varying by model, reproducible under deterministic decode."
- "Consistent with behavior near a chaos boundary. Not a proof of chaos."
- "The frame is the contribution. The experiment is a probe, not a theorem."

*[Remember: only use if someone presses hard. Don't volunteer this.]*

---

## Slide 26, Backup: Related work I came across late

**SAY if asked for citations:**
- "Salinas and Morstatter 2024, 'Butterfly Effect of Altering Prompts.' They published something close to my whitespace example."
- "Sclar et al. 2023, formatting sensitivity, up to 76-point swings on LLaMA-2-13B."
- "Lu et al. 2021, example ordering alone moves few-shot from near-random to near-SOTA."
- "PromptRobust, POSIX, RobustAlpacaEval, published sensitivity benchmarks."
- "Dynamical-systems side: Poole 2016, Schoenholz 2017, edge-of-chaos signal propagation. Geshkovski 2023, attention as particle dynamics. Tomihari and Karakida 2025, Jacobian-Lyapunov in self-attention."

*[Remember: don't oversell any single one as proving your result. Frame honestly as "found late, complementary."]*

---

## Slide 27, Backup: Statistical honesty

**SAY if asked about robustness:**
- "n is small, 9 prompt pairs per model in the panel, 24 in the hardened Qwen wave."
- "Robust at n=24: Qwen 4B vs 0.8B, p less than 0.001. 4B vs 2B, p equals 0.012. Cluster-level differences."
- "Weak at this n: 4B vs 9B, p equals 0.78. Middle-pack ordering. Standalone quantization flip."
- "Scaffold vs non-scaffold is confounded with post-training recipe, needs different *models*, not more prompts."
- "Specific model orderings are underclaimed here. The broad clusters would need a much larger prompt set to overturn."

*[Remember: "This is a measurement proposal and a teaching talk, not a benchmark paper."]*

---

## Slide 28, Backup: Failed experiments

**SAY if asked:**
- "A few models didn't run."
- "gpt-oss-20b hit an MXFP4 / Triton driver mismatch on the SageMaker image."
- "Nemotron Nano 9B v2, container lacked mamba-ssm."
- "Phi-4 mini, Transformers version / custom-code import failure."
- "These are tooling misses rather than stability findings."

*[Remember: don't let this turn into a debugging retrospective.]*

---

## Slide 29, Backup: Full bootstrap readout (512 tokens)

**SAY if asked for numbers:**
- "Stable cluster: DeepSeek-R1 Qwen 7B 0.027. Qwen 4B 0.050. LLaMA-1 7B 0.053. Gemma E2B instruct 0.056. Qwen 9B 0.057. Mistral 7B 0.068. Gemma E4B instruct 0.072. Qwen 2B 0.075."
- "Higher sensitivity: OLMo 2 7B 0.088. Qwen 0.8B 0.103. OLMo 3 7B 0.104. Gemma E4B base 0.119. GPT-2 XL 0.144. **Phi-4 reasoning+ 0.161.** Gemma E2B base 0.199."
- "n equals 24 pairs at 512 tokens."
- "Phi-4 is scaffolded *and* brittle. Scaffold does not mean stable."
- "Cluster view."

*[Remember: do NOT present this as a leaderboard. If pressed, give a cluster and stop.]*

---

## Glossary (if you blank mid-word)

- **Argmax decode**, pick highest-logit token each step. Deterministic.
- **Attractor**, region of state space trajectories settle into.
- **Bootstrap CI**, resample prompt pairs for uncertainty.
- **Collapse**, model stops responding to input; looks stable.
- **Embedding distance**, semantic proxy from sentence embeddings.
- **JS divergence**, symmetric bounded distribution distance.
- **Levenshtein**, edit distance; insertions/deletions/substitutions.
- **Lyapunov exponent**, rate nearby trajectories separate.
- **QLE**, quasi-Lyapunov; finite-depth analog (Li et al.).
- **Scaffold**, deterministic preamble like `<think>` or "Thinking Process:".
- **Sensitivity**, output change per unit input change.
- **Temperature**, logit rescaling before sampling.

---

## If you get totally lost

1. Pause. Sip water. (Teams hides the pause better than a room does.)
2. Say: **"Let me get back to the spine."**
3. Say the thesis verbatim:
   > "LLMs are hybrid sequential systems, continuous activations feeding a discrete branching process. Small input changes can flip branches. Measuring that has specific traps."
4. Ask: "Where was I?"
5. Move on. Nobody remembers the stumble.
