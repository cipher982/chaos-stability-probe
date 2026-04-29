# Speaker Notes

Background, likely questions, rebuttals, glossary. Keep open on a second monitor or phone. Audience never sees this.

## Opening mindset

**What you are claiming (say this tone out loud in your head before slide 1):**

- LLMs have a measurable property called *input sensitivity*. It is not sampling randomness.
- I built a probe for it, ran it across 18 models, and found the naive version of the probe has two failure modes that will fool people. Naming those failure modes is the contribution.
- Chaos is the *lens*, not the *theorem*. Never say "we proved LLMs are chaotic."

**What you must not claim:**
- "I measured the Lyapunov exponent of [any LLM]."
- "Bigger is more stable" (9B ≈ 4B in your data).
- "Modern post-training made models stable" (LLaMA-1 is the counterexample).
- "Older models are more stable" (token-edit and semantic metrics split).
- "Quantization makes models more robust" (that was the collapse confound).

**Single-sentence elevator pitch:** *Chaos gives us vocabulary for when small input changes get absorbed vs. amplified in LLMs — and the obvious way to measure that has two confounds the field needs to name before quoting numbers.*


## Slide 1 — Title. "Nearby Prompts, Distant Trajectories"

**Background:** The Lorenz attractor on the slide is the canonical 3-variable dynamical system (Lorenz 1963, "Deterministic Nonperiodic Flow") whose trajectories stay bounded but never repeat — the archetype of deterministic chaos. Shown here to signal the frame before saying a word.

**Likely opening question:** *"Chaos theory and LLMs — is this metaphor or math?"*
**Answer:** Both. The math is real (Li et al. 2025 compute quasi-Lyapunov exponents inside Qwen2-14B). My experiment is the output-level shadow of that internal dynamics, in a deliberately discrete setting where the formal math doesn't quite apply.

**Delivery:** 60 seconds. Don't mention compression. Tease the "two ways to mis-measure" line — that's what should make them lean forward.


## Slide 2 — "Same input. Same weights. Different output."

**Background:** The outputs shown are real argmax-decoded generations from OLMo-3 7B (allenai/OLMo-3-7B-Instruct). The only difference between prompt A and prompt B is a trailing space character. Under `do_sample=False` there is no stochasticity — the generation is fully deterministic given the input tokens.

**Likely question:** *"Is OLMo just broken?"*
**Answer:** No — this is an existence proof that happens on many models. SmolLM3 flipped its `<think>` mode on a leading newline. Falcon-3 changed its second reasoning step after a leading newline. OLMo is just the sharpest, most teachable example.

**Likely pushback:** *"Different tokens → different outputs, that's how transformers work."*
**Answer:** Yes at the token level. The interesting claim is about *magnitude* — some models move a lot, some barely move, and the variation is not predictable from size or era.

**Delivery:** Read A and B aloud if the room didn't catch the style difference.


## Slide 3 — "This isn't just temperature."

**Background — critical distinction:**
- **Temperature `T`:** scales logits before softmax: `softmax(logits / T)`. `T=0` is argmax (deterministic). `T>0` samples.
- **Sensitivity:** how much the *pre-sampling* distribution moves when the input moves.

These are orthogonal. You can have high temperature + low sensitivity (noisy draws from a stable distribution), or zero temperature + high sensitivity (deterministic, but the distribution itself moved).

**The "two clouds" rigorous version (backup if pushed):** Sample prompt A 50 times at T=0.7, sample prompt B 50 times at T=0.7. Embed all 100 outputs, project to 2D. If the clouds are disjoint, sampling noise cannot bridge the gap — the shift is larger than the sampling envelope. That's the airtight version of "the distribution moved."

**Likely question:** *"If I set temperature to zero always, doesn't sensitivity disappear?"*
**Answer:** No — sensitivity is a property of the distribution shape across inputs, not the sampling procedure. Argmax just makes it cleanly observable. At T>0 sensitivity is still present but gets mixed with sampling variance.

**One-liner to repeat:** *"Temperature samples from a distribution. Sensitivity asks how far the distribution moved."*


## Slide 4 — "This is what dynamical systems theory studies."

**Background — the minimum chaos primer:**
- **Chaos (Lorenz, Strogatz):** deterministic systems where small input differences grow over time, usually exponentially until saturation.
- **Lyapunov exponent `λ`:** the average exponential rate of separation. `|δ(t)| ≈ |δ(0)| · e^(λt)`. `λ>0` means chaotic.
- **Classical examples:** Lorenz system (weather), double pendulum, logistic map at `r > 3.57`.

**Key teaching move:** chaos is *deterministic*, not random. Same input → same output. But neighboring inputs → drastically different outputs after enough time.

**Weather analogy:** atmospheric forecasts aren't random — they're sensitive to initial conditions. A 0.1°C perturbation can lead to a qualitatively different forecast a week later. The system is deterministic; it just amplifies small differences.

**Likely question:** *"So are LLMs chaotic or not?"*
**Answer:** Formally, unclear. Classical chaos needs continuous state + iteration to infinity. LLMs have mixed continuous (activations) + discrete (tokens) state and finite depth/length. The *phenomenology* matches: deterministic, small-input amplification. Li et al. call their version "quasi-Lyapunov" for exactly this reason.

**Terms the audience may or may not know:**
- **Attractor:** a region of state space that trajectories are drawn to.
- **Bifurcation:** a qualitative change in system behavior as a parameter changes.
- **Strange attractor:** an attractor with fractal structure (Lorenz butterfly).


## Slide 5 — "Both outputs can be correct."

**Background — this slide is the answer to the "isn't this just regenerate?" objection.**

The trap: if you show correctness flips (A gets 3381, B gets 3281), the audience thinks "the model is buggy." If you show stance flips on a taste question, the audience thinks "both are fine, same model — that IS interesting."

**The double pendulum analogy is load-bearing:** no swing of a double pendulum is "wrong" — both obey physics, they just land in different places. Same bar for LLMs: both the Foundation and Hyperion recommendation are defensible book picks. The model did its job twice. It just picked a different basin.

**Why sampling doesn't explain this:** if you sample prompt A many times, you'll get variations of the Foundation recommendation (different phrasings, different justifications). You won't jump to Hyperion — that's in a different mode of the distribution. To reach Hyperion you need to perturb the *input*, not the *sampling*.

**Likely question:** *"How do you quantify a meaning-preserving change?"*
**Answer:** Two operational definitions. (1) Structured no-ops by construction: whitespace, punctuation, Oxford-comma, ASCII vs Unicode quote — the set of edits a human judge would call "same meaning." (2) Embedding distance threshold: input embedding cosine distance below some ε. The second is softer but measurable.

**Math bridge (for a dynamicist):** we're computing a finite-difference proxy for `‖∂output/∂input‖` where the denominator is forced small by the perturbation class.


## Slide 6 — "State, and prior work"

**Background — the Li et al. 2025 paper (arXiv:2503.13530):**

**What they actually did:**
- Paper title: *Cognitive Activation and Chaotic Dynamics in Large Language Models: A Quasi-Lyapunov Analysis of Reasoning Mechanisms.*
- Model: Qwen2-14B (40 layers, hidden dim 5120).
- Perturb an activation element at one layer (e.g. `h_{17,4074}` at layer 10), run forward, measure how the perturbation grows by later layers.
- Formula: `λ_{m,n} = (1/(n-m)) · ln(|δ_n| / |δ_m|)` in the limit `|δ_m| → 0`. This is Definition 2, Eq. 9-10 of the paper.

**Their findings:**
1. Hidden-state magnitude grows exponentially with depth: factor **1.32× per layer** for layers 0–9 (this is your "~1.3×" number), 1.08× for layers 10–38. Piecewise exponential fit.
2. Shallow layers contract perturbations (λ<0, convergent); deep layers amplify them (λ>0, divergent).
3. Decomposing the residual stream at output: **MLP 55.8%, attention 44.2%, initial input 0.0009%**. The input is a "minor perturbation" to the network by the time you reach the top.
4. Behavioral test on CMMLU (11,528 multiple-choice): zeroing the smallest 5% of activations drops accuracy >20%. Further 0.5% steps drop another 14%. "Sensitivity to initial conditions."

**What they explicitly didn't do:**
- Only one model (Qwen2-14B).
- Defined an *iterative* QLE (Definition 3) over token generation steps but never computed it.
- Never compared across models.
- No connection to quantization or compression.

**Why this matters for your talk:** you're doing the axis they *defined but never ran* — token-level sensitivity across 18 models. Genuinely complementary.

**Likely question:** *"How does your probe relate to the QLE?"*
**Answer:** Li et al. work in continuous activation space inside the forward pass; I work in discrete token space across the whole generation. Their math is cleaner because activations are continuous. My measurements are the downstream shadow of what they measure internally — and I can run it on models they couldn't get to.

**Vasić et al. 2026:** Lyapunov spectra of embedding trajectories for human speech — note the trap, this measures *human* speech trajectories embedded by an LLM, not the LLM's own generation dynamics.

**Edge-of-chaos (Langton 1990, Zhang et al. 2024 "Intelligence at the Edge of Chaos"):** trained neural networks tend to sit near the order/chaos boundary. Theoretical support for "this framing isn't a stretch."


## Slide 7 — "The experiment"

**Background — methodology details you might be asked:**

- **Model loading:** `AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.bfloat16)`. Pinned HF revisions.
- **Decoding:** `model.generate(..., do_sample=False, max_new_tokens=N)`. Argmax. Seed irrelevant.
- **Chat templates:** deliberately disabled for the main ladder to keep the input surface identical across models. Chat template is a separate axis.
- **Primary metric:** `1 - cos(embed(A), embed(B))` using `all-MiniLM-L6-v2` from sentence-transformers.
- **Secondary metrics:** token edit distance (Levenshtein on tokens), common prefix length, final-layer hidden-state cosine distance, logit JS/KL divergence at prompt-end.
- **Prompts:** 21 pairs spanning identical / no-op formatting / punctuation / synonym / paraphrase / small semantic / positive control. Hardened Qwen wave: 24 pairs.
- **Stats:** bootstrap 95% CIs (1000 resamples), paired permutation tests over prompts.

**Likely question:** *"Why sentence-embedding and not BLEU or ROUGE?"*
**Answer:** BLEU/ROUGE reward surface overlap. A model that paraphrases the same answer would score "different." Sentence-embedding distance is closer to "did the semantic content shift." Still a proxy — backed up with token edit distance and raw examples.

**Likely question:** *"Why disable chat templates?"*
**Answer:** Templates vary wildly across models (`<|im_start|>`, `[INST]`, no template at all). Keeping templates would mean measuring "model + tokenizer + template" as a bundle. I deliberately probe the raw model surface. Templates are a separate (important) axis.

**Control sanity check:** identical prompt + argmax → 0.000 divergence. Framework nondeterminism is not dominating the signal.


## Slide 8 — "Within-Qwen: first finding (with caveat)"

**Background — the numbers:**
- n=24 hardened wave. Paired permutation test over prompts (not over models).
- Means: 4B=0.034, 9B=0.037, 2B=0.073, 0.8B=0.089.
- Key p-values: 4B vs 9B = 0.78 (not distinguishable). 4B vs 2B = 0.012 (real). 4B vs 0.8B < 0.001 (real).

**CRITICAL caveat — confirmed today:** Qwen3 has a thinking-mode chat-template flag. `enable_thinking=True` is default on **all sizes including 0.6B and 1.7B**. Hard switch: `tokenizer.apply_chat_template(..., enable_thinking=False)` — produces clean output with no `<think>` tags. Per official HF model card.

So the right Slide 7 experiment is: rerun the Qwen ladder with thinking *off* on all sizes. You kicked this off on SageMaker.

**Likely question:** *"Why doesn't 9B beat 4B?"*
**Answer:** At this n, I can't distinguish them. There are two real possibilities: (1) the capacity-stability curve is flat past ~4B; (2) sample size is too small to detect a small difference. I'm not claiming 4B is better — only that both are in the same stable cluster and dramatically more stable than 0.8B.

**Likely question:** *"Could this be Qwen-specific?"*
**Answer:** Possibly. That's why Slide 9 looks across 18 models. Within-family is where causality is cleanest (same pretraining corpus, same architecture, only scale varies); cross-family tests whether the signature generalizes.


## Slide 9 — "The scaffold confound"

**Background:** Five models in the panel emit a visible reasoning scaffold (Qwen 4B/9B with `<think>` or `Thinking Process:`, Phi-4 reasoning+, DeepSeek-R1-Distill, SmolLM3). Sixteen don't.

**The arithmetic of the confound:** if two generations share 38 tokens of identical "Thinking Process:" boilerplate before diverging, the first ~38 tokens contribute zero to semantic distance. If the total is 100 tokens, that's 38% of the output score that was forced to zero by construction.

**What this doesn't prove:** that scaffolded models are content-*unstable* underneath. They might still be stable. We just can't tell from this metric.

**Fix:** strip the scaffold before scoring. Or extend generations to 512 tokens so scaffold dilutes. Or run the same models with thinking disabled (the Qwen experiment you're kicking off).

**Likely question:** *"So is the scaffold finding real or not?"*
**Answer:** The surface observation is real (4× mean gap, bootstrap CI [−0.140, −0.076]). The causal interpretation is uncertain — scaffold presence and modern post-training recipe are collinear in this panel. Need models that vary one without the other to untangle.

**Cross-check that rescues the Qwen 4B finding:** the *prompt-end logit wave* (Slide 13) also ranks 4B as stable — with no scaffold in the next-token distribution. So 4B isn't only scaffold-stable, there's some underlying distribution-level stability too.


## Slide 10 — "Concrete failure: no-op formatting"

**Background — what OLMo actually did:**
- Prompt A: `Write a concise Python function that checks whether a string is a palindrome.`
- Prompt B: same + trailing ` ` (one space).
- Decoder: `do_sample=False`, bf16, no chat template.

Output A: long docstring, normalization pipeline (lowercase, strip non-alphanumeric), two-pointer or character-filter implementation.
Output B: conversational ("Certainly! Here's..."), simpler `s == s[::-1]` body.

**Why this is *not* "OLMo is bad":** OLMo 3 7B is a legitimate modern research model (AllenAI). Every model in the panel has some version of this failure somewhere in the 21 prompt pairs — this one is just the sharpest, most legible example.

**Why the audience will remember this slide:** it's the only slide with actual Python code they can read. Use it. Read A and B aloud.

**Likely question:** *"Is this a chat-template problem?"*
**Answer:** Possibly. Chat templates can be sensitive to whitespace around role markers. But the effect shows up with templates disabled too. This is a property of the model's response function, not just its tokenizer hygiene.

**Likely question:** *"Why would a trailing space cause this?"*
**Honest answer:** I don't fully know mechanistically. Hypothesis: the trailing space changes which position is "last prompt token," which perturbs the final hidden state enough to cross a decision boundary in early token logits, which sets the generation into a different textual basin.


## Slide 11 — "Era, recipe, and the LLaMA-1 surprise"

**Background — the LLaMA-1 thing:**
- Model: `huggyllama/llama-7b` (community conversion of Meta's original 2023 weights).
- Score: 0.058 mean semantic distance.
- No scaffold. No chain-of-thought. Pre-RLHF base model by modern standards.

**Why this breaks the "post-RLHF made models stable" story:** LLaMA-1 predates modern chat post-training by ~2 years and it's still content-stable. Something else is doing the work.

**Candidate explanations (I don't know which):**
- Pretraining corpus composition (LLaMA-1 was trained on a mix that maybe hit a stability sweet spot).
- Model size matters more than recipe (but then GPT-J 6B, same size, is brittle).
- Tokenizer or vocabulary effects.
- The community conversion path happens to produce stable weights.

**Within-Gemma comparison:** Gemma E4B instruct 0.069 vs Gemma E4B base 0.202. Same weights modulo instruction tuning → 3× sensitivity difference. This is the cleanest "instruction tuning measurably helps" result you have.

**Follow-up on the older-model hunch:** token edit distance at `t=60` weakly
points toward modern/instruct models being more surface-divergent, but
512-token semantic distance points the other way. Do not say older models are
more stable. Say metric choice revealed a split: older/base models often wander
or echo templates at the token level, while modern/instruct models can preserve
answer semantics better.

**Likely question:** *"Is LLaMA-1 just an outlier artifact?"*
**Answer:** Possible. n=24 prompts, one model, community weights. But it's the one base model that's content-stable without a scaffold, which makes it the interesting data point — not the one you'd expect based on era.


## Slide 12 — "Quantization: the result that surprised me"

**Background:**
- BF16 → 8-bit → 4-bit using `bitsandbytes` via Transformers integration.
- 8-bit: `load_in_8bit=True`, LLM.int8() quantization.
- 4-bit: `load_in_4bit=True`, NF4 with double quantization.
- Same 24 prompts, argmax, same metric.

**The data:**
- Qwen 4B: BF16 0.013 → 8-bit 0.025 → 4-bit 0.026 (drifts slightly less stable).
- Qwen 0.8B: BF16 0.138 → 8-bit 0.110 → 4-bit 0.091 (looks *more* stable).

**Why "more stable" is wrong:** see next slide. Collapse confound.

**Likely question:** *"Why bitsandbytes and not GPTQ/AWQ/HQQ?"*
**Answer:** Single-backend for comparability. Different backends have different quantization dynamics that would muddy the comparison. bitsandbytes is the Transformers default and easiest to reproduce. Cross-backend is future work.

**Likely question:** *"Is n=9 enough for this?"*
**Answer:** The within-system 0.8B flip is p=0.19 — not significant as a standalone claim. The real evidence is the drift-from-BF16 number on the next slide, which is large and has a clear mechanical explanation.


## Slide 13 — "Stability isn't responsiveness"

**Background — this is the most load-bearing slide in the talk.**

**The argument:**
- A model that always says "the the the the" has 0.000 divergence under any perturbation. Perfectly stable. Useless.
- A collapsed model that snaps to one answer for every input is also 0.000 divergence. Also stable. Also useless.
- You want models that *respond differently to different inputs* (high responsiveness) but *respond similarly to small perturbations of the same input* (high stability).
- A single-axis stability metric cannot distinguish these.

**The fix:** pair perturbation divergence with *distance from baseline* (here: BF16 version of the same model on identical prompts). High perturbation stability + high baseline drift = collapse. High perturbation stability + low baseline drift = genuine robustness.

**Why this is the headline finding:** the field is about to start running stability probes at scale (capability benchmarks are saturating; stability is the natural next thing to score). If people report single-axis numbers naively, they'll mistake degeneration for robustness and the literature will be noisy for years. Saying "measure both axes" now is preemptive.

**Likely question:** *"How do you measure output diversity?"*
**Answer:** Two ways I've considered: (a) distance from baseline on identical prompts (implemented); (b) within-model output diversity across genuinely different prompts (not yet). Both catch different forms of collapse.


## Slide 14 — "Measuring is the hard part."

**Background — summary table of the two confounds:**

| Confound | Mechanism | Detection |
|---|---|---|
| Collapse | Outputs stop responding to input; degenerate distribution | Distance from baseline on identical prompts |
| Scaffold | Deterministic preamble dominates similarity metric | Scaffold stripping; longer continuations; logit-level check |

**If you had to say one thing about this whole talk:** this is it. If they forget everything else, leave them with "measuring stability naively gives wrong answers in two specific ways."

**Likely question:** *"Any other confounds you expect?"*
**Answer:** A few I'd want to rule out with more data:
- *Tokenizer confound:* if model A has a coarser tokenizer, perturbations land in different places token-wise. Hard to normalize.
- *Length confound:* longer outputs accumulate more divergence mechanically. I use fixed `max_new_tokens` but natural early-stopping varies.
- *Task confound:* some prompt types are inherently more constrained (arithmetic) vs open (creative writing). My ladder mixes both.


## Slide 15 — "Long-generation trajectories"

**Background — what the plot shows:**
- x-axis: token position in generation (0 to 512).
- y-axis: cumulative semantic distance between A and B generations up to position t.
- Three lines: Qwen 0.8B, Qwen 4B, DeepSeek-R1-Distill-Qwen-7B.

**Pattern:**
- Qwen 0.8B: branches early (~first 20 tokens), grows fast, saturates high.
- Qwen 4B: branches later, grows slower.
- DeepSeek-R1: long shared prefix (sometimes 138 tokens on a punctuation-only perturbation, mostly scaffold), then diverges, often semantically reconverges even when token paths differ.

**The dynamical-regime story:** these shapes are *different*. Not just different amplitudes — different *shapes*. Which is the signature of different underlying dynamics.

**Caveat:** I'm not fitting exponentials to these. They often look piecewise linear, which is consistent with a saturating process (branch → reach basin → stay in basin). Don't oversell the "exponential" framing.

**Likely question:** *"Is this the Lyapunov exponent?"*
**Answer:** No. A real Lyapunov estimate would need initial condition pairs arbitrarily close, averaged over a proper measure on the state space. These are specific prompt pairs in discrete token space. It's "the qualitative thing a Lyapunov exponent measures," not the quantity itself.


## Slide 16 — "From text to logits"

**Background — what's being measured:**
- At the *final prompt position* (before any generation), compute full-vocab next-token distribution `p_A` and `p_B`.
- **JS divergence:** `JS(p_A, p_B) = 0.5·KL(p_A || M) + 0.5·KL(p_B || M)` where `M = 0.5(p_A + p_B)`. Symmetric, bounded in [0, log 2].
- **Top-1 flip rate:** what fraction of prompt pairs have `argmax(p_A) ≠ argmax(p_B)` at the final position.

**Why this matters:** if the text-level ranking were an argmax-flip artifact (two nearly-tied logits flipping 50/50), the JS values would be small and the flip rate would explain most of the variance. Instead:
- Stable cluster (Qwen 9B, 4B, LLaMA-1): JS very small, flip rate 0%.
- Brittle cluster (Gemma E4B base, Qwen 0.8B, OLMo-3): JS 10×–1000× larger, flip rate 20%+.

**The distribution is actually moving**, not just tie-breaking.

**Likely question:** *"Why not full KL?"*
**Answer:** KL is asymmetric and blows up when `p_A` puts mass somewhere `p_B` doesn't. JS is bounded and symmetric — the right choice for "how different are these distributions" aggregated across prompts.


## Slide 17 — "The two floors"

**Background — the compression frame:**

**Static floor (well-studied):**
- *Rate-distortion theory:* Shannon's result on minimum bits needed to represent a signal within fixed distortion. Applied to LLM weights/KV cache.
- *TurboQuant (Google 2025):* compresses KV cache to ~3 bits with reported near-zero accuracy loss.
- *KIVI:* 2-bit KV cache quantization baseline.
- The question *"how few bits can store the model?"* is close to solved.

**Dynamical floor (conjecture):**
- Even if you store weights faithfully, quantization injects per-operation noise during inference.
- A contracting (λ<0) model absorbs that noise. A divergent (λ>0) model amplifies it.
- So a model's *compressibility at fixed behavioral quality* should depend on its dynamical regime.
- Corollary: stable models should tolerate more aggressive compression than sensitive ones — at equal downstream quality.

**This is a conjecture, not a result.** My own data is suggestive but contains the 0.8B collapse counterexample. The naive version of the claim fails. The more careful version (measured in responsiveness + drift together) is still open.

**Likely question:** *"Isn't this just saying quantization-aware training matters?"*
**Answer:** Related but different. QAT adapts the model to quantization noise at training time. My conjecture is about an inherent property of the trained model — some models are dynamically more tolerant of injected noise than others, independent of whether they've been QAT'd.


## Slide 18 — "Statistical honesty"

**Background — the stats details:**

- **Bootstrap:** resample prompt pairs with replacement 1000 times, recompute mean, take 2.5th and 97.5th percentiles for 95% CI.
- **Paired permutation test:** for each pair of models A, B, compute per-prompt difference `d_i = score_A(prompt_i) - score_B(prompt_i)`. Under null hypothesis, signs of `d_i` are random. Permute signs 10,000 times, count how often the permuted mean exceeds the observed mean.
- Paired (vs. unpaired) because the same prompts are scored on both models — controls for prompt-level variance.

**What's robust, what's not** (reread from the slide before Q&A):
- Robust: cluster membership; Qwen 4B vs 0.8B; Qwen 4B vs 2B; Gemma E4B-it vs Qwen 4B.
- Not robust at this n: 4B vs 9B ordering; 2B vs 0.8B ordering; middle-pack positions; standalone 0.8B quant flip.

**Likely question:** *"Why not report effect sizes?"*
**Answer:** Fair. Cohen's d or similar would be a cleaner presentation. I reported raw means + CIs because the absolute scale of the metric is what matters for the collapse argument.

**Likely question:** *"How many prompts would it take to make 4B vs 9B robust?"*
**Answer:** Rough back-of-envelope: observed difference ~0.003, pooled SD ~0.04, to reach p=0.05 I'd need n ≈ (1.96 · 0.04 / 0.003)² ≈ 680 prompt pairs. That's why I'm not claiming 4B beats 9B.


## Slide 19 — "What's solid, what's next"

**Background — the SageMaker-scale extensions:**

1. **Iterative QLE across models.** Li et al. defined the formula (Definition 3) but only on one model. Perturb the input embedding vector `X_0` by a small Gaussian of norm ε, run `m` generation steps, embed the outputs, measure `ln(|δ_m| / |δ_0|) / m`. Take ε small. Run across the 18-model panel. That's the *iterative* axis they never computed.

2. **Meaning-preserving sensitivity slope.** For each prompt pair, plot (input embedding distance, output semantic distance). Fit the slope near origin. That slope is the discrete-token local Lyapunov analogue. Defensible chaos-adjacent quantity.

3. **Activation-space perturbation probe.** Adapt Li et al.'s intra-network QLE to the full 18-model panel — so you connect the output shadow you measure to the activation-space chaos they measure. Formal machinery works here because activations are continuous.

4. **Scaffold-stripped numbers** from the Qwen-thinking-off rerun kicked off today.

**Likely question:** *"What would it take to publish this?"*
**Answer:** Larger prompt set (100s–1000s), pre-registered metrics, task-specific validators, multi-backend quantization controls, statistical analysis over prompt distributions. This is a learning-club talk and a measurement proposal, not a benchmark paper.


## Slide 20 — "Questions?"

**Most likely questions — quick-reference card:**

**"Is this really chaos?"**
→ Backup slide 2. Formal chaos needs continuous state + iteration to infinity. I'm measuring the output-level shadow. Li et al. do the formal version in activation space.

**"Isn't deterministic decode unrealistic?"**
→ It's the right first probe. Under sampling, sampling variance swamps the perturbation signal. Deterministic decode cleanly isolates the input sensitivity. Sampling is a second, separable variance axis.

**"Couldn't this all be chat templates / tokenizers?"**
→ Templates were disabled. Tokenizer differences remain as a possible confound — that's part of why I emphasize cluster membership over precise ordering.

**"Is the semantic metric meaningful?"**
→ It's a proxy. Backed up by token edit distance, logit JS at prompt end (Slide 16), and raw examples. No single metric is trusted alone.

**"Does this generalize to production prompts?"**
→ No claim of that. Teams should run this style of probe on their own distributions. I'm demonstrating a method, not releasing a benchmark.

**"Is lower divergence always better?"**
→ No. Task-dependent. Creative work wants diversity; operational systems want consistency. Sensitivity is a property, not a universal score.

**"Does this show bigger = more stable?"**
→ No. 4B ≈ 9B in my data. Within-family capacity matters up to a threshold, then plateaus. Cross-family, size doesn't predict sensitivity.

**"Any evidence that scaffolds *cause* stability?"**
→ Not yet. Scaffold presence is collinear with modern post-training recipe in this panel. The Qwen thinking-off rerun (SageMaker, today) will let me test this directly — same weights, scaffold on vs off.

**"What's the one thing I should take away?"**
→ Stability is a measurable property of LLMs. The obvious way to measure it has two failure modes (collapse, scaffold) that make brittle models look stable. Name the failure modes before you trust the numbers.


## Backup — Glossary

**Argmax decoding:** at each step, pick the token with the highest logit. Deterministic.

**Attractor:** a subset of state space that trajectories get drawn into over time. Fixed-point attractor (one point), limit cycle (periodic), strange attractor (fractal).

**Bifurcation:** qualitative change in behavior as a parameter crosses a threshold.

**Bootstrap CI:** resample with replacement, compute statistic, take percentiles. Nonparametric confidence intervals.

**Butterfly effect:** pop-culture name for sensitive dependence on initial conditions. Technically imprecise but useful as a hook.

**Chaos:** deterministic sensitive dependence. Small input changes → divergent outputs. Not randomness.

**Collapse (confound):** model outputs stop varying with input. Scores "stable" on a perturbation probe because there's no responsiveness.

**Edge of chaos:** boundary between ordered and chaotic regimes. Trained networks tend to sit near this boundary.

**Embedding distance:** cosine distance between sentence embeddings. Proxy for semantic similarity.

**JS divergence (Jensen-Shannon):** symmetric, bounded version of KL. Measures distance between two probability distributions.

**KL divergence:** asymmetric measure of how one distribution differs from another. `KL(p || q) = Σ p(x) log(p(x)/q(x))`.

**KV cache:** stored key-value tensors from attention during generation; enables fast autoregressive decoding.

**Lorenz system:** three coupled ODEs (Lorenz 1963) that produce a strange attractor. Archetype of chaos.

**Lyapunov exponent:** average exponential rate of trajectory separation. λ>0 = chaotic.

**MLP (feed-forward):** the two-linear-layers-with-nonlinearity block after attention in a transformer layer.

**Paired permutation test:** significance test that permutes the sign of paired differences. Controls for pair-level structure.

**Quasi-Lyapunov exponent (QLE):** Li et al. 2025's adaptation of Lyapunov concept for finite-depth neural networks. Not a full Lyapunov because of finite layers, heterogeneous per-layer functions, and high-dimensional state.

**Rate-distortion theory:** Shannon's framework for minimum bits to represent a signal at a given distortion level. Applied to model compression.

**Residual stream:** the accumulating hidden state in a transformer, where each layer adds its contribution via residual connections.

**Scaffold (confound):** deterministic preamble (e.g. `<think>...` reasoning block) that inflates common prefix length and suppresses semantic distance before answer content appears.

**Sensitivity:** how much the output distribution moves when the input moves. Distinct from sampling variance.

**Temperature:** logit rescaling factor before softmax. T=0 deterministic, T>0 stochastic.


## Backup — Paper citations cheat-sheet

- **Li et al. 2025** — arXiv:2503.13530 — *Cognitive Activation and Chaotic Dynamics in Large Language Models: A Quasi-Lyapunov Analysis of Reasoning Mechanisms.* Qwen2-14B, QLE, 1.32×/layer, MLP 55.8% vs attention 44.2%.
- **Tomihari & Karakida 2025** — arXiv:2505.19458 — *Recurrent Self-Attention Dynamics: An Energy-Agnostic Perspective from Jacobians.* Self-attention as dynamical system; normalization suppresses Lipschitz blow-up.
- **Wang et al. 2025** — arXiv:2502.15208 — *Unveiling Attractor Cycles in LLMs.* Iterated paraphrasing converges to 2-cycle attractors.
- **Zhang et al. 2024** — arXiv:2410.02536 — *Intelligence at the Edge of Chaos.* Trained networks sit near order/chaos boundary; optimal task performance at the edge.
- **Cohen et al. 2021** — arXiv:2103.00065 — *Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability.* Training dynamics operate near the `2/step-size` stability threshold.
- **Lorenz 1963** — *Deterministic Nonperiodic Flow.* Foundational chaos paper.
- **TurboQuant (Google 2025)** — ~3-bit KV compression with near-zero quality loss.
- **KIVI 2024** — arXiv:2402.02750 — 2-bit KV cache quantization baseline.


## Backup — If someone asks you to explain a Lyapunov exponent

**Minimum explanation:** imagine two copies of the same system, started at almost-identical initial conditions. Their trajectories diverge over time. If the divergence grows exponentially, the exponent in that growth is the Lyapunov exponent. Positive = chaotic. Zero = neutral (like planetary orbits). Negative = converging to an attractor.

**Formula:** `|δ(t)| ≈ |δ(0)| · e^(λt)`, so `λ = lim_{t→∞} lim_{|δ(0)|→0} (1/t) · ln(|δ(t)|/|δ(0)|)`.

**Why it's hard for LLMs:** the two limits don't cleanly exist. `t→∞` doesn't work because depth is finite and generation ends. `|δ(0)|→0` doesn't work because token space is discrete. That's why Li et al. say "quasi."


## Backup — If someone pushes on "why deterministic decode?"

**Short:** sampling variance at any realistic temperature swamps the perturbation signal. At T=0.7 on OLMo-3, within-prompt sampling distance is the same order of magnitude as between-prompt distance. Can't separate signal from noise.

**Medium:** deterministic decode is a control, not the production setting. Once you understand the deterministic floor (what the model *would* do without sampling), you can add sampling back as a separable variance component. Doing both simultaneously conflates them.

**Long (if asked for the rigorous version):** the deterministic output is the argmax of the output distribution. Two prompts with identical distributions have identical argmaxes. Two prompts with meaningfully different distributions often have different argmaxes. So argmax divergence is a lower bound on distributional divergence — if argmaxes differ, distributions differ. The logit-level JS on Slide 16 confirms this isn't a tie-breaking artifact.


## Backup — Recovery lines if a slide lands badly

- If they don't react to Slide 2: *"I'll come back to why this isn't just randomness on the next slide."*
- If they stall on Slide 4: *"Think weather forecasts, not coin flips."*
- If they push on "is this chaos" at Slide 5: *"I'm using chaos as the lens. The formal version is in Li et al. — I'll get to that."*
- If Slide 10 gets "that's just a bug": *"It's not a bug — the same behavior shows up on every model, OLMo is just the clearest example. It's the probe surfacing something real."*
- If they get lost on Slide 13: *"Low divergence can mean 'stable' or 'collapsed.' The metric can't tell."*
- If they're skeptical of Slide 17 two-floors: *"This is a conjecture. My data is suggestive, not definitive."*


## Backup — Your own confidence levels

**Very confident:**
- Dynamical sensitivity is measurable.
- Deterministic decode with argmax produces reproducible signal.
- Broad semantic clusters are more reliable than precise ranks at n=24.
- Collapse confound is real in Qwen 0.8B 4-bit case.
- Scaffold confound is real as a surface observation.
- Token-edit and semantic-distance metrics can disagree; report both when
  making era or recipe claims.

**Moderately confident:**
- LLaMA-1 is genuinely content-stable (one model, one conversion — small risk of artifact).
- Logit-level and text-level rankings agree (sanity check, not proof).

**Speculative — label as such:**
- Chaos is the right *cause* of compression floors.
- Scaffold *causes* stability (vs just correlates).
- Results generalize to production prompt distributions.

**Not claiming:**
- Any specific Lyapunov number.
- Bigger = more stable (it doesn't).
- Modern post-training = stable (LLaMA-1 counterexample).
