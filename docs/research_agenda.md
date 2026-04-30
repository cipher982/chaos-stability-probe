# Research Agenda

This is the post-talk working plan for turning the stability probe into a
GitHub-ready project and, later, a paper-shaped artifact. It is intentionally
not a claim of proof. The current contribution is a measurement frame: small
prompt-token perturbations can produce model-dependent trajectory changes, and
naive stability metrics are easy to fool.

## Success Criteria

- The public repo can be cloned, smoke-tested, and understood without knowing
  the rushed presentation context.
- Every headline chart traces back to a config, raw artifact, and processing
  script.
- Claims distinguish prompt-token perturbations from raw character edits.
- Scaffolded deliberation streams are not compared directly to answer-first
  generations unless the chart labels that distinction.
- "Stable" always means insensitive to the tested perturbation, not better,
  more accurate, more faithful, or more useful.

## Current Defensible Claims

1. Deterministic LLM generation can branch under small effective prompt-token
   perturbations; this is input sensitivity, not sampling noise.
2. Stability is model- and recipe-dependent. The Qwen3.5 ladder gives a real
   contrast, but not a clean monotonic size law.
3. Scaffolded outputs can look stable on short-output metrics because shared
   preambles dominate the text distance.
4. Quantized or collapsed systems can appear stable while drifting away from
   the BF16/reference behavior, so perturbation stability and fidelity must be
   reported separately.
5. Tokenization matters. A raw edit that becomes token-identical after the
   template/tokenizer is not evidence about model sensitivity.

## Paper Spine

1. Motivation: production LLM systems need local behavioral stability, not only
   benchmark quality.
2. Lens: autoregressive inference as a hybrid sequential system, with continuous
   activations/logits feeding discrete token decisions.
3. Probe: deterministic prompt-neighborhood tests across model families,
   reporting semantic distance, token-path divergence, hidden-state distance,
   and logit-boundary diagnostics.
4. Confounds: scaffold stability, collapse stability, raw-character edits that
   vanish at tokenization, and metric scale dependence.
5. Findings: Qwen size-family contrast, Gemma base-vs-instruct recipe contrast,
   token-certified micro perturbations, logit-boundary signals, and
   quantization/fidelity split.
6. Roadmap: stronger prompt neighborhoods, scaffold/content boundaries,
   larger paired panels, and calibrated accuracy/responsiveness checks.

## Next Experiments

### 1. Finish Token-Certified Micro v3

- Finish Qwen3.5 4B thinking-off.
- Rerun Gemma4 E2B base or mark it partial; the completed SageMaker job timed
  out before `summary.csv`, so it is not valid for the v3 table yet.
- Regenerate the v3 summary and talk figures only after the table is complete.

Done criteria:

- `runs/rankings/token_micro_v3/combined_model_summary.csv` has all selected
  valid models.
- Each model has 25 controls and 500 effective non-control token perturbations.
- Any partial/timed-out run is excluded or explicitly labeled.

### 2. Base vs Instruct Recipe Contrast

Gemma is currently the cleanest same-family recipe test. Use token-certified
prompt pairs and report base/instruct separately for E2B and E4B.

Done criteria:

- Matched prompt set and max-token budget.
- Same processing path for base and instruct.
- Examples inspected from raw `generations.jsonl` for the largest divergences.

### 3. Scaffold Boundary Extraction

Build auditable scaffold/content splits instead of relying on full-output
distance for reasoning-style models.

Done criteria:

- Per-generation boundary span and confidence label.
- Raw/full, scaffold-only, and post-boundary metrics emitted side by side.
- Failed or heuristic boundaries remain visible in downstream tables.

### 4. Logit-Boundary Follow-Up

Run logit probes on the certified micro set. The current best mechanism story
is decision-boundary fragility, not full-vocab JS as a single scalar.

Done criteria:

- Prompt-end top-1 probability, top-1 margin, flip rate, and JS divergence.
- Teacher-forced divergence along matched continuations.
- Correlation table against 512-token semantic divergence.

### 5. Stability vs Responsiveness

Pair every perturbation-stability claim with a responsiveness/fidelity check so
collapse cannot masquerade as robustness.

Done criteria:

- Identical-prompt drift from reference for quantization comparisons.
- Optional task-answer accuracy or rubric score on a small labeled prompt set.
- Two-axis chart: perturbation distance vs reference drift/quality.

