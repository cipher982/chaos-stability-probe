# Task List

## Source Of Truth

- `configs/models.json` is the canonical model registry.
  - `observed_behavior.annotation_status=observed_in_current_harness` means we
    inspected raw outputs for that model in this repo's harness.
  - `dominant_prefix_kind` records observed behavior such as
    `thinking_process`, `think_tag`, `visible_cot`, `template_echo`, or `none`.
  - `visible_reasoning_scaffold` is an observed-output label, not a claim that
    the model is intrinsically a reasoning model in every serving setup.
  - `boundary_detection` records whether answer-boundary extraction is clean,
    heuristic, failed-prone, or not applicable.
  - `unknown` means uninspected; do not use those models in
    reasoning/scaffold claims until raw-prefix inspection exists.
- `docs/results_digest.md` is the canonical current talk readout.
- `docs/experiment_journal.md` is the chronological lab notebook.
- `docs/task_list.md` is operational state and next actions.

## Done

- Read and consolidated `raw_initial_discussion.txt`.
- Wrote [north_star.md](north_star.md) with talk framing, claims, caveats, and model panel.
- Wrote [experiment_journal.md](experiment_journal.md) as the live lab notebook.
- Built local Hugging Face/Transformers harness:
  - deterministic and sampled generation
  - prompt-pair perturbation ladder
  - same-prompt baseline controls
  - hidden-state layer distance extraction
  - JSONL/CSV outputs
  - per-model failure recording
- Added plotting tools:
  - per-run divergence plots
  - hidden-state heatmaps
  - cross-run comparison bars
  - cross-run comparison boxplots
- Ran local smoke on `Qwen/Qwen3.5-0.8B`.
- Ran expanded local comparison for:
  - `Qwen/Qwen3.5-0.8B`
  - `Qwen/Qwen3.5-4B`
- Pulled completed SageMaker panel and added expanded `Qwen/Qwen3.5-9B`,
  `Gemma 4 E2B`, and `OLMo 3 7B` results to comparison plots.
- Reran `Gemma 4 E4B` after fixing the temporary Triton/Torch dependency issue
  and added it to the full cross-model comparison.
- Doubled the successful sample set from 6 to 13 stability profiles in wave 2.
- Added `scripts/rank_runs.py` and generated the 13-model stability ranking.
- Added `scripts/bootstrap_stability.py` and generated the bucketed bootstrap
  ranking artifact.
- Added [rebuttals.md](rebuttals.md) for steel man objections and talk Q&A prep.
- Added [prior_art.md](prior_art.md) for literature anchors and talk references.
- Added opt-in bitsandbytes quantization support to the harness.
- Added Wave 3 model registry entries for Qwen3.5, Qwen3.6, Gemma 4 base, and
  quantized stretch candidates.
- Found an early empirical result:
  - `Qwen3.5-4B` is much more stable than `Qwen3.5-0.8B` on the expanded deterministic prompt-perturbation ladder.
- Added SageMaker packaging/launch/status/artifact scripts.
- Launched first SageMaker smoke, diagnosed script-mode parameter bug, and patched it.
- Launched second SageMaker smoke with corrected parameters.

## In Progress

Latest experimental framing:

- Treat each model/checkpoint/quantization as its own dynamical system.
- Primary question: within that fixed system, do nearby prompts stay nearby,
  branch, or reconverge?
- For the talk, prioritize **micro-perturbations** that look identical or
  nearly identical to a human: whitespace, line wraps, duplicated punctuation,
  CRLF/newline suffixes, and tiny repeated words. These are more intuitive than
  synonym/paraphrase edits for showing that the model sees a different token
  stream where the human sees the same prompt.
- Cross-system fidelity, such as BF16 output vs 4-bit output on the exact same
  prompt, is not the main talk question. Keep it as a caveat for quantization,
  not as a load-bearing result.
- "Stable" means insensitive to the tested prompt perturbation, not high
  quality, faithful, or useful.

Current micro-perturbation work:

- Added `configs/prompt_pairs_micro_500.json` with 525 prompt pairs:
  25 identical controls plus 500 tiny surface edits across whitespace,
  punctuation, line-wrap, tab, parenthesis, and duplicated-small-word
  categories.
- Local Qwen3.5 0.8B direct-answer run completed and processed:
  - raw run: `runs/micro_qwen35_08b_500/qwen35_08b`
  - processed summary: `runs/rankings/micro_qwen35_08b_500`
  - chart: `runs/rankings/micro_qwen35_08b_500/micro_category_semantic_bar.png`
  - readout: internal layout/syntax edits branch; prefix/suffix whitespace
    mostly does not.
- SageMaker 512-token micro sweeps launched on 24 GB lanes:
  - `chaos-micro-qwen08-512-20260429-001` (`qwen35_08b`,
    thinking disabled, QA account)
  - `chaos-micro-qwen2b-512-20260429-001` (`qwen35_2b`,
    thinking disabled, ML production account)
  - `chaos-micro-gemma-e2b-it-512-20260429-001` (`gemma4_e2b_it`,
    marketing production account)
  - `chaos-micro-gemma-e2b-base-512-20260429-001` (`gemma4_e2b_base`,
    preprod `g6e`)
  - `chaos-micro-gemma-e4b-it-512-20260429-001` (`gemma4_e4b_it`,
    preprod `g6e`)
  - `chaos-micro-gemma-e4b-base-512-20260429-001` (`gemma4_e4b_base`,
    queued for preprod `g6e`)
  - `chaos-micro-qwen4b-thinkoff-512-20260429-001` (`qwen35_4b`,
    thinking disabled, preprod `g6e`)
  - `chaos-micro-qwen9b-thinkoff-512-20260429-001` (`qwen35_9b`,
    thinking disabled, preprod `g6e`)
- Reprocess with:

```bash
uv run python scripts/process_micro_sweep.py \
  runs/micro_qwen35_08b_500/qwen35_08b \
  --out-dir runs/rankings/micro_qwen35_08b_500
```

Next bounded talk-polish tasks:

- Build one clean slide/table around the trajectory-ratio chart:
  - x-axis: generated token position
  - y-axis: fraction of generated prefix that has diverged
  - zero means token-for-token identical so far
- Update quantization slide language:
  - compare sensitivity profiles within BF16/8-bit/4-bit systems
  - mention BF16-fidelity only as a caveat against confusing stability with
    quality

Current artifact state:

- Legacy lane is pulled and semantically processed through GPT-2 XL and Pythia.
- Qwen quantization fidelity analysis is complete:
  - `runs/quantization_fidelity/qwen_quantized_vs_bf16_small_semantic.png`
  - `runs/quantization_fidelity/qwen_quantized_vs_bf16_small_summary.csv`
- Final expanded bootstrap readout is complete:
  - `runs/rankings/final_21model_readout/small_perturbation_bootstrap.csv`
- Robust prompt ladder added:
  - `configs/prompt_pairs_robust.json`
  - 42 prompt pairs total
  - 24 small-perturbation pairs (`noop_format`, `punctuation`, `synonym`)
    instead of the earlier 9

Completed robustness wave:

- Replaced the first pending robustness jobs before training started. Reason:
  loading/downloading dominates wall time, so each model load should do more
  useful measurement.
- Relaunched with 128 generated tokens and hidden-state capture; all five jobs
  completed and were processed:
  - `chaos-robust-qwen35-08b-20260429-002`
  - `chaos-robust-qwen35-2b-20260429-002`
  - `chaos-robust-qwen35-4b-20260429-002`
  - `chaos-robust-qwen35-9b-20260429-002`
  - `chaos-robust-gemma4-e4b-it-20260429-002`

Robust-wave result on 24 small-perturbation prompt pairs:

| Model | Mean semantic distance | 95% bootstrap CI |
| --- | ---: | ---: |
| Qwen3.5 4B | 0.0345 | 0.0182-0.0528 |
| Qwen3.5 9B | 0.0368 | 0.0165-0.0606 |
| Qwen3.5 2B | 0.0728 | 0.0389-0.1150 |
| Gemma4 E4B it | 0.0778 | 0.0406-0.1221 |
| Qwen3.5 0.8B | 0.0887 | 0.0484-0.1368 |

Paired permutation readout:

- `Qwen3.5 4B` vs `Qwen3.5 0.8B`: p = 0.0004
- `Qwen3.5 4B` vs `Qwen3.5 2B`: p = 0.0123
- `Qwen3.5 4B` vs `Qwen3.5 9B`: p = 0.7781
- `Gemma4 E4B it` vs `Qwen3.5 4B`: p = 0.0143

Why not `--repeats` here:

- Deterministic decoding with the same prompt/pair settings repeats the same
  trajectory, so repeat count does not buy statistical independence.
- Better spend per-load work on more prompt pairs, longer generation, and
  hidden-state/logit-style diagnostics.

Post-processing command:

```bash
uv run python scripts/process_robust_wave.py
```

This downloads ready artifacts, adds semantic metrics, builds the merged
summary, bootstraps confidence intervals over prompt pairs, runs paired
permutation tests for planned contrasts, and writes:

- `runs/rankings/robust_wave/merged_summary.csv`
- `runs/rankings/robust_wave/small_perturbation_bootstrap.csv`
- `runs/rankings/robust_wave/small_perturbation_bootstrap.png`
- `runs/rankings/robust_wave/paired_permutation_tests.csv`

Completed logit-probe wave:

- Added logit-level capture to the harness:
  - full-vocab KL/JS divergence computed at each probe point
  - top-token agreement, top-token margins, and winner-rank shifts
  - top-k token/logit/probability snapshots
  - teacher-forced trajectories along both generated continuations
- Launched and processed the robust five-model set with `--logit-probe`,
  `--logit-top-k 10`, and `--logit-max-steps 128`:
  - `chaos-logit-robust-qwen35-08b-20260429-001`
  - `chaos-logit-robust-qwen35-2b-20260429-001`
  - `chaos-logit-robust-qwen35-4b-20260429-001`
  - `chaos-logit-robust-qwen35-9b-20260429-001`
  - `chaos-logit-robust-gemma4-e4b-it-20260429-001`
- Also pulled and included the completed contrast logit jobs:
  - `chaos-logit-robust-gemma4-e4b-base-20260429-001`
  - `chaos-logit-robust-olmo3-20260429-001`
  - `chaos-logit-legacy-gpt2-xl-20260429-001`
  - `chaos-logit-legacy-opt-6p7b-20260429-001`
  - `chaos-logit-legacy-llama1-7b-20260429-001`

Post-processing command:

```bash
uv run python scripts/process_logit_wave.py
```

This writes:

- `runs/rankings/logit_wave/merged_logit_probes_light.csv`
- `runs/rankings/logit_wave/prompt_end_logit_summary.csv`
- `runs/rankings/logit_wave/teacher_forced_trajectory_logit_summary.csv`
- `runs/rankings/logit_wave/teacher_forced_js_by_t.csv`
- `runs/rankings/logit_wave/prompt_end_js_summary.png`
- `runs/rankings/logit_wave/teacher_forced_js_by_t.png`

Prompt-end logit sanity result:

- `Qwen3.5 9B`, `Qwen3.5 4B`, and `LLaMA1 7B` have very small prompt-end
  JS divergence and zero top-1 flips on the robust small-perturbation set.
- `GPT-2 XL`, `OPT`, base Gemma, Qwen 0.8B/2B, Gemma instruct, and OLMo3 show
  larger next-token distribution shifts and nonzero top-1 flip rates.
- This supports the claim that sensitivity is visible below text decoding, but
  it does not by itself resolve the scaffold/content confound.

Queue/dispatch policy:

- Added `configs/sagemaker_queue.json` and `scripts/dispatch_sagemaker_queue.py`
  so we can keep GPU lanes filled without re-deciding the queue by hand.
- Dispatcher behavior:
  - checks live SageMaker state
  - counts active `chaos-*` jobs
  - launches queued jobs until the `--max-active` lane count is full
  - skips jobs whose names already exist in SageMaker
- Run with:

```bash
uv run python scripts/dispatch_sagemaker_queue.py
```

Currently queued/staged jobs cover:

- sampling-vs-input-sensitivity demos on `OLMo 3` and `Qwen3.5 4B`
- Gemma E4B base logit run for base-vs-instruct comparison
- OLMo 3 robust logit run for the no-op formatting demo
- legacy logit probes for GPT-2 XL, OPT, and LLaMA1
- Qwen3.5 0.8B sampling demo
- scaled prompt-ladder logit runs on `Qwen3.5 4B`, `Qwen3.5 0.8B`,
  `Qwen3.5 9B`, `Gemma4 E4B it`, and `OLMo 3`
- temperature sweep sampling demos at `temperature=0.1` and `0.3` for OLMo,
  Qwen3.5 4B, and Qwen3.5 0.8B
- quantized-logit controls for Qwen3.5 4B and 0.8B at 8-bit and 4-bit

Latest dispatch launched:

- `chaos-sample-demo-olmo3-20260429-001`
- `chaos-sample-demo-qwen35-4b-20260429-001`
- `chaos-logit-robust-gemma4-e4b-base-20260429-001`
- `chaos-logit-robust-olmo3-20260429-001`
- `chaos-scaled-logit-qwen35-4b-20260429-001`

Scaled prompt ladder:

- Added `scripts/make_scaled_prompt_pairs.py`.
- Generated `configs/prompt_pairs_scaled.json` with 113 prompt pairs:
  - 10 identical controls
  - 40 no-op formatting pairs
  - 40 punctuation pairs
  - 10 synonym pairs
  - 8 small semantic changes
  - 5 positive controls

Sampling controls:

- Added explicit `--temperature` and `--top-p` flags to the local, panel, and
  SageMaker launchers.
- This lets the talk compare input sensitivity against same-prompt sampling
  variance at lower temperatures instead of only `temperature=0.7`.
- Pulled and processed the `temperature=0.7` sampling demo for OLMo3,
  Qwen3.5 4B, and Qwen3.5 0.8B:
  - `runs/rankings/sampling_demo/sampling_distance_summary.csv`
  - At this temperature, between-prompt distances are generally the same order
    as within-prompt sampling distances.
  - Talk implication: do not claim user-facing sampling variance is small; say
    deterministic probes isolate a different axis.

Reasoning/scaffold analysis:

- Added `scripts/analyze_scaffold_correlation.py`.
- Wrote scaffold annotations and correlation artifacts under
  `runs/rankings/scaffold_analysis/`.
- Added a raw-prefix sanity artifact:
  - `runs/inspection/generation_prefixes_final21.csv`
  - `runs/inspection/generation_prefix_summary_final21.csv`
  - This scans the first generated words/token IDs from every output in the
    current 21-model readout and labels obvious prefix classes.
- Observed reasoning scaffold is now a first-class candidate variable:
  - `thinking_process`: `Qwen3.5 4B`, `Qwen3.5 9B`
  - `think_tag`: `Phi-4 reasoning plus`, `SmolLM3 3B`
  - `visible_cot`: `DeepSeek R1 Qwen 7B`
  - `template_echo`: several older/base models; track separately from
    reasoning scaffolds
- Current result: scaffolded models have much lower small-perturbation semantic
  distance on the final 21-model readout (`0.033` vs `0.141`), but this may be
  format adherence rather than content robustness.
- Raw-prefix read confirms this is not subtle:
  - `Qwen3.5 4B` and `Qwen3.5 9B` are `Thinking Process:` on every checked row.
  - `Phi-4 reasoning plus` and `SmolLM3 3B` are `<think>` on every checked row.
  - `DeepSeek R1 Qwen 7B` emits visible chain-of-thought-style prose.
  - legacy/base GPT/OPT/Pythia/LLaMA often echo templates, which is different
    from modern reasoning scaffolds and is generally brittle.
- Existing 64/128-token outputs are often too short to reach post-scaffold
  answer content, so content-only analysis needs longer generations.

Current active scaffold/content capture:

- Queued a 512-token `chaos-scaffold-long-*` wave with logit probes
  (`logit_max_steps=256`) so scaffolded models have room to reach actual
  answer content.
- Goal: 512-token coverage for all 21 final-panel models, not just the Qwen
  ladder. Use 48 GB `g6e` lanes for larger/reasoning models; use 24 GB `g5`
  and smaller `g4dn` lanes for smaller/base models where feasible.
- First preprod `g6e` jobs launched:
  - `chaos-scaffold-long-qwen35-4b-20260429-001`
  - `chaos-scaffold-long-qwen35-9b-20260429-001`
  - `chaos-scaffold-long-qwen35-08b-20260429-001`
  - `chaos-scaffold-long-qwen35-2b-20260429-001`
  - `chaos-scaffold-long-deepseek-r1-qwen7b-20260429-001`
- Auxiliary 24 GB / smaller lanes launched:
  - preprod `ml.g5.2xlarge`: `chaos-scaffold-long-gptj-6b-g5-20260429-001`
  - preprod `ml.g5.xlarge`: `chaos-scaffold-long-gpt2-xl-g5x-20260429-001`
  - preprod `ml.g4dn.2xlarge`:
    `chaos-scaffold-long-gemma4-e2b-it-g4dn-20260429-001`
  - zh-ml `ml.g5.2xlarge`:
    `chaos-scaffold-long-pythia-6p9b-ml-g5-20260429-001`
  - zh-qa AIEngineer `ml.g5.2xlarge`:
    `chaos-scaffold-long-opt-6p7b-qa-ai-g5-20260429-001`
  - marketing prod `ml.g5.2xlarge`:
    `chaos-scaffold-long-llama1-7b-prod-g5-20260429-001`
- Remaining final-panel 512-token jobs are staged in
  `configs/sagemaker_queue.json`:
  - `chaos-scaffold-long-phi4-reasoning-plus-20260429-001`
  - `chaos-scaffold-long-smollm3-3b-20260429-001`
  - `chaos-scaffold-long-gemma4-e4b-it-20260429-001`
  - `chaos-scaffold-long-olmo3-7b-20260429-001`
  - `chaos-scaffold-long-mistral7b-v03-20260429-001`
  - `chaos-scaffold-long-gemma4-e2b-base-20260429-001`
  - `chaos-scaffold-long-gemma4-e4b-base-20260429-001`
  - `chaos-scaffold-long-granite33-8b-20260429-001`
  - `chaos-scaffold-long-falcon3-10b-20260429-001`
  - `chaos-scaffold-long-olmo2-7b-20260429-001`
- `zh-qa-engineer` can observe SageMaker but cannot create training jobs;
  `zh-qa-aiengineer` can create jobs if launched with `--no-tags`.
- `scripts/launch_sagemaker_panel.py` and the queue dispatcher now support
  per-job `profile`, `bucket`, `role_arn`, `instance_type`, and `no_tags`.
- Added `scripts/process_scaffold_long_wave.py`.
  - Reads `configs/sagemaker_queue.json`.
  - Uses each job's configured AWS profile/region.
  - Pulls only completed 512-token scaffold-long artifacts.
  - Adds semantic metrics if missing.
  - Writes:
    - `runs/rankings/scaffold_long_wave/job_manifest.csv`
    - `runs/rankings/scaffold_long_wave/merged_summary.csv`
    - `runs/rankings/scaffold_long_wave/small_perturbation_bootstrap.csv`
    - `runs/rankings/scaffold_long_wave/token_budget_summary.csv`
  - Current processed run has 17/21 models ready.
- Added `scripts/process_scaffold_long_logits.py`.
  - Reads the ready `chaos-scaffold-long-*` `logit_probes.jsonl` artifacts.
  - Writes prompt-end logit margins/flip rates, teacher-forced trajectory
    summaries, branch-window rows, and semantic-vs-logit correlation artifacts
    under `runs/rankings/scaffold_long_logits/`.
  - Current 17-model result: prompt-end top-1 flip rate correlates with
    512-token semantic divergence (`r ~= 0.75`), while top-1 margin/probability
    is strongly anti-correlated (`margin r ~= -0.77`, top-1 probability
    `r ~= -0.93`). This supports the decision-boundary /
    response-attractor framing more than raw JS divergence alone.
- Added Qwen thinking-mode controls:
  - `scripts/run_stability_probe.py`, `scripts/run_panel.py`,
    `scripts/launch_sagemaker_panel.py`, and the dispatcher accept
    `--thinking-mode default|enabled|disabled`.
  - `disabled` passes `enable_thinking=False` to chat templates that support
    it. In the current Qwen3.5 tokenizer this still emits an empty
    `<think>...</think>` prefix, so call it "thinking-off / empty-think prefix"
    rather than "no scaffold."
  - Launched 512-token/logit thinking-off controls for Qwen3.5 0.8B, 2B, and
    4B across extra `g5.2xlarge` lanes. Qwen3.5 9B thinking-off is staged in
    `configs/sagemaker_queue.json` for the next `g6e` slot.

Latest live queue check, 2026-04-29:

- Processed scaffold-long wave now has 20/25 configured jobs ready.
- Newly pulled/processed since prior status:
  - `chaos-scaled-logit-qwen35-08b-20260429-001`
  - `chaos-scaffold-long-phi4-reasoning-plus-20260429-001`
  - `chaos-scaffold-long-olmo3-7b-20260429-001`
  - `chaos-scaffold-long-gemma4-e4b-base-20260429-001`
- Current scaffold-long ranking artifact is refreshed at
  `runs/rankings/scaffold_long_wave/small_perturbation_bootstrap.csv`.
- Current scaffold-long logit artifacts are refreshed under
  `runs/rankings/scaffold_long_logits/`.
- Active / queued in SageMaker:
  - preprod `g6e`: `chaos-scaled-logit-qwen35-9b-20260429-001` training.
  - preprod `g6e`: `chaos-scaled-logit-gemma4-e4b-it-20260429-001`
    training.
  - preprod `g6e`: `chaos-scaffold-long-qwen35-9b-thinkoff-20260429-001`
    launched and pending capacity.
  - preprod `g6e`: `chaos-scaled-logit-olmo3-7b-20260429-001` launched and
    pending capacity.
  - preprod `g6e`: `chaos-sample-demo-olmo3-t01-20260429-001` launched and
    pending capacity.
  - preprod `g4dn`: `chaos-scaffold-long-gemma4-e2b-base-20260429-001`
    training.
  - QA `g5`: `chaos-scaffold-long-qwen35-08b-thinkoff-20260429-001`
    training.
  - ML prod `g5`: `chaos-scaffold-long-qwen35-2b-thinkoff-20260429-001`
    training.
  - marketing prod `g5`:
    `chaos-scaffold-long-qwen35-4b-thinkoff-20260429-001` training.

Latest refresh after AWS SSO reload:

- All scaffold-long jobs completed and processed: 25/25 ready.
- Qwen thinking-off controls are now in the main scaffold-long ranking.
- Key Qwen default vs thinking-off means:
  - `Qwen3.5 4B`: `0.050`; think-off: `0.067`
  - `Qwen3.5 9B`: `0.057`; think-off: `0.072`
  - `Qwen3.5 2B`: `0.075`; think-off: `0.072`
  - `Qwen3.5 0.8B`: `0.103`; think-off: `0.079`
- Interpretation: Qwen scaffold contributes to 4B/9B apparent stability, but
  does not fully explain it; smaller Qwens do not follow a simple scaffold
  story.
- Refreshed logit correlation with 20 models:
  - top-1 probability vs semantic divergence: `r = -0.842`
  - top-1 flip rate vs semantic divergence: `r = 0.570`
  - top-1 margin vs semantic divergence: `r = -0.388`
  - prompt-end JS vs semantic divergence: `r = -0.096`
- Launched next preprod `g6e` sampling-control batch:
  - `chaos-sample-demo-olmo3-t03-20260429-001`
  - `chaos-sample-demo-qwen35-4b-t01-20260429-001`
  - `chaos-sample-demo-qwen35-4b-t03-20260429-001`
  - `chaos-sample-demo-qwen35-08b-t01-20260429-001`
  - `chaos-sample-demo-qwen35-08b-t03-20260429-001`
- Remaining queued after sampling batch: Qwen quant-logit controls.

Latest progress check:

- Completed and pulled:
  - `chaos-sample-demo-qwen35-08b-t01-20260429-001`
  - `chaos-sample-demo-qwen35-08b-t03-20260429-001`
- Refreshed `runs/rankings/sampling_demo/` after the Qwen 0.8B temperature
  runs landed.
- Still training:
  - `chaos-sample-demo-olmo3-t03-20260429-001`
  - `chaos-sample-demo-qwen35-4b-t01-20260429-001`
  - `chaos-sample-demo-qwen35-4b-t03-20260429-001`
- Newly launched into open `g6e` slots:
  - `chaos-quant-logit-qwen35-4b-bnb8-20260429-001`
  - `chaos-quant-logit-qwen35-4b-bnb4-20260429-001`
- Remaining unlaunched after these finish:
  - Qwen3.5 0.8B quant-logit 8-bit
  - Qwen3.5 0.8B quant-logit 4-bit

Critical interpretation correction:

- Do not use Qwen3.5 4B/9B reasoning-on 512-token outputs in answer-aligned
  trajectory charts. Raw inspection shows they are long draft/critique/polish
  streams and hit the 512-token cap on every row.
- Treat reasoning-on full-output metrics as visible-deliberation/scaffold
  stability, not direct-answer stability.
- Use Qwen thinking-off runs for answer-first Qwen comparisons.
- Show reasoning-on models separately if useful: the finding is that reasoning
  mode can create a strong deliberation attractor that is not directly
  comparable to direct-answer models.

Talk asset triage after checking `talk/`:

- Safe / useful for current slides:
  - `talk/slides.md` conceptual chaos framing.
  - sampling-variance caveat slide, backed by `runs/rankings/sampling_demo/`.
  - non-scaffold/answer-first model table, with Qwen reasoning-on excluded.
  - quantization/collapse caveat slide as a warning, not a core claim.
  - logit mechanism slide if phrased conservatively: top-token confidence is
    the strongest current predictor; margin/flip are weaker after adding more
    models.
- Needs revision before presenting:
  - Qwen table using old robust-wave 4B/9B reasoning-on numbers as a "clean"
    answer-stability contrast. Replace or relabel with Qwen thinking-off runs.
  - scaffold slide language that says 512-token reasoning outputs expose
    "content robustness"; raw outputs show many are still deliberation streams.
  - Phi-4 language: say visible-thinking trajectory diverges, not that final
    answer content is brittle.
- Avoid unless rebuilt:
  - `talk/family_aligned.html` and `talk/data/family_aligned.json`; these assume
    answer alignment that is not supported for Qwen/Phi/DeepSeek reasoning-on.
  - `talk/family_fork.html` / `talk/data/family_curves.json` for current claims,
    because reasoning vs non-reasoning curves mix deliberation streams with
    answer-first outputs.
  - `talk/branching.html` / `talk/data/branching.json` for current Qwen claims;
    it was built from older local all-pairs runs, not the final think-off
    answer-first panel.
  - `runs/trajectory_figures/longprobe_output_trajectory_divergence.png` as an
    answer-trajectory slide; it includes old reasoning-on trajectories and
    should be replaced with think-off/direct-answer trajectories if used.

Statistical robustness plan:

- Treat prompt pair as the unit of analysis. Generated tokens within one output
  are not independent samples.
- Use paired comparisons: every selected model sees the exact same prompt
  pairs.
- Primary robust claims to test:
  - `Qwen3.5 4B` vs `Qwen3.5 0.8B`
  - Qwen size ladder cluster structure (`0.8B`, `2B`, `4B`, `9B`)
  - Gemma E4B instruction vs base
  - legacy/base vs modern/instruct recipe signatures, explicitly split by
    token-edit trajectory metrics vs semantic-content metrics
- Use bootstrap intervals and paired permutation tests over prompt pairs.
- Present clusters and large contrasts, not exact leaderboards.
- If the 42-pair robust ladder still leaves ambiguous contrasts, expand prompt
  count further before adding more models.

## Current Completed Compute

Wave 4 first five-slot wave launched at 2026-04-29 10:36 and completed:

- Long-generation trajectory probes first, because they are the highest-value
  presentation addition:
  - `chaos-longprobe-qwen35-4b-20260429-001`
  - `chaos-longprobe-qwen35-08b-20260429-001`
  - `chaos-longprobe-deepseek-r1-qwen7b-20260429-001`
- Then clean controls that fill missing axes:
  - `chaos-stability-qwen35-2b-20260429-001`
  - `chaos-stability-qwen35-4b-bnb8-20260429-001`

Second fill wave launched at 2026-04-29 11:12 and completed:

- `chaos-stability-qwen35-4b-bnb4-20260429-001`
- `chaos-stability-qwen35-08b-bnb8-20260429-001`
- `chaos-stability-qwen35-08b-bnb4-20260429-001`

Status after follow-up artifact pull:

- Completed and pulled: all three long probes, `qwen35_2b`,
  `qwen35_4b_bnb8`, `qwen35_4b_bnb4`, `qwen35_08b_bnb8`,
  `qwen35_08b_bnb4`.
- Semantic metrics added for the completed stability controls.

Gemma base-vs-instruct controls launched and completed:

1. `chaos-stability-gemma4-e2b-base-20260429-001`
2. `chaos-stability-gemma4-e4b-base-20260429-001`

Legacy/pre-chat contrast lane completed:

- `gpt2_xl`
- `gptj_6b`
- `opt_6p7b`
- `pythia_6p9b_deduped`
- `llama1_7b_huggyllama` as best-effort / clearly labeled community
  conversion

Reason for the legacy lane:

- The current panel is biased toward recent instruction/chat/reasoning models.
- Older base models test whether modern post-training creates stronger response
  attractors or different sensitivity signatures.
- This is more directly connected to the talk's chaos/dynamical-systems thesis
  than adding more same-era leaderboard models.

Launched first legacy fill wave at 2026-04-29:

- `chaos-stability-opt-6p7b-legacy-20260429-001`
- `chaos-stability-gptj-6b-legacy-20260429-001`
- `chaos-stability-llama1-7b-legacy-20260429-001`

Completed and pulled:

- `chaos-stability-opt-6p7b-legacy-20260429-001`
- `chaos-stability-gptj-6b-legacy-20260429-001`
- `chaos-stability-llama1-7b-legacy-20260429-001`
- `chaos-stability-gemma4-e2b-base-20260429-001`
- `chaos-stability-gemma4-e4b-base-20260429-001`

Generated trajectory/ranking figures:

- `runs/trajectory_figures/longprobe_output_trajectory_divergence.png`
- `runs/trajectory_figures/qwen_quantized_output_trajectory_divergence.png`
- `runs/trajectory_figures/current_small_perturbation_semantic_ranking.png`

Launched and completed remaining legacy jobs:

- `chaos-stability-pythia-6p9b-legacy-20260429-001`
- `chaos-stability-gpt2-xl-legacy-20260429-001`

Legacy/base readout after semantic processing:

| Model | Small-perturbation semantic mean |
| --- | ---: |
| Qwen3.5 4B | 0.0133 |
| Qwen3.5 9B | 0.0255 |
| LLaMA1 7B | 0.0582 |
| Gemma4 E4B it | 0.0693 |
| Gemma4 E2B it | 0.0858 |
| Qwen3.5 2B | 0.0972 |
| Gemma4 E2B base | 0.1356 |
| Qwen3.5 0.8B | 0.1381 |
| GPT-J 6B | 0.1936 |
| Gemma4 E4B base | 0.2023 |
| Pythia 6.9B | 0.2370 |
| OPT 6.7B | 0.2560 |
| GPT-2 XL | 0.2822 |

Interpretation: older/base models often look more sensitive, but era alone is
not the whole story because LLaMA1 7B is relatively stable. The cleaner claim
is that post-training/base-model recipe can change response attractors; the
within-family Gemma base-vs-instruct contrast is especially useful.

Follow-up reanalysis for `docs/findings/older_models_more_stable.md`:
token edit distance around `t=60` weakly suggests modern/instruct models are
more surface-divergent, but 512-token semantic distance flips the sign and
shows modern/instruct models as more semantically contractive. Do not present
this as "older models are more stable"; present it as a metric split and recipe
confound.

Parked / not needed for tomorrow:

- Investigate HQQ/GPTQ/GGUF support for 3-bit and 2-bit Qwen3.5 quantization.
- Do not mix low-bit backend results into the clean bitsandbytes grid without
  labeling the backend clearly.
- Reason for parking: the talk no longer depends on compression as the central
  thesis. Existing BF16/8-bit/4-bit results are enough to show that
  quantization changes sensitivity profiles.

## Wave 2 Final State

Done in wave 2:

- `chaos-stability-granite33-8b-20260429-001`: success
- `chaos-stability-deepseek-r1-qwen7b-20260429-001`: success
- `chaos-stability-smollm3-3b-20260429-001`: success
- `chaos-stability-mistral7b-v03-20260429-001`: success
- `chaos-stability-olmo2-7b-20260429-001`: success
- `chaos-stability-phi4-reasoning-plus-20260429-001`: success
- `chaos-stability-falcon3-10b-20260429-001`: success
- `chaos-stability-phi4-mini-20260429-002`: model-load failure

`gpt-oss-20b` remains a recorded tooling miss after four attempts. The final
attempt preserved the container Torch/CUDA stack and installed Triton 3.4.0 with
`--no-deps`, but generation still failed on CPU/CUDA tensor splits.

## Next

### Reasoning-scaffold confound (surfaced 2026-04-29 while building talk viz)

Qwen 3.5 4B/9B emit a deterministic "Thinking Process:" preamble; Qwen 0.8B
does not. Current `common_prefix_tokens` metrics count scaffold-matching as
stability. See the 2026-04-29 journal entry for full context.

Before the talk (required):

- [ ] Write `scripts/strip_scaffold.py` — detects end of reasoning preamble
      (regex on "Thinking Process:" / numbered-list opener / first content
      line) and returns scaffold-free token offsets per generation.
- [ ] Recompute `common_prefix` and `token_edit_distance_norm` on aligned
      content-only tokens for Qwen 2B/4B/9B and DeepSeek-R1-Qwen-7B.
      Write to `runs/rankings/scaffold_aligned_summary.csv`.
- [ ] Rebuild `talk/branching.html` with a "show scaffold / hide scaffold"
      toggle. Audience sees "coupled: 25 → 4" when toggle flips — the
      animation demonstrates the methodology correction itself.
- [ ] Decide whether to match the side-by-side panels to the same family
      (both reasoning or both non-reasoning) for clarity.

Refinements from 2026-04-29 critique (see journal entry for detail):

- [ ] Boundary-detection rules are scaffold-kind specific:
      - `<think>...</think>` (Phi-4, SmolLM3): clean regex.
      - `Thinking Process:` (Qwen): heuristic close, flag as `heuristic`.
      - `visible_cot` (DeepSeek-R1): model-specific; may be `failed`.
      - Emit per-generation boundary confidence (`clean` / `heuristic` /
        `failed`) so downstream analyses can filter.
- [ ] **Truncate all models to first 32 post-scaffold tokens** before
      comparison. Otherwise scaffold models are compared on a ~14-token
      content budget vs ~64-token budget for non-scaffold models.
- [ ] MVP visual for the talk: two-bar chart per model — "full output" vs
      "content-only" semantic divergence. The gap is the scaffold effect.
      Single chart, honest either way.
- [ ] Note the collinearity caveat in the slide: scaffold presence in our
      panel is almost perfectly collinear with modern 2025-era
      post-training. Cannot yet separate scaffold-vs-post-training.
      Template-echo models (GPT-2 XL, OPT, Pythia, LLaMA1, GPT-J) are
      brittle, so format-adherence *per se* does not produce stability —
      effect is specific to modern reasoning scaffolds.
- [ ] Three-tier decomposition is the right long-term split
      (boilerplate prefix / scaffold content / answer content) but too
      ambitious for tonight. Two-tier (stripped vs not) is the MVP.

After the talk (research follow-up):

- [ ] Test H2: do reasoning-tuned models show *delayed* divergence rather
      than *reduced* divergence? Compare early-token vs late-token
      divergence for reasoning vs size-matched non-reasoning peers.
      Candidates in current panel: DeepSeek-R1-Qwen-7B, Phi-4 reasoning
      plus, Qwen3.5 4B/9B (reasoning) vs Mistral 7B, LLaMA1 7B,
      OLMo2/3 7B, Granite 3.3 8B, Falcon3 10B (non-reasoning).
- [ ] Test H3: within a single reasoning model, compute
      scaffold-phase vs content-phase divergence separately. If scaffold
      divergence ≈ 0 and content divergence ≈ non-reasoning peers, that is
      a publishable finding: "reasoning-tune apparent stability is scaffold
      adherence, not robustness."

### Talk polish

1. Use [results_digest.md](results_digest.md) as the compact talk readout.
2. Prefer these presentation artifacts:
   - `runs/trajectory_figures/longprobe_output_trajectory_divergence.png`
   - `runs/trajectory_figures/current_small_perturbation_semantic_ranking.png`
   - `runs/quantization_fidelity/qwen_quantized_vs_bf16_small_semantic.png`
   - `runs/rankings/wave2_13model_bootstrap/small_perturbation_bootstrap_buckets.png`
3. Pull 2-3 concrete text examples for the slide narrative:
   - no-op formatting failure
   - Qwen 0.8B vs 4B tiny perturbation
   - DeepSeek delayed branch / semantic convergence
4. For new compute, prefer high-N micro-perturbation sweeps or direct-answer
   controls. Avoid more broad model-ranking jobs unless a slide has a specific
   missing answer.

## Commands Worth Reusing

Local expanded Qwen comparison:

```bash
uv run python scripts/run_panel.py \
  --model qwen35_08b \
  --prompt-pairs configs/prompt_pairs_expanded.json \
  --max-new-tokens 64 \
  --timeout-s 900 \
  --out-root runs/qwen35_08b_expanded

uv run python scripts/run_panel.py \
  --model qwen35_4b \
  --prompt-pairs configs/prompt_pairs_expanded.json \
  --max-new-tokens 64 \
  --timeout-s 1200 \
  --out-root runs/qwen35_4b_expanded

uv run python scripts/compare_runs.py \
  --run qwen35_08b runs/qwen35_08b_expanded/qwen35_08b \
  --run qwen35_4b runs/qwen35_4b_expanded/qwen35_4b \
  --out-dir runs/comparisons/qwen35_expanded_size_ladder
```

SageMaker status:

```bash
uv run python scripts/sagemaker_status.py \
  --job-name chaos-stability-smoke-20260428-002
```

Download SageMaker artifact:

```bash
uv run python scripts/download_sagemaker_artifact.py \
  chaos-stability-smoke-20260428-002 \
  --extract
```

Launched SageMaker panel:

```bash
uv run python scripts/launch_sagemaker_panel.py \
  --job-name chaos-stability-panel-qwen-gemma-001 \
  --prompt-pairs configs/prompt_pairs_expanded.json \
  --model qwen35_08b \
  --model qwen35_4b \
  --model qwen35_9b \
  --model gemma4_e2b_it \
  --max-new-tokens 64 \
  --timeout-s 3600 \
  --max-runtime-s 28800
```
