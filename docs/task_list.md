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
- `docs/experiment_index.md` is the one-row-per-experiment tracker.
- `docs/task_list.md` is operational state and next actions.

## Done

- Read and consolidated `raw_initial_discussion.txt`.
- Captured the initial talk framing, claims, caveats, and model panel in the
  journal/digest.
- Wrote the legacy lab notebook now archived at
  [experiment_journal_legacy.md](archive/experiment_journal_legacy.md).
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
- For the talk, prioritize **effective token perturbations**, not raw character
  edits. A prompt pair is valid only after the chat template/tokenizer produces
  different prompt token IDs. Character-only categories that normalize away are
  tokenizer/wrapper artifacts, not model stability evidence.
- Cross-system fidelity, such as BF16 output vs 4-bit output on the exact same
  prompt, is not the main talk question. Keep it as a caveat for quantization,
  not as a load-bearing result.
- "Stable" means insensitive to the tested prompt perturbation, not high
  quality, faithful, or useful.

Token-audited pivot:

Success criteria for the overnight rerun:

- Every processed run has `prompt_tokens.jsonl`.
- Every non-control row in `summary.csv` has `prompt_token_edit_distance > 0`.
- `skipped_pairs.jsonl` records token-identical raw edits when any were
  skipped; absence means zero skipped pairs after validation.
- Slides use only `runs/rankings/token_micro_v2/` or explicitly labeled
  historical/audit artifacts for the tiny-edit claim.
- Any chart label says prompt-token perturbation, not raw character edit.

- Added prompt-token capture to `scripts/run_stability_probe.py`.
  New runs write:
  - `prompt_tokens.jsonl` with post-template prompt token IDs for A/B
  - prompt-token delta columns in `summary.csv`
  - `skipped_pairs.jsonl` for non-control pairs that are token-identical
- Added `--skip-token-identical-non-controls`; all new token-micro jobs must
  use it.
- Added `scripts/audit_micro_token_deltas.py` to reclassify old micro runs.
  Result: for the five completed micro sweeps, 229/500 non-control raw edits
  were token-identical after formatting; 271/model remain as real prompt-token
  perturbations.
- Added `configs/sagemaker_queue_token_micro.json` with 19 token-enforced
  reruns and `scripts/sagemaker_queue_supervisor.py` to keep profile-specific
  lanes moving.
- Added `scripts/process_token_micro_queue.py` to download/process completed
  token-micro jobs into `runs/rankings/token_micro_v2/`.
  - Processor now refuses artifacts missing `prompt_tokens.jsonl`,
    or containing token-identical non-control rows; missing
    `skipped_pairs.jsonl` is treated as zero skipped pairs.
- Added `scripts/summarize_token_micro_v2.py`; the processor updates
  `runs/rankings/token_micro_v2/combined_model_summary.csv` whenever it
  processes new completions.
- Added `scripts/plot_token_micro_v2.py`; the processor regenerates
  `talk/micro_visuals/token_micro_v2_model_bar.png` and
  `talk/micro_visuals/token_micro_v2_category_heatmap.png` after new
  completions.
- Bad launch note: initial `-001` token-micro jobs were stopped because
  `run_panel.py` did not yet forward `--skip-token-identical-non-controls`.
  Use `configs/sagemaker_queue_token_micro_v2.json` (`-002` names).
- Launched corrected token-enforced jobs:
  - `chaos-token-micro-qwen2b-thinkoff-20260430-002`
  - `chaos-token-micro-qwen4b-thinkoff-20260430-002`
  - `chaos-token-micro-qwen9b-thinkoff-20260430-002`
  - `chaos-token-micro-gemma-e2b-it-20260430-002`
  - `chaos-token-micro-gemma-e2b-base-20260430-002`
  - `chaos-token-micro-qwen08-thinkoff-20260430-002`
  - `chaos-token-micro-gemma-e4b-it-20260430-002`
- First corrected processed result:
  - `gemma4_e2b_it`: 296 processed pairs = 25 controls + 271 effective
    prompt-token perturbations. `skipped_pairs.jsonl` contains 229
    token-identical non-controls. Non-control semantic mean: `0.0559`;
    p90: `0.1117`; controls: effectively zero.
- Corrected processed results so far:
  - `qwen35_08b`: 271 effective prompt-token perturbations. Mean `0.0914`;
    p90 `0.1601`; max `0.4517`; controls effectively zero.
  - `gemma4_e2b_it`: 271 effective prompt-token perturbations. Mean `0.0559`;
    p90 `0.1117`; max `0.4861`; controls effectively zero.
  - `qwen35_2b`: 271 effective prompt-token perturbations. Mean `0.0904`;
    p90 `0.1740`; max `0.1998`; controls effectively zero.
  - `qwen35_4b`: 271 effective prompt-token perturbations. Mean `0.0809`;
    p90 `0.1396`; max `0.2933`; controls effectively zero.
  - `qwen35_9b`: 271 effective prompt-token perturbations. Mean `0.0732`;
    p90 `0.1307`; max `0.3280`; controls effectively zero.
  - `olmo2_7b_instruct`: 500 effective prompt-token perturbations. Mean
    `0.0715`; p90 `0.1799`; max `0.3468`; controls effectively zero. This
    run had no skipped pairs because all non-control raw edits changed prompt
    tokens.
  - `gemma4_e4b_it`: 271 effective prompt-token perturbations. Mean `0.0652`;
    p90 `0.1646`; max `0.3689`; controls effectively zero.
  - `mistral7b_instruct_v03`: 500 effective prompt-token perturbations. Mean
    `0.0637`; p90 `0.1495`; max `0.2206`; controls effectively zero.
  - `granite33_8b_instruct`: 500 effective prompt-token perturbations. Mean
    `0.0668`; p90 `0.1474`; max `0.4043`; controls effectively zero.
  - `falcon3_10b_instruct`: 500 effective prompt-token perturbations. Mean
    `0.0535`; p90 `0.1355`; max `0.2449`; controls effectively zero. This
    run had no skipped pairs because all non-control raw edits changed prompt
    tokens.
- Newly launched after wider routing:
  - `chaos-token-micro-mistral7b-v03-20260430-002` on the ML production
    `g5.2xlarge` lane.
  - `chaos-token-micro-falcon3-10b-20260430-002` on the preprod `g6e.2xlarge`
    lane.
  - `chaos-token-micro-olmo2-7b-20260430-002` on the preprod `g6e.2xlarge`
    lane after rerouting from a blocked 24 GB lane.
  - `chaos-token-micro-gpt2-xl-20260430-002` on the marketing production
    `g5.2xlarge` lane after Qwen4B completed.
  - `chaos-token-micro-olmo3-7b-20260430-002` on the QA `g5.2xlarge` lane
    after Qwen0.8B completed.
  - `chaos-token-micro-pythia-6p9b-20260430-002` on a freed preprod
    `g6e.2xlarge` lane.
  - `chaos-token-micro-granite33-8b-20260430-002` on a freed preprod
    `g6e.2xlarge` lane.
  - `chaos-token-micro-llama1-7b-20260430-002` on a freed preprod
    `g6e.2xlarge` lane.
  - `chaos-token-micro-gptj-6b-20260430-002` on a freed preprod
    `g6e.2xlarge` lane.
  - `chaos-token-micro-opt-6p7b-20260430-002` on the ML production
    `g5.2xlarge` lane.
- Live status at 2026-04-30 05:54 -0300:
  - Processed/completed: 10 models.
  - Active training: Gemma E2B base, Gemma E4B base, OLMo3, LLaMA-1 7B,
    GPT-2 XL, GPT-J 6B, OPT 6.7B, Pythia 6.9B.
  - Queued behind QA: Phi-4 mini.
  - Failed: none.
- Stopped stale pre-tokenization jobs that were still occupying preprod `g6e`
  lanes:
  - `chaos-micro-gemma-e2b-base-512-20260429-001`
  - `chaos-micro-gemma-e4b-base-512-20260429-001`
- Pending token-enforced queue includes Gemma E4B base, Mistral, OLMo2/3,
  LLaMA-1, GPT-2 XL, GPT-J, OPT, Pythia, Granite, Falcon, Phi mini.
- Queue routing was widened after the first valid completions:
  - 7B-ish and smaller no-hidden runs are routed across idle 24 GB lanes
    (`zh-ml-productionengineer`, `zh-marketing-productionengineer`,
    `zh-qa-aiengineer`).
  - Preprod `g6e` remains reserved for currently running Gemma jobs and the
    largest stretch run (`Falcon3 10B`).
- Running local loops:
  - `scripts/sagemaker_queue_supervisor.py --queue configs/sagemaker_queue_token_micro_v2.json --passes 0 --sleep-s 600`
  - `scripts/process_token_micro_queue.py --queue configs/sagemaker_queue_token_micro_v2.json --passes 0 --sleep-s 900`

Token-certified v3 reinforcement wave:

- Rationale: the corrected v2 run is valid, but Qwen/Gemma instruct models only
  kept 271 effective non-control rows after tokenizer/template filtering. To
  remove that sample-size weakness, v3 uses model-specific prompt-pair files
  certified against each model's exact tokenizer and thinking mode before
  launch.
- Added `scripts/make_token_certified_micro_pairs.py`.
- Added `configs/prompt_pairs_token_certified/*.json`:
  25 identical controls + 500 effective non-control prompt-token
  perturbations per selected model.
- Added `configs/sagemaker_queue_token_certified_v3.json` with 8 follow-up
  jobs:
  - Qwen3.5 0.8B/2B/4B/9B, all `thinking_mode=disabled`
  - Gemma4 E2B/E4B instruct and base
- Running local v3 loops in tmux session `chaos-token-night`, window
  `token-cert-v3`:
  - `scripts/sagemaker_queue_supervisor.py --queue configs/sagemaker_queue_token_certified_v3.json --passes 0 --sleep-s 600`
  - `scripts/process_token_micro_queue.py --queue configs/sagemaker_queue_token_certified_v3.json --rank-dir runs/rankings/token_micro_v3 --passes 0 --sleep-s 900`
- Live status at 2026-04-30 19:35 -0300:
  - Processed in `runs/rankings/token_micro_v3/`: Qwen3.5 0.8B, 2B, 4B, 9B
    thinking-off; Gemma4 E2B/E4B instruct; Gemma4 E4B base; partial OLMo3 7B
    instruct.
  - Completed but still incomplete for this analysis: Gemma4 E2B base
    `-003` produced partial raw rows but no `summary.csv`. A processing retry
    recorded
    `runs/rankings/token_micro_v3/_processing_errors/chaos-token-micro-gemma-e2b-base-token-cert-20260430-003.json`.
  - Repair jobs currently in progress:
    `chaos-token-micro-gemma-e2b-base-token-cert-20260430-004`,
    `chaos-token-micro-opt-6p7b-20260430-004`.
  - Recovered partial repair: `chaos-token-micro-olmo3-7b-20260430-004`
    produced no `summary.csv`, but `scripts/process_micro_sweep.py` recovered
    one from `generations.jsonl`. It has 177 summary rows = 25 controls + 152
    effective non-controls after CUDA errors near the end, so keep it caveated
    in v3 claims.
  - Logit mechanism jobs currently in progress:
    `chaos-logit-token-cert-gemma-e2b-base-20260430-001`,
    `chaos-logit-token-cert-gemma-e4b-it-20260430-001`,
    `chaos-logit-token-cert-gemma-e4b-base-20260430-001`.
  - Current v3 processed means:
    Gemma4 E4B base `0.1286`, Qwen3.5 0.8B `0.0930`, Qwen3.5 2B `0.0912`,
    OLMo3 7B instruct `0.0860` over 152 effective rows, Qwen3.5 4B `0.0855`,
    Qwen3.5 9B `0.0786`, Gemma4 E4B instruct `0.0684`, Gemma4 E2B instruct
    `0.0591`.
  - Added `scripts/analyze_token_micro_panel.py`; current bootstrap output is
    under `runs/rankings/token_micro_v3/stats/`.

Paper-grade follow-up wave launched 2026-04-30:

- Added `configs/sagemaker_queue_paper_repairs_v1.json` for incomplete
  token-micro rows:
  - `chaos-token-micro-gemma-e2b-base-token-cert-20260430-004`
  - `chaos-token-micro-olmo3-7b-20260430-004`
  - `chaos-token-micro-opt-6p7b-20260430-004`
- Added `configs/sagemaker_queue_logit_token_cert_v1.json` and
  `scripts/process_logit_queue.py` for the mechanism wave: token-certified
  prompt neighborhoods plus prompt-end and teacher-forced logit diagnostics.
- Launched immediately:
  - Qwen3.5 0.8B thinking-off logit probe on QA `g5.2xlarge`
  - Qwen3.5 2B thinking-off logit probe on ML prod `g5.2xlarge`
  - Qwen3.5 4B thinking-off logit probe on marketing prod `g5.2xlarge`
  - Qwen3.5 9B thinking-off logit probe on preprod `g6e.2xlarge`
  - Gemma4 E2B instruct logit probe on preprod `g6e.2xlarge`
- Queued behind full lanes:
  - Gemma4 E2B base, E4B instruct, E4B base logit probes

Live operations at 2026-04-30 19:42 -0300:

- E06/E09 processed jobs: Qwen3.5 0.8B, Qwen3.5 2B, Qwen3.5 4B, Qwen3.5 9B,
  Gemma4 E2B instruct.
- E06/E09 still running: Gemma4 E2B base/E4B instruct/E4B base on preprod.
- E10 local captures completed: Qwen3.5 2B and Qwen3.5 4B.
- E10 SageMaker metadata captures launched:
  - `chaos-silent-div-qwen2b-20260430-001` on ML production `g5.2xlarge`
    (`InProgress`, pending as of launch check).
  - `chaos-silent-div-qwen4b-20260430-001` on marketing production
    `g5.2xlarge` (`InProgress`, pending as of launch check).
- Paper-repair note: OLMo3 `-004` was recovered from raw generations but is
  partial: 25 controls + 152 effective non-controls.
- Experiment readouts now belong in `docs/experiment_index.md` and
  `experiments/E##_*/README.md`, not this operational list.
- Next step when lanes free: rerun
  `uv run python scripts/sagemaker_queue_supervisor.py --queue configs/sagemaker_queue_logit_token_cert_v1.json --passes 1`.
- Process completed logit jobs with:

```bash
uv run python scripts/process_logit_queue.py \
  --queue configs/sagemaker_queue_logit_token_cert_v1.json \
  --out-dir runs/rankings/logit_token_cert_v1
```

Trajectory-branching research pivot:

- Main frame: **TrajectoryScope**, a paired-generation microscope for branch
  points, cliffs, basin switches, scaffold masking, and amplification under
  tiny token-visible prompt edits.
- Committed experiment structure now mirrors the active mechanism threads:
  E05-E10 short notes live under `experiments/`; E09/E10 implementation files
  moved there, while the old `scripts/` command paths remain compatibility
  wrappers for docs, queues, and active SageMaker jobs.
- Boundary labels are metadata, not the headline. Use nuisance/task-relevant
  labels to interpret whether a branch is semantically appropriate, but keep
  the primary analysis on event localization, warning signals, and
  interventions.
- Hypotheses under active test:
  - H1: high-divergence pairs localize to branch windows.
  - H2: logits and/or hidden states can warn before visible token divergence.
  - H3: branch points are enriched near margin cliffs or high-confidence basin
    switches.
  - H4: scaffolds can mask internal divergence.
  - H5: forced-prefix or activation patching can move, delay, or suppress
    selected branches.
- Added `scripts/analyze_trajectory_events.py`.
  - Input: one or more run dirs with `summary.csv`/`summary_with_semantic.csv`
    and `logit_probes.jsonl`.
  - Output:
    - `trajectory_events.csv`
    - `branch_prediction_windows.csv`
    - `trajectory_event_summary.csv`
  - First local validation:

```bash
uv run python scripts/analyze_trajectory_events.py \
  runs/mechinterp_seed/qwen35_08b \
  runs/mechinterp_seed/qwen35_2b \
  --out-dir runs/trajectory_events/mechinterp_seed
```

- Added `scripts/capture_silent_divergence.py` for the focused hidden-state
  pilot along the common-prefix window. Start locally on Qwen3.5 2B selected
  patch targets, then move broader hidden capture to SageMaker if the pilot is
  useful.
- Added SageMaker support for that pilot:
  - `sagemaker_entry.py` supports `CHAOS_ENTRYPOINT=silent_divergence`.
  - `scripts/run_silent_divergence_panel.py` runs selected branch cases
    model-by-model.
  - `scripts/launch_sagemaker_panel.py` and queue dispatch support
    `entrypoint` plus `pair_ids`.
  - `configs/sagemaker_queue_silent_divergence_pilot_v1.json` stages Qwen3.5
    2B/4B/9B over five replayable branch cases.
  - `scripts/process_silent_divergence_queue.py` downloads and merges completed
    silent-divergence artifacts.
  - Launched `chaos-silent-div-qwen9b-20260430-001` on preprod `g6e.2xlarge`
    at 2026-04-30 16:50 -0300.
  - `chaos-silent-div-qwen9b-20260430-001` completed at 2026-04-30 17:04 -0300
    and was processed to `runs/rankings/silent_divergence_pilot_v1/`.
  - Launched metadata-backed Qwen2B/Qwen4B SageMaker entries at 2026-04-30
    19:40 -0300:
    `chaos-silent-div-qwen2b-20260430-001` on ML production `g5.2xlarge`,
    `chaos-silent-div-qwen4b-20260430-001` on marketing production
    `g5.2xlarge`.
  - Fixed E10 capture for no-visible-branch controls: it now captures the full
    shared generated prefix instead of only `t=0`.
  - Fixed E10 queue processing so auth/profile failures are written as explicit
    errors and missing jobs are recorded as `not_found`.
  - Recaptured local Qwen2B/Qwen4B E10 from commit `5128d18` with runtime
    metadata under
    `runs/rankings/silent_divergence_local_qwen_ladder_meta_20260430/`.
- Added `scripts/analyze_branch_prediction.py` to score simple AUROC baselines
  for branch decision-window and pre-branch targets from
  `branch_prediction_windows.csv`.
- The first event-mining validation found mostly `silent_logit_divergence`
  cases in the local Qwen 0.8B/2B seed set. Treat this as a machinery check,
  not a final result; the SageMaker token-certified logit queue is the higher-N
  panel.
- First local branch-prediction sanity check on the Qwen 0.8B/2B seed set:
  logit JS and centered logit L2 predict visible branch within 1-10 tokens with
  AUROC around `0.80-0.86` overall. This is promising enough to run on the
  higher-N SageMaker logit queue, but too small and target-selected to claim.
- First local Qwen3.5 2B hidden pilot wrote:
  - `runs/silent_divergence_pilot/qwen35_2b_silent_divergence_summary.csv`
  - `runs/silent_divergence_pilot/qwen35_2b_silent_divergence_layers.csv`
  The blank-line case branches at token 9 and shows logit/hidden movement
  before the visible split; the parenthesized case is an immediate token-0
  branch and is less useful for silent-warning claims.
- Next compute steps:
  - Process `configs/sagemaker_queue_logit_token_cert_v1.json` whenever jobs
    complete, then run `analyze_trajectory_events.py` against the extracted
    run dirs.
  - Run margin-cliff prediction from `branch_prediction_windows.csv`.
  - If the hidden pilot shows pre-visible hidden warning, launch a SageMaker
    hidden-state queue for selected Qwen/Gemma branch cases.
  - If event mining finds stable branch candidates, expand forced-prefix
    interventions before doing more broad model rankings.

Local mech-interp seed setup:

- Added `configs/prompt_pairs_mechinterp_seed.json`,
  `scripts/analyze_branch_points.py`, `scripts/select_patch_targets.py`,
  `scripts/activation_patch_branch.py`, `scripts/summarize_patch_results.py`,
  and `scripts/plot_patch_heatmap.py`.
- Ran local Qwen3.5 0.8B and 2B thinking-off seed panels under
  `runs/mechinterp_seed/`.
- Ran residual activation patching under `runs/mechinterp_patch/`.
- Current strongest finding: parenthesized `(a)` and tab-after-space prompt
  edits can flip answer-opening/early-branch tokens, and last-layer
  final-context residual patching fully rescues the clean branch in both Qwen
  0.8B and 2B.
- Broader selector added and run:
  `runs/mechinterp_patch/selected_patch_targets.csv`.
- Compact patch readout:
  `runs/mechinterp_patch/patch_summary.csv`.
- Aligned-position sweeps added after discovering that raw same-index
  all-position patching misaligns tokens after insertions. Use
  `--positions aligned` for insertion/deletion prompt deltas.
- Aligned heatmaps:
  - `runs/mechinterp_patch_aligned/qwen35_08b__token_cert_parenthesize_word_0434.heatmap.png`
  - `runs/mechinterp_patch_aligned/qwen35_08b__token_cert_tab_after_space_0572.heatmap.png`
  - `runs/mechinterp_patch_aligned/qwen35_2b__token_cert_parenthesize_word_0434.heatmap.png`
  - `runs/mechinterp_patch_aligned/qwen35_2b__token_cert_tab_after_space_0572.heatmap.png`
- Next useful step: attach SAE feature activations to the selected aligned
  branch cases. Qwen-Scope now has an official Qwen3.5 2B Base residual-stream
  SAE release that is close enough to try first on the local 2B branch cases;
  Gemma Scope is the parallel path if we want Gemma-family feature labels.
- Qwen-Scope SAE pilot completed for Qwen3.5 2B:
  - `runs/mechinterp_sae/qwen35_2b__token_cert_parenthesize_word_0434__sae_features.csv`
  - `runs/mechinterp_sae/qwen35_2b__token_cert_tab_after_space_0572__sae_features.csv`
  - `runs/mechinterp_sae/sae_feature_delta_summary.csv`
- Next useful step after the pilot: add feature-label lookup if Qwen-Scope or
  Neuronpedia exposes labels for these feature IDs; otherwise keep the claim at
  feature-overlap/delta level and avoid naming the features.

Micro-perturbation work started:

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
- SageMaker 512-token micro sweeps:
  - Completed + processed:
    - `chaos-micro-qwen2b-512-20260429-001` -> `runs/rankings/micro_sweep/qwen35_2b`
    - `chaos-micro-qwen4b-thinkoff-512-20260429-001` -> `runs/rankings/micro_sweep/qwen35_4b`
    - `chaos-micro-qwen9b-thinkoff-512-20260429-001` -> `runs/rankings/micro_sweep/qwen35_9b`
    - `chaos-micro-gemma-e2b-it-512-20260429-001` -> `runs/rankings/micro_sweep/gemma4_e2b_it`
    - `chaos-micro-gemma-e4b-it-512-20260429-001` -> `runs/rankings/micro_sweep/gemma4_e4b_it`
  - Still running as of live AWS check:
    - `chaos-micro-qwen08-512-20260429-001` (`qwen35_08b`, thinking disabled, QA)
    - `chaos-micro-gemma-e2b-base-512-20260429-001` (`gemma4_e2b_base`, preprod `g6e`)
    - `chaos-micro-gemma-e4b-base-512-20260429-001` (`gemma4_e4b_base`, preprod `g6e`)
- Current processed micro means, excluding identical controls:
  - `gemma4_e2b_it`: mean 0.0276, p90 0.0886, max 0.4892
  - `gemma4_e4b_it`: mean 0.0353, p90 0.1177, max 0.3689
  - `qwen35_9b`: mean 0.0397, p90 0.1089, max 0.3280
  - `qwen35_4b`: mean 0.0464, p90 0.1229, max 0.3603
  - `qwen35_2b`: mean 0.0490, p90 0.1482, max 0.1998
  - identical controls are effectively zero for all processed runs.
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

Artifact state at scaffold-analysis pass:

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

At queue creation time, queued/staged jobs covered:

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
    21-model readout and labels obvious prefix classes.
- Observed reasoning scaffold is now a first-class candidate variable:
  - `thinking_process`: `Qwen3.5 4B`, `Qwen3.5 9B`
  - `think_tag`: `Phi-4 reasoning plus`, `SmolLM3 3B`
  - `visible_cot`: `DeepSeek R1 Qwen 7B`
  - `template_echo`: several older/base models; track separately from
    reasoning scaffolds
- Scaffold-analysis result: scaffolded models have much lower
  small-perturbation semantic distance on the final 21-model readout (`0.033`
  vs `0.141`), but this may be format adherence rather than content robustness.
- Raw-prefix read confirms this is not subtle:
  - `Qwen3.5 4B` and `Qwen3.5 9B` are `Thinking Process:` on every checked row.
  - `Phi-4 reasoning plus` and `SmolLM3 3B` are `<think>` on every checked row.
  - `DeepSeek R1 Qwen 7B` emits visible chain-of-thought-style prose.
  - legacy/base GPT/OPT/Pythia/LLaMA often echo templates, which is different
    from modern reasoning scaffolds and is generally brittle.
- Existing 64/128-token outputs are often too short to reach post-scaffold
  answer content, so content-only analysis needs longer generations.

Scaffold/content capture launched, now superseded by completed refresh below:

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
- At that point, remaining final-panel 512-token jobs were staged in
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
  - Initial processed run had 17/21 models ready.
- Added `scripts/process_scaffold_long_logits.py`.
  - Reads the ready `chaos-scaffold-long-*` `logit_probes.jsonl` artifacts.
  - Writes prompt-end logit margins/flip rates, teacher-forced trajectory
    summaries, branch-window rows, and semantic-vs-logit correlation artifacts
    under `runs/rankings/scaffold_long_logits/`.
  - Initial 17-model result, superseded below by the 20-model refresh:
    prompt-end top-1 flip rate correlates with
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

Earlier live queue check, 2026-04-29, superseded by the completed refresh below:

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
- Revised for current deck / notes:
  - Qwen table is now caveated as scaffold-confounded, with thinking-off
    controls in the following slide/notes.
  - scaffold slide says "metric artifact / mixed bag," not content robustness.
  - Phi-4 language is framed as prompt-end confidence vs 512-token trajectory,
    not proof that final answer content is brittle.
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
    has been replaced in `talk/slides.md` by the Qwen thinking-off control
    figure.

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
- `runs/trajectory_figures/qwen_thinkoff_trajectory_and_semantic.png`
- `runs/trajectory_figures/qwen_quantized_output_trajectory_divergence.png`
- `runs/trajectory_figures/current_small_perturbation_semantic_ranking.png`

Launched and completed remaining legacy jobs:

- `chaos-stability-pythia-6p9b-legacy-20260429-001`
- `chaos-stability-gpt2-xl-legacy-20260429-001`

Legacy/base readout after semantic processing. This is the historical
short-output/small-perturbation panel; the current talk's era/recipe slide uses
the 512-token semantic panel and should not mix these point estimates:

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

Follow-up reanalysis: token edit distance around `t=60` weakly suggests
modern/instruct models are more surface-divergent, but 512-token semantic
distance flips the sign and shows modern/instruct models as more semantically
contractive. Do not present this as "older models are more stable"; present it
as a metric split and recipe confound.

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

Current priority is structured mechanism work, not more broad model ranking.

1. Keep `docs/experiment_index.md` current with one row per experiment. Put
   restart details in the relevant `experiments/E##_*/README.md`.
2. Mine structured divergence events from completed logit runs:
   - visible branch token;
   - first silent logit warning while output text is still identical;
   - margin/entropy at branch;
   - persistence/reconvergence;
   - scaffold/content confidence;
   - final semantic divergence.
3. Run margin-cliff prediction: while continuations are still identical, test
   whether branch probability within 1/2/5/10 tokens is predictable from
   margin, entropy, top-1 probability, or rank instability.
4. Expand branch patching by mechanism type:
   - edit-boundary shocks;
   - accumulated branch bias;
   - token-effective but inert edits;
   - replay-unstable false positives.
5. Add negative controls for the SAE pilot: prompt-token-effective edits with
   similar token deltas but no output branch, so feature deltas are not confused
   with generic tokenization differences.
6. Keep scaffold/content boundary extraction as a separate measurement cleanup:
   preserve raw text, boundary span, confidence label, and score-before/after.
7. Finish or explicitly retire stale token-certified/SageMaker queues before
   using them in current claims. Check live SageMaker state before changing
   queue language.
8. Use the old talk-polish tasks below this point as historical context only;
   the current interpretation lives in `docs/results_digest.md`.

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
