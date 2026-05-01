# E07 Mechanistic Branch Patching

Question: can residual activation patching move selected tiny-edit branch
tokens, and where is the strongest causal position/layer signal?

Inputs:
- `configs/prompt_pairs_mechinterp_seed.json`
- branch-point outputs from `scripts/analyze_branch_points.py`
- selected targets in `runs/mechinterp_patch/`

Commands:
- `uv run python scripts/analyze_branch_points.py ...`
- `uv run python scripts/select_patch_targets.py`
- `uv run python scripts/activation_patch_branch.py --positions aligned ...`
- `uv run python scripts/run_activation_patch_panel.py --targets-json configs/activation_patch_targets_v1.json ...`
- `uv run python scripts/summarize_patch_results.py`
- `uv run python scripts/process_activation_patch_queue.py --queue configs/sagemaker_queue_activation_patch_v1.json --out-dir runs/rankings/activation_patch_v1`
- `uv run python scripts/compare_activation_patch_waves.py`

Outputs:
- patch CSVs and heatmaps under `runs/mechinterp_patch*/`
- merged SageMaker readouts under `runs/rankings/activation_patch_v*/`
- cross-wave comparison tables/figure under
  `runs/rankings/activation_patch_comparison/`

Status: active causal-intervention wave.
- Qwen2B/4B/9B v1 SageMaker jobs are processed in
  `runs/rankings/activation_patch_v1/`: 17/18 finite rescue cases and 16/18
  replayable full-or-overshoot rescues.
- Qwen0.8B v2 is processed in `runs/rankings/activation_patch_v2/`: 6/6
  finite, replayable, full-or-overshoot rescues.
- Gemma E2B-IT v2 job `chaos-activation-patch-gemma-e2b-it-20260430-002`
  completed but all cases failed because Gemma4 exposes decoder blocks at
  `model.language_model.layers`.
- `activation_patch_branch.py` now supports that Gemma4 block path, and retry
  `chaos-activation-patch-gemma-e2b-it-20260430-003` is processed in
  `runs/rankings/activation_patch_v2/`: 6/6 finite, replayable,
  full-or-overshoot rescues.
- Gemma base v3 jobs are processed in `runs/rankings/activation_patch_v3/`
  from `configs/activation_patch_targets_v3.json`:
  - `chaos-activation-patch-gemma-e2b-base-20260430-001`
  - `chaos-activation-patch-gemma-e4b-base-20260430-001`
- Gemma E2B base: 6/6 finite, replayable, full-or-overshoot rescues; all best
  rescues are final-context at layer 34.
- Gemma E4B base: 6/6 finite/full-or-overshoot rescues, 5/6 replayable; best
  positions split 3 prompt-LCP and 3 final-context.
- Forward comparison across v1-v3: 41/42 selected cases have full final-context
  rescue, but only 20/42 have full prompt-LCP rescue. In 27/42 cases,
  prompt-LCP rescue is stronger than every aligned prompt-control position.
  Treat final-context rescue as broad causal movability, and prompt-LCP rescue
  as the more specific edit-boundary mechanism signal.
- Reverse-direction v4 jobs are running from
  `configs/activation_patch_targets_v4_reverse.json` and
  `configs/sagemaker_queue_activation_patch_v4_reverse.json`, including the
  prompt-LCP-heavy Qwen2B/4B cases. These patch B activations into clean A runs
  to test whether the causal branch movement is directional/symmetric rather
  than a one-way final-state overwrite artifact.
- Scripts remain in `scripts/` for now because later E08 scripts import them
  directly.

SageMaker status: `sagemaker_entry.py` now supports
`CHAOS_ENTRYPOINT=activation_patch` for aligned residual-patching panels.
