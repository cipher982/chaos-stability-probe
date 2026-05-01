# Task List

Operational board only. Keep historical narrative out of this file; use
`docs/results_digest.md`, `docs/experiment_index.md`, and per-experiment
`experiments/E##_*/README.md` files for durable readouts.

## Source Of Truth

- `docs/results_digest.md` - current interpretation and safest claims.
- `docs/experiment_index.md` - one row per experiment.
- `configs/models.json` - model registry and observed scaffold behavior.
- `configs/sagemaker_queue_*.json` - explicit SageMaker queues.
- `runs/` - raw and derived artifacts; generated source of truth, not committed.

## Active Framing

- Main research object: structured trajectory branching under tiny
  token-certified prompt edits.
- Avoid broad leaderboard expansion unless it tests a specific mechanism.
- Treat prompt pair as the statistical unit.
- Do not describe `branch_within_N` as pre-branch warning; it includes the
  branch timestep. Use strict `pre_branch_within_N` for warning claims.
- Current strongest claim: decision-boundary fragility is localized and
  model/recipe dependent; Qwen branch timing is not monotonic with size.

## Live Operations

Last checked: 2026-04-30 21:13 -0300.

### SageMaker Running

- Preprod `g6e.2xlarge`:
  - `chaos-logit-token-cert-gemma-e2b-base-20260430-001`
  - `chaos-logit-token-cert-gemma-e4b-it-20260430-001`
  - `chaos-logit-token-cert-gemma-e4b-base-20260430-001`
  - `chaos-token-micro-gemma-e2b-base-token-cert-20260430-004`
- Marketing production `g5.2xlarge`:
  - `chaos-activation-patch-gemma-e2b-it-20260430-002`
- QA `g5.2xlarge`:
  - `chaos-activation-patch-qwen08-20260430-002`

Recent stopped jobs are known superseded early attempts from earlier waves; no
new failed/stopped jobs were found in the latest active queues.

### Newly Processed

- E10 SageMaker metadata recaptures completed and were processed:
  - `chaos-silent-div-qwen2b-20260430-001`
  - `chaos-silent-div-qwen4b-20260430-001`
  - `chaos-silent-div-qwen9b-meta-20260430-001`
  - Output: `runs/rankings/silent_divergence_pilot_v1/`
- E07 activation-patching jobs were launched from selected Qwen2B/4B/9B
  branch cases:
  - target config: `configs/activation_patch_targets_v1.json`
  - queue: `configs/sagemaker_queue_activation_patch_v1.json`
- E07 Qwen2B/4B/9B activation-patching jobs completed and were processed:
  - output: `runs/rankings/activation_patch_v1/`
  - summary: 17/18 finite rescue cases; 16/18 replayable full-or-overshoot rescues.
- E07 v2 activation-patching jobs were launched for Qwen0.8B and Gemma E2B-IT:
  - target config: `configs/activation_patch_targets_v2.json`
  - queue: `configs/sagemaker_queue_activation_patch_v2.json`
- E05 paper repair completed and was processed:
  - `chaos-token-micro-opt-6p7b-20260430-004`
  - Output: `runs/rankings/token_micro_v3/`
  - OPT 6.7B is now the fragile v3 baseline: mean `0.2894`, p90 `0.5146`.
- E09 branch prediction artifacts were rebuilt with explicit target labels and
  long-prefix scoring:
  - `branch_window_including_branch` includes the branch timestep.
  - `strict_pre_branch_warning` excludes the branch timestep.
  - Long-prefix subset (`branch_t >= 5`) weakens strict pre-branch-within-1 to
    centered L2 `0.568`, JS `0.558`.
- E10 backend comparison was built:
  - `runs/rankings/e10_backend_comparison_20260430/`
  - Qwen2B mean absolute branch-t delta local-vs-SageMaker: `4.25`.
  - Qwen4B mean absolute branch-t delta local-vs-SageMaker: `8.80`.

### Pending Processing

- Process when complete:
  - Qwen0.8B/Gemma E2B-IT activation-patching jobs
  - Gemma E2B base/E4B instruct/E4B base logit-token jobs
  - Gemma E2B base token-micro repair

## Current Readouts

### E05 Token-Certified Micro

`runs/rankings/token_micro_v3/combined_model_summary.csv`

- OPT 6.7B: `0.2894` mean over 500 effective rows.
- Gemma4 E4B base: `0.1286` mean over 500 effective rows.
- Qwen0.8B/2B/4B/9B thinking-off: `0.0930`, `0.0912`, `0.0855`, `0.0786`.
- Gemma4 E4B/E2B instruct: `0.0684`, `0.0591`.
- OLMo3 7B instruct is partial: `0.0860` mean over 152 effective rows.

### E09 Trajectory Events

`runs/trajectory_artifacts/logit_token_cert_v1/`

- At-branch AUROC is strong: low-margin `0.953`, JS `0.891`.
- Strict pre-branch-within-1 is weaker: centered L2 `0.649`, JS `0.620`.
- On long-prefix cases (`branch_t >= 5`), strict pre-branch-within-1 weakens to
  centered L2 `0.568`, JS `0.558`.
- Qwen ladder branch timing is not monotonic with size; only `10.4%` of shared
  cases are monotonic earlier-with-size and `10.4%` monotonic later-with-size.

### E10 Silent Divergence

`runs/rankings/silent_divergence_pilot_v1/`

- Qwen2B/Qwen4B SageMaker CUDA/bfloat16 metadata now exists.
- Qwen9 SageMaker CUDA/bfloat16 metadata now exists.
- Local MPS/float16 branch timing is not identical to SageMaker CUDA/bfloat16:
  mean absolute branch-t delta is `4.25` for Qwen2B and `8.80` for Qwen4B.
- On the five selected E10 cases, SageMaker mean branch-t is `9.0` for Qwen2B
  over four visible-branch cases, `7.8` for Qwen4B, and `1.8` for Qwen9.
  Treat this as case-selection evidence, not a general scaling law.

### E07 Mechanistic Branch Patching

`runs/rankings/activation_patch_v1/`

- Qwen2B/4B/9B v1 patch wave processed from 18 selected branch cases.
- Finite rescue exists for 17/18 cases; 16/18 are replayable
  full-or-overshoot rescues.
- Best rescue position classes differ in this selected set:
  - Qwen2B: 3 prompt-LCP, 2 final-context, 1 aligned-prompt.
  - Qwen4B: 4 prompt-LCP, 2 final-context.
  - Qwen9: 2 prompt-LCP, 4 final-context.
- Treat rescue fractions above 1.0 as overshoot, not better-than-perfect
  semantic recovery.

## Next Actions

1. Process Qwen0.8B/Gemma E2B-IT activation-patching jobs when they land.
2. Process Gemma logit-token jobs and Gemma E2B base token-micro repair when
   they land.
3. Build the paper-grade figures:
   - single-case trajectory anatomy,
   - Qwen branch-timing parallel coordinates,
   - at-branch vs strict pre-branch AUROC forest plot.
4. Compare E07 v1/v2 by mechanism type and model family before launching the
   next causal wave.

## Useful Commands

```bash
uv run python scripts/sagemaker_status.py --prefix chaos --max-results 20 --details
uv run python scripts/process_logit_queue.py --queue configs/sagemaker_queue_logit_token_cert_v1.json --out-dir runs/rankings/logit_token_cert_v1
uv run python scripts/process_silent_divergence_queue.py --queue configs/sagemaker_queue_silent_divergence_pilot_v1.json --out-dir runs/rankings/silent_divergence_pilot_v1
uv run python scripts/dispatch_sagemaker_queue.py --queue configs/sagemaker_queue_activation_patch_v1.json --profile zh-marketing-preprod-aiengineer --max-active 5
uv run python scripts/process_activation_patch_queue.py --queue configs/sagemaker_queue_activation_patch_v1.json --out-dir runs/rankings/activation_patch_v1
uv run python scripts/process_activation_patch_queue.py --queue configs/sagemaker_queue_activation_patch_v2.json --out-dir runs/rankings/activation_patch_v2
uv run python scripts/process_token_micro_queue.py --queue configs/sagemaker_queue_paper_repairs_v1.json --rank-dir runs/rankings/token_micro_v3 --passes 1 --sleep-s 0
uv run python scripts/build_trajectory_artifacts.py --trajectory-dir runs/trajectory_events/logit_token_cert_v1 --silent-summary runs/rankings/silent_divergence_local_qwen_ladder_meta_20260430/silent_divergence_readout.csv
```
