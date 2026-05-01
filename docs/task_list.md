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

Last checked: 2026-05-01 10:34 -0300.

### SageMaker Running

E07 v5 randomized replication is running.

- Preprod `g6e.2xlarge`:
  - `chaos-activation-patch-rep-qwen4b-20260501-001` - downloading
  - `chaos-activation-patch-rep-qwen9b-20260501-001` - downloading
  - `chaos-activation-patch-rep-gemma-e2b-it-20260501-001` - downloading
  - `chaos-activation-patch-rep-gemma-e2b-base-20260501-001` - pending
  - `chaos-activation-patch-rep-gemma-e4b-it-20260501-001` - pending
- Marketing production `g5.2xlarge`:
  - `chaos-activation-patch-rep-qwen2b-20260501-001` - downloading
- QA `g5.2xlarge`:
  - `chaos-activation-patch-rep-qwen08-20260501-001` - downloading
- Queued, not yet launched:
  - `chaos-activation-patch-rep-gemma-e4b-base-20260501-001`

No failed SageMaker jobs were found in the latest scan. The only recent stopped
job surfaced outside the reverse queue was
`chaos-token-micro-qwen4b-thinkoff-20260430-001`, already superseded by later
completed Qwen4B token-micro jobs.

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
    centered L2 `0.581`, JS `0.558`.
- E10 backend comparison was built:
  - `runs/rankings/e10_backend_comparison_20260430/`
  - Qwen2B mean absolute branch-t delta local-vs-SageMaker: `4.25`.
  - Qwen4B mean absolute branch-t delta local-vs-SageMaker: `8.80`.
- Batch-determinism checks completed and were pulled:
  - `chaos-batch-det-qwen08-g6e-20260430-001`
  - `chaos-batch-det-qwen08-g5-20260430-001`
  - output: `runs/sagemaker_artifacts/chaos-batch-det-*/runs/batch_determinism.json`
- E07 v2 activation-patching completed and was processed:
  - output: `runs/rankings/activation_patch_v2/`
  - Qwen0.8B: 6/6 finite, replayable, full-or-overshoot rescues.
  - Gemma E2B-IT: job completed but patch cases failed because
    `activation_patch_branch.py` did not find Gemma's transformer block list.
- E05 Gemma E2B base token-micro repair completed and was processed:
  - output: `runs/rankings/token_micro_v3/`
  - Gemma E2B base is now the second-most fragile v3 model:
    mean `0.1786`, p90 `0.3505`.
- E06 logit-token queue completed and was processed:
  - output: `runs/rankings/logit_token_cert_v1/`
  - All Qwen0.8B/2B/4B/9B and Gemma E2B/E4B instruct/base rows are now present.
- E09 trajectory artifacts were rebuilt against the complete 8-model logit
  panel:
  - output: `runs/trajectory_events/logit_token_cert_v1/`
  - artifact bundle: `runs/trajectory_artifacts/logit_token_cert_v1/`
- Gemma E2B-IT E07 retry completed and was processed:
  - `chaos-activation-patch-gemma-e2b-it-20260430-003`
  - output: `runs/rankings/activation_patch_v2/`
  - Gemma E2B-IT: 6/6 finite, replayable, full-or-overshoot rescues.
- E07 v3 Gemma base causal wave was launched from the rebuilt E09 case
  selection:
  - target config: `configs/activation_patch_targets_v3.json`
  - queue: `configs/sagemaker_queue_activation_patch_v3.json`
  - completed jobs: `chaos-activation-patch-gemma-e2b-base-20260430-001`,
    `chaos-activation-patch-gemma-e4b-base-20260430-001`
  - output: `runs/rankings/activation_patch_v3/`
  - Gemma E2B base: 6/6 finite, replayable, full-or-overshoot rescues; all
    best rescues were final-context at layer 34.
  - Gemma E4B base: 6/6 finite/full-or-overshoot rescues, 5/6 replayable; best
    positions split 3 prompt-LCP and 3 final-context.
- E07 v4 reverse-direction causal wave was launched as a specificity check:
  - target config: `configs/activation_patch_targets_v4_reverse.json`
  - queue: `configs/sagemaker_queue_activation_patch_v4_reverse.json`
  - expanded to include Qwen2B/4B because those were the strongest
    prompt-LCP forward-rescue models.
  - purpose: test whether B activations can push clean A runs toward B's branch,
    rather than only showing that A activations can rescue A inside B.
- E07 v4 reverse-direction causal wave completed and was processed:
  - output: `runs/rankings/activation_patch_v4_reverse/`
  - reverse wave: 21/21 finite full-or-overshoot rescue cases, 19/21 replayable
    full-or-overshoot cases.
  - matched forward/reverse comparison:
    `runs/rankings/activation_patch_comparison/directional_case_comparison.csv`
  - directional readout: 21/21 matched cases have full-or-overshoot rescue in
    both directions; 12/21 have full prompt-LCP rescue in both directions.
  - caveat: two Qwen9B reverse cases did not replay the branch, so count them
    as broad patchability evidence, not replay-clean causal examples.
- E07 v5 randomized replication wave was launched to test selection bias:
  - target config: `configs/activation_patch_targets_v5_replication.json`
  - queue: `configs/sagemaker_queue_activation_patch_v5_replication.json`
  - selection: five held-out token-certified branch cases per model from the
    E09 candidate pool, stratified across immediate/early/mid/long branch
    timing where available.
  - launched: 7/8 jobs; Gemma E4B base is waiting for a preprod `g6e` slot.

### Pending Processing

- Launch remaining E07 v5 Gemma E4B base when a preprod `g6e` slot opens.
- Process E07 v5 replication jobs when they complete.

## Current Readouts

### E05 Token-Certified Micro

`runs/rankings/token_micro_v3/combined_model_summary.csv`

- OPT 6.7B: `0.2894` mean over 500 effective rows.
- Gemma4 E2B base: `0.1786` mean over 500 effective rows.
- Gemma4 E4B base: `0.1286` mean over 500 effective rows.
- Qwen0.8B/2B/4B/9B thinking-off: `0.0930`, `0.0912`, `0.0855`, `0.0786`.
- Gemma4 E4B/E2B instruct: `0.0684`, `0.0591`.
- OLMo3 7B instruct is partial: `0.0860` mean over 152 effective rows.

### E09 Trajectory Events

`runs/trajectory_artifacts/logit_token_cert_v1/`

- Complete 8-model panel: Qwen0.8B/2B/4B/9B plus Gemma E2B/E4B
  instruct/base.
- At-branch AUROC is strong: low-margin `0.947`, JS `0.883`.
- Strict pre-branch-within-1 is weaker: centered L2 `0.661`, JS `0.618`.
- On long-prefix cases (`branch_t >= 5`), strict pre-branch-within-1 weakens to
  centered L2 `0.581`, JS `0.558`.
- Qwen ladder branch timing is not monotonic with size; only `10.4%` of shared
  cases are monotonic earlier-with-size and `10.4%` monotonic later-with-size.
- Gemma base models branch earlier and more often immediately than their
  instruction-tuned siblings: immediate visible branch rate is `18.6%` for
  Gemma E2B base and `31.6%` for Gemma E4B base, versus `0.6%`/`11.0%` for
  E2B/E4B instruct.

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
- Qwen0.8B v2 patch wave processed from 6 selected branch cases; all 6 are
  replayable full-or-overshoot rescues.
- Gemma E2B-IT v2 retry processed from 6 selected branch cases; all 6 are
  replayable full-or-overshoot rescues.
- Gemma E2B/E4B base v3 patch waves processed from 12 selected branch cases;
  all 12 have finite full-or-overshoot rescues, with 11/12 replayable.
- Best rescue position classes differ in this selected set:
  - Qwen0.8B: 4 prompt-LCP, 2 final-context.
  - Gemma E2B-IT: 2 prompt-LCP, 2 generated-prefix, 2 final-context.
  - Gemma E2B base: 6 final-context.
  - Gemma E4B base: 3 prompt-LCP, 3 final-context.
  - Qwen2B: 3 prompt-LCP, 2 final-context, 1 aligned-prompt.
  - Qwen4B: 4 prompt-LCP, 2 final-context.
  - Qwen9: 2 prompt-LCP, 4 final-context.
- Treat rescue fractions above 1.0 as overshoot, not better-than-perfect
  semantic recovery.

### Batch Determinism

`runs/sagemaker_artifacts/chaos-batch-det-*/runs/batch_determinism.json`

- Qwen0.8B CUDA/bfloat16 singleton-vs-batch checks completed on both
  `ml.g6e.2xlarge` and `ml.g5.2xlarge`.
- Batching is faster but not token-exact:
  - `g6e`: batch size 2/4/8 mismatched `1/8`, `3-4/8`, `5/8` prompts.
  - `g5`: batch size 2/4/8 mismatched `3/8`, `2/8`, `5/8` prompts.
- Do not batch science-critical branch-timing generations unless the paper
  explicitly treats batch shape as part of the runtime condition.

## Next Actions

1. Build the paper-grade figures:
   - single-case trajectory anatomy,
   - Qwen branch-timing parallel coordinates,
   - at-branch vs strict pre-branch AUROC forest plot.
2. Process E07 v5 replication. If it broadly reproduces v1-v4, stop launching
   activation-patching waves and pivot to paper figures/writing.
3. Start paper outline once figures and caveats are frozen.

## Useful Commands

```bash
uv run python scripts/sagemaker_status.py --prefix chaos --max-results 20 --details
uv run python scripts/process_logit_queue.py --queue configs/sagemaker_queue_logit_token_cert_v1.json --out-dir runs/rankings/logit_token_cert_v1
uv run python scripts/process_silent_divergence_queue.py --queue configs/sagemaker_queue_silent_divergence_pilot_v1.json --out-dir runs/rankings/silent_divergence_pilot_v1
uv run python scripts/dispatch_sagemaker_queue.py --queue configs/sagemaker_queue_activation_patch_v1.json --profile zh-marketing-preprod-aiengineer --max-active 5
uv run python scripts/process_activation_patch_queue.py --queue configs/sagemaker_queue_activation_patch_v1.json --out-dir runs/rankings/activation_patch_v1
uv run python scripts/process_activation_patch_queue.py --queue configs/sagemaker_queue_activation_patch_v2.json --out-dir runs/rankings/activation_patch_v2
uv run python scripts/process_activation_patch_queue.py --queue configs/sagemaker_queue_activation_patch_v4_reverse.json --out-dir runs/rankings/activation_patch_v4_reverse
uv run python scripts/process_activation_patch_queue.py --queue configs/sagemaker_queue_activation_patch_v5_replication.json --out-dir runs/rankings/activation_patch_v5_replication
uv run python scripts/process_token_micro_queue.py --queue configs/sagemaker_queue_paper_repairs_v1.json --rank-dir runs/rankings/token_micro_v3 --passes 1 --sleep-s 0
uv run python scripts/build_trajectory_artifacts.py --trajectory-dir runs/trajectory_events/logit_token_cert_v1 --silent-summary runs/rankings/silent_divergence_local_qwen_ladder_meta_20260430/silent_divergence_readout.csv
```
