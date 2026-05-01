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
- Practical tool direction: BranchTrace, a branch-level debugger for LLM
  behavior regressions.
- Avoid broad leaderboard expansion unless it tests a specific mechanism.
- Treat prompt pair as the statistical unit.
- Do not describe `branch_within_N` as pre-branch warning; it includes the
  branch timestep. Use strict `pre_branch_within_N` for warning claims.
- Current strongest claim: tiny-edit branch events are often causally movable,
  but prompt-LCP/edit-boundary rescue is a subset, not the universal mechanism.

## Live Operations

Last checked: 2026-05-01 after processing E07 v5.

- No active SageMaker jobs found in the latest `chaos` scan.
- No newly failed jobs require rerun.
- Recent stopped token-micro jobs were superseded by later completed repair
  jobs; do not rerun without a new reason.

## Current Readouts

### E07 Mechanistic Branch Patching

Artifacts:

- `runs/rankings/activation_patch_v1/`
- `runs/rankings/activation_patch_v2/`
- `runs/rankings/activation_patch_v3/`
- `runs/rankings/activation_patch_v4_reverse/`
- `runs/rankings/activation_patch_v5_replication/`
- `runs/rankings/activation_patch_comparison/`

Readout:

- Forward selected waves: 80/82 cases have full final-context rescue.
- Prompt-LCP/edit-boundary rescue is real but not universal: 41/82 full,
  65/82 at least 0.5, and 51/82 stronger than every aligned prompt-control
  position.
- Strict late-only cases remain 0/82 under the 0.5 rescue cutoff after
  aligned-prompt and generated-prefix controls.
- Reverse-direction controls: 21/21 matched cases have full-or-overshoot rescue
  in both directions; 19/21 are replayable in both directions; 12/21 have full
  prompt-LCP rescue in both directions.
- Randomized held-out V5 replication: 39/40 finite full-or-overshoot rescue
  cases and 35/40 replayable full-or-overshoot rescue cases.
- V5 best-position split: 11 prompt-LCP, 22 final-context, 5 generated-prefix,
  2 aligned-prompt.

Decision: stop expanding activation-patching waves unless a new run tests a
specific objection. The next value is synthesis, figures, negative controls,
and BranchTrace-style artifacts.

### E09 Trajectory Events

Artifacts:

- `runs/trajectory_events/logit_token_cert_v1/`
- `runs/trajectory_artifacts/logit_token_cert_v1/`

Readout:

- Complete 8-model panel: Qwen0.8B/2B/4B/9B plus Gemma E2B/E4B instruct/base.
- At-branch AUROC is strong: low-margin `0.947`, JS `0.883`.
- Strict pre-branch-within-1 is modest: centered L2 `0.661`, JS `0.618`.
- On long-prefix cases (`branch_t >= 5`), strict pre-branch-within-1 weakens to
  centered L2 `0.581`, JS `0.558`.
- Qwen ladder branch timing is not monotonic with size; only `10.4%` of shared
  cases are monotonic earlier-with-size and `10.4%` monotonic later-with-size.
- Gemma base models branch earlier and more often immediately than their
  instruction-tuned siblings.

### E10 Silent Divergence

Artifacts:

- `runs/rankings/silent_divergence_pilot_v1/`
- `runs/rankings/silent_divergence_local_qwen_ladder_meta_20260430/`
- `runs/rankings/e10_backend_comparison_20260430/`

Readout:

- Qwen2B/Qwen4B/Qwen9 CUDA/bfloat16 metadata exists.
- Backend/dtype can move branch timing materially: local-vs-SageMaker mean
  absolute branch-t delta is `4.25` for Qwen2B and `8.80` for Qwen4B.
- Treat E10 as case-selection/intervention evidence, not a general scaling law.

### E05 Token-Certified Micro

Artifact:

- `runs/rankings/token_micro_v3/combined_model_summary.csv`

Readout:

- OPT 6.7B remains the fragile v3 baseline: mean `0.2894`, p90 `0.5146`.
- Gemma4 E2B base is second-most fragile: mean `0.1786`, p90 `0.3505`.
- Qwen0.8B/2B/4B/9B thinking-off means: `0.0930`, `0.0912`, `0.0855`,
  `0.0786`.
- Gemma4 E4B/E2B instruct means: `0.0684`, `0.0591`.
- OLMo3 7B instruct is partial: `0.0860` mean over 152 effective rows.

### Batch Determinism

Artifacts:

- `runs/sagemaker_artifacts/chaos-batch-det-*/runs/batch_determinism.json`

Readout:

- Qwen0.8B CUDA/bfloat16 singleton-vs-batch checks completed on both
  `ml.g6e.2xlarge` and `ml.g5.2xlarge`.
- Batching is faster but not token-exact:
  - `g6e`: batch size 2/4/8 mismatched `1/8`, `3-4/8`, `5/8` prompts.
  - `g5`: batch size 2/4/8 mismatched `3/8`, `2/8`, `5/8` prompts.
- Do not batch science-critical branch-timing generations unless the paper
  explicitly treats batch shape as part of the runtime condition.

## Next Actions

1. Build a Branch Card artifact from an existing high-signal case:
   first branch, margins/top-k, replay/patch evidence, suspected prompt span,
   caveats, and stable-fix placeholder.
2. Build paper/product figures:
   single-case trajectory anatomy,
   Qwen branch-timing parallel coordinates,
   at-branch vs strict pre-branch AUROC,
   E07 best rescue position classes.
3. Add forced-prefix replay and prompt-delta bisect before more hidden-state
   patching. These are the practical primitives for API/server workflows.
4. Add focused negative controls:
   prompt-token-effective edits that do not branch,
   and replay-unstable branch cases.
5. Start a compact paper outline only after the first Branch Card and figures
   exist.

## Useful Commands

```bash
uv run python scripts/sagemaker_status.py --prefix chaos --max-results 60
uv run python scripts/process_activation_patch_queue.py --queue configs/sagemaker_queue_activation_patch_v5_replication.json --out-dir runs/rankings/activation_patch_v5_replication
uv run python scripts/compare_activation_patch_waves.py
uv run python scripts/process_logit_queue.py --queue configs/sagemaker_queue_logit_token_cert_v1.json --out-dir runs/rankings/logit_token_cert_v1
uv run python scripts/process_silent_divergence_queue.py --queue configs/sagemaker_queue_silent_divergence_pilot_v1.json --out-dir runs/rankings/silent_divergence_pilot_v1
uv run python scripts/build_trajectory_artifacts.py --trajectory-dir runs/trajectory_events/logit_token_cert_v1 --silent-summary runs/rankings/silent_divergence_local_qwen_ladder_meta_20260430/silent_divergence_readout.csv
```
