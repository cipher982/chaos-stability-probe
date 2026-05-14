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

Last checked: 2026-05-14 after launching the E12 hidden-warning vector wave.

- Active SageMaker jobs:
  - `chaos-hidden-warning-qwen08-20260514-001` (`ml.g6e.2xlarge`, preprod)
  - `chaos-hidden-warning-qwen2b-20260514-001` (`ml.g6e.2xlarge`, preprod)
  - `chaos-hidden-warning-vectors-qwen9b-20260514-001` (`ml.g6e.2xlarge`, preprod)
  - `chaos-hidden-warning-vectors-qwen4b-20260514-001` (`ml.g6e.2xlarge`, preprod)
  - `chaos-hidden-warning-vectors-gemma-e4b-base-20260514-001` (`ml.g6e.2xlarge`, preprod)
  - `chaos-hidden-warning-vectors-qwen2b-20260514-001` (`ml.g5.2xlarge`, ML prod)
  - `chaos-hidden-warning-vectors-qwen08-20260514-001` (`ml.g4dn.2xlarge`, ML prod)
  - `chaos-hidden-warning-vectors-gemma-e2b-it-20260514-001` (`ml.g5.2xlarge`, marketing prod)
  - `chaos-hidden-warning-vectors-gemma-e2b-base-20260514-001` (`ml.g4dn.2xlarge`, marketing prod)
  - `chaos-hidden-warning-vectors-gemma-e4b-it-20260514-001` (`ml.g5.2xlarge`, QA)
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

### E12 Hidden Warning Probe

Artifacts:

- `experiments/E12_hidden_warning_probe/`
- `configs/sagemaker_queue_hidden_warning_probe_v1.json`
- `configs/sagemaker_queue_hidden_warning_probe_v2_vectors.json`
- `runs/rankings/hidden_warning_probe_e10_sagemaker/`
- active scalar jobs `chaos-hidden-warning-qwen08-20260514-001` and
  `chaos-hidden-warning-qwen2b-20260514-001`
- active vector jobs listed in Live Operations above

Readout:

- E09 skipped hidden capture, so the broad logit-token panel cannot answer
  hidden-state warning from saved artifacts.
- Existing E10 selected cases are only a smoke test, but hidden-distance
  features beat JS on strict pre-branch windows in that tiny sample.
- A broader local Qwen0.8B/MPS 60-pair sanity run is much weaker:
  best summary hidden AUROC is about `0.56` for within 2/5/10-token warning
  windows; best layer-picked exact-offset hidden features are about `0.59-0.61`.
  Treat this as evidence against scalar hidden-distance monitoring.
- A larger local Qwen0.8B/MPS 111-pair vector recapture is positive:
  pair-grouped mean-difference probes over residual deltas reach AUROC `0.74`
  within 1/2 tokens, `0.71` within 5, and `0.70` within 10. Same-artifact
  JS/logit and scalar residual-distance features are much weaker on strict
  pre-branch windows.
- The vector wave uses larger v2 pair lists across the 8-model Qwen/Gemma
  panel and captures float16 residual-delta vectors at exact horizons
  `0/1/2/5/10/20/32/64/96/128`. This is the first run that can support a real
  residual-vector probe rather than scalar drift AUROCs.

Next:

- Pull both jobs when complete, process with `scripts/build_silent_divergence_readout.py`,
  then score with `experiments/E12_hidden_warning_probe/analyze_hidden_warning.py`.
- Pull vector jobs when complete and score with
  `experiments/E12_hidden_warning_probe/analyze_hidden_vectors.py` using
  pair-grouped splits.
- Use the resulting within-window AUROCs to decide whether the blog can claim
  a narrow residual-vector warning signal. Do not rely on exact-offset numbers
  until the no-branch controls are large enough; the current local exact-offset
  comparison is class-imbalanced.

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

Current E11 state: paper synthesis is the active track. `paper/draft.md` is in
submission-ready markdown form and the interactive signature lab has been
rewritten as a public explainer (see E11 row in `experiment_index.md`).

Open decisions (not work) before the next step can start:

- Workshop venue / LaTeX template. Blocks step 2 and sets the deadline.
- Public BranchTrace release scope. Step 3 is conditional.
- Whether to automate case selection in `branchtrace/signature_lab.py`
  (currently three hand-picked pair IDs in `CASE_SPECS`). Not a submission
  blocker; makes future regens self-correcting.

Next concrete steps, in order:

1. Read `paper/draft_review.pdf` end-to-end and note anything to fix in
   `paper/draft.md`.
2. Pick the workshop LaTeX template, then convert `paper/draft.md`.
3. Tag/release BranchTrace examples if public cards are still useful.
4. Submit the paper.

E12 follow-up command skeleton after the hidden-warning jobs finish:

```bash
uv run python scripts/download_sagemaker_artifact.py chaos-hidden-warning-qwen08-20260514-001 --extract
uv run python scripts/download_sagemaker_artifact.py chaos-hidden-warning-qwen2b-20260514-001 --extract
uv run python scripts/build_silent_divergence_readout.py --capture-root runs/sagemaker_artifacts/chaos-hidden-warning-qwen08-20260514-001/runs --out-dir runs/rankings/hidden_warning_probe_qwen08
uv run python scripts/build_silent_divergence_readout.py --capture-root runs/sagemaker_artifacts/chaos-hidden-warning-qwen2b-20260514-001/runs --out-dir runs/rankings/hidden_warning_probe_qwen2b
uv run python experiments/E12_hidden_warning_probe/analyze_hidden_warning.py --summary runs/rankings/hidden_warning_probe_qwen08/merged_silent_divergence_summary.csv --layers runs/rankings/hidden_warning_probe_qwen08/merged_silent_divergence_layers.csv --summary runs/rankings/hidden_warning_probe_qwen2b/merged_silent_divergence_summary.csv --layers runs/rankings/hidden_warning_probe_qwen2b/merged_silent_divergence_layers.csv --out-dir runs/rankings/hidden_warning_probe_qwen08_qwen2b_auc

# Vector jobs need their launch profile when downloading cross-account artifacts.
uv run python scripts/download_sagemaker_artifact.py chaos-hidden-warning-vectors-qwen2b-20260514-001 --profile zh-ml-productionengineer --extract
uv run python experiments/E12_hidden_warning_probe/analyze_hidden_vectors.py --artifact-dir runs/sagemaker_artifacts/chaos-hidden-warning-vectors-qwen2b-20260514-001/runs/qwen35_2b --out-dir runs/rankings/hidden_warning_vectors_qwen2b
```

Rebuild the interactive lab from artifacts with:

```bash
uv run python -m branchtrace.cli signature-lab --repo-root .
```

The lab data bundle is `paper/signature_lab_data.json` (`signature-lab/0.1`).
It is generated from the activation-patching comparison table, trajectory
events, logit probes, patch CSV/JSON files, and SAE pilot CSVs. Do not hand-edit
the embedded HTML data.

Session-note archive for longer-running epics lives under
`~/git/obsidian_vault/AI-Sessions/` (e.g. the 2026-05-13 branch-signatures
public-explainer epic). The session notes hold context that would bloat these
docs.

## Useful Commands

```bash
uv run python scripts/sagemaker_status.py --prefix chaos --max-results 60
uv run python scripts/process_activation_patch_queue.py --queue configs/sagemaker_queue_activation_patch_v5_replication.json --out-dir runs/rankings/activation_patch_v5_replication
uv run python scripts/compare_activation_patch_waves.py
uv run python scripts/process_logit_queue.py --queue configs/sagemaker_queue_logit_token_cert_v1.json --out-dir runs/rankings/logit_token_cert_v1
uv run python scripts/process_silent_divergence_queue.py --queue configs/sagemaker_queue_silent_divergence_pilot_v1.json --out-dir runs/rankings/silent_divergence_pilot_v1
uv run python scripts/build_trajectory_artifacts.py --trajectory-dir runs/trajectory_events/logit_token_cert_v1 --silent-summary runs/rankings/silent_divergence_local_qwen_ladder_meta_20260430/silent_divergence_readout.csv
```
