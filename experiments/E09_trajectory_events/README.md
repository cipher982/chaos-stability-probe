# E09 Trajectory Events

Question: are large downstream output differences concentrated around
localized visible or silent branch events?

Inputs:
- completed runs with `summary.csv` or `summary_with_semantic.csv`
- matching `logit_probes.jsonl`
- usually token-certified E06 logit jobs

Commands:
- `uv run python scripts/analyze_trajectory_events.py RUN_DIR ... --out-dir runs/trajectory_events/NAME`
- `uv run python scripts/build_trajectory_artifacts.py --trajectory-dir runs/trajectory_events/NAME --out-root runs/trajectory_artifacts`
- `uv run python scripts/analyze_branch_prediction.py runs/trajectory_events/NAME/branch_prediction_windows.csv --bootstrap-samples 200`
  bootstraps aggregate AUROC by prompt-pair group; add `--bootstrap-scope
  groups` only when per-model confidence intervals are needed.
- `uv run python scripts/select_trajectory_cases.py --events runs/trajectory_events/NAME/trajectory_events.csv --windows runs/trajectory_events/NAME/branch_prediction_windows.csv --out-dir runs/trajectory_case_selection/NAME`

Outputs:
- `run_metadata.json` with input hashes, build arguments, git commit, and dirty
  state
- `trajectory_events.csv`
- `trajectory_event_summary.csv`
- `branch_prediction_windows.csv`
- `branch_prediction/branch_prediction_auc.csv`
- `branch_prediction_long_prefix/branch_prediction_auc.csv` for visible-branch
  cases whose branch timestep is at least the configured long-prefix minimum
- `case_candidates.csv` and `recommended_cases.csv` for figure/story selection
- `model_comparison/` with paired branch timing and Qwen ladder comparisons
- optional HTML casebook under `runs/casebooks/...`

Inspection:
- `uv run python scripts/render_trajectory_casebook.py --events runs/trajectory_events/NAME/trajectory_events.csv --windows runs/trajectory_events/NAME/branch_prediction_windows.csv --out-dir runs/casebooks/NAME`
- Add `--figure-dir runs/figures/NAME` to emit one standalone HTML panel per
  selected case plus a `manifest.csv`.
- Add `--cases-from runs/trajectory_case_selection/NAME/recommended_cases.csv`
  to render the exact recommended rows from case selection.

Status:
- Higher-N logit-token readout currently includes Qwen3.5 0.8B, Qwen3.5 2B,
  Qwen3.5 4B, Qwen3.5 9B, and Gemma4 E2B instruct.
- Current output is `runs/trajectory_events/logit_token_cert_v1/`; bundled
  branch-prediction/casebook/comparison artifacts are under
  `runs/trajectory_artifacts/logit_token_cert_v1/`.
- `branch_within_N` is a decision-window target and includes the branch
  timestep itself. In `branch_prediction_auc.csv`, this is labeled
  `branch_window_including_branch`.
- `pre_branch_within_N` is the strict warning target and excludes the branch
  timestep. In `branch_prediction_auc.csv`, this is labeled
  `strict_pre_branch_warning`.
- At-branch classification is strong: low-margin AUROC `0.953`
  (`0.951-0.956` clustered CI), JS AUROC `0.891` (`0.885-0.898`).
- Decision-window branch-within-1 is moderate: JS AUROC `0.766`
  (`0.758-0.774`), low-margin AUROC `0.746` (`0.740-0.755`).
- Pure pre-branch-within-1 warning is weaker: centered-logit-L2 AUROC `0.649`
  (`0.638-0.661`), JS AUROC `0.620` (`0.605-0.634`).
- On the long-prefix subset (`branch_t >= 5`), pure pre-branch-within-1 warning
  weakens further: centered-logit-L2 AUROC `0.568` (`0.554-0.579`), JS AUROC
  `0.558` (`0.545-0.570`).
- Paired Qwen ladder branch timing is not monotonic: only `10.4%` of shared
  non-control cases are monotonic earlier with size, and only `10.4%` are
  monotonic later with size.

Caveat: the warning threshold often fires at `t=0`, so use this as a branch
risk detector, not yet a precise "silent lead time" claim. Do not call
`branch_within_1` a one-token-ahead metric.

Next: rerun after the newly launched Gemma base/E4B logit jobs finish.
