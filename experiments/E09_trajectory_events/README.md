# E09 Trajectory Events

Question: are large downstream output differences concentrated around
localized visible or silent branch events?

Inputs:
- completed runs with `summary.csv` or `summary_with_semantic.csv`
- matching `logit_probes.jsonl`
- usually token-certified E06 logit jobs

Commands:
- `uv run python scripts/analyze_trajectory_events.py RUN_DIR ... --out-dir runs/trajectory_events/NAME`
- `uv run python scripts/analyze_branch_prediction.py runs/trajectory_events/NAME/branch_prediction_windows.csv`

Outputs:
- `trajectory_events.csv`
- `trajectory_event_summary.csv`
- `branch_prediction_windows.csv`
- `branch_prediction_auc.csv`
- optional HTML casebook under `runs/casebooks/...`

Inspection:
- `uv run python scripts/render_trajectory_casebook.py --events runs/trajectory_events/NAME/trajectory_events.csv --windows runs/trajectory_events/NAME/branch_prediction_windows.csv --out-dir runs/casebooks/NAME`

Status:
- Higher-N logit-token readout currently includes Qwen3.5 0.8B, Qwen3.5 2B,
  Qwen3.5 9B, and Gemma4 E2B instruct.
- Current output is `runs/trajectory_events/logit_token_cert_v1/`.
- `branch_within_N` is a decision-window target and includes the branch
  timestep itself.
- At-branch classification is strong: low-margin AUROC `0.950`, JS AUROC
  `0.891`.
- Decision-window branch-within-1 is moderate: JS AUROC `0.759`, low-margin
  AUROC `0.739`.
- Pure pre-branch-within-1 warning is weaker: centered-logit-L2 AUROC `0.650`,
  JS AUROC `0.607`.

Caveat: the warning threshold often fires at `t=0`, so use this as a branch
risk detector, not yet a precise "silent lead time" claim. Do not call
`branch_within_1` a one-token-ahead metric.

Next: rerun after Qwen4B and the newly launched Gemma logit jobs finish.
