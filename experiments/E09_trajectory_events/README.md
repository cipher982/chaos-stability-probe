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

Status: local seed check passed; promote only after the larger logit queue is
processed. Generated artifacts stay under `runs/trajectory_events/`.
