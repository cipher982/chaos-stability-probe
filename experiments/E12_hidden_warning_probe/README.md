# E12 Hidden Warning Probe

Question: do residual-stream distance features expose pre-branch warning that
the E09 logit-only analysis misses?

Initial state:
- E09 `logit_token_cert_v1` skipped hidden-state capture, so the broad
  525-pair panel cannot answer this directly from saved artifacts.
- E10 already captured common-prefix hidden/logit distances for five selected
  Qwen cases per model. That is useful as a design check, not as the final
  blog claim.

Commands:
- Score existing E10 captures:
  `uv run python experiments/E12_hidden_warning_probe/analyze_hidden_warning.py --summary runs/rankings/silent_divergence_pilot_v1/merged_silent_divergence_summary.csv --layers runs/rankings/silent_divergence_pilot_v1/merged_silent_divergence_layers.csv --out-dir runs/rankings/hidden_warning_probe_e10_sagemaker`
- Select broader recapture pairs:
  `uv run python experiments/E12_hidden_warning_probe/select_probe_pairs.py --model qwen35_08b --out-csv runs/hidden_warning_probe/qwen35_08b_pairs.csv --out-json runs/hidden_warning_probe/qwen35_08b_pair_ids.json`
- Launch broader recaptures:
  `uv run python scripts/dispatch_sagemaker_queue.py --queue configs/sagemaker_queue_hidden_warning_probe_v1.json --profile zh-marketing-preprod-aiengineer --max-active 2`

Outputs:
- `runs/rankings/hidden_warning_probe_*/hidden_warning_auc.csv`
- `runs/rankings/hidden_warning_probe_*/hidden_warning_best_layers.csv`
- selected recapture pair lists under `runs/hidden_warning_probe/`
- committed pair-list snapshots:
  `experiments/E12_hidden_warning_probe/qwen35_08b_pairs.csv` and
  `experiments/E12_hidden_warning_probe/qwen35_2b_pairs.csv`
- combined Qwen0.8B/2B analysis is supported by passing `--summary` and
  `--layers` twice to `analyze_hidden_warning.py`.

Current readout:
- E10 selected SageMaker cases show strong-looking hidden-distance warning
  (`~0.73-0.76` within 5/10 tokens), but that is a tiny hand-picked sample.
- Local Qwen0.8B/MPS 60-pair recapture is much weaker:
  `runs/rankings/hidden_warning_probe_local_qwen08_60pair_auc/` has best
  summary hidden AUROC around `0.56` for within 2/5/10-token windows, with
  best layer-picked exact-offset hidden features around `0.59-0.61`.
  This does not support a publishable "hidden states give early warning" claim
  on its own.

Caveats:
- Existing E10 captures save distances, not raw residual vectors, so they can
  test residual drift/norm-style features but not a full linear probe.
- Use strict pre-branch targets (`tokens_until_branch > 0`) for warning claims.
  The branch timestep itself is discrimination at the fork, not foresight.
