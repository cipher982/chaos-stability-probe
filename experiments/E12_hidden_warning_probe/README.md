# E12 Hidden Warning Probe

Question: do residual-stream distance features expose pre-branch warning that
the E09 logit-only analysis misses?

Initial state:
- E09 `logit_token_cert_v1` skipped hidden-state capture, so the broad
  525-pair panel cannot answer this directly from saved artifacts.
- E10 already captured common-prefix hidden/logit distances for five selected
  Qwen cases per model. That is useful as a design check, not as the final
  blog claim.
- Prior art argues for looking below logits: tuned-lens/future-lens style
  work decodes intermediate hidden states into present or future-token
  predictions, while causal tracing/state-tracking work warns that probes
  should be treated as evidence of available information, not the mechanism.

Commands:
- Score existing E10 captures:
  `uv run python experiments/E12_hidden_warning_probe/analyze_hidden_warning.py --summary runs/rankings/silent_divergence_pilot_v1/merged_silent_divergence_summary.csv --layers runs/rankings/silent_divergence_pilot_v1/merged_silent_divergence_layers.csv --out-dir runs/rankings/hidden_warning_probe_e10_sagemaker`
- Select broader recapture pairs:
  `uv run python experiments/E12_hidden_warning_probe/select_probe_pairs.py --model qwen35_08b --out-csv runs/hidden_warning_probe/qwen35_08b_pairs.csv --out-json runs/hidden_warning_probe/qwen35_08b_pair_ids.json`
- Launch broader recaptures:
  `uv run python scripts/dispatch_sagemaker_queue.py --queue configs/sagemaker_queue_hidden_warning_probe_v1.json --profile zh-marketing-preprod-aiengineer --max-active 2`
- Launch cross-account vector recaptures:
  `uv run python scripts/dispatch_sagemaker_queue.py --queue configs/sagemaker_queue_hidden_warning_probe_v2_vectors.json --include-cross-account --continue-on-error`
- Score vector artifacts after download/extract:
  `uv run python experiments/E12_hidden_warning_probe/analyze_hidden_vectors.py --artifact-dir runs/sagemaker_artifacts/<job>/runs/<model> --out-dir runs/rankings/hidden_warning_vectors_<model>`

Outputs:
- `runs/rankings/hidden_warning_probe_*/hidden_warning_auc.csv`
- `runs/rankings/hidden_warning_probe_*/hidden_warning_best_layers.csv`
- selected recapture pair lists under `runs/hidden_warning_probe/`
- committed pair-list snapshots:
  `experiments/E12_hidden_warning_probe/qwen35_08b_pairs.csv` and
  `experiments/E12_hidden_warning_probe/qwen35_2b_pairs.csv`
- v2 committed pair-list snapshots:
  `experiments/E12_hidden_warning_probe/*_pairs_v2.csv`
- v2 vector queue:
  `configs/sagemaker_queue_hidden_warning_probe_v2_vectors.json`
- vector artifacts per model:
  `*_hidden_vector_features.csv` plus `*_hidden_vector_features.npz`
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
- A v2 cross-account SageMaker wave is in flight as of 2026-05-14. It captures
  float16 last-token residual deltas at exact horizons
  `0/1/2/5/10/20/32/64/96/128`, using sparse branch-relative forwards to keep
  compute tractable. The queue covers the 8-model Qwen/Gemma panel with
  larger pair selections (`96` visible long-prefix targets plus available
  no-visible controls per model).

Caveats:
- Existing E10 captures save distances, not raw residual vectors, so they can
  test residual drift/norm-style features but not a full linear probe.
- V2 vector captures save `hidden_b - hidden_a` deltas, not full clean and
  perturbed states. That is enough for warning probes over prompt-pair
  separation, but not for arbitrary post-hoc tuned-lens decoding.
- Use strict pre-branch targets (`tokens_until_branch > 0`) for warning claims.
  The branch timestep itself is discrimination at the fork, not foresight.
- Use pair-grouped splits for vector probes; generated timesteps from the same
  prompt pair are not independent samples.
