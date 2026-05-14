# E12 Hidden Warning Probe

Question: do residual-stream distance features expose pre-branch warning that
the E09 logit-only analysis misses?

Hypothesis:
- H1: paired residual-state deltas contain a branch-proximity signal before the
  visible token fork. The signal should be decodable from `hidden_b -
  hidden_a` several tokens before the branch, and should beat logits/JS and
  scalar residual drift features on strict pre-branch windows.
- H0: the local Qwen0.8B vector result is a separability artifact: edit
  category, generally unstable pairs, repeated horizon rows, backend details,
  or immediate branch cases make the probe look better than real forecasting.

Full-slate design:
- Models: Qwen3.5 0.8B/2B/4B/9B plus Gemma4 E2B/E4B instruct/base.
- Inputs: v2 pair lists with up to 96 long-prefix visible branches plus all
  available no-visible controls per model.
- Captures: last-token residual delta vectors at exact horizons
  `0/1/2/5/10/20/32/64/96/128`, with logits and scalar residual summaries
  from the same forwards.
- Primary target: strict `pre_branch_within_H`, where branch rows are positive
  only when `0 < tokens_until_branch <= H`; the branch step itself is not
  foresight.
- Primary probes: pair-grouped, pair-weighted mean-difference probes over raw
  and row-normalized residual deltas, layerwise.
- Controls: same-artifact JS/logit and scalar residual summaries; pair-hash
  splits; edit-category holdout splits; no-visible and far-from-branch rows as
  negatives; remove/impoverish exact-offset claims when controls are too
  imbalanced.
- Success bar: call this a publishable hidden-state warning result only if
  residual-vector probes beat logits/scalars by a meaningful margin at
  `H=2/5/10` on multiple models or at least multiple Qwen sizes, and survive
  category holdout.

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
- Launch full v3 cross-account vector recaptures:
  `uv run python scripts/dispatch_sagemaker_queue.py --queue configs/sagemaker_queue_hidden_warning_probe_v3_full_vectors.json --include-cross-account --continue-on-error`
- Score vector artifacts after download/extract:
  `uv run python experiments/E12_hidden_warning_probe/analyze_hidden_vectors.py --artifact-dir runs/sagemaker_artifacts/<job>/runs/<model> --out-dir runs/rankings/hidden_warning_vectors_<model>`
- Score nested layer-selection controls:
  `uv run python experiments/E12_hidden_warning_probe/analyze_hidden_vectors_nested.py --artifact-dir runs/sagemaker_artifacts/<job>/runs/<model> --out-dir runs/rankings/hidden_warning_vectors_<panel>_nested --horizon 1 --horizon 2 --horizon 5 --horizon 10 --split-mode pair_hash --split-mode category_holdout`

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
- v3 full vector queue:
  `configs/sagemaker_queue_hidden_warning_probe_v3_full_vectors.json`
- vector artifacts per model:
  `*_hidden_vector_features.csv` plus `*_hidden_vector_features.npz`
- full v3 vector readout:
  `runs/rankings/hidden_warning_vectors_v3_full_controls/`
- full v3 nested-layer readout:
  `runs/rankings/hidden_warning_vectors_v3_full_nested/`
- full v3 scalar/logit baseline:
  `runs/rankings/hidden_warning_scalar_v3_full/`
- combined Qwen0.8B/2B analysis is supported by passing `--summary` and
  `--layers` twice to `analyze_hidden_warning.py`.

Current readout:
- E10 selected SageMaker cases show strong-looking hidden-distance warning
  (`~0.73-0.76` within 5/10 tokens), but that is a tiny hand-picked sample.
- Local Qwen0.8B/MPS 60-pair recapture is much weaker:
  `runs/rankings/hidden_warning_probe_local_qwen08_60pair_auc/` has best
  summary hidden AUROC around `0.56` for within 2/5/10-token windows, with
  best layer-picked exact-offset hidden features around `0.59-0.61`.
  This does not support a scalar hidden-distance warning claim.
- Local Qwen0.8B/MPS 111-pair vector recapture is positive:
  `runs/rankings/hidden_warning_vectors_local_qwen08_full/` shows
  pair-grouped residual-delta probes at AUROC `0.74` within 1/2 tokens,
  `0.71` within 5, and `0.70` within 10. Same-artifact JS/logit and scalar
  residual-distance features are much weaker on strict pre-branch windows.
  This is the first evidence that monitoring the full state helps.
- Full v3 CUDA panel, 8/8 models processed:
  `runs/rankings/hidden_warning_vectors_v3_full_controls/` covers
  Qwen0.8B/2B/4B/9B plus Gemma E2B/E4B instruct/base, 915 prompt pairs total.
  Best-layer pair-weighted residual-vector AUROC on strict pre-branch warning
  windows is well above logits/scalars across every model. With pair-hash
  splits, median AUROC across models is `0.790` at H=1, `0.767` at H=2,
  `0.765` at H=5, and `0.738` at H=10. With edit-category holdout, the
  medians are `0.788`, `0.773`, `0.764`, and `0.740`. The weakest best-layer
  cases are the Gemma base models, but they remain above `0.68` on the main
  H=1/2/5/10 windows.
- The stronger nested-layer control also stays positive:
  `runs/rankings/hidden_warning_vectors_v3_full_nested/` chooses the layer
  inside each training fold before scoring the held-out pair/category fold.
  Pair-hash nested medians are `0.768` at H=1, `0.750` at H=2, `0.731` at
  H=5, and `0.719` at H=10. Category-holdout nested medians are `0.763`,
  `0.761`, `0.757`, and `0.728`. This reduces the best-layer inflation concern
  without removing the signal.
- Same-run logits/scalars remain weak on strict pre-branch windows:
  `runs/rankings/hidden_warning_scalar_v3_full/` has aggregate best summary
  AUROC `0.534` at H=1, `0.526` at H=2, `0.509` at H=5, and `0.527` at H=10.
  Per-model best summary AUROC medians are only `0.541`, `0.551`, `0.534`,
  and `0.535`; even best layer-picked scalar residual distances stay around
  `0.55` median. At the branch timestep, logit features still discriminate
  strongly (`0.821` effective branching factor, `0.782` JS), so the negative
  result is specifically about advance warning from final logits/scalar drift.
- Answer to the spike: yes, monitoring the full residual-state delta helped.
  The honest claim is not "we can predict branches far in advance from logits";
  it is "a supervised probe over paired residual-state deltas detects
  pre-branch proximity several tokens before the visible fork, while final
  logits and scalar hidden-distance summaries largely miss it."

Caveats:
- Existing E10 captures save distances, not raw residual vectors, so they can
  test residual drift/norm-style features but not a full linear probe.
- V2 vector captures save `hidden_b - hidden_a` deltas, not full clean and
  perturbed states. That is enough for warning probes over prompt-pair
  separation, but not for arbitrary post-hoc tuned-lens decoding.
- The vector probe is a supervised paired-run discriminator over
  `hidden_b - hidden_a`; it is not yet an online single-run branch predictor.
- Use strict pre-branch targets (`tokens_until_branch > 0`) for warning claims.
  The branch timestep itself is discrimination at the fork, not foresight.
- Use pair-grouped splits for vector probes; generated timesteps from the same
  prompt pair are not independent samples.
- Do not rely on current exact-offset AUROCs as a blog claim: the local
  exact-offset comparison has many branch positives and only a small no-branch
  control set. The within-window target is the more meaningful warning readout.
