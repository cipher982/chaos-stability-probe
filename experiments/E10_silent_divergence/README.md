# E10 Silent Divergence

Question: do hidden states or logits separate along a shared generated prefix
before the visible branch token appears?

Inputs:
- `configs/prompt_pairs_mechinterp_seed.json`
- selected `pair_ids` from replayable E07 branch cases
- `configs/sagemaker_queue_silent_divergence_pilot_v1.json`

Commands:
- `uv run python scripts/capture_silent_divergence.py --model qwen35_2b`
- `uv run python scripts/run_silent_divergence_panel.py --model qwen35_2b`
- `uv run python scripts/build_silent_divergence_readout.py --capture-root runs/silent_divergence_panel/NAME --out-dir runs/rankings/NAME`
- `uv run python scripts/process_silent_divergence_queue.py`
- `uv run python scripts/compare_silent_divergence_backends.py --left runs/rankings/LOCAL/silent_divergence_readout.csv --right runs/rankings/SAGEMAKER/silent_divergence_readout.csv --left-label local --right-label sagemaker --out-dir runs/rankings/e10_backend_comparison`

Outputs:
- per-model capture CSVs under `runs/silent_divergence_pilot/`
- new captures include effective branching factor columns for both prompt sides
  at each shared-prefix step
- `run_metadata.json` beside new captures, recording resolved backend/device/
  dtype, model ID, torch/transformers versions, git SHA, and dirty-worktree
  state
- merged SageMaker readouts under `runs/rankings/silent_divergence_pilot_v1/`
- local Qwen ladder readout under
  `runs/rankings/silent_divergence_local_qwen_ladder_meta_20260430/`
- raw SageMaker artifacts under `runs/sagemaker_artifacts/`

Status:
- Qwen2B/Qwen4B/Qwen9 SageMaker recaptures completed and processed under
  `runs/rankings/silent_divergence_pilot_v1/`.
- Qwen2B/Qwen4B local captures were recaptured from commit `5128d18` with
  runtime metadata under
  `runs/silent_divergence_panel/local_qwen_ladder_meta_20260430/`.

Current readout: metadata-backed local Qwen2B/Qwen4B captures and SageMaker
CUDA/bfloat16 recaptures differ on the same five selected branch cases. Local
vs SageMaker mean absolute branch-t delta is `4.25` for Qwen2B and `8.80` for
Qwen4B, so branch timing claims need backend/dtype caveats. Under SageMaker
CUDA/bfloat16, mean visible branch-t on the five selected cases is `9.0` for
Qwen2B over four visible-branch cases, `7.8` for Qwen4B, and `1.8` for Qwen9.
Older captures will show blank/NaN effective branching factor in readouts
because the raw capture rows predate that field.

Next: use these cases as intervention targets, not as a general Qwen scaling
claim. Keep the original queue in `configs/`; the copy here records the
experiment-owned config snapshot.
