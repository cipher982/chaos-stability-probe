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
- `uv run python scripts/process_silent_divergence_queue.py`

Outputs:
- per-model capture CSVs under `runs/silent_divergence_pilot/`
- `run_metadata.json` beside new captures, recording resolved backend/device/
  dtype, model ID, torch/transformers versions, git SHA, and dirty-worktree
  state
- merged SageMaker readouts under `runs/rankings/silent_divergence_pilot_v1/`
- local Qwen ladder readout under
  `runs/rankings/silent_divergence_local_qwen_ladder/`
- raw SageMaker artifacts under `runs/sagemaker_artifacts/`

Status:
- Qwen9 SageMaker pilot completed and processed under
  `runs/rankings/silent_divergence_pilot_v1/`.
- Qwen2B local capture completed under
  `runs/silent_divergence_panel/local_qwen2b_20260430/`.
- Qwen4B local capture completed under
  `runs/silent_divergence_panel/local_qwen4b_20260430/`.

Current readout: Qwen2B, Qwen4B, and Qwen9 differ on the same five selected
branch cases, but the old captures predate `run_metadata.json`. Treat the
cross-model branch timing as exploratory until the key comparisons are
recaptured or otherwise backed by resolved runtime metadata.

Queue state: the SageMaker Qwen2B/Qwen4B E10 entries are currently `not_found`;
Qwen9 is `processed` but missing runtime metadata.

Next: recapture the key local comparisons with metadata, then decide whether
launching SageMaker E10 adds anything beyond local captures. Keep the original
queue in `configs/`; the copy here records the experiment-owned config snapshot.
