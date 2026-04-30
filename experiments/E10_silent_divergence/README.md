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
  `runs/rankings/silent_divergence_local_qwen_ladder_meta_20260430/`
- raw SageMaker artifacts under `runs/sagemaker_artifacts/`

Status:
- Qwen9 SageMaker pilot completed and processed under
  `runs/rankings/silent_divergence_pilot_v1/`.
- Qwen2B/Qwen4B local captures were recaptured from commit `5128d18` with
  runtime metadata under
  `runs/silent_divergence_panel/local_qwen_ladder_meta_20260430/`.

Current readout: metadata-backed local Qwen2B and Qwen4B captures differ on
the same five selected branch cases. Qwen4B branches earlier than Qwen2B on
several cases, both branch at token 0 for the parenthesized-word case, and
Qwen2B has one no-visible-branch case in the 64-step logged window. Qwen9
SageMaker still branches earlier on the old artifact, but that artifact predates
`run_metadata.json`, so keep the larger-model timing caveated.

Queue state: the SageMaker Qwen2B/Qwen4B E10 entries are currently `not_found`;
Qwen9 is `processed` but missing runtime metadata.

Next: decide whether relaunching Qwen9 E10 with metadata adds enough beyond the
local captures. Keep the original queue in `configs/`; the copy here records the
experiment-owned config snapshot.
