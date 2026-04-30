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
- merged SageMaker readouts under `runs/rankings/silent_divergence_pilot_v1/`
- raw SageMaker artifacts under `runs/sagemaker_artifacts/`

Status: SageMaker pilot is active. Keep the original queue in `configs/`;
the copy here records the experiment-owned config snapshot.
