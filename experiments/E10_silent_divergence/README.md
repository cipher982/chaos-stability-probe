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

Status:
- Qwen9 SageMaker pilot completed and processed under
  `runs/rankings/silent_divergence_pilot_v1/`.
- Qwen2B local capture completed under
  `runs/silent_divergence_panel/local_qwen2b_20260430/`.
- Qwen4B local capture completed under
  `runs/silent_divergence_panel/local_qwen4b_20260430/`.

Current readout: Qwen2B and Qwen9 differ on the same five selected branch
cases. Qwen2B branches later or not at all on several cases where Qwen9
branches immediately or by token 3. This makes branch timing itself a
model-dependent observable.

Next: merge the Qwen4B local readout, then decide whether launching the
SageMaker Qwen4B E10 job adds anything beyond the local capture. Keep the
original queue in `configs/`; the copy here records the experiment-owned config
snapshot.
