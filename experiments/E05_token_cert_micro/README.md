# E05 Token-Certified Micro Perturbations

Question: which tiny edits still exist after tokenizer and chat-template
formatting, and how much do those effective prompt-token perturbations move
deterministic outputs?

Inputs:
- `configs/prompt_pairs_token_certified/*.json`
- `configs/sagemaker_queue_token_certified_v3.json`
- model registry entries in `configs/models.json`

Commands:
- `uv run python scripts/make_token_certified_micro_pairs.py`
- `uv run python scripts/sagemaker_queue_supervisor.py --queue configs/sagemaker_queue_token_certified_v3.json`
- `uv run python scripts/process_token_micro_queue.py --queue configs/sagemaker_queue_token_certified_v3.json --rank-dir runs/rankings/token_micro_v3`

Outputs:
- raw job artifacts under `runs/sagemaker_artifacts/`
- derived rankings under `runs/rankings/token_micro_v3/`
- token-certified prompt-pair configs under `configs/prompt_pairs_token_certified/`

Status: active measurement hygiene thread. Keep queue/config paths in
`configs/` while jobs or docs reference them.
