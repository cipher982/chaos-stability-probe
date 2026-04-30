# E06 Logit Boundary Probe

Question: do prompt-end and teacher-forced logit metrics explain later
generation divergence better than text distance alone?

Inputs:
- token-certified prompt-pair configs in `configs/prompt_pairs_token_certified/`
- `configs/sagemaker_queue_logit_token_cert_v1.json`
- completed panel runs with `logit_probes.jsonl`

Commands:
- `uv run python scripts/sagemaker_queue_supervisor.py --queue configs/sagemaker_queue_logit_token_cert_v1.json`
- `uv run python scripts/process_logit_queue.py --queue configs/sagemaker_queue_logit_token_cert_v1.json`

Outputs:
- raw SageMaker extracts under `runs/sagemaker_artifacts/`
- merged logit summaries under `runs/rankings/logit_token_cert_v1/`

Status: active mechanism wave. Leave launch queues in `configs/` for
compatibility with current supervisors and historical commands.
