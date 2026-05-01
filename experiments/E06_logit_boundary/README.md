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

Current readout:
- `runs/rankings/logit_token_cert_v1/semantic_vs_prompt_end_logits.csv`
- Complete processed panel: Qwen0.8B/2B/4B/9B plus Gemma E2B/E4B
  instruct/base.
- Gemma base models are the prompt-end outliers:
  - Gemma E2B base: JS `0.0618`, top-1 flip `0.164`, mean BF `15.6`.
  - Gemma E4B base: JS `0.0452`, top-1 flip `0.303`, mean BF `16.0`.
- Gemma instruct models are much more concentrated:
  - E2B instruct: JS `0.00637`, top-1 flip `0.011`, mean BF `1.23`.
  - E4B instruct: JS `0.01025`, top-1 flip `0.067`, mean BF `1.53`.
- Qwen remains non-monotonic across size: Qwen9 has lower prompt-end top-1
  flip than Qwen4, while Qwen0.8B has the widest effective support.

Status: processed mechanism wave. Leave launch queues in `configs/` for
compatibility with current supervisors and historical commands.
