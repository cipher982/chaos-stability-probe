# E08 SAE Feature Pilot

Question: at causal branch positions from E07, do Qwen-Scope SAE feature sets
change in a way that matches the patching story?

Inputs:
- selected Qwen3.5 2B branch cases from E07
- `configs/prompt_pairs_mechinterp_seed.json`
- Qwen-Scope residual-stream SAE checkpoints

Commands:
- `uv run python scripts/extract_qwen_sae_branch_features.py ...`
- `uv run python scripts/summarize_sae_feature_deltas.py ...`

Outputs:
- feature rows and overlap summaries under `runs/mechinterp_sae/`

Status: feature-ID pilot, not human-labeled interpretation. Keep claims tied
to overlap/delta evidence at causal branch positions.
Generated artifacts stay under `runs/`, not this experiment directory.
