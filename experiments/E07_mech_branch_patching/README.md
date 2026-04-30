# E07 Mechanistic Branch Patching

Question: can residual activation patching move selected tiny-edit branch
tokens, and where is the strongest causal position/layer signal?

Inputs:
- `configs/prompt_pairs_mechinterp_seed.json`
- branch-point outputs from `scripts/analyze_branch_points.py`
- selected targets in `runs/mechinterp_patch/`

Commands:
- `uv run python scripts/analyze_branch_points.py ...`
- `uv run python scripts/select_patch_targets.py`
- `uv run python scripts/activation_patch_branch.py --positions aligned ...`
- `uv run python scripts/summarize_patch_results.py`

Outputs:
- patch CSVs and heatmaps under `runs/mechinterp_patch*/`

Status: local causal pilot. Scripts remain in `scripts/` for now because later
E08 scripts import them directly.
