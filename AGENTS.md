# AGENTS.md

## Project Overview

`chaos-stability-probe` is an exploratory Python/Transformers harness for a
talk on LLM dynamical sensitivity, prompt perturbations, logit/hidden-state
divergence, and chaos as a conceptual lens.

## Build & Test

| Action | Command |
| --- | --- |
| Install dependencies | `uv sync` |
| Run local smoke | `uv run python scripts/run_stability_probe.py --smoke --limit-pairs 3 --max-new-tokens 48 --out-dir runs/local_smoke` |
| Plot a run | `uv run python scripts/plot_results.py runs/local_smoke` |
| Inspect a run | `uv run python scripts/inspect_run.py runs/local_smoke` |
| Compile scripts | `uv run python -m py_compile scripts/*.py` |
| Process robust wave | `uv run python scripts/process_robust_wave.py` |
| Process logit wave | `uv run python scripts/process_logit_wave.py` |
| Process sampling demo | `uv run python scripts/process_sampling_demo.py` |

## Source Of Truth

Read in this order:

1. `AGENTS.md` - stable agent entry point and boundaries.
2. `README.md` - command overview and repo layout.
3. `docs/results_digest.md` - current talk readout and safest claims.
4. `docs/task_list.md` - operational next actions.
5. `docs/experiment_journal.md` - chronological lab notebook and historical context.

Do not encode live experiment status in `AGENTS.md`. Put dynamic state in
`docs/task_list.md` or append-only notes in `docs/experiment_journal.md`.

## Current-State Config

- `configs/models.json` is the canonical model registry.
  - `observed_behavior` records inspected scaffold/template/reasoning-prefix
    behavior for this repo's harness.
  - `unknown` means uninspected, not non-reasoning.
- `configs/sagemaker_queue.json` is the ordered SageMaker queue.
- Prompt sets live in `configs/prompt_pairs*.json`.

## Architecture

- `scripts/run_stability_probe.py` runs one model against prompt pairs and
  writes raw generations, summaries, curves, hidden states, and optional logits.
- `scripts/run_panel.py` runs multiple models locally with isolation.
- `scripts/launch_sagemaker_panel.py`, `scripts/dispatch_sagemaker_queue.py`,
  and `scripts/download_sagemaker_artifact.py` operate SageMaker GPU jobs.
- `scripts/process_*.py` scripts pull artifacts and generate analysis outputs
  under `runs/rankings/`.
- `talk/` contains slide and visualization artifacts.

## SageMaker

Use the repo scripts instead of raw AWS job creation. Before assuming capacity
or job completion, query live SageMaker state:

```bash
uv run python scripts/sagemaker_status.py --prefix chaos --max-results 30
```

Dispatch queued work with:

```bash
uv run python scripts/dispatch_sagemaker_queue.py --max-active 5
```

When adding jobs, prefer high-signal captures per model load: more prompts,
longer continuations, hidden states, logits, and raw token/text artifacts.

## Conventions

- Use `uv` for Python commands.
- Keep changes small and directly tied to the talk or experiment harness.
- Prefer structured JSON/CSV artifacts over ad hoc text parsing.
- Treat prompt pair as the statistical unit; generated tokens are not
  independent samples.
- Distinguish raw-output metrics from scaffold/content-only metrics.
- Qualify cross-model stability numbers by observed reasoning/scaffold behavior:
  raw/full-output stability can measure deterministic scaffolds (`<think>`,
  `Thinking Process:`, visible deliberation) rather than answer-content
  robustness. Do not compare scaffolded and non-scaffolded models naively.
- Label heuristic boundary detection as heuristic; do not hide failed
  scaffold/answer boundaries.

## Documentation Boundaries

- `AGENTS.md`: stable instructions only.
- `README.md`: command and layout overview.
- `docs/results_digest.md`: compact current interpretation.
- `docs/task_list.md`: current operational state.
- `docs/experiment_journal.md`: append-only experiment history.
- `docs/rebuttals.md`: talk Q&A and objections.
- `docs/prior_art.md`: citations and conceptual anchors.

Avoid creating new top-level docs unless the existing files cannot reasonably
hold the information.

## Agent Boundaries

Always:

- Read the relevant source-of-truth docs before changing framing or experiments.
- Check live SageMaker state before discussing queues or GPU utilization.
- Sanity-check raw `generations.jsonl` examples when interpreting results.
- Preserve raw artifacts; analysis can be recomputed later.
- For small `talk/slides.md` edits, keep QA scoped: render the deck if needed,
  then inspect only the affected slide image(s). Do not run full-deck visual QA
  or fixer agents unless the change is deck-wide, changes slide numbering/theme,
  or the user explicitly asks.

Ask first:

- Changing the core metric definitions.
- Deleting or overwriting run artifacts.
- Adding large new dependency families or switching inference backends.

Never:

- Store secrets or credentials in this repo.
- Treat derived CSVs as more authoritative than raw generations plus config.
- Present scaffold/content heuristics as ground truth without confidence labels.
- Put transient queue status or speculative conclusions in `AGENTS.md`.
