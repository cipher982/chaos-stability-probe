# AGENTS.md

## Project Overview

`chaos-stability-probe` is an exploratory Python/Transformers harness for a
talk on LLM dynamical sensitivity, prompt perturbations, logit/hidden-state
divergence, and chaos as a conceptual lens.

## Audience & Tone

This is **not** a research paper, symposium, thesis defense, or publication.
The talk is a **lightweight casual presentation to the AI/ML team at work** —
a mix of analysts, engineers, frontend folks, and data scientists who happen
to work on AI projects. Friendly, exploratory, "look at this weird thing I
found" energy. Some past team talks have been kinda goofy.

Implications for agent behavior on this repo:

- Bias toward **keeping interesting findings** even if weakly defensible,
  with honest caveats ("n=6", "one prompt family", "didn't reproduce Li et
  al. on my setup"). Don't strip 90% of the fun to eke out 5% more rigor.
- Frame broken hypotheses as part of the story (the reveal is fun) rather
  than quietly deleting them.
- Favor visually striking / conceptually surprising vizzes over
  publication-grade ones. If a synthetic viz is clearly labeled
  illustrative, that's fine.
- Be direct in writeups and docs, but don't assume a hostile reviewer.

## Build & Test

| Action | Command |
| --- | --- |
| Install dependencies | `uv sync` |
| Run local smoke | `uv run python scripts/run_stability_probe.py --smoke --limit-pairs 3 --max-new-tokens 48 --out-dir runs/local_smoke` |
| Plot a run | `uv run python scripts/plot_results.py runs/local_smoke` |
| Inspect a run | `uv run python scripts/inspect_run.py runs/local_smoke` |
| Compile scripts | `uv run python -m py_compile scripts/*.py experiments/E*/*.py` |
| Process robust wave | `uv run python scripts/process_robust_wave.py` |
| Process logit wave | `uv run python scripts/process_logit_wave.py` |
| Process sampling demo | `uv run python scripts/process_sampling_demo.py` |

## Source Of Truth

Read in this order:

1. `AGENTS.md` - stable agent entry point and boundaries.
2. `README.md` - command overview and repo layout.
3. `docs/results_digest.md` - current talk readout and safest claims.
4. `docs/experiment_index.md` - one-row-per-experiment tracker.
5. `docs/task_list.md` - operational next actions.
6. `experiments/E##_*/README.md` - experiment-specific restart notes.

Do not encode live experiment status in `AGENTS.md`. Put dynamic queue/current
work state in `docs/task_list.md`, compact experiment state in
`docs/experiment_index.md`, and restart details in the relevant
`experiments/E##_*/README.md`.

## Current-State Config

- `configs/models.json` is the canonical model registry.
  - `observed_behavior` records inspected scaffold/template/reasoning-prefix
    behavior for this repo's harness.
  - `unknown` means uninspected, not non-reasoning.
- `configs/sagemaker_queue*.json` holds active/compatibility SageMaker queues.
- Prompt sets live in `configs/prompt_pairs*.json`.

## Architecture

- `scripts/run_stability_probe.py` runs one model against prompt pairs and
  writes raw generations, summaries, curves, hidden states, and optional logits.
- `scripts/run_panel.py` runs multiple models locally with isolation.
- `scripts/launch_sagemaker_panel.py`, `scripts/dispatch_sagemaker_queue.py`,
  and `scripts/download_sagemaker_artifact.py` operate SageMaker GPU jobs.
- `scripts/process_*.py` scripts pull artifacts and generate analysis outputs
  under `runs/rankings/`.
- `experiments/E##_*/` holds committed experiment-specific code/config
  snapshots and short restart notes. Keep stable command shims in `scripts/`
  when docs, queues, or SageMaker jobs reference those paths.
- Generated outputs stay under `runs/`, not `experiments/`.
- `talk/` contains slide and visualization artifacts. `talk/slides.md` is the
  Marp source of truth; present from `talk/browser.html`, which displays
  high-resolution PNGs from `talk/slide_images/`.

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

## Experiment Design Protocol

Every experiment thread must have exactly one compact row in
`docs/experiment_index.md`. Keep that table high signal: question, status,
primary artifacts, current readout, and next decision.

Put experiment-specific restart notes in `experiments/E##_*/README.md`:
question, inputs, commands, outputs, current readout, caveats, and next action.
Do not append long chronology to `docs/archive/experiment_journal_legacy.md`;
it is historical only.

Before launching new compute, check whether the proposed run answers a current
open question in `docs/results_digest.md` or `docs/task_list.md`. Prefer
discriminating tests over broader leaderboards:

- controls that can falsify the current interpretation;
- paired comparisons within the same model family or recipe;
- token-certified prompt perturbations instead of raw character edits;
- replayable branch cases before SAE/feature interpretation;
- negative controls that change prompt tokens but do not change behavior.

When cleaning up prior work, do not rewrite or delete the raw chronology. Add
structured summaries, index entries, and decision notes that point to existing
artifacts. Treat raw run directories and `generations.jsonl` as the audit log;
derived CSVs and charts are summaries, not source of truth.

## Documentation Boundaries

- `AGENTS.md`: stable instructions only.
- `README.md`: command and layout overview.
- `docs/results_digest.md`: compact current interpretation.
- `docs/experiment_index.md`: one-row-per-experiment tracker.
- `docs/task_list.md`: current operational state.
- `experiments/E##_*/README.md`: experiment-specific restart notes.
- `docs/archive/experiment_journal_legacy.md`: legacy archive only; do not
  append.
- `docs/rebuttals.md` and `docs/prior_art.md`: supporting reference artifacts,
  not living source-of-truth docs.

Do not create new living docs for research direction, paper plans, or next
steps. Put current interpretation in `results_digest.md`, the experiment row in
`experiment_index.md`, active work in `task_list.md`, and restart details in
the relevant experiment README. Extra docs must be clearly dated
artifacts/snapshots, and should be avoided unless the existing files cannot
reasonably hold the information.

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
