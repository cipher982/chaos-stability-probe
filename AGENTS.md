# AGENTS.md

## Overview

`chaos-stability-probe` is an exploratory Python/Transformers harness for LLM
dynamical sensitivity, prompt perturbations, logit/hidden-state divergence, and
chaos as a lens. Post-talk direction: turn trajectory branching into
**BranchTrace**, a branch-level debugger for LLM behavior regressions; treat
activation patching as open-model X-ray evidence, not the whole product.

Tone: keep the Learning Club talk materials friendly, exploratory, visually
clear. Bias toward keeping interesting findings with honest caveats (`n=6`,
"one prompt family"), frame broken hypotheses as part of the story, and don't
strip 90% of the fun for 5% more rigor. Paper/prototype work favors replayable
artifacts, exact counts, explicit caveats. Be direct but don't assume a hostile
reviewer.

## Build & Test

| Action | Command |
| --- | --- |
| Install | `uv sync` |
| Local smoke | `uv run python scripts/run_stability_probe.py --smoke --limit-pairs 3 --max-new-tokens 48 --out-dir runs/local_smoke` |
| Plot | `uv run python scripts/plot_results.py runs/local_smoke` |
| Inspect | `uv run python scripts/inspect_run.py runs/local_smoke` |
| Compile | `uv run python -m py_compile scripts/*.py experiments/E*/*.py` |
| Process robust wave | `uv run python scripts/process_robust_wave.py` |
| Process logit wave | `uv run python scripts/process_logit_wave.py` |
| Process sampling demo | `uv run python scripts/process_sampling_demo.py` |

## Docs (read in this order)

- `README.md` - command/layout overview.
- `docs/results_digest.md` - current interpretation, tool framing, safest claims.
- `docs/experiment_index.md` - one-row-per-experiment tracker.
- `docs/task_list.md` - operational next actions.
- `experiments/E##_*/README.md` - experiment restart notes.
- `docs/archive/experiment_journal_legacy.md` - legacy only; do not append.
- `docs/rebuttals.md`, `docs/prior_art.md` - supporting references, not source of truth.

Do not encode live experiment status in `AGENTS.md`. Living state: current
interpretation → `results_digest.md`; one row per experiment → `experiment_index.md`;
active work → `task_list.md`; restart details → experiment README. Do not create
new living docs for research direction/plans; extra docs must be dated
snapshots, only if existing files can't hold the info.

## Config

- `configs/models.json` - canonical model registry. `observed_behavior` records
  inspected scaffold/template/reasoning-prefix behavior. `unknown` = uninspected,
  not non-reasoning.
- `configs/sagemaker_queue*.json` - active/compatibility SageMaker queues.
- `configs/prompt_pairs*.json` - prompt sets.

## Architecture

- `scripts/run_stability_probe.py` - one model vs prompt pairs; writes raw
  generations, summaries, curves, hidden states, optional logits.
- `scripts/run_panel.py` - multiple models locally, isolated.
- `scripts/launch_sagemaker_panel.py`, `scripts/dispatch_sagemaker_queue.py`,
  `scripts/download_sagemaker_artifact.py` - SageMaker GPU jobs.
- `scripts/process_*.py` - pull artifacts, generate outputs under `runs/rankings/`.
- `experiments/E##_*/` - committed experiment code/config snapshots + short
  restart notes. Keep stable command shims in `scripts/` when docs, queues, or
  SageMaker jobs reference those paths.
- Generated outputs live under `runs/`, not `experiments/`.
- `talk/slides.md` - Marp source of truth; present from `talk/browser.html`,
  which displays high-res PNGs from `talk/slide_images/`.

## SageMaker

Use repo scripts, not raw AWS job creation. Query live state before assuming
capacity/completion:

```bash
uv run python scripts/sagemaker_status.py --prefix chaos --max-results 30
uv run python scripts/dispatch_sagemaker_queue.py --max-active 5
```

Prefer high-signal captures per model load: more prompts, longer continuations,
hidden states, logits, raw token/text artifacts.

## Conventions

- Use `uv` for Python.
- Prefer structured JSON/CSV artifacts over ad hoc text parsing.
- Prompt pair is the statistical unit; generated tokens are not independent samples.
- Distinguish raw-output metrics from scaffold/content-only metrics. Qualify
  cross-model stability by observed reasoning/scaffold behavior: raw/full-output
  stability can measure deterministic scaffolds (` thinking`, `Thinking Process:`,
  visible deliberation) rather than answer-content robustness. Do not compare
  scaffolded and non-scaffolded models naively.
- Label heuristic boundary detection as heuristic; don't hide failed
  scaffold/answer boundaries.

## Experiment Design

One compact row per experiment in `docs/experiment_index.md`: question,
status, primary artifacts, current readout, next decision. Restart
notes (question, inputs, commands, outputs, readout, caveats, next action) go in
`experiments/E##_*/README.md`.

Before new compute, check it answers an open question in `results_digest.md` or
`task_list.md`. Prefer discriminating tests over broader leaderboards: falsifying
controls, paired comparisons within a model family/recipe, token-certified prompt
perturbations over character edits, replayable branch cases before SAE/feature
interpretation, negative controls that change prompt tokens but not behavior.

Cleanup: don't rewrite/delete raw chronology. Add structured summaries, index
entries, decision notes pointing at existing artifacts. Treat raw run dirs and
`generations.jsonl` as the audit log; derived CSVs/charts are summaries, not
source of truth.

## Hygiene (after substantial changes)

- `docs/task_list.md` - trim stale checklist/history; leave active ops, blockers, next actions.
- `docs/experiment_index.md` - update affected row on any status/artifact/readout change.
- `docs/results_digest.md` - update only when interpretation/safest claim changed.
- `experiments/E##_*/README.md` - update when commands, queue configs, inputs/outputs, caveats changed.
- Remove/rewrite notes that became false; don't rely on the user to notice stale docs.
- Prefer deleting stale prose over adding explanation. Moment-in-time detail → `runs/` artifact or dated snapshot, not a living doc.

## Agent Boundaries

Always:

- Read relevant source-of-truth docs before changing framing or experiments.
- Check live SageMaker state before discussing queues or GPU utilization.
- Sanity-check raw `generations.jsonl` when interpreting results.
- Preserve raw artifacts (analysis is recomputable).
- Small `talk/slides.md` edits: render the deck, inspect only affected slide(s).
  No full-deck visual QA/fixer agents unless deck-wide, slide numbering/theme
  changed, or the user asks.

Ask first:

- Changing core metric definitions.
- Deleting or overwriting run artifacts.
- Adding large new dependency families or switching inference backends.

Never:

- Store secrets or credentials in this repo.
- Treat derived CSVs as more authoritative than raw generations plus config.
- Present scaffold/content heuristics as ground truth without confidence labels.
- Put transient queue status or speculative conclusions in `AGENTS.md`.