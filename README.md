# LLM Stability Probe

Exploratory harness for a learning-club talk on LLMs as systems in motion.

The first goal is not to estimate a formal Lyapunov exponent. It is to produce
a defensible stability profile:

- same-prompt deterministic baseline variance
- near-prompt generation divergence
- hidden-state divergence across layers

## Local Smoke Run

```bash
uv run python scripts/run_stability_probe.py \
  --smoke \
  --limit-pairs 3 \
  --max-new-tokens 48 \
  --out-dir runs/local_smoke

uv run python scripts/plot_results.py runs/local_smoke
uv run python scripts/inspect_run.py runs/local_smoke
```

Run a panel with per-model timeout isolation:

```bash
uv run python scripts/run_panel.py \
  --model qwen35_08b \
  --model gemma4_e2b_it \
  --limit-pairs 2 \
  --max-new-tokens 24 \
  --timeout-s 600 \
  --out-root runs/local_panel_smoke
```

If the smoke model is unavailable, pass any accessible model:

```bash
uv run python scripts/run_stability_probe.py \
  --model Qwen/Qwen3.5-0.8B \
  --prompt-pairs configs/prompt_pairs_expanded.json \
  --limit-pairs 3 \
  --max-new-tokens 48 \
  --out-dir runs/qwen35_08b_smoke
```

## Outputs

Each run directory contains:

- `metadata.json`: run settings
- `generations.jsonl`: raw generated token IDs and text
- `summary.csv`: pair-level divergence metrics
- `curves.jsonl`: prefix-length divergence curves
- `hidden_states.jsonl`: layer-wise hidden-state distances
- `failures.jsonl`: per-model/per-pair failures
- `plot_*.png`: generated plots

Optional semantic metrics:

```bash
uv run python scripts/add_semantic_metrics.py runs/local_smoke
```

Panel runs also include `panel_manifest.jsonl`, which records per-model status,
stdout/stderr tails, elapsed time, and timeouts.

Rank multiple completed runs:

```bash
uv run python scripts/rank_runs.py \
  --run qwen35_08b runs/qwen35_08b_expanded/qwen35_08b \
  --run qwen35_4b runs/qwen35_4b_expanded/qwen35_4b \
  --out-dir runs/rankings/example
```

Quantized loading is opt-in per model registry entry via `quantization`:

- `bnb_8bit`
- `bnb_4bit`

Use quantized runs as controlled comparisons against the exact same checkpoint,
not as interchangeable replacements for the main BF16 model-size ladder.

## Talk Prep

Current-state docs:

- `talk/slides.md`: live Marp deck (renders to `talk/slides.pdf`).
- `talk/speaker_notes.md`: per-slide background, likely questions, glossary.
- `docs/results_digest.md`: canonical current talk readout.
- `docs/experiment_journal.md`: chronological lab notebook.
- `docs/task_list.md`: operational next actions.

Current-state config:

- `configs/models.json`: canonical model registry. Each entry includes an
  `observed_behavior` block for scaffold/template/reasoning-prefix behavior
  when we have inspected raw outputs. Treat `unknown` entries as unobserved,
  not as non-reasoning.
- `configs/sagemaker_queue.json`: ordered SageMaker work queue.

Supporting/historical docs:

- `docs/presenter_notes.md`: earlier speaker notes.
- `docs/talk_outline.md`: earlier draft slide flow.
- `docs/north_star.md`: initial framing.
- `docs/rebuttals.md`: steel man objections and sober answers.
- `docs/prior_art.md`: supporting literature and talks/explainers.

If these conflict, trust `docs/results_digest.md` first, then the latest entry
in `docs/experiment_journal.md`.

Slide assets:

- `runs/talk_figures/`: slide-friendly PNGs

## Notes

Use Transformers directly for hidden-state access. vLLM can be useful later for
generation-only throughput, but it is not the right first tool for mechanistic
layer analysis.

## SageMaker

The SageMaker path uses a "training job" as a generic GPU batch runner. It does
not train a model.

Dry-run the job request:

```bash
uv run python scripts/launch_sagemaker_panel.py \
  --dry-run \
  --job-name chaos-stability-smoke \
  --model qwen35_08b \
  --limit-pairs 2 \
  --max-new-tokens 24
```

Launch a small smoke job:

```bash
uv run python scripts/launch_sagemaker_panel.py \
  --job-name chaos-stability-smoke \
  --model qwen35_08b \
  --limit-pairs 2 \
  --max-new-tokens 24
```

Check status and pull artifacts:

```bash
uv run python scripts/sagemaker_status.py --job-name chaos-stability-smoke
uv run python scripts/download_sagemaker_artifact.py chaos-stability-smoke --extract
```
