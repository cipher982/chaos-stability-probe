# Experiment Journal

This is the live lab notebook for the LLM stability probe.

## Entry Format Going Forward

Use this structure for each new experiment thread. Keep it concise, but make it
restartable without reading the whole repo history.

- **Question / hypothesis:** what uncertainty this experiment is meant to
  reduce.
- **Design:** models, prompt set, controls, decoding mode, token budget, and
  the statistical unit.
- **Commands / artifacts:** exact commands or queue names plus the raw run
  directories and derived summaries.
- **Results:** compact numbers and the most important concrete examples.
- **Interpretation:** what changed in the story, including failed hypotheses.
- **Caveats / guardrails:** confounds, weak rows, replay failures, heuristic
  boundaries, or partial runs.
- **Decision / next test:** whether to expand, pivot, stop, or turn into a
  slide/writeup claim.

## 2026-04-28 Late Night: Setup

### Objective

Build a reproducible harness to measure model stability/divergence across open-weight LLMs.

The immediate plan:

1. Persist the talk and experiment north star.
2. Build a local Hugging Face/Transformers harness.
3. Run a tiny-model dry run on the MacBook to debug prompt formatting, generation, hidden-state extraction, metric output, and plotting.
4. Expand to a curated model panel.
5. Package the same harness for SageMaker GPU jobs.

### Current Hypotheses

- Same-prompt deterministic repeats should define a low baseline variance for local open models.
- Near-identical prompts will produce measurable extra divergence beyond that baseline.
- Instruction-tuned models may be more contractive semantically than base models because they are pulled toward assistant-style attractors.
- Larger models within a family may be more stable on constrained/factual/code tasks, but creative prompts may still diverge quickly.
- Hidden-state distances may show where close inputs separate across transformer layers.

### Controls

- **Control A:** identical prompt, deterministic decode. Measures implementation nondeterminism.
- **Control B:** identical prompt, sampled decode. Measures sampling variance.
- **Test C:** near-identical prompts, deterministic decode. Measures input sensitivity.
- **Test D:** near-identical prompts, sampled decode with fixed seed if feasible. Measures perturbation sensitivity under realistic generation.
- **Test E:** hidden-state distance by layer for prompt pairs. Measures depth dynamics.

### Perturbation Ladder

Use prompt pairs with increasing perturbation strength:

1. No-op / identical.
2. Whitespace or trailing newline.
3. Punctuation or capitalization.
4. Synonym swap.
5. Semantic-preserving paraphrase.
6. Small semantic change.
7. Positive control: clearly different task/topic.

### Initial Dry-Run Model

Prefer the smallest model that works easily with Hugging Face Transformers. Candidate order:

1. `Qwen/Qwen3.5-0.8B` if available and accessible.
2. `Qwen/Qwen3.5-4B` if local memory/runtime is acceptable.
3. `google/gemma-4-E2B-it` if latest Transformers support and HF access are clean.
4. Fallback only if needed: a stable tiny HF model for harness debugging, clearly labeled as not part of final panel.

### Design Decisions

- Use `uv` for Python environment management.
- Use Transformers directly for the mechanistic hidden-state pass.
- Save raw rows as JSONL before aggregating.
- Model registry controls model-specific quirks.
- Fail one model without failing the whole panel.

### Dry-Run Log

#### `runs/qwen35_08b_smoke`

Command:

```bash
uv run python scripts/run_stability_probe.py \
  --smoke \
  --limit-pairs 2 \
  --max-new-tokens 24 \
  --out-dir runs/qwen35_08b_smoke
```

Result:

- Model `Qwen/Qwen3.5-0.8B` loaded successfully on local MPS with float16.
- HF marked the model as `image-text-to-text`, but `AutoModelForCausalLM` still worked.
- Transformers warned that the model has an optional fast path requiring extra FLA/causal-conv packages; torch fallback worked.
- Identical prompt and trailing-newline prompt both generated identical text under deterministic decode.
- Hidden-state distances for identical/no-op controls were zero except for floating-point epsilon.

#### `runs/qwen35_08b_allpairs_v2`

Command:

```bash
uv run python scripts/run_stability_probe.py \
  --smoke \
  --max-new-tokens 64 \
  --out-dir runs/qwen35_08b_allpairs_v2

uv run python scripts/plot_results.py runs/qwen35_08b_allpairs_v2
uv run python scripts/inspect_run.py runs/qwen35_08b_allpairs_v2
```

Pair-level output divergence:

| Pair | Category | Common prefix tokens | Normalized token edit distance |
| --- | --- | ---: | ---: |
| `control_identical_weather` | `control_identical` | 62 | 0.000 |
| `noop_trailing_newline` | `noop_format` | 62 | 0.000 |
| `punctuation_comma` | `punctuation` | 2 | 0.938 |
| `synonym_happy_joyful` | `synonym` | 0 | 0.938 |
| `paraphrase_stability` | `paraphrase` | 5 | 0.766 |
| `semantic_small_change` | `semantic_small` | 5 | 0.844 |
| `positive_control_different` | `positive_control` | 0 | 1.000 |

Hidden-state observations:

- Identical and trailing-newline prompts had zero hidden-state distance.
- Tiny lexical/punctuation changes had low final-layer hidden cosine distance (`~0.003-0.005`) while output text diverged sharply.
- Paraphrase had higher final-layer hidden distance (`~0.069`).
- Positive control had much higher final-layer hidden distance (`~0.449`).

Interpretation:

- The harness is working and already produces a useful distinction: output text can diverge strongly even when final prompt hidden states remain close.
- This is a good warning for the talk: "generation divergence" and "depth hidden-state divergence" are different measurements.
- Need more models and repeats before making any model-level claims.

#### `runs/gemma4_e2b_smoke`

Command:

```bash
uv run python scripts/run_stability_probe.py \
  --model gemma4_e2b_it \
  --limit-pairs 2 \
  --max-new-tokens 24 \
  --out-dir runs/gemma4_e2b_smoke
```

Result:

- HF model metadata was reachable and public.
- Download/cache reached about 3.4 GB locally.
- Local MPS run stalled before writing generation rows and was killed.
- Shutdown emitted a multiprocessing resource tracker semaphore warning.

Interpretation:

- Do not spend more Mac time on Gemma 4 tonight.
- Keep Gemma 4 in the model panel, but retry on SageMaker CUDA before calling it unsupported.
- This validates the need for per-model failure isolation in the harness.

#### Sampling Controls

Commands:

```bash
uv run python scripts/run_stability_probe.py \
  --smoke \
  --limit-pairs 3 \
  --max-new-tokens 48 \
  --sample \
  --repeats 3 \
  --skip-hidden \
  --out-dir runs/qwen35_08b_sampled_controls

uv run python scripts/run_stability_probe.py \
  --smoke \
  --limit-pairs 3 \
  --max-new-tokens 48 \
  --sample \
  --repeats 3 \
  --different-seeds-within-pair \
  --skip-hidden \
  --out-dir runs/qwen35_08b_sampled_different_seeds
```

Results:

- With the same seed inside each prompt pair, identical and no-op prompts had zero divergence even under sampled decoding.
- With different seeds inside each prompt pair, identical prompts diverged heavily (`mean normalized token edit distance ~0.82`).
- Different-seed sampling variance was about as large as the punctuation perturbation itself.

Interpretation:

- For measuring input sensitivity, deterministic decode or same-seed paired sampling is mandatory.
- Different-seed sampling mostly measures stochastic generation variance, not prompt perturbation sensitivity.
- This is a strong slide candidate because it shows why controls matter.

#### Qwen3.5 Size Ladder: `0.8B` vs `4B`

Commands:

```bash
uv run python scripts/run_panel.py \
  --model qwen35_4b \
  --max-new-tokens 64 \
  --timeout-s 900 \
  --out-root runs/qwen35_4b_allpairs

uv run python scripts/compare_runs.py \
  --run qwen35_08b runs/qwen35_08b_allpairs_v2 \
  --run qwen35_4b runs/qwen35_4b_allpairs/qwen35_4b \
  --out-dir runs/comparisons/qwen35_size_ladder
```

Result:

| Category | Qwen3.5 0.8B token edit | Qwen3.5 4B token edit | 0.8B common prefix | 4B common prefix |
| --- | ---: | ---: | ---: | ---: |
| `control_identical` | 0.000 | 0.000 | 62 | 64 |
| `noop_format` | 0.000 | 0.000 | 62 | 64 |
| `punctuation` | 0.938 | 0.000 | 2 | 64 |
| `synonym` | 0.938 | 0.094 | 0 | 25 |
| `paraphrase` | 0.766 | 0.484 | 5 | 19 |
| `semantic_small` | 0.844 | 0.328 | 5 | 23 |
| `positive_control` | 1.000 | 0.563 | 0 | 18 |

Interpretation:

- This is the first genuinely interesting empirical result.
- Within the same model family, the larger model stayed much closer under deterministic near-prompt perturbations.
- The punctuation case is especially striking: `0.8B` diverged after two generated tokens; `4B` stayed identical for the full 64-token generation.
- Do not overclaim from one prompt ladder, but this supports a concrete talk hypothesis: model size may correlate with local output stability on simple tasks.

Artifacts:

- `runs/comparisons/qwen35_size_ladder/compare_output_divergence.png`
- `runs/comparisons/qwen35_size_ladder/compare_common_prefix.png`

#### Expanded Qwen3.5 Size Ladder

Commands:

```bash
uv run python scripts/run_panel.py \
  --model qwen35_08b \
  --prompt-pairs configs/prompt_pairs_expanded.json \
  --max-new-tokens 64 \
  --timeout-s 900 \
  --out-root runs/qwen35_08b_expanded

uv run python scripts/run_panel.py \
  --model qwen35_4b \
  --prompt-pairs configs/prompt_pairs_expanded.json \
  --max-new-tokens 64 \
  --timeout-s 1200 \
  --out-root runs/qwen35_4b_expanded

uv run python scripts/compare_runs.py \
  --run qwen35_08b runs/qwen35_08b_expanded/qwen35_08b \
  --run qwen35_4b runs/qwen35_4b_expanded/qwen35_4b \
  --out-dir runs/comparisons/qwen35_expanded_size_ladder
```

Mean normalized token edit distance over 3 prompt pairs per category:

| Category | Qwen3.5 0.8B | Qwen3.5 4B |
| --- | ---: | ---: |
| `control_identical` | 0.000 | 0.000 |
| `noop_format` | 0.000 | 0.000 |
| `punctuation` | 0.552 | 0.036 |
| `synonym` | 0.458 | 0.057 |
| `paraphrase` | 0.901 | 0.526 |
| `semantic_small` | 0.810 | 0.198 |
| `positive_control` | 0.990 | 0.521 |

Mean common generated-token prefix:

| Category | Qwen3.5 0.8B | Qwen3.5 4B |
| --- | ---: | ---: |
| `control_identical` | 58.3 | 64.0 |
| `noop_format` | 58.3 | 64.0 |
| `punctuation` | 16.3 | 58.7 |
| `synonym` | 2.0 | 25.7 |
| `paraphrase` | 1.7 | 18.7 |
| `semantic_small` | 2.0 | 20.7 |
| `positive_control` | 0.0 | 18.0 |

Interpretation:

- The size effect survived the expanded prompt set.
- `4B` is much more output-stable under small perturbations than `0.8B`.
- The positive-control common-prefix result is a caution: larger models may share a generic assistant opening across even different tasks, so common-prefix length alone is not a sufficient semantic divergence metric.
- Normalized edit distance plus raw examples are more informative than any single scalar.

Artifacts:

- `runs/comparisons/qwen35_expanded_size_ladder/compare_output_divergence.png`
- `runs/comparisons/qwen35_expanded_size_ladder/compare_common_prefix.png`

#### Semantic Metric Add-On

Commands:

```bash
uv run python scripts/add_semantic_metrics.py runs/qwen35_08b_expanded/qwen35_08b
uv run python scripts/add_semantic_metrics.py runs/qwen35_4b_expanded/qwen35_4b
uv run python scripts/compare_runs.py \
  --run qwen35_08b runs/qwen35_08b_expanded/qwen35_08b \
  --run qwen35_4b runs/qwen35_4b_expanded/qwen35_4b \
  --out-dir runs/comparisons/qwen35_expanded_size_ladder
```

Mean sentence-embedding cosine distance:

| Category | Qwen3.5 0.8B | Qwen3.5 4B |
| --- | ---: | ---: |
| `control_identical` | 0.000 | 0.000 |
| `noop_format` | 0.000 | 0.000 |
| `punctuation` | 0.132 | 0.005 |
| `synonym` | 0.242 | 0.035 |
| `paraphrase` | 0.192 | 0.104 |
| `semantic_small` | 0.589 | 0.416 |
| `positive_control` | 1.027 | 0.732 |

Interpretation:

- Semantic distance agrees with the edit-distance result directionally: `4B` is more stable under small prompt perturbations.
- Semantic distance also dampens pure wording differences, which makes it a better slide metric for non-code audiences.
- The embedding model is `sentence-transformers/all-MiniLM-L6-v2`; this is a pragmatic metric, not ground truth.

Artifact:

- `runs/comparisons/qwen35_expanded_size_ladder/compare_semantic_divergence.png`

#### Final-Layer Hidden-State Comparison

Mean final-layer last-token hidden cosine distance:

| Category | Qwen3.5 0.8B | Qwen3.5 4B |
| --- | ---: | ---: |
| `control_identical` | 0.000 | 0.000 |
| `noop_format` | 0.000 | 0.000 |
| `punctuation` | 0.003 | 0.005 |
| `synonym` | 0.036 | 0.002 |
| `paraphrase` | 0.135 | 0.020 |
| `semantic_small` | 0.116 | 0.023 |
| `positive_control` | 0.582 | 0.229 |

Interpretation:

- The hidden-state metric broadly matches the semantic/output stability story.
- Punctuation is interesting: hidden-state distances are tiny for both models, yet output divergence is much larger in `0.8B`. That suggests a small representation difference can still cross a generation decision boundary in the smaller model.

Artifact:

- `runs/comparisons/qwen35_expanded_size_ladder/compare_final_layer_hidden_divergence.png`

#### Local Qwen3.5 9B Probe

Command:

```bash
uv run python scripts/run_panel.py \
  --model qwen35_9b \
  --max-new-tokens 64 \
  --timeout-s 1800 \
  --out-root runs/qwen35_9b_allpairs
```

Result:

- `Qwen/Qwen3.5-9B` completed locally on the Mac after a large download.
- Download dominated runtime: ~18 GB cache, total elapsed ~523 seconds.
- The run used the original 7-pair ladder, not the expanded 21-pair set.

Three-size comparison on the 7-pair ladder:

| Category | 0.8B edit | 4B edit | 9B edit |
| --- | ---: | ---: | ---: |
| `control_identical` | 0.000 | 0.000 | 0.000 |
| `noop_format` | 0.000 | 0.000 | 0.000 |
| `punctuation` | 0.938 | 0.000 | 0.375 |
| `synonym` | 0.938 | 0.094 | 0.156 |
| `paraphrase` | 0.766 | 0.484 | 0.359 |
| `semantic_small` | 0.844 | 0.328 | 0.375 |
| `positive_control` | 1.000 | 0.563 | 0.578 |

Interpretation:

- The size/stability story is not strictly monotonic on every prompt.
- `9B` is much more stable than `0.8B`, but not uniformly more stable than `4B` on the small 7-pair ladder.
- This is a useful caveat: stability is model-size related, but probably mediated by training, decoding defaults, and prompt class.

Artifacts:

- `runs/comparisons/qwen35_three_size_ladder/compare_output_divergence.png`
- `runs/comparisons/qwen35_three_size_ladder/compare_semantic_divergence.png`
- `runs/comparisons/qwen35_three_size_ladder/compare_final_layer_hidden_divergence.png`

### Open Questions

- Which models require HF auth/license acceptance?
- Which model classes expose `hidden_states` cleanly with `output_hidden_states=True`?
- Do Gemma 4 and Qwen3.5 require unreleased or nightly Transformers?
- Should the SageMaker job use one model per job or one sequential panel job?

## SageMaker Log

### `chaos-stability-smoke-20260428-001`

Status: failed quickly.

Root cause:

- The PyTorch training container launched correctly.
- Source was uploaded to S3 correctly.
- `scripts/launch_sagemaker_panel.py` incorrectly passed script-mode fields as environment variables instead of SageMaker hyperparameters.
- The training toolkit therefore tried to execute `/opt/ml/code/sagemaker_entry.py` before downloading/unpacking the submitted source and failed with "No such file or directory."

Fix:

- Move `sagemaker_program`, `sagemaker_submit_directory`, `sagemaker_region`, and `sagemaker_container_log_level` into `HyperParameters`.

### `chaos-stability-smoke-20260428-002`

Status: completed successfully after fixing script-mode hyperparameters.

Command:

```bash
uv run python scripts/launch_sagemaker_panel.py \
  --job-name chaos-stability-smoke-20260428-002 \
  --model qwen35_08b \
  --limit-pairs 2 \
  --max-new-tokens 24 \
  --max-runtime-s 3600
```

Result:

- SageMaker CUDA path validated end to end.
- Container used `torch 2.7.1+cu128`, CUDA, bf16.
- `qwen35_08b` smoke completed in ~21 seconds inside the container after provisioning/download.
- Artifact contained expected `panel_manifest.jsonl`, `summary.csv`, `generations.jsonl`, `curves.jsonl`, `metadata.json`, and `hidden_states.jsonl`.

### `chaos-stability-panel-20260429-001`

Status: launched.

Command:

```bash
uv run python scripts/launch_sagemaker_panel.py \
  --job-name chaos-stability-panel-20260429-001 \
  --prompt-pairs configs/prompt_pairs_expanded.json \
  --model qwen35_08b \
  --model qwen35_4b \
  --model qwen35_9b \
  --model gemma4_e2b_it \
  --model olmo3_7b_instruct \
  --model nemotron_nano_9b_v2 \
  --model gpt_oss_20b \
  --max-new-tokens 64 \
  --timeout-s 3600 \
  --max-runtime-s 28800
```

Panel intent:

- Validate a Qwen size ladder on GPU.
- Retry Gemma 4 E2B on CUDA after local MPS stall.
- Add non-Qwen diversity: Ai2 OLMo, NVIDIA Nemotron, OpenAI gpt-oss.
- Keep per-model timeout isolation so one quirky model cannot kill the whole panel.

### Parallel single-model SageMaker jobs

Status: launched to use the available parallel GPU lane and avoid sequential panel blockage.

Jobs:

- `chaos-stability-gemma4-e2b-20260429-001`
- `chaos-stability-olmo3-7b-20260429-001`
- `chaos-stability-nemotron-9b-20260429-001`
- `chaos-stability-gptoss-20b-20260429-001`

All use:

- `configs/prompt_pairs_expanded.json`
- `max_new_tokens=64`
- one model per job
- per-model timeout `3600s`
- max job runtime `14400s`

Follow-up status:

- Broad panel is `Training`.
- Four single-model jobs have all started and are in SageMaker `Downloading`.

Follow-up result:

- `chaos-stability-nemotron-9b-20260429-001` completed as a harness-level success but model load failed.
- Failure reason: `mamba-ssm is required by the Mamba model but cannot be imported`.
- Treat Nemotron as a dependency-specific miss unless we decide to add/build `mamba-ssm` in the SageMaker image.
- `chaos-stability-gptoss-20b-20260429-001` completed as a harness-level success but every pair failed.
- Failure reason: MXFP4 fell back without Triton and the model split across CPU/CUDA, causing mixed-device matmul errors.
- Patched SageMaker `requirements.txt` to include `triton>=3.4.0`, added `device_map: "single"` for `gpt_oss_20b`, and relaunched `chaos-stability-gptoss-20b-20260429-002` generation-only with `--skip-hidden`.

### Notes for Future Self

Do not conflate:

- prompt-output divergence
- hidden-state divergence across layers
- sampling randomness
- hardware/framework nondeterminism
- training edge-of-chaos
- static quantization/rate-distortion

The talk is stronger if those are kept separate.

### Results Digest

Added [results_digest.md](results_digest.md) as the compact talk-facing readout.

Current strongest claims:

- `Qwen3.5-4B` is much more stable than `Qwen3.5-0.8B` on the expanded deterministic prompt ladder.
- `Gemma 4 E2B` behaves more like `Qwen3.5-0.8B` on token edit distance, but semantic distance dampens some synonym/paraphrase differences.
- `OLMo 3 7B` shows a real no-op formatting sensitivity, especially on the palindrome code prompt, but this should be framed as model-specific fragility rather than a universal law.

### Additional Standalone SageMaker Jobs

Launched two more standalone expanded-ladder jobs to avoid waiting on the broad sequential panel:

```bash
uv run python scripts/launch_sagemaker_panel.py \
  --job-name chaos-stability-qwen35-9b-20260429-001 \
  --prompt-pairs configs/prompt_pairs_expanded.json \
  --model qwen35_9b \
  --max-new-tokens 64 \
  --timeout-s 3600 \
  --max-runtime-s 14400

uv run python scripts/launch_sagemaker_panel.py \
  --job-name chaos-stability-gemma4-e4b-20260429-001 \
  --prompt-pairs configs/prompt_pairs_expanded.json \
  --model gemma4_e4b_it \
  --max-new-tokens 64 \
  --timeout-s 3600 \
  --max-runtime-s 14400
```

Why these two:

- `qwen35_9b` gives a direct expanded-ladder Qwen size comparison against `0.8B` and `4B`.
- `gemma4_e4b_it` gives a useful Gemma E2B/E4B comparison if it loads cleanly.

### Broad Panel Completed

Pulled `chaos-stability-panel-20260429-001`.

Successful expanded-ladder model outputs:

- `qwen35_08b`
- `qwen35_4b`
- `qwen35_9b`
- `gemma4_e2b_it`
- `olmo3_7b_instruct`

Failure records:

- `gpt_oss_20b`: same CPU/CUDA split issue as the first standalone run.
- `nemotron_nano_9b_v2`: missing `mamba-ssm`.

Added semantic metrics and regenerated:

- `runs/comparisons/qwen35_panel_expanded_size_ladder`
- `runs/comparisons/panel_cross_lab_expanded`
- `runs/talk_figures`

Expanded panel Qwen semantic distance means:

| Category | Qwen3.5 0.8B | Qwen3.5 4B | Qwen3.5 9B |
| --- | ---: | ---: | ---: |
| `control_identical` | 0.000 | 0.000 | 0.000 |
| `noop_format` | 0.000 | 0.000 | 0.000 |
| `punctuation` | 0.154 | 0.005 | 0.047 |
| `synonym` | 0.261 | 0.035 | 0.030 |
| `paraphrase` | 0.196 | 0.100 | 0.091 |
| `semantic_small` | 0.596 | 0.414 | 0.374 |
| `positive_control` | 1.022 | 0.722 | 0.708 |

Interpretation:

- `9B` is much more stable than `0.8B`.
- `9B` is comparable to `4B`, not uniformly better.
- This keeps the size story honest: scale helps in this family, but it is not a
  simple monotonic law from this small probe.

### `gpt-oss-20b` Retry Notes

`chaos-stability-gptoss-20b-20260429-002` fixed the SageMaker launch path but
still produced only `failures.jsonl`.

Root cause:

- Adding `triton>=3.4.0` caused pip to upgrade the SageMaker image from its
  CUDA-compatible PyTorch 2.7 build to PyTorch 2.11 / CUDA 13.
- The `ml.g6e.2xlarge` driver exposed by SageMaker was too old for that CUDA
  build, so CUDA initialization broke and `GptOssForCausalLM` failed to import.

Patch:

- Pin `triton==3.3.1` in `requirements.txt`.
- Keep `device_map: "single"` for `gpt_oss_20b`.
- Relaunched `chaos-stability-gptoss-20b-20260429-003` with `--skip-hidden`.

`003` kept the container's PyTorch 2.7 / CUDA 12.8 stack intact, but
Transformers still refused the MXFP4 path because CUDA MXFP4 requires Triton
`>=3.4.0`; it dequantized to bf16 and hit the same CPU/CUDA split during
generation.

Next patch:

- `sagemaker_entry.py` now installs `triton==3.4.0` with `--no-deps` only for
  `gpt_oss_20b`, so pip does not replace the SageMaker image's PyTorch build.
- Relaunched `chaos-stability-gptoss-20b-20260429-004`.

Final `gpt-oss-20b` result:

- `004` preserved PyTorch 2.7 / CUDA 12.8 and installed Triton 3.4.0 with
  `--no-deps`.
- The model loaded far enough to run pair evaluation, but every pair still
  failed with a CPU/CUDA tensor split during matmul.
- I am treating `gpt-oss-20b` as a tooling miss for tonight rather than burning
  more time on a model-specific loading path.

The same temporary dependency issue affected standalone jobs launched while
`triton>=3.4.0` was still in `requirements.txt`:

- `chaos-stability-qwen35-9b-20260429-001`
- `chaos-stability-gemma4-e4b-20260429-001`

`qwen35_9b` already has a valid expanded result from the broad panel, so I did
not relaunch it. Relaunched Gemma E4B as:

- `chaos-stability-gemma4-e4b-20260429-002`

### Gemma E4B Result

`chaos-stability-gemma4-e4b-20260429-002` completed successfully after the
dependency pin.

Added:

- `runs/sagemaker_artifacts/chaos-stability-gemma4-e4b-20260429-002`
- `runs/comparisons/gemma4_expanded_size_ladder`
- `runs/comparisons/full_success_expanded`
- refreshed `runs/talk_figures`

Gemma semantic distance means:

| Category | Gemma 4 E2B | Gemma 4 E4B |
| --- | ---: | ---: |
| `control_identical` | 0.000 | 0.000 |
| `noop_format` | 0.000 | 0.000 |
| `punctuation` | 0.146 | 0.035 |
| `synonym` | 0.111 | 0.173 |
| `paraphrase` | 0.128 | 0.162 |
| `semantic_small` | 0.606 | 0.614 |
| `positive_control` | 1.000 | 1.004 |

Interpretation:

- E4B is much more stable than E2B on punctuation.
- E4B is not uniformly more stable than E2B on this prompt ladder.
- This reinforces the main caveat: stability is measurable, but not a simple
  size-only story.

## Wave 2: Doubling the Model Set

Goal: move from six successful stability profiles toward roughly twelve.
Final result: 13 successful profiles.

Added second-wave registry entries after checking Hugging Face model metadata:

- `microsoft/Phi-4-mini-instruct`
- `microsoft/Phi-4-reasoning-plus`
- `ibm-granite/granite-3.3-8b-instruct`
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
- `HuggingFaceTB/SmolLM3-3B`
- `mistralai/Mistral-7B-Instruct-v0.3`
- `allenai/OLMo-2-1124-7B-Instruct`
- `tiiuae/Falcon3-10B-Instruct`

Launched first five parallel jobs, filling the observed `ml.g6e.2xlarge` quota:

```bash
uv run python scripts/launch_sagemaker_panel.py --job-name chaos-stability-granite33-8b-20260429-001 --prompt-pairs configs/prompt_pairs_expanded.json --model granite33_8b_instruct --max-new-tokens 64 --timeout-s 3600 --max-runtime-s 14400
uv run python scripts/launch_sagemaker_panel.py --job-name chaos-stability-deepseek-r1-qwen7b-20260429-001 --prompt-pairs configs/prompt_pairs_expanded.json --model deepseek_r1_distill_qwen_7b --max-new-tokens 64 --timeout-s 3600 --max-runtime-s 14400
uv run python scripts/launch_sagemaker_panel.py --job-name chaos-stability-smollm3-3b-20260429-001 --prompt-pairs configs/prompt_pairs_expanded.json --model smollm3_3b --max-new-tokens 64 --timeout-s 3600 --max-runtime-s 14400
uv run python scripts/launch_sagemaker_panel.py --job-name chaos-stability-mistral7b-v03-20260429-001 --prompt-pairs configs/prompt_pairs_expanded.json --model mistral7b_instruct_v03 --max-new-tokens 64 --timeout-s 3600 --max-runtime-s 14400
uv run python scripts/launch_sagemaker_panel.py --job-name chaos-stability-olmo2-7b-20260429-001 --prompt-pairs configs/prompt_pairs_expanded.json --model olmo2_7b_instruct --max-new-tokens 64 --timeout-s 3600 --max-runtime-s 14400
```

`phi4_mini_instruct` launch hit the quota limit because the five other jobs
were already active. It was launched after a slot freed:

```bash
uv run python scripts/launch_sagemaker_panel.py \
  --job-name chaos-stability-phi4-mini-20260429-002 \
  --prompt-pairs configs/prompt_pairs_expanded.json \
  --model phi4_mini_instruct \
  --max-new-tokens 64 \
  --timeout-s 3600 \
  --max-runtime-s 14400
```

### Wave 2 Results

Successful additions:

- `smollm3_3b`
- `mistral7b_instruct_v03`
- `granite33_8b_instruct`
- `deepseek_r1_distill_qwen_7b`
- `olmo2_7b_instruct`
- `phi4_reasoning_plus`
- `falcon3_10b_instruct`

Failure / miss:

- `phi4_mini_instruct`: model load failed with
  `ImportError("cannot import name 'LossKwargs' from 'transformers.utils' ...")`.
  Treat this as a custom-code / Transformers compatibility miss.

Small-perturbation ranking artifact:

- `runs/rankings/wave2_13model/small_perturbation_ranking.png`
- `runs/rankings/wave2_13model/stability_rankings.csv`

Ranking summary:

| Rank | Model | Mean semantic distance over no-op + punctuation + synonym |
| ---: | --- | ---: |
| 1 | `qwen35_4b` | 0.013 |
| 2 | `phi4_reasoning_plus` | 0.024 |
| 3 | `qwen35_9b` | 0.026 |
| 4 | `deepseek_r1_qwen7b` | 0.030 |
| 5 | `mistral7b_v03` | 0.055 |
| 6 | `gemma4_e4b` | 0.069 |
| 7 | `smollm3_3b` | 0.074 |
| 8 | `granite33_8b` | 0.079 |
| 9 | `falcon3_10b` | 0.082 |
| 10 | `gemma4_e2b` | 0.086 |
| 11 | `olmo3_7b` | 0.135 |
| 12 | `qwen35_08b` | 0.138 |
| 13 | `olmo2_7b` | 0.144 |

Interpretation:

- The original goal was to double from 6 to roughly 12 successful profiles; the
  final wave now has 13 successful profiles.
- The top four under small perturbations are `qwen35_4b`,
  `phi4_reasoning_plus`, `qwen35_9b`, and `deepseek_r1_qwen7b`.
- `deepseek_r1_qwen7b` has the best broader meaningful-perturbation mean in
  this set, so it is worth calling out separately from the small-perturbation
  ranking.
- `phi4_mini_instruct`, `gpt_oss_20b`, and `nemotron_nano_9b_v2` remain tooling
  or dependency misses, not clean model-stability results.

## Wave 3 Plan: Family Ladders and Quantization Controls

Question from follow-up: should we keep adding models, especially more Qwen3.5
and Gemma 4 sizes, and should quantization become part of the experiment?

Decision:

- Yes, add more family-ladder points.
- Do not globally switch the experiment to quantized loading.
- Treat quantization as a separate controlled axis: same checkpoint, same
  prompts, same decoding, different numeric representation.

New registry entries:

- `qwen35_2b`: fills the Qwen3.5 ladder between `0.8B` and `4B`.
- `qwen35_4b_bnb8`: same checkpoint as `qwen35_4b`, loaded with bitsandbytes
  8-bit quantization.
- `qwen35_4b_bnb4`: same checkpoint as `qwen35_4b`, loaded with bitsandbytes
  4-bit quantization.
- `qwen35_27b_fp8`: larger Qwen3.5 FP8 checkpoint that should fit a 48 GB GPU
  by weight size.
- `qwen35_35b_a3b_fp8`: Qwen3.5 MoE FP8 stretch candidate.
- `qwen36_27b_fp8` and `qwen36_35b_a3b_fp8`: current Hugging Face trending
  Qwen-family stretch candidates. Keep these in a trending/new-model section,
  not in the clean Qwen3.5 size ladder.
- `gemma4_e2b_base` and `gemma4_e4b_base`: base-model pairs for the already-run
  Gemma 4 instruction-tuned models.
- `gemma4_31b_it_nvfp4`: large quantized Gemma 4 stretch candidate.

Recommended next jobs once SageMaker auth is refreshed:

```bash
uv run python scripts/launch_sagemaker_panel.py --job-name chaos-stability-qwen35-2b-20260429-001 --prompt-pairs configs/prompt_pairs_expanded.json --model qwen35_2b --max-new-tokens 64 --timeout-s 3600 --max-runtime-s 14400
uv run python scripts/launch_sagemaker_panel.py --job-name chaos-stability-qwen35-4b-bnb8-20260429-001 --prompt-pairs configs/prompt_pairs_expanded.json --model qwen35_4b_bnb8 --max-new-tokens 64 --timeout-s 3600 --max-runtime-s 14400
uv run python scripts/launch_sagemaker_panel.py --job-name chaos-stability-qwen35-4b-bnb4-20260429-001 --prompt-pairs configs/prompt_pairs_expanded.json --model qwen35_4b_bnb4 --max-new-tokens 64 --timeout-s 3600 --max-runtime-s 14400
uv run python scripts/launch_sagemaker_panel.py --job-name chaos-stability-gemma4-e2b-base-20260429-001 --prompt-pairs configs/prompt_pairs_expanded.json --model gemma4_e2b_base --max-new-tokens 64 --timeout-s 3600 --max-runtime-s 14400
uv run python scripts/launch_sagemaker_panel.py --job-name chaos-stability-gemma4-e4b-base-20260429-001 --prompt-pairs configs/prompt_pairs_expanded.json --model gemma4_e4b_base --max-new-tokens 64 --timeout-s 3600 --max-runtime-s 14400
```

Stretch jobs after the clean controls:

```bash
uv run python scripts/launch_sagemaker_panel.py --job-name chaos-stability-qwen35-27b-fp8-20260429-001 --prompt-pairs configs/prompt_pairs_expanded.json --model qwen35_27b_fp8 --max-new-tokens 64 --timeout-s 5400 --max-runtime-s 14400 --skip-hidden
uv run python scripts/launch_sagemaker_panel.py --job-name chaos-stability-qwen35-35b-a3b-fp8-20260429-001 --prompt-pairs configs/prompt_pairs_expanded.json --model qwen35_35b_a3b_fp8 --max-new-tokens 64 --timeout-s 5400 --max-runtime-s 14400 --skip-hidden
uv run python scripts/launch_sagemaker_panel.py --job-name chaos-stability-qwen36-27b-fp8-20260429-001 --prompt-pairs configs/prompt_pairs_expanded.json --model qwen36_27b_fp8 --max-new-tokens 64 --timeout-s 5400 --max-runtime-s 14400 --skip-hidden
uv run python scripts/launch_sagemaker_panel.py --job-name chaos-stability-qwen36-35b-a3b-fp8-20260429-001 --prompt-pairs configs/prompt_pairs_expanded.json --model qwen36_35b_a3b_fp8 --max-new-tokens 64 --timeout-s 5400 --max-runtime-s 14400 --skip-hidden
uv run python scripts/launch_sagemaker_panel.py --job-name chaos-stability-gemma4-31b-nvfp4-20260429-001 --prompt-pairs configs/prompt_pairs_expanded.json --model gemma4_31b_it_nvfp4 --max-new-tokens 64 --timeout-s 5400 --max-runtime-s 14400 --skip-hidden
```

Interpretation rules:

- Qwen3.5 size ladder should use the non-quantized points where possible:
  `0.8B`, `2B`, `4B`, `9B`.
- Qwen3.5 27B/35B FP8 results are useful, but they mix size and quantization.
- `qwen35_4b` vs `qwen35_4b_bnb8` vs `qwen35_4b_bnb4` is the clean
  quantization result.
- Gemma 4 base vs instruction-tuned pairs answer whether instruction tuning
  itself appears to contract or amplify this probe's prompt perturbations.
- Hidden-state extraction should be skipped on the larger FP8/NVFP4 stretch
  models unless there is spare runtime.

## 2026-04-29 Documentation and Statistical Reset

User concern: avoid holding current state in chat context and avoid spreading
the "current truth" across many docs that can drift.

Documentation policy from here:

- `docs/results_digest.md` is the pinned current-state talk readout.
- `docs/experiment_journal.md` is the chronological lab notebook.
- `docs/task_list.md` is operational only.
- Superseded planning docs should be folded into the digest or journal, then
  deleted rather than kept as additional living surfaces.

Statistical reset:

- The 13-model point ranking is useful, but should become a bucketed plot with
  confidence intervals.
- Robust: `Qwen3.5-4B` is much more stable than `Qwen3.5-0.8B`.
- Robust: stable top cluster is `qwen35_4b`, `phi4_reasoning_plus`,
  `qwen35_9b`, and `deepseek_r1_qwen7b`.
- Not robust: exact order inside the top cluster or middle pack.
- Not robust: a simple monotonic size law.

Next priority order:

1. Bucketed ranking plot with bootstrap confidence intervals.
2. No-op formatting raw examples.
3. Long-generation divergence-vs-token-position plot for a few models.
4. Chat-template control.
5. Additional model / quantization jobs after the presentation-critical plots.

## 2026-04-29 Wave 4 Background Compute Launch

Launched first five-slot background wave on `ml.g6e.2xlarge`:

- `chaos-longprobe-qwen35-4b-20260429-001`
- `chaos-longprobe-qwen35-08b-20260429-001`
- `chaos-longprobe-deepseek-r1-qwen7b-20260429-001`
- `chaos-stability-qwen35-2b-20260429-001`
- `chaos-stability-qwen35-4b-bnb8-20260429-001`

Rationale:

- Use three slots for long-generation trajectory probes, because a
  divergence-vs-position plot is the highest-leverage addition to the talk.
- Use one slot to fill the Qwen3.5 size ladder at 2B.
- Use one slot for the first quantization control against the already-run
  `qwen35_4b` baseline.

Next launch order as slots free:

1. `qwen35_4b_bnb4`
2. `qwen35_08b_bnb8`
3. `qwen35_08b_bnb4`
4. `gemma4_e2b_base`
5. `gemma4_e4b_base`
6. Larger stretch candidates only after clean controls have artifacts.

Added Qwen 0.8B quantization controls after the Act 3 discussion:

- `qwen35_08b_bnb8`
- `qwen35_08b_bnb4`

Reason: the clean quantization grid should be at least 2x3:

| Model size | BF16 | 8-bit | 4-bit |
| --- | --- | --- | --- |
| Qwen3.5 0.8B | done | queued | queued |
| Qwen3.5 4B | done | running | queued |

Do not interpret low-bit results with stability alone. A destructive
quantization can produce stable but low-quality or repetitive output, so the
analysis must also measure distance from the BF16 baseline / answer quality.

Possible destructive-low-bit lane:

- HQQ/GPTQ/GGUF for 3-bit and 2-bit variants.
- Keep this clearly labeled by backend and quantizer.
- Treat it as an exploratory stress test unless the runtime path is close enough
  to the Transformers path to compare fairly.

Update after slots freed:

Launched at 2026-04-29 11:12:

- `chaos-stability-qwen35-4b-bnb4-20260429-001`
- `chaos-stability-qwen35-08b-bnb8-20260429-001`
- `chaos-stability-qwen35-08b-bnb4-20260429-001`

## 2026-04-29 Pre-Result Hypotheses

Logged before reading Wave 4 results.

Long-generation trajectory probes:

- `Qwen3.5-4B` should diverge more slowly and/or reconverge more than
  `Qwen3.5-0.8B`.
- `Qwen3.5-0.8B` should separate earlier and show higher sustained divergence.
- `DeepSeek-R1-Distill-Qwen-7B` may show lexical divergence without equivalent
  semantic divergence, especially on prompts with a clear answer target.
- If we see Lyapunov-like behavior, it should be local: early rising-window
  growth followed by saturation, not an exponential fit over the entire 512
  tokens.

Qwen3.5 size ladder:

- `Qwen3.5-2B` should land between `0.8B` and the `4B/9B` cluster, but the
  relationship is expected to be nonlinear.
- Possible outcome A: `2B` close to `0.8B`, implying a stability transition
  somewhere between 2B and 4B.
- Possible outcome B: `2B` close to `4B`, implying `0.8B` is the fragile
  endpoint.
- Do not expect a simple monotonic size law.

Quantization controls:

- `Qwen3.5-4B` 8-bit should be close to BF16.
- `Qwen3.5-4B` 4-bit may show measurable drift, but likely modest drift.
- `Qwen3.5-0.8B` should be more sensitive to 4-bit quantization than `4B`.
- Track distance from BF16 outputs in addition to perturbation stability; stable
  degradation should not be counted as improved stability.
- The main Act 3 hypothesis is an interaction: smaller/brittle models lose more
  stability per bit removed than larger/stable models.

Gemma base vs instruction-tuned:

- Instruction-tuned Gemma should be more semantically contractive on
  assistant-like prompts.
- Base Gemma should show more stylistic/content variance.
- Treat this as secondary unless it strongly contradicts the Qwen story.

No-op formatting / chat-template sensitivity:

- Highest no-op failures should correlate with response-style flips, not just
  small semantic differences.
- Raw prompt formatting may reduce some no-op sensitivity but could hurt
  instruction-following consistency overall.

## 2026-04-29 Wave 4 Status Check

Checked live SageMaker state around 11:28 local time.

Completed and pulled locally:

- `chaos-longprobe-qwen35-08b-20260429-001`
- `chaos-longprobe-qwen35-4b-20260429-001`
- `chaos-longprobe-deepseek-r1-qwen7b-20260429-001`
- `chaos-stability-qwen35-2b-20260429-001`
- `chaos-stability-qwen35-4b-bnb8-20260429-001`
- `chaos-stability-qwen35-4b-bnb4-20260429-001`
- `chaos-stability-qwen35-08b-bnb4-20260429-001`

Still running:

- `chaos-stability-qwen35-08b-bnb8-20260429-001`

Launched after slots freed:

- `chaos-stability-gemma4-e2b-base-20260429-001`
- `chaos-stability-gemma4-e4b-base-20260429-001`

First readout from semantic post-processing:

| Model | Small-perturbation semantic mean |
| --- | ---: |
| Qwen3.5 0.8B BF16 | 0.1381 |
| Qwen3.5 2B BF16 | 0.0972 |
| Qwen3.5 4B BF16 | 0.0133 |
| Qwen3.5 4B 8-bit | 0.0249 |
| Qwen3.5 4B 4-bit | 0.0262 |
| Qwen3.5 0.8B 8-bit | 0.1096 |
| Qwen3.5 0.8B 4-bit | 0.0911 |
| Qwen3.5 9B BF16 | 0.0255 |

Interpretation is provisional. The 2B point supports a nonlinear size ladder:
0.8B and 2B are far more sensitive than 4B/9B. The 4B quantization points show
modest added sensitivity, not collapse. The 0.8B quantized results are lower
divergence than the BF16 baseline on this metric, decreasing from BF16 to 8-bit
to 4-bit. That makes the quantization story non-obvious: we cannot present
"lower bits always worse"; we need compare against BF16 outputs and inspect
quality/style before making a talk claim.

Long-probe first readout:

- No-op newline is perfectly stable for all three long-probe models.
- `Qwen3.5-0.8B` branches almost immediately on punctuation/synonym/semantic
  prompts by token-prefix distance, but semantic distances remain moderate.
- `Qwen3.5-4B` branches later on punctuation/synonym than 0.8B, matching the
  stability story.
- `DeepSeek-R1-Distill-Qwen-7B` is the most interesting qualitatively:
  punctuation keeps a 138-token prefix and very low semantic distance, while
  token edit distance can still be high. This supports the talk distinction
  between lexical trajectory divergence and semantic convergence.

## 2026-04-29 Framing Pivot: Legacy Models

The talk should not get trapped inside the experiment mechanics or the
quantization sub-story. The primary object is chaos/dynamical sensitivity in
LLMs: under small input perturbations, do response trajectories stay nearby,
branch, or reconverge?

Quantization remains useful as one perturbation source, but it is not the
center of gravity. A cleaner next experimental contrast is model era and
post-training recipe. The current panel is mostly recent instruction/chat-era
models, so add older base models:

- `gpt2_xl`
- `gptj_6b`
- `opt_6p7b`
- `pythia_6p9b_deduped`
- `llama1_7b_huggyllama` as a best-effort community LLaMA-1 conversion

Hypothesis:

- Older base models may show less instruction-following contraction and more
  surface-form drift.
- Modern chat/instruction models may have stronger attractor basins around
  assistant-style answers.
- This should be framed as a dynamical signature of post-training, not a model
  quality ranking.

Added model config entries and launched first legacy wave:

- `chaos-stability-opt-6p7b-legacy-20260429-001`
- `chaos-stability-gptj-6b-legacy-20260429-001`
- `chaos-stability-llama1-7b-legacy-20260429-001`

Queued for the next free slots:

- `pythia_6p9b_deduped`
- `gpt2_xl`

First legacy/base readout after artifact pull:

| Model | Small-perturbation semantic mean |
| --- | ---: |
| Qwen3.5 4B | 0.0133 |
| Qwen3.5 9B | 0.0255 |
| LLaMA1 7B | 0.0582 |
| Gemma4 E4B it | 0.0693 |
| Gemma4 E2B it | 0.0858 |
| Qwen3.5 2B | 0.0972 |
| Gemma4 E2B base | 0.1356 |
| Qwen3.5 0.8B | 0.1381 |
| GPT-J 6B | 0.1936 |
| Gemma4 E4B base | 0.2023 |
| OPT 6.7B | 0.2560 |

This is a strong early signal for the talk frame: older/base models and Gemma
base controls are more sensitive than the strongest modern chat/instruction
models on this metric. Do not overclaim yet; Pythia and GPT-2 XL are still
running, and LLaMA1 is a community conversion.

Generated trajectory figures:

- `runs/trajectory_figures/longprobe_output_trajectory_divergence.png`
- `runs/trajectory_figures/qwen_quantized_output_trajectory_divergence.png`
- `runs/trajectory_figures/current_small_perturbation_semantic_ranking.png`

Launched remaining legacy jobs:

- `chaos-stability-pythia-6p9b-legacy-20260429-001`
- `chaos-stability-gpt2-xl-legacy-20260429-001`

Pulled and processed the remaining legacy jobs. Full small-perturbation semantic
ranking now ends with:

- `Pythia 6.9B`: 0.2370
- `OPT 6.7B`: 0.2560
- `GPT-2 XL`: 0.2822

This reinforces "older/base often more sensitive," while `LLaMA1 7B` remains
the caveat against an era-only story.

## 2026-04-29 Quantization Fidelity Check

Added `scripts/compare_quantized_to_bf16.py` to compare quantized outputs to
the exact BF16 model outputs on the same prompt side.

Small-perturbation distance from BF16 output:

| Model | Precision | Semantic distance from BF16 | Token distance from BF16 |
| --- | --- | ---: | ---: |
| Qwen3.5 0.8B | 8-bit | 0.0982 | 0.5381 |
| Qwen3.5 0.8B | 4-bit | 0.1315 | 0.6213 |
| Qwen3.5 4B | 8-bit | 0.0194 | 0.1606 |
| Qwen3.5 4B | 4-bit | 0.0555 | 0.3082 |

Artifact:

- `runs/quantization_fidelity/qwen_quantized_vs_bf16_small_semantic.png`

Interpretation: this rescues the quantization section. Qwen3.5 0.8B looked
more stable under quantization by prompt-perturbation distance, but its outputs
move farther from BF16 than Qwen3.5 4B does. The better phrasing is:
"stability is cheap; fidelity is not." Apparent stability can come from a
smaller or altered response manifold, not true robustness.

## 2026-04-29 Robust Ladder Completion and Logit Probe Pivot

Processed the completed robust five-model wave with
`scripts/process_robust_wave.py`.

Small-perturbation semantic readout over 24 prompt pairs:

| Model | Mean | 95% bootstrap CI |
| --- | ---: | ---: |
| Qwen3.5 4B | 0.0345 | 0.0182-0.0528 |
| Qwen3.5 9B | 0.0368 | 0.0165-0.0606 |
| Qwen3.5 2B | 0.0728 | 0.0389-0.1150 |
| Gemma4 E4B it | 0.0778 | 0.0406-0.1221 |
| Qwen3.5 0.8B | 0.0887 | 0.0484-0.1368 |

Paired permutation tests:

- `Qwen3.5 4B` vs `Qwen3.5 0.8B`: p = 0.0004
- `Qwen3.5 4B` vs `Qwen3.5 2B`: p = 0.0123
- `Qwen3.5 4B` vs `Qwen3.5 9B`: p = 0.7781
- `Gemma4 E4B it` vs `Qwen3.5 4B`: p = 0.0143

Interpretation: the stronger prompt ladder supports cluster-level claims. The
talk should say `4B` and `9B` are indistinguishable here, while `4B` separates
from `0.8B` and `2B`.

Added logit-level probing to the harness because text distance is a lossy
downstream measurement. New output file: `logit_probes.jsonl`.

What gets captured:

- full-vocab KL and Jensen-Shannon divergence at each probe point
- top-token agreement and winner-rank shifts
- top-token logit margins, so argmax flips can be interpreted as either large
  distribution shifts or tiny changes near a decision boundary
- top-k token/logit/probability snapshots
- teacher-forced logit divergence along both generated continuations

Launched the robust five-model set again with logit probes:

- `chaos-logit-robust-qwen35-08b-20260429-001`
- `chaos-logit-robust-qwen35-2b-20260429-001`
- `chaos-logit-robust-qwen35-4b-20260429-001`
- `chaos-logit-robust-qwen35-9b-20260429-001`
- `chaos-logit-robust-gemma4-e4b-it-20260429-001`

Post-process with:

```bash
uv run python scripts/process_logit_wave.py
```

## 2026-04-29 Reasoning/Scaffold Correlation Pass

Visual inspection of the branching chart surfaced a confound: `Qwen3.5 4B`
and `Qwen3.5 9B` often start with a deterministic `Thinking Process:` scaffold.
This can make common-prefix and text-distance metrics look artificially stable,
because the first tokens are shared boilerplate rather than answer content.

Added `scripts/analyze_scaffold_correlation.py`.

Artifacts:

- `runs/rankings/scaffold_analysis/model_scaffold_annotations.csv`
- `runs/rankings/scaffold_analysis/scaffold_vs_stability.csv`
- `runs/rankings/scaffold_analysis/scaffold_group_summary.csv`
- `runs/rankings/scaffold_analysis/scaffold_bootstrap_difference.csv`

Observed scaffold labels in the final 21-model readout:

| Scaffold kind | Models |
| --- | --- |
| `thinking_process` | `Qwen3.5 4B`, `Qwen3.5 9B` |
| `think_tag` | `Phi-4 reasoning plus`, `SmolLM3 3B` |
| `visible_cot` | `DeepSeek R1 Qwen 7B` |
| `template_echo` | `LLaMA1`, `GPT-J`, `Pythia`, `OPT`, `GPT-2 XL` |
| `none` | Mistral, Gemma IT/base, Granite, Falcon, Qwen 0.8B/2B, OLMo |

Small-perturbation correlation:

| Group | Models | Mean semantic distance | Mean common prefix |
| --- | ---: | ---: | ---: |
| Observed reasoning scaffold | 5 | 0.033 | 38.3 |
| No observed reasoning scaffold | 16 | 0.141 | 20.5 |

Bootstrap difference, scaffold minus non-scaffold: `-0.107` semantic distance
with 95% interval `[-0.140, -0.076]`.

Interpretation:

- The correlation is strong enough to treat reasoning/scaffold behavior as a
  candidate explanatory variable.
- It does not prove reasoning models are truly more content-stable.
- The safer talk line is that scaffolded post-training creates a strong format
  attractor. That may delay divergence, and separating scaffold stability from
  content stability is the next analysis.

## 2026-04-29: Reasoning-scaffold confound surfaced while building talk viz

While building `talk/branching.html` (side-by-side token-stream divergence
demo), a significant confound became visible.

### Observation

Qwen 3.5 4B and 9B emit a deterministic reasoning scaffold before content:

```
Thinking Process:

1.  **Analyze the Request:**
    *   **Topic:** ...
    *   **Tone:** ...
```

Qwen 3.5 0.8B does not. It goes straight to prose.

For the `synonym_happy_joyful` pair, 4B shows `common_prefix = 25 tokens`
before the branch — but the first ~20 of those tokens are the scaffold
matching itself, not robustness to the perturbation. The perturbed word
(`happy` vs `joyful`) only appears inside the scaffold's `**Topic:**` line,
after which divergence begins.

### Why this matters

1. **Cross-model stability comparisons are partially contaminated when the
   panel mixes reasoning-tuned and non-reasoning models.** The 2B→4B cliff
   in the Qwen ladder is probably still real (0.8B diverges inside the
   first content token) but the *magnitude* is inflated by scaffold
   adherence.
2. **Within-model perturbation-tier comparisons are unaffected.** Scaffold
   is constant across tiers for a given model, so the scrubber viz
   (`talk/scrubber.html`) is clean.
3. **This may be a real finding in its own right.** Reasoning-tuned models
   may appear more stable on aggregate metrics simply because they share a
   deterministic preamble. Content-phase stability is the part that actually
   answers the research question.

### Testable hypotheses

- H1: If we strip the scaffold and recompute `common_prefix` from the first
  post-scaffold token, the 2B→4B cliff in `merged_summary.csv` shrinks
  measurably but remains present.
- H2: Reasoning-tuned models (DeepSeek-R1-Qwen-7B, Phi-4 reasoning plus,
  Qwen3.5 4B/9B) will show lower early-token divergence than size-matched
  non-reasoning peers, and comparable or higher late-token divergence —
  i.e. divergence is *delayed*, not *reduced*.
- H3: The scaffold-phase vs content-phase divergence split, measured
  within a single reasoning model, will show near-zero divergence during
  thinking and normal divergence during content.

### Implication for the talk

This is the kind of methodology caveat that strengthens credibility if
presented. Path forward: strip scaffolds, recompute, show both raw and
aligned versions. Target the reasoning-vs-non-reasoning comparison as a
potentially publishable secondary finding.

### Presentation-clarity note

When doing side-by-side token-stream demos, both panels should be the same
model family (reasoning or non-reasoning) — otherwise the eye latches onto
"one has a Thinking block and one doesn't" rather than the divergence
dynamics the demo is supposed to show.

### Follow-up critique (2026-04-29, after scaffold correlation pass)

The scaffold/non-scaffold `-0.107` bootstrap effect is real in our panel,
but four nuances need to be kept in mind when interpreting and presenting
it:

1. **Scaffold presence is collinear with "modern 2025-era post-training."**
   The 5 scaffold models (Qwen 4B/9B, Phi-4 reasoning plus, SmolLM3,
   DeepSeek-R1-Qwen-7B) are also the 5 most heavily modern-post-trained
   models in the panel. We cannot yet attribute the stability effect to
   scaffold presence vs post-training recency — those variables are not
   separable in this data.

2. **`template_echo` models are brittle.** GPT-2 XL, OPT, Pythia, LLaMA1,
   GPT-J all echo templates and are among the least stable models measured.
   That refines the claim: format-adherence *per se* does not produce
   stability. The effect is specific to *modern reasoning scaffolds*.

3. **The scaffold is not pure boilerplate — it reflects the prompt.** In
   Qwen 4B's `Thinking Process`, lines like `**Topic:** happy scientist`
   vs `**Topic:** joyful scientist` contain the perturbation. The scaffold
   itself is doing input-sensitive work. The right decomposition is three
   tiers, not two:
   - *Boilerplate prefix:* the literal `Thinking Process:\n\n1. **Analyze
     the Request:**` frame. Pure format.
   - *Scaffold content:* prompt-reflective lines inside the frame
     (`Topic:`, `Tone:`). Measures sensitivity during reasoning.
   - *Answer content:* post-scaffold output. Measures sensitivity during
     actual answer.
   Only the first tier is pure format stability. Tiers 2 and 3 are both
   content sensitivity at different generation phases.

4. **Token-budget mismatch after stripping.** If a scaffold model spends 50
   of 64 tokens on preamble, content-only comparison gives 14 content
   tokens vs 64 full tokens for non-scaffold models. Two possible fixes:
   (a) truncate all models to first N post-scaffold tokens (e.g. 32), or
   (b) regenerate scaffold models at higher max_tokens so they emit ~64
   *content* tokens. Option (a) is simpler for the talk.

### Boundary detection note

Clean delimiters vary by scaffold kind:

- `<think>...</think>` (Phi-4, SmolLM3): clean regex boundary.
- `Thinking Process:` (Qwen): no reliable close delimiter; trails into
  content. Needs heuristic (first double newline after list, first
  non-numbered-list line, etc.).
- `visible_cot` (DeepSeek-R1): messy, model-specific.

Boundary-detection confidence should be recorded per generation
(`clean` / `heuristic` / `failed`) so downstream analyses can filter or
caveat. Do not pretend the strip is perfect.

### MVP for the talk tomorrow

Rather than the full three-tier decomposition tonight:

1. Strip boilerplate prefix only (regex: `Thinking Process:\n\n1.  \*\*`
   and `<think>.*?</think>`). Don't try to separate scaffold-content from
   answer-content.
2. Truncate all models to first 32 post-scaffold tokens for a fair window.
3. Recompute semantic distance on stripped/truncated generations.
4. One visual: two-bar chart per model — "full output" vs "content-only"
   divergence. The gap is the scaffold effect. If Qwen 4B's content-only
   bar stays low → real robustness. If it jumps to match 0.8B → scaffold
   was doing all the work. Chart is honest either way.

### Safer talk framing

Given the collinearity with post-training recency, avoid:

> "The strongest predictor wasn't size or age — it was whether the model
> enters a stable reasoning scaffold."

Prefer:

> "Among our 21 models, the ones with modern reasoning scaffolds were
> also the most stable. We can't yet separate whether that stability is
> the scaffold itself, the post-training that produced the scaffold, or
> both. Here's what happens when we strip the scaffold: [chart]."

The softer version is bulletproof under audience pushback and invites
exactly the follow-up questions the data can answer.

## 2026-04-29 Raw Prefix Sanity Pass and Long Scaffold Wave

User explicitly asked to stop trusting aggregate summaries and read the
generated outputs. I scanned the first generated words/token IDs for every
output feeding the current 21-model readout and wrote:

- `runs/inspection/generation_prefixes_final21.csv`
- `runs/inspection/generation_prefix_summary_final21.csv`

The sanity pass confirms the scaffold confound is real:

- `Qwen3.5 4B` and `Qwen3.5 9B`: every checked row begins with
  `Thinking Process:`.
- `Phi-4 reasoning plus` and `SmolLM3 3B`: every checked row begins with
  `<think>`.
- `DeepSeek R1 Qwen 7B`: every checked row begins with visible reasoning-style
  prose (`Okay, so...`, `I need...`).
- legacy/base GPT-2 XL, GPT-J, OPT, Pythia, and LLaMA1 often echo templates;
  this is format adherence but not the same phenomenon as modern reasoning
  scaffolds, and it generally correlates with brittle outputs.

Concrete example: on `synonym_happy_joyful`, Qwen 4B/9B first restate the
perturbed word inside the scaffold's `Topic`/`Tone` lines before producing
answer content. That means early common-prefix stability is partly wrapper
stability, but the scaffold itself is also prompt-sensitive content. The
right split remains:

1. boilerplate prefix,
2. scaffold content,
3. answer content.

The existing 64/128-token outputs are often not long enough for content-only
analysis. For example, Qwen 4B robust outputs can still be inside
`3. **Drafting - Attempt 1:**` at the cutoff. Therefore the immediate compute
pivot is to capture longer continuations rather than trying to over-interpret
short outputs.

Launched a 512-token scaffold/content wave with logit probes:

- `chaos-scaffold-long-qwen35-4b-20260429-001`
- `chaos-scaffold-long-qwen35-9b-20260429-001`
- `chaos-scaffold-long-qwen35-08b-20260429-001`
- `chaos-scaffold-long-qwen35-2b-20260429-001`

These run on the robust prompt set with `logit_probe=true`,
`logit_top_k=10`, and `logit_max_steps=256`. The goal is to preserve enough
raw material to later compute:

- raw full-output divergence,
- boilerplate-stripped divergence,
- scaffold-content divergence,
- answer-content divergence,
- prompt-end and trajectory logit divergence.

Sampling demo updated after pulling the Qwen 0.8B job. At temperature `0.7`,
between-prompt distances are generally the same order as within-prompt
sampling distances. That does not invalidate deterministic sensitivity probes,
but it means the talk should not pretend sampled user-facing variance is small.
Better framing: sampling variance and input sensitivity are distinct axes that
can have comparable magnitudes in product settings.

## 2026-04-29 Multi-Account 512-Token Expansion

Checked live SageMaker state and quotas across configured AWS SSO profiles.
Findings:

- preprod (`zh-marketing-preprod-aiengineer`) has the main `ml.g6e.2xlarge`
  quota of 5 and currently runs the heavy 48 GB lane.
- several other accounts expose one `ml.g5.2xlarge` training-job quota; these
  are 24 GB lanes and are useful for smaller/base models.
- `zh-qa-engineer` can list jobs but cannot create them; `zh-qa-aiengineer`
  can create jobs when the request omits tags.
- Some accounts have SageMaker access but no obvious role/bucket pairing; do
  not assume cross-account launchability just because `list_training_jobs`
  works.

Patched:

- `scripts/launch_sagemaker_panel.py` now supports `--no-tags`.
- `scripts/dispatch_sagemaker_queue.py` now passes optional per-job
  `profile`, `region`, `bucket`, `role_arn`, `instance_type`, and `no_tags`.

Launched auxiliary 512-token jobs with logit probes:

- preprod `ml.g5.2xlarge`: `chaos-scaffold-long-gptj-6b-g5-20260429-001`
- preprod `ml.g5.xlarge`: `chaos-scaffold-long-gpt2-xl-g5x-20260429-001`
- preprod `ml.g4dn.2xlarge`:
  `chaos-scaffold-long-gemma4-e2b-it-g4dn-20260429-001`
- zh-ml `ml.g5.2xlarge`:
  `chaos-scaffold-long-pythia-6p9b-ml-g5-20260429-001`
- zh-qa AIEngineer `ml.g5.2xlarge`:
  `chaos-scaffold-long-opt-6p7b-qa-ai-g5-20260429-001`
- marketing prod `ml.g5.2xlarge`:
  `chaos-scaffold-long-llama1-7b-prod-g5-20260429-001`

The target is now 512-token coverage for all 21 final-panel models. Remaining
models are staged in `configs/sagemaker_queue.json`.

## 2026-04-29 Progress Check: 20/25 Scaffold-Long Jobs Ready

Live SageMaker check showed the scaffold-long panel mostly landed. Pulled and
processed newly completed artifacts for Phi-4 reasoning plus, OLMo3, Gemma E4B
base, and the scaled Qwen 0.8B logit run. `scripts/process_scaffold_long_wave.py`
now reports 20/25 configured scaffold-long jobs ready; the missing set is Gemma
E2B base plus the four Qwen thinking-off controls.

Refreshed artifacts:

- `runs/rankings/scaffold_long_wave/small_perturbation_bootstrap.csv`
- `runs/rankings/scaffold_long_logits/semantic_logit_correlations.csv`
- `runs/rankings/scaffold_long_logits/semantic_vs_prompt_end_logits.csv`

Current 512-token semantic ordering, using the ready 20 models, still supports
clusters rather than a leaderboard. DeepSeek R1 Qwen 7B, Qwen3.5 4B/9B,
LLaMA1, Gemma E2B it, and Mistral are in the low-divergence band. Qwen3.5
0.8B, OLMo3, Gemma E4B base, GPT-2 XL, Phi-4 reasoning plus, and legacy
GPT/OPT/Pythia occupy higher-divergence bands. Phi-4 is now an important
cautionary example: "reasoning model" does not automatically mean stable in
this harness.

The logit result remains the cleaner mechanistic lead: prompt-end top-1
probability and margin strongly anti-correlate with later semantic divergence,
while top-1 flip rate correlates positively. That keeps the best explanatory
thread as local decision-boundary / response-attractor margin, with
reasoning-scaffold behavior treated as a qualifier rather than the whole story.

To keep the queue full, launched three additional preprod `g6e` jobs:

- `chaos-scaffold-long-qwen35-9b-thinkoff-20260429-001`
- `chaos-scaled-logit-olmo3-7b-20260429-001`
- `chaos-sample-demo-olmo3-t01-20260429-001`

The side-account Qwen thinking-off controls for 0.8B, 2B, and 4B are training.
These runs are the key controlled test for whether Qwen's apparent stability
survives when visible thinking is disabled via the chat-template flag.

## 2026-04-29 Scaffold-Long Panel Complete

After AWS SSO refresh, all previously active SageMaker jobs had completed.
Pulled and processed the remaining 512-token scaffold-long artifacts:

- Gemma E2B base
- Qwen3.5 0.8B / 2B / 4B / 9B thinking-off controls
- Qwen3.5 9B scaled-logit, Gemma E4B-it scaled-logit, OLMo3 scaled-logit
- OLMo3 temperature-0.1 sampling demo

`scripts/process_scaffold_long_wave.py` now reports 25/25 configured
scaffold-long jobs ready.

Most relevant current readout:

| Run | Mean semantic distance |
| --- | ---: |
| Qwen3.5 4B | 0.050 |
| Qwen3.5 4B think-off | 0.067 |
| Qwen3.5 9B | 0.057 |
| Qwen3.5 9B think-off | 0.072 |
| Qwen3.5 2B | 0.075 |
| Qwen3.5 2B think-off | 0.072 |
| Qwen3.5 0.8B | 0.103 |
| Qwen3.5 0.8B think-off | 0.079 |

Interpretation: visible Qwen thinking/scaffold appears to help 4B/9B look
somewhat more stable, but it is not the entire explanation; the thinking-off
versions remain in the broad stable/mid band. For 0.8B, thinking-off improves
the score, which argues against a simple "thinking scaffold always stabilizes"
story. The correct slide line is: scaffold is a qualifier and sometimes an
attractor, not a universal cause of stability.

The refreshed scaffold-long logit pass includes 20 models with logit artifacts.
Adding Phi-4, OLMo3, and Gemma E4B base weakened the earlier cross-model margin
correlations but did not erase the direction:

- top-1 probability vs semantic divergence: `r = -0.842`
- top-1 flip rate vs semantic divergence: `r = 0.570`
- top-1 margin vs semantic divergence: `r = -0.388`
- prompt-end JS vs semantic divergence: `r = -0.096`

This keeps the decision-boundary story viable, but it should now be phrased
more conservatively: top-token confidence is the cleanest logit predictor so
far; margin/flip rate are useful but more heterogeneous once more models are
included.

Sampling temperature controls:

- Processed OLMo3 temperature-0.1 sampling demo.
- Existing sampling readout still says within-prompt sampling variance is often
  the same order as between-prompt perturbation variance. This is a talk
  caveat, not a failure: deterministic probes isolate prompt sensitivity, while
  sampled product behavior mixes prompt sensitivity with sampling variance.

To keep GPUs warm, launched the next five queued preprod `g6e` jobs:

- `chaos-sample-demo-olmo3-t03-20260429-001`
- `chaos-sample-demo-qwen35-4b-t01-20260429-001`
- `chaos-sample-demo-qwen35-4b-t03-20260429-001`
- `chaos-sample-demo-qwen35-08b-t01-20260429-001`
- `chaos-sample-demo-qwen35-08b-t03-20260429-001`

These were pending capacity immediately after launch. The remaining queued jobs
after this batch are the Qwen quant-logit controls.

## 2026-04-29 Progress Check: Sampling Controls Partially Landed

Live SageMaker check:

- Still training on preprod `g6e`:
  - `chaos-sample-demo-olmo3-t03-20260429-001`
  - `chaos-sample-demo-qwen35-4b-t01-20260429-001`
  - `chaos-sample-demo-qwen35-4b-t03-20260429-001`
- Completed and pulled:
  - `chaos-sample-demo-qwen35-08b-t01-20260429-001`
  - `chaos-sample-demo-qwen35-08b-t03-20260429-001`
- Newly launched to keep the two open `g6e` lanes warm:
  - `chaos-quant-logit-qwen35-4b-bnb8-20260429-001`
  - `chaos-quant-logit-qwen35-4b-bnb4-20260429-001`

Refreshed `scripts/process_sampling_demo.py` after pulling the Qwen 0.8B temp
runs. The current sampling summary remains qualitatively unchanged: for these
demo prompts, within-prompt sampling distances and between-prompt perturbation
distances are often the same order. This supports using sampling variance as a
separate baseline/axis in the talk rather than pretending deterministic
perturbation results map directly to sampled product behavior.

## 2026-04-29 Pivot: High-N Micro-Perturbation Sweep

Reframed the next experiment around edits that are visually/semantically
nearly invisible to humans rather than synonyms or paraphrases. Added
`configs/prompt_pairs_micro_500.json` with 525 pairs: 25 identical controls
and 500 tiny surface perturbations covering trailing/leading whitespace,
newlines/CRLF, line wraps, tabs, duplicate punctuation, spaces around
punctuation, parenthesized words, and duplicated small words.

Started a local Qwen3.5 0.8B run with thinking disabled:
`runs/micro_qwen35_08b_500/qwen35_08b`. Early raw-output inspection shows the
controls are byte-identical while several human-equivalent edits already change
deterministic answer structure or wording, including duplicate punctuation,
extra internal spaces, line wraps, and duplicated small words. This is a better
talk hook than broad model ranking because it makes the perturbation concrete:
the human sees the same instruction, the model sees a different token stream.

Promoted the idea to SageMaker with 512-token direct-answer sweeps and hidden
state capture disabled:

- `chaos-micro-qwen08-512-20260429-001` (`qwen35_08b`, thinking disabled, QA)
- `chaos-micro-qwen2b-512-20260429-001` (`qwen35_2b`, thinking disabled, ML prod)
- `chaos-micro-gemma-e2b-it-512-20260429-001` (`gemma4_e2b_it`, marketing prod)
- `chaos-micro-qwen4b-thinkoff-512-20260429-001` (`qwen35_4b`, thinking disabled, preprod)
- `chaos-micro-qwen9b-thinkoff-512-20260429-001` (`qwen35_9b`, thinking disabled, preprod)
- `chaos-micro-gemma-e2b-base-512-20260429-001` (`gemma4_e2b_base`, preprod)
- `chaos-micro-gemma-e4b-it-512-20260429-001` (`gemma4_e4b_it`, preprod)
- `chaos-micro-gemma-e4b-base-512-20260429-001` (`gemma4_e4b_base`, queued)

Added `scripts/process_micro_sweep.py` to summarize category-level means and
worst examples. It now includes exact `prompt_a` / `prompt_b` text in the
worst-example output so the result can become a slide without reverse
engineering the prompt config.

Local Qwen3.5 0.8B micro sweep completed and processed:
`runs/rankings/micro_qwen35_08b_500/`.

Most important result: tiny edits are not uniformly dangerous. Prefix/suffix
formatting edits were effectively inert in this setup, while internal layout
and syntactic surface changes caused large deterministic branches.

Top category means by semantic output distance:

| Category | Mean semantic distance |
| --- | ---: |
| parenthesize word | 0.077 |
| tab after space | 0.073 |
| blank line wrap | 0.069 |
| line wrap | 0.069 |
| space before punctuation | 0.052 |
| double internal space | 0.051 |

Near-zero categories included leading/trailing spaces, leading/trailing
newlines, CRLF suffixes, tab indentation, and spaces after punctuation. The
slide-grade chart is
`runs/rankings/micro_qwen35_08b_500/micro_category_semantic_bar.png`.

## 2026-04-30 Post-Talk Paper Direction

The next paper-shaped question is mechanism, not leaderboard expansion. The
current data suggests several separable axes:

- prompt-token perturbations, not raw character edits;
- recipe/post-training effects, especially Gemma base vs instruct;
- scaffold/deliberation attractors vs answer-first behavior;
- logit decision-boundary fragility vs bulk full-vocab distribution movement;
- stability vs responsiveness/fidelity, so collapse is not mistaken for
  robustness.

Actions launched:

- Repair queue for partial token-micro jobs with 12-hour per-model timeouts:
  Gemma4 E2B base token-certified v3, OLMo3 v2, and OPT v2.
- Token-certified logit queue for Qwen3.5 thinking-off and Gemma base/instruct
  models. This captures prompt-end logit metrics plus teacher-forced logit
  divergence over the first 64 generated tokens.
- Added `scripts/process_logit_queue.py` so the logit wave has a reusable
  processing path rather than a one-off notebook/script.

## 2026-04-30 Local Mechanistic-Interpretability Seed Tests

Started a local, laptop-sized mechanism pass for the whitespace/punctuation
question. Added:

- `configs/prompt_pairs_mechinterp_seed.json`: hand-picked high-signal
  token-certified micro pairs plus controls.
- `scripts/analyze_branch_points.py`: reads `logit_probes.jsonl` and summarizes
  `pre_branch` / `branch` / `post_branch` logit behavior around the first
  generated-token divergence.
- `scripts/activation_patch_branch.py`: residual activation patching for one
  prompt pair, measuring whether patching clean activations into the corrupted
  run rescues the clean branch token.

Local Qwen3.5 0.8B and 2B thinking-off seed runs completed under
`runs/mechinterp_seed/`. Both reproduce the same qualitative mechanism:
semantically inert formatting edits often leave pre-branch distributions close,
then hit a low-margin branch step where top-1 flips.

Best current causal evidence:

- `qwen35_08b`, `token_cert_parenthesize_word_0434`: adding `(a)` to
  "Describe when a cache..." changes the first token from `A` to `Cache`.
  Patching the clean final-context residual into the corrupted run at the last
  layer fully rescues the clean branch (`rescue_fraction=1.0`).
- `qwen35_08b`, `token_cert_tab_after_space_0572`: tab after space changes the
  branch token from ` critical` to ` key` at generated token 3. Last-layer
  final-context patch fully rescues the clean branch.
- `qwen35_2b`, same two cases: parenthesized `(a)` and tab-after-space are also
  fully rescued by last-layer final-context patching.

Important guardrail: `token_cert_triple_internal_space_0029` looked like a good
late-branch example by output text, but full-forward replay at the nominal
branch did not favor the observed corrupted branch token. Treat such cases as
weak patching targets until the replay/branch-token metric is aligned.

## 2026-04-30 Mechanistic-Interpretability Follow-Through

Added a systematic patch-target selector and patch-result summarizer:

- `scripts/select_patch_targets.py`
- `scripts/summarize_patch_results.py`
- `scripts/plot_patch_heatmap.py`

Selector output:

- `runs/mechinterp_patch/selected_patch_targets.csv`

Additional selected patch cases completed:

- `qwen35_2b`, `token_cert_blank_line_wrap_0357`
- `qwen35_08b`, `token_cert_blank_line_wrap_0212`
- `qwen35_08b`, `token_cert_line_wrap_0406`

Patch summary:

- `runs/mechinterp_patch/patch_summary.csv`

All three newly selected cases were replayable and rescuable. The eight-case
summary now shows a robust but partly expected pattern: final-context residual
patches at late layers rescue the clean branch in every replayable high-signal
case. That is good causal evidence that the branch preference is in the residual
state, but it is not yet a satisfying localization story.

Ran deeper position sweeps and found an important methodological guardrail:
raw same-index all-position patching is misleading for insertion/deletion
prompt deltas because positions after the edit boundary no longer refer to the
same token. Updated `scripts/activation_patch_branch.py` with
`--positions aligned`, using token alignment for equal prompt spans plus common
generated prefixes.

Aligned-position results:

- `qwen35_08b`, `token_cert_parenthesize_word_0434`: strongest non-final rescue
  is the prompt LCP/edit-boundary token (`rescue_fraction ~= 0.92` at layer 1).
  Matched positions away from the edit boundary mostly do nothing. This supports
  an edit-boundary residual-state mechanism rather than a vague global
  punctuation story.
- `qwen35_08b`, `token_cert_tab_after_space_0572`: less localized. The clean
  branch is rescued most strongly at the shared generated-prefix/final context;
  prompt-boundary rescue reaches about `0.60` at layer 19. This looks more like
  a branch-bias accumulated by the shared prefix than a clean single-boundary
  feature.

Heatmaps:

- `runs/mechinterp_patch_aligned/qwen35_08b__token_cert_parenthesize_word_0434.heatmap.png`
- `runs/mechinterp_patch_aligned/qwen35_08b__token_cert_tab_after_space_0572.heatmap.png`
- `runs/mechinterp_patch_aligned/qwen35_2b__token_cert_parenthesize_word_0434.heatmap.png`
- `runs/mechinterp_patch_aligned/qwen35_2b__token_cert_tab_after_space_0572.heatmap.png`

After the SAE pilot, ran Qwen3.5 2B aligned-position sweeps for the same two
cases. They match the feature-overlap story: parenthesized `(a)` has strong
prompt-boundary rescue at the edit/LCP token (`rescue_fraction ~= 0.86` at
layer 0), while tab-after-space is again dominated by the last shared generated
prefix/final context (`rescue_fraction=1.0` at layers 22-23).

Research-lens update: current SAE/circuit-tracing work suggests the next step
should not be "name the whitespace feature" immediately. First keep the branch
target causal and replayable; then inspect SAE activations at the same
layer/position. Qwen-Scope now has an official Qwen3.5 2B Base residual-stream
SAE release close enough to try against the local Qwen3.5 2B branch cases.
Recent SAE robustness work warns that feature interpretations can themselves be
brittle under tiny input perturbations, so causal patching is the guardrail
before trusting a feature label.

Implemented `scripts/extract_qwen_sae_branch_features.py` for the Qwen-Scope
Qwen3.5 2B residual-stream SAEs. Pilot runs:

- `qwen35_2b`, `token_cert_parenthesize_word_0434`, layers 0 and 23.
- `qwen35_2b`, `token_cert_tab_after_space_0572`, layers 18 and 23.

Artifacts:

- `runs/mechinterp_sae/qwen35_2b__token_cert_parenthesize_word_0434__sae_features.csv`
- `runs/mechinterp_sae/qwen35_2b__token_cert_tab_after_space_0572__sae_features.csv`
- `runs/mechinterp_sae/sae_feature_delta_summary.csv`

Readout:

- Parenthesized `(a)` prompt-boundary features are almost disjoint between the
  clean token `" a"` and corrupted token `" ("`: top-20 feature overlap is
  `1/20` at layer 0 and `3/20` at layer 23.
- The same parenthesized case has more overlap at the final context but large
  branch-specific feature deltas at layer 23.
- Tab-after-space has low prompt-boundary feature overlap (`5/20`) but much
  higher final/generated-prefix overlap (`16-18/20`), matching the aligned
  patching result that it is less localized to a single edit-boundary position.

Keep the caveat explicit: these are Qwen-Scope feature IDs, not human-readable
feature labels. The useful current claim is about feature-set overlap/deltas at
causal branch positions, not named "whitespace features."
