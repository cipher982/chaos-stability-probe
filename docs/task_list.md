# Task List

Operational board only. Keep historical narrative out of this file; use
`docs/results_digest.md`, `docs/experiment_index.md`, and per-experiment
`experiments/E##_*/README.md` files for durable readouts.

## Source Of Truth

- `docs/results_digest.md` - current interpretation and safest claims.
- `docs/experiment_index.md` - one row per experiment.
- `configs/models.json` - model registry and observed scaffold behavior.
- `configs/sagemaker_queue_*.json` - explicit SageMaker queues.
- `runs/` - raw and derived artifacts; generated source of truth, not committed.

## Active Framing

- Main research object: structured trajectory branching under tiny
  token-certified prompt edits.
- Practical tool direction: BranchTrace, a branch-level debugger for LLM
  behavior regressions.
- Avoid broad leaderboard expansion unless it tests a specific mechanism.
- Treat prompt pair as the statistical unit.
- Do not describe `branch_within_N` as pre-branch warning; it includes the
  branch timestep. Use strict `pre_branch_within_N` for warning claims.
- Current strongest claim: tiny-edit branch events are often causally movable,
  but prompt-LCP/edit-boundary rescue is a subset, not the universal mechanism.

## Live Operations

Last checked: 2026-05-01 after processing E07 v5.

- No active SageMaker jobs found in the latest `chaos` scan.
- No newly failed jobs require rerun.
- Recent stopped token-micro jobs were superseded by later completed repair
  jobs; do not rerun without a new reason.

## Current Readouts

### E07 Mechanistic Branch Patching

Artifacts:

- `runs/rankings/activation_patch_v1/`
- `runs/rankings/activation_patch_v2/`
- `runs/rankings/activation_patch_v3/`
- `runs/rankings/activation_patch_v4_reverse/`
- `runs/rankings/activation_patch_v5_replication/`
- `runs/rankings/activation_patch_comparison/`

Readout:

- Forward selected waves: 80/82 cases have full final-context rescue.
- Prompt-LCP/edit-boundary rescue is real but not universal: 41/82 full,
  65/82 at least 0.5, and 51/82 stronger than every aligned prompt-control
  position.
- Strict late-only cases remain 0/82 under the 0.5 rescue cutoff after
  aligned-prompt and generated-prefix controls.
- Reverse-direction controls: 21/21 matched cases have full-or-overshoot rescue
  in both directions; 19/21 are replayable in both directions; 12/21 have full
  prompt-LCP rescue in both directions.
- Randomized held-out V5 replication: 39/40 finite full-or-overshoot rescue
  cases and 35/40 replayable full-or-overshoot rescue cases.
- V5 best-position split: 11 prompt-LCP, 22 final-context, 5 generated-prefix,
  2 aligned-prompt.

Decision: stop expanding activation-patching waves unless a new run tests a
specific objection. The next value is synthesis, figures, negative controls,
and BranchTrace-style artifacts.

### E09 Trajectory Events

Artifacts:

- `runs/trajectory_events/logit_token_cert_v1/`
- `runs/trajectory_artifacts/logit_token_cert_v1/`

Readout:

- Complete 8-model panel: Qwen0.8B/2B/4B/9B plus Gemma E2B/E4B instruct/base.
- At-branch AUROC is strong: low-margin `0.947`, JS `0.883`.
- Strict pre-branch-within-1 is modest: centered L2 `0.661`, JS `0.618`.
- On long-prefix cases (`branch_t >= 5`), strict pre-branch-within-1 weakens to
  centered L2 `0.581`, JS `0.558`.
- Qwen ladder branch timing is not monotonic with size; only `10.4%` of shared
  cases are monotonic earlier-with-size and `10.4%` monotonic later-with-size.
- Gemma base models branch earlier and more often immediately than their
  instruction-tuned siblings.

### E10 Silent Divergence

Artifacts:

- `runs/rankings/silent_divergence_pilot_v1/`
- `runs/rankings/silent_divergence_local_qwen_ladder_meta_20260430/`
- `runs/rankings/e10_backend_comparison_20260430/`

Readout:

- Qwen2B/Qwen4B/Qwen9 CUDA/bfloat16 metadata exists.
- Backend/dtype can move branch timing materially: local-vs-SageMaker mean
  absolute branch-t delta is `4.25` for Qwen2B and `8.80` for Qwen4B.
- Treat E10 as case-selection/intervention evidence, not a general scaling law.

### E05 Token-Certified Micro

Artifact:

- `runs/rankings/token_micro_v3/combined_model_summary.csv`

Readout:

- OPT 6.7B remains the fragile v3 baseline: mean `0.2894`, p90 `0.5146`.
- Gemma4 E2B base is second-most fragile: mean `0.1786`, p90 `0.3505`.
- Qwen0.8B/2B/4B/9B thinking-off means: `0.0930`, `0.0912`, `0.0855`,
  `0.0786`.
- Gemma4 E4B/E2B instruct means: `0.0684`, `0.0591`.
- OLMo3 7B instruct is partial: `0.0860` mean over 152 effective rows.

### Batch Determinism

Artifacts:

- `runs/sagemaker_artifacts/chaos-batch-det-*/runs/batch_determinism.json`

Readout:

- Qwen0.8B CUDA/bfloat16 singleton-vs-batch checks completed on both
  `ml.g6e.2xlarge` and `ml.g5.2xlarge`.
- Batching is faster but not token-exact:
  - `g6e`: batch size 2/4/8 mismatched `1/8`, `3-4/8`, `5/8` prompts.
  - `g5`: batch size 2/4/8 mismatched `3/8`, `2/8`, `5/8` prompts.
- Do not batch science-critical branch-timing generations unless the paper
  explicitly treats batch shape as part of the runtime condition.

## Next Actions

Planning spec with Codex review: `docs/branchtrace_spec_20260506.md`. Until a
Phase is complete, all work goes through that spec; do not open a new plan.

### Phase 0.5 — OTHER/imm audit (completed 2026-05-07)

**Result: 3-regime taxonomy survives; rename OTHER/imm regime.**

All 30 immediate-branch cases have `first_diff_token=0` (tokenization
shift at start of prompt) and full rescue at the last prompt position
(30/30). EDIT/imm = "token-0 edit representation also works"; OTHER/imm
= "only last-prompt-position works." No multi-boundary to redefine LCP
around, so no headline bump. Rename the regime from "prompt-position-
shifted" to **"prompt-accumulation"** — the edit's effect only becomes
a sufficient handle after it has propagated through the prompt to the
final position. Last-prompt-position at branch_t=0 is a near-proxy for
final-context.

Headline stays 41/82 edit-boundary + 27/82 trajectory-migration +
14/82 prompt-accumulation. Pre-registered gen-prefix test stands at
50/52.

### Phase 1 — BranchTrace v1, one hero card (active)

1. Claim `E11_branchtrace_card` row in `docs/experiment_index.md`.
2. Scaffold `branchtrace/` package: `schema.py`, `loaders/stability_run.py`,
   `loaders/logit_run.py`, `loaders/patch_wave.py`, `build_card.py`,
   `render_card.py`, `cli.py`. Jinja2 for HTML, Pydantic for schema.
3. Build hero Branch Card JSON from existing artifacts for
   `qwen35_2b__token_cert_parenthesize_word_0434`:
   - logit/trajectory: `runs/trajectory_events/logit_token_cert_v1/`
     and `runs/sagemaker_artifacts/chaos-logit-token-cert-qwen2b-thinkoff-*/`
   - patch evidence: `runs/mechinterp_patch_aligned/qwen35_2b__token_cert_parenthesize_word_0434.*`
     and `runs/rankings/activation_patch_v2/`
   - required fields: full runtime env metadata, artifact sha256s,
     `source` tag on every block (observed|derived|not_run),
     `selection_provenance.pool` set to `hand_selected_v2`.
4. Render static HTML; smoke test locally.
5. Commit as `E11_branchtrace_card/` with short restart README.

### Phase 2 — second archetype card

6. Build a Branch Card for a long-prefix / silent case, e.g.
   `gemma4_e4b_base__token_cert_blank_line_wrap_0212` from
   `runs/trajectory_artifacts/logit_token_cert_v1/case_selection/recommended_cases.csv`.
7. Confirm schema handles `silent_logit_divergence` events and long
   common prefixes without special-casing.

### Phase 3 — forced-prefix replay (local only, kill criteria pre-committed)

8. Implement `branchtrace replay --mode forced-prefix` in local
   Transformers. API adapters out of scope for v1.
9. Run replay on both hero cards plus ~20 E07 prompt-LCP-rescue cases.
10. Apply kill criteria from spec §9:
    - <30% forced-prefix flips → demote replay; paper stays
      activation-patching-centered.
    - 30-70% → keep as a secondary primitive; note overlap.
    - >70% → note redundancy with patching for this subset; keep
      patching for the rest.
11. Record outcome in `experiments/E11_branchtrace_card/README.md`.

### Phase 4 — paper draft (6-8pp workshop)

**Venue committed: NeurIPS interp workshop** (expert confirmed this is
both more honest and more likely to land than BlackboxNLP given the
evidence). BlackboxNLP remains the fallback only if the tool gets much
cleaner than expected; do not draft for both venues.

**Center of gravity: three-regime taxonomy from Opus deep-dive 2026-05-07.**
Thesis line: "Token-certified tiny edits induce branch events that are
causally localizable, and the locus depends on whether the edit's effect
has been integrated into generated context. Three regimes: edit-boundary
rescue (41/82), trajectory-migration rescue (27/82, generated-prefix
rescues 25/27), and prompt-position-shifted rescue (14/82, multi-boundary
edits). Strict late-only rescue is 0/82 — a continuum of downstream
causal handles, not late overwrites."

12. Rebuild figures from existing CSVs via scripts under
    `experiments/E##_*/`:
    - **Three-regime rescue figure (NEW, central):** per-case scatter or
      bar set showing prompt-LCP / generated-prefix / aligned-prompt /
      final-context rescue fractions, faceted by regime (EDIT-silent,
      EDIT-imm, OTHER-silent / trajectory-migration, OTHER-imm /
      prompt-shifted). Source: `runs/rankings/activation_patch_comparison/case_level_summary.csv`.
    - E07 best rescue position classes (existing:
      `runs/rankings/activation_patch_comparison/e07_best_rescue_position_classes.png`;
      re-emit for paper sizing).
    - Pre-registered gen-prefix figure: 50/52 silent cases with full
      generated-prefix rescue, split EDIT-silent 25/25 vs OTHER-silent
      25/27.
    - Soft edit-integration-time figure: prompt-LCP full rate vs
      gen-prefix full rate binned by branch_t (0-2, 3-5, 6-10, 11-25, 26+)
      on silent cases.
    - at-branch vs strict pre-branch AUROC bars from
      `runs/trajectory_events/logit_token_cert_v1/branch_prediction_auc.csv`
      — supporting/contextual result, NOT central.
    - Branch Card figure (hero case, small inset).
    - Negative-control table: prompt-token-effective edits with no
      branch (pull from E05 token-certified).
13. Commit claims to three-regime framing. Do NOT sell early-vs-late
    causality (0/82 blocks it) or basin-switch/cliff-slip (not in data).
14. Write ethics / licenses / data release section.
15. Limitations section must explicitly call out: 0/82 strict late-only
    as continuum-not-dichotomy evidence; backend/dtype branch-timing
    shifts (E10 mean 4.25 on Qwen2B); scaffold confound (E03);
    selection provenance (42 hand-selected vs 40 held-out V5);
    prompt-pair is statistical unit; OTHER/imm may be LCP-definition
    artifact (see Phase 0.5 audit).

### Phase 5 — release

16. Tag `branchtrace v0.1`; ship cards + minimal README.
17. Submit paper. Third blog post optional.

### Explicit non-goals (v1)

- No OpenAI / Anthropic API replay adapters.
- No `branchtrace bisect` prompt-delta binary search.
- No new SageMaker waves.
- No broad model leaderboard expansion.
- No standalone PyPI polish.

## Useful Commands

```bash
uv run python scripts/sagemaker_status.py --prefix chaos --max-results 60
uv run python scripts/process_activation_patch_queue.py --queue configs/sagemaker_queue_activation_patch_v5_replication.json --out-dir runs/rankings/activation_patch_v5_replication
uv run python scripts/compare_activation_patch_waves.py
uv run python scripts/process_logit_queue.py --queue configs/sagemaker_queue_logit_token_cert_v1.json --out-dir runs/rankings/logit_token_cert_v1
uv run python scripts/process_silent_divergence_queue.py --queue configs/sagemaker_queue_silent_divergence_pilot_v1.json --out-dir runs/rankings/silent_divergence_pilot_v1
uv run python scripts/build_trajectory_artifacts.py --trajectory-dir runs/trajectory_events/logit_token_cert_v1 --silent-summary runs/rankings/silent_divergence_local_qwen_ladder_meta_20260430/silent_divergence_readout.csv
```
