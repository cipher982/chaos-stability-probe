# E11 — BranchTrace v1 + paper

Restart notes for the Branch Card build. Planning spec:
`docs/branchtrace_spec_20260506.md` (includes 2026-05-06 Codex review
response and kill criteria).

## Question

Can existing E05/E07/E09 artifacts be packaged as reusable **Branch Cards**
(JSON + HTML) that a reviewer can read end-to-end without re-running any
model? And in Phase 3, does forced-prefix replay recover branch control in
the E07 prompt-LCP-rescue subset?

## Inputs

- Hero (edit-boundary / immediate-branch):
  `qwen35_2b__token_cert_parenthesize_word_0434`
  - Logit/trajectory: `runs/trajectory_events/logit_token_cert_v1/`
    and the SageMaker run under
    `runs/sagemaker_artifacts/chaos-logit-token-cert-qwen2b-thinkoff-*/`.
  - Patch evidence: `runs/mechinterp_patch_aligned/qwen35_2b__token_cert_parenthesize_word_0434.{csv,json,heatmap.png}`
    and `runs/rankings/activation_patch_v2/`.
- Long-prefix / silent hero (Phase 2 — pick one): candidates from
  `runs/trajectory_artifacts/logit_token_cert_v1/case_selection/recommended_cases.csv`,
  e.g. `gemma4_e4b_base__token_cert_blank_line_wrap_0212`
  (`branch_t=20`, `silent_logit_lead=58`, `long_shared_prefix_branch`).
- Spec-mandated metadata: runtime env (torch/transformers/dtype/device/batch/
  attn_impl/seed), tokenizer + model revision, `max_new_tokens`,
  chat template, artifact sha256s.

## Commands (Phase 1 target)

```bash
# Scaffold package (not yet committed)
branchtrace/
  schema.py
  loaders/{stability_run.py,logit_run.py,patch_wave.py}
  build_card.py
  render_card.py
  cli.py

# Build hero card
uv run python -m branchtrace.cli build \
  --run-a runs/sagemaker_artifacts/chaos-logit-token-cert-qwen2b-thinkoff-20260430-001/runs/qwen35_2b \
  --run-b same-dir-with-perturbed-pair \
  --pair-id token_cert_parenthesize_word_0434 \
  --patch-wave runs/rankings/activation_patch_v2 \
  --patch-aligned runs/mechinterp_patch_aligned \
  --out cards/qwen35_2b__parenthesize_word_0434.json

uv run python -m branchtrace.cli render \
  cards/qwen35_2b__parenthesize_word_0434.json \
  --out cards/qwen35_2b__parenthesize_word_0434.html
```

## Outputs

- `cards/<id>.json` — Branch Card (schema `branchcard/0.1`, per spec §4).
- `cards/<id>.html` — self-contained HTML view.
- `branchtrace/` — reusable package.
- Paper figures will sit under `experiments/E11_branchtrace_card/figures/`.
- `paper/signature_lab_data.json` — generated interactive-lab data bundle
  (schema `signature-lab/0.1`) built from patch, trajectory, logit, and SAE
  artifacts.
- `paper/signature_explainer.html` — self-contained interactive lab. Rebuild
  with `uv run python -m branchtrace.cli signature-lab --repo-root .`.

## Current readout

- Spec drafted 2026-05-06. Codex review folded in: "82 selected" corrected
  to "42 hand-selected + 40 held-out V5"; `max_new_tokens` in hero card is
  `256` (not `512`); runtime metadata is mandatory; every schema field
  tagged `observed|derived|not_run`; API adapters + bisect + blog moved
  to backlog.
- Hatch expert consultation 2026-05-06: **ship it with pivot.** Paper
  center of gravity moves from tool-adjacent framing to the 0/82 strict
  late-only finding as central intellectual bite. Venue committed:
  NeurIPS interp workshop. BranchTrace demoted to "Branch Card format
  + renderer," not "general branch debugging tool."
- Opus deep-dive 2026-05-07: rev 2 of paper framing. The clean finding
  is a **three-regime taxonomy** (edit-boundary 41, trajectory-migration
  27, prompt-position-shifted 14), not a two-class split. 0/82 demoted
  from headline to control that licenses the continuum framing. Thesis
  line now in spec §10 rev 2. Pre-registered gen-prefix test: 50/52.
  Basin-switch/cliff-slip killed (not in data).
- Phase 0.5 audit completed 2026-05-07: all 30 immediate-branch cases
  have `first_diff_token=0` and full last-prompt-position rescue (30/30).
  No multi-boundary to redefine LCP around; no headline bump. Regime
  renamed from "prompt-position-shifted" to "prompt-accumulation":
  the token-0 edit handle is insufficient, but the last-prompt-position
  (near-proxy for final-context at branch_t=0) rescues fully.
- Phase 3 forced-prefix replay kill criteria pre-committed (<30% demotes
  replay; 30-70% keeps it secondary; >70% notes redundancy).

## Caveats

- Backend/dtype shifts branch_t materially (E10 mean absolute delta 4.25
  on Qwen2B, 8.80 on Qwen4B). Every Branch Card must record exact
  runtime; refuse to merge cards across backends without this.
- `suspected_controlling_span` is a heuristic (prompt-LCP vs aligned
  prompt controls). Do not phrase as a mechanism claim.
- `(a)` hero is an immediate visible branch at `t=0`. Phase 2 pair must
  be a silent/long-prefix case for paper hero status.
- Two-tier specificity = specificity axis, not disjoint mechanisms.
  0/82 strict late-only is a hard constraint.

## Next action

Phase 3 minimal run complete (2026-05-10):
`scripts/run_forced_prefix_replay.py` →
`runs/forced_prefix_replay/phase3_qwen2b/{summary.csv,detail.jsonl}`.
11 qwen35_2b cases with local branches + 1 no-local-branch (MPS vs
SageMaker backend drift). Post-force free-decode of 10 tokens, measured
by token-LCP vs A's continuation.

Preliminary by regime (N too small to claim):

  edit_boundary        mean rejoin 8.0 / 10  (n=8)
  prompt_accumulation  mean rejoin 10 / 10   (n=1)
  trajectory_migration mean rejoin 5.0 / 10  (n=2, split 10 & 0)

Caveats: token-LCP under-counts semantically-identical rejoins that pick
different surface tokens (e.g. 0344 continues same topic in 2 tokens
then diverges lexically, rejoin=2). Expanding to more models or adding
a logprob-of-suffix metric are options if the paper needs it; for now
this is a support figure.

Phase 4 rev 4: `paper/draft.md` (2026-05-12) — submission-cleanup.
Third expert pass verdict: "submission-ready after cleanup, not
after new science."

Applied: prompt-LCP defined unambiguously (first differing prompt
token at index k); aligned-prompt controls labeled sampled
non-boundary; §3.2 reframed as "reserved-wave rule transfer" with
explicit note that V5 patch CSVs existed before the rule was
written, so the transfer check is not prospective preregistration;
rule-of-three given explicit iid caveat (pairs aren't iid); abstract
softened ("arguing against a purely generic-perturbation explanation
at least for this one-model control"); conclusion softened
("makes a purely generic-perturbation account unlikely, though it
does not settle the question"); held-out summary merged into §3.2;
reverse-direction moved to §4 exploratory; §3 trimmed to six
subsections; denominator-flow table added at §3.1; all three
wrong-donor strict-replay matches tabulated in §3.5 with target /
donor token pairs and the "common continuation word" pattern
discussed.

Subsequent cleanup completed Fig 2, real citations, ethics/license/data
release text, and final prose readthrough. Current paper draft is
`paper/draft.md`; review renders are generated artifacts under `paper/`.

Interactive signature lab (2026-05-13):
`branchtrace/signature_lab.py` now generates `paper/signature_lab_data.json`
and `paper/signature_explainer.html` from raw artifacts. The default case is
`qwen35_2b__token_cert_line_wrap_0573`, a non-token-zero boundary-rescue
example (`branch_t=9`). The lab also includes
`gemma4_e2b_base__token_cert_blank_line_wrap_0212` for generated-prefix rescue,
`qwen35_2b__token_cert_line_wrap_0378` for the immediate tokenization-shift
edge case, and observed SAE pilot feature bars where available.
