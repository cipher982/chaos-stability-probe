# BranchTrace + Paper — Spec Snapshot

Dated snapshot, 2026-05-06. This is a one-time planning artifact, not a living
doc. If it survives useful, fold it into `docs/results_digest.md` /
`docs/task_list.md` / `experiments/E##_*/README.md`.

## 1. Goal

Turn the existing chaos-stability-probe artifacts into two linked outputs:

1. **BranchTrace** — a small open-source tool that emits **Branch Cards**
   from already-logged run artifacts. Branch Card = a single
   human/machine-readable record of "where these two runs first diverged, how
   confident was the model, can we replay/flip the branch, which prompt span
   likely controls it."
2. **Short paper (6-8pp, workshop tier)** — "Causally Movable Branch Events
   under Token-Certified Prompt Edits." Core claims drawn from existing E05,
   E07, and E09 artifacts: held-out replication of bidirectional branch
   control, and a two-tier specificity split (edit-boundary vs
   trajectory-state rescue).

Both share the same underlying data and code. The tool is the demo; the paper
is the claim.

## 2. Non-goals

- No new broad model leaderboard. `docs/task_list.md` explicitly froze panel
  expansion. This spec honors that.
- No new SageMaker waves before a Branch Card exists end-to-end on existing
  artifacts.
- Not a generic prompt-regression linter. BranchTrace's unit of work is a
  pair of runs (prompt/model/template A vs B) plus optional open-model
  activation evidence. If the user only logged text, degrade gracefully.
- No SAE/feature-labeling story in v1. E08 is feature-ID only; keep it as
  optional sidecar evidence, not a primary claim.
- Not a web UI. Static HTML + JSON is enough. A notebook viewer is fine.

## 3. Strongest existing evidence (summary)

- **E07 forward (82 cases = 42 hand-selected v1-v3 + 40 held-out random V5,
  across Qwen0.8B/2B/4B/9B + Gemma E2B/E4B instruct/base):** 80/82 full
  final-context rescue; 41/82 full prompt-LCP rescue; 65/82 ≥0.5 prompt-LCP;
  51/82 prompt-LCP beats every aligned prompt-control position. Keep the
  selected vs held-out split explicit in the paper — "82 selected" would
  weaken the selection-bias defense.
- **Three-regime taxonomy of the 82 cases (Opus deep-dive, 2026-05-07).**
  The cleanest carve is a 2×2 crossing of `{prompt_lcp_full?}` ×
  `{immediate_visible_branch?}`:

  | Regime | n | prompt-LCP | generated-prefix | aligned-prompt (best) | Interpretation |
  | --- | ---: | --- | --- | --- | --- |
  | Edit-boundary (EDIT) | 41 | full | full (25/25 silent) | fails in silent (5/25 full) | Edit-boundary is the specific causal handle |
  | Trajectory-migration (OTHER/silent) | 27 | partial (median 0.54) | full (25/27) | weak (3/27 full) | Edit's causal handle has migrated downstream |
  | Prompt-accumulation (OTHER/imm) | 14 | partial (median 0.62) | n/a (branch_t=0) | full (14/14 at last prompt position) | Token-0 edit representation is not a sufficient handle; last-prompt-position is (near-proxy for final-context at branch_t=0) |

  All three regimes replicate on the held-out V5 subset. `event_kind`
  (silent vs immediate) does NOT by itself predict prompt-LCP success;
  what it predicts is which non-LCP position substitutes (aligned-prompt
  when immediate, generated-prefix when silent). `branch_min_margin_logit`
  and branching factor distributions overlap across regimes — the
  basin-switch / cliff-slip hypothesis is NOT in the data.
- **E07 reverse (21 matched cases):** 21/21 full-or-overshoot both
  directions; 19/21 replayable both; 12/21 full prompt-LCP both; best
  position class agrees in 16/21.
- **E07 V5 held-out random (40 cases):** 39/40 finite full-or-overshoot;
  35/40 replayable. Position split: 11 prompt-LCP, 22 final-context,
  5 generated-prefix, 2 aligned-prompt.
- **E09 logit/trajectory (~4200 effective pairs, 8 models):** at-branch
  low-margin AUROC 0.947, JS 0.883. Strict pre-branch-within-1 modest: L2
  0.661, JS 0.618; weakens on long-prefix subset (L2 0.581, JS 0.558).
- **E05 hygiene:** 10-model token-certified ladder (≤500 effective non-
  control pairs/model); OPT 6.7B is fragility baseline (mean 0.289), Gemma
  instruct is most stable (0.059-0.068).

Open asymmetry: 0/82 strict late-only cases under 0.5 rescue cutoff. Claim
shape is "edit-boundary rescue is a specific subset, not early-vs-late
exclusivity."

## 4. Branch Card — schema

Single JSON object per divergence pair; one HTML page renders it.

```jsonc
{
  "schema_version": "branchcard/0.1",
  "id": "qwen35_2b__token_cert_parenthesize_word_0434",
  "generated_at": "2026-05-06T00:00:00Z",

  "run_a": {
    "model_name": "Qwen/Qwen3.5-2B",
    "model_label": "qwen35_2b",
    "decode": {"do_sample": false, "max_new_tokens": 256},
    "runtime": {
      "torch": "2.4.0",
      "transformers": "4.46.x",
      "tokenizer_revision": "...",
      "model_revision": "...",
      "dtype": "bfloat16",
      "device": "cuda",
      "batch_size": 1,
      "attn_impl": "sdpa",
      "seed": 0
    },
    "prompt_text": "...",
    "prompt_token_ids": [...],
    "generated_text": "...",
    "generated_token_ids": [...],
    "run_dir": "runs/sagemaker_artifacts/.../qwen35_2b"
  },
  "run_b": { ... same shape ... },

  "edit": {
    "kind": "micro_parenthesize_word",
    "prompt_token_edit_distance": 3,
    "prompt_token_delta_kind": "small_token_delta_2_3",
    "prompt_token_lcp": 20,
    "visible_diff_span": {"a": "... a ...", "b": "... (a) ..."}
  },

  "branch": {
    "branch_t": 4,
    "event_kind": "silent_logit_divergence",
    "silent_logit_lead": 4,
    "common_prefix_tokens": 4,
    "branch_token_a": {"id": 123, "text": " first", "top1_prob": 0.64,
                       "top1_margin": 0.18, "js_vs_b": 0.31,
                       "effective_branching_factor": 1.41},
    "branch_token_b": {"id": 456, "text": " let",    "top1_prob": 0.52, ...},
    "topk_a": [[123, 0.64], [456, 0.22], ...],
    "topk_b": [[456, 0.52], [123, 0.28], ...]
  },

  "replay": {
    "deterministic_reproducible_a": true,
    "deterministic_reproducible_b": true,
    "forced_prefix_a_flips_to_b_branch_token": null,  // filled by replay prim
    "forced_prefix_b_flips_to_a_branch_token": null
  },

  "selection_provenance": {
    "source": "observed",
    "wave": "activation_patch_v2",
    "pool": "hand_selected_v2",                // hand_selected_v1..v3 | reverse_v4 | heldout_v5 | recommended_casebook
    "archetype": "immediate_branch_control"
  },

  "patch_evidence": {      // source: observed if E07 artifacts exist; else not_run
    "best_position_class": "prompt_lcp",
    "best_layer": 0,
    "best_rescue_fraction": 0.86,
    "final_context_rescue_fraction": 1.00,
    "aligned_prompt_control_max": 0.12,
    "generated_prefix_rescue": 0.20,
    "wave": "activation_patch_v2",
    "heatmap_path": "runs/.../parenthesize_word_0434.heatmap.png"
  },

  "suspected_controlling_span": {
    "source": "derived",                       // observed | derived | not_run
    "token_indices_a": [18, 19, 20],
    "token_indices_b": [18, 19, 20, 21],
    "confidence": "edit_boundary",             // edit_boundary | trajectory_state | ambiguous
    "method": "prompt_lcp_vs_aligned_controls_heuristic"
  },

  "caveats": [
    "Selected case; held-out V5 replication for the class is 35/40 replayable.",
    "Backend/dtype can move branch_t; E10 mean absolute delta 4.25 on Qwen2B."
  ],

  "artifacts": {
    "trajectory_csv": "...",
    "patch_csv": "...",
    "events_csv": "..."
  }
}
```

HTML view (1 page): edit diff + branch token top-k bar + margin/JS sparkline
across common prefix + patch heatmap thumbnail + replay table + caveats
footer.

## 5. Architecture

Repo layout (additive, no refactor of existing scripts):

```
branchtrace/
  __init__.py
  schema.py         # Pydantic models; emit+validate schema_version
  loaders/
    stability_run.py   # read runs/.../{generations,curves,hidden_states}.jsonl
    logit_run.py       # read runs/trajectory_events/*
    patch_wave.py      # read runs/rankings/activation_patch_*/
  build_card.py     # pair(run_a, run_b) -> BranchCard
  render_card.py    # BranchCard -> standalone HTML (jinja2, self-contained)
  replay.py         # forced-prefix replay + prompt-delta bisect primitive
  cli.py            # `branchtrace build` / `branchtrace render` / `branchtrace replay`
scripts/
  build_branch_card.py   # thin shim for one specific case; reuses branchtrace
```

CLI surface (v1, minimal):

- `branchtrace build --run-a RUN_DIR --run-b RUN_DIR --pair-id ID
   [--patch-wave runs/rankings/activation_patch_v2]
   --out cards/<id>.json`
- `branchtrace render cards/<id>.json --out cards/<id>.html`
- `branchtrace replay cards/<id>.json --mode forced-prefix
   [--provider local|openai-api|anthropic-api]`
- `branchtrace bisect cards/<id>.json
   [--granularity token|span]`

Conventions:

- Input = existing run directories and rankings CSVs. Do not recompute.
- Output = one self-contained `.json` + one self-contained `.html` per card,
  under `cards/` (new top-level dir; gitignore by default).
- `replay` must work in three modes: local Transformers (full), open API with
  logprobs (top-k branch only, no patching), text-only API (replay/bisect
  only, document the degradation).

## 6. Paper outline (6-8pp)

Working title: **Causally Movable Branch Events under Token-Certified Prompt
Edits.**

1. **Intro.** Hybrid sequential-system framing (1p). Contribution:
   replayable Branch Card + two-tier specificity claim + held-out rep.
2. **Method.** Token-certified prompt pairs; deterministic decode; aligned
   residual activation patching (prompt-LCP / aligned-prompt-control /
   generated-prefix / final-context positions); reverse-direction and
   random held-out controls. Reference existing configs and scripts by path.
3. **Results.**
   - 3.1 Selected forward waves (E07 v1-v3): 80/82 final, 41/82 prompt-LCP.
   - 3.2 Reverse-direction (v4): 21/21 both; best position class agrees 16/21.
   - 3.3 Held-out random (v5): 35/40 replayable; position split.
   - 3.4 At-branch vs pre-branch prediction (E09): AUROC table.
   - 3.5 Branch timing is not monotonic with parameter count (10.4%/10.4%
     monotone share across Qwen ladder).
4. **Discussion.** Two-tier specificity split. Scaffold confound (E03).
   Collapse confound (E04). Non-universality of branch timing.
5. **Tool.** BranchTrace and one worked Branch Card example (figure).
6. **Limits.** Held-out V5 is 40 cases; pre-branch warning is modest; no
   SAE feature labels; backend/dtype shifts branch timing (E10).
7. **Related.** Li et al. quasi-Lyapunov; activation patching (Meng et al.,
   Zhang & Nanda, Heimersheim & Nanda); paraphrase attractor work;
   formatting sensitivity literature (NeurIPS 2024 attention/format bias).
8. **Conclusion.** One paragraph; defer product framing to BranchTrace repo.

Target venue ordered by fit: BlackboxNLP 2026 (late summer), NeurIPS
interp workshops (fall), ICLR Tiny Papers. Workshop > conference; don't try
to force a full conference submission on this scope.

## 7. Phased plan

**Phase 0 — spec review (now, ≤1h).** Get hatch codex review on this doc.
Resolve must-fix issues, leave nice-to-haves as backlog.

**Phase 1 — one Branch Card, end to end (~1-2 days).**
- Pick hero case: `qwen35_2b__token_cert_parenthesize_word_0434` (sharp
  edit-boundary, layer-0 prompt-LCP rescue ≈0.86, heatmap already exists at
  `runs/mechinterp_patch_aligned/qwen35_2b__token_cert_parenthesize_word_0434.heatmap.png`).
- Build `branchtrace/schema.py` + `loaders/*` for this one case only.
- Build `branchtrace/render_card.py` to emit self-contained HTML.
- Output: `cards/qwen35_2b__parenthesize_word_0434.{json,html}`.
- Smoke test: open HTML locally; confirm edit diff, branch top-k, patch
  evidence, caveats all render.

**Phase 2 — card for second archetype (~0.5 day).** One `silent_logit_lead`
case from recommended_cases.csv to confirm schema handles silent-divergence
shape (the `(a)` case is an `immediate_branch_control` archetype).

**Phase 3 — replay primitive (~2 days).**
- `branchtrace replay --mode forced-prefix` in local Transformers first.
  Given a card, force run A to emit run B's branch token and continue;
  record whether the continuation recovers run B's behavior.
- `branchtrace bisect`: binary-search over the prompt-token edit to find
  the minimum token-visible delta that still reproduces the branch. Output:
  minimal-edit JSON back into the card as a new field.
- Stretch: OpenAI API + Anthropic API adapters for logprob-only replay.
  Scope reduction: v1 can be local-only.

**Phase 4 — paper draft (~3-4 days).**
- Freeze claims against current artifacts only.
- Rebuild figures from existing CSVs via small plotting scripts under
  `experiments/E##_*/` (no new living `scripts/` entries).
- One Branch Card figure in the "tool" section.
- Internal review pass (ask hatch expert or a human reader).

**Phase 5 — public release (~1 day).**
- Clean up README and tag v0.1.
- Push BranchTrace card + spec excerpt as a third blog post.
- Submit paper to chosen workshop.

Total: ~2 weeks of focused effort. Can compress Phase 3 if API adapters slip.

## 8. Open questions for review

1. **Scope of v1 replay.** Is local-only replay acceptable for a workshop
   claim, or does the paper need at least one hosted-API demo to claim
   generality? My bias: local-only is fine for paper; API adapters can land
   in BranchTrace v0.2.
2. **Venue.** BlackboxNLP 2026 vs NeurIPS interp workshop vs ICLR Tiny —
   which matches the two-tier specificity claim best given the sample
   size (82 selected + 40 held-out)?
3. **Branch Card schema surface.** Is `suspected_controlling_span` too
   confident given that 0/82 strict late-only cases exist? Should
   `confidence` be limited to `edit_boundary` / `trajectory_state` /
   `ambiguous` and omit `localized` language?
4. **Reproducibility guarantee.** E10 found backend/dtype shifts branch_t
   materially. Should every Branch Card record the exact `torch.__version__`,
   dtype, device, and batch size used in both runs and refuse to merge
   cards from different backends? My bias: yes; make it a schema-required
   field set.
5. **Paper vs tool ordering.** Submit paper first with BranchTrace repo
   link, or release BranchTrace first as a demo and cite from paper? My
   bias: release tool first (Phase 1-3 ship before Phase 4 submission);
   paper gets a working URL to point at.
6. **Negative-control expansion.** Task-list item #4 asks for "prompt-token-
   effective edits that do not branch." Those cases are already sitting in
   E05 data (identical-behavior prompt-token-effective pairs). Should the
   paper include a small negative-control table, or is it enough to cite
   E05 means and leave the individual cases as BranchTrace test fixtures?
7. **Kill criteria.** If forced-prefix replay in Phase 3 reveals that most
   "causally movable" E07 cases are *not* movable via prefix alone (i.e.,
   need activation patching), does that weaken the paper or strengthen it?
   My reading: strengthens (it makes patching a specific capability, not a
   generic effect); but we need to commit to that framing before we run.

## 9. Codex review response — 2026-05-06

Full review in session log; prioritized punch list folded in here.

**Must-fix before building (applied above or in §10):**

- "82 selected" phrasing is wrong. It is 42 hand-selected forward (v1-v3) plus
  40 held-out random V5. Keep the split explicit. **(applied in §3.)**
- `max_new_tokens: 512` in the example card is false for the logit-probe
  hero runs, which used `256`. Do not let illustrative JSON contain false
  hero values. **(applied in §4.)**
- Every Branch Card must record `torch`, `transformers`, tokenizer + model
  revision, dtype, device, batch size, attn impl, seed, `max_new_tokens`,
  chat template, artifact sha256. E10 shows backend/dtype shifts branch_t
  materially (mean absolute delta 4.25 on Qwen2B, 8.80 on Qwen4B). Refuse
  to merge cards across backends without this metadata. **(applied in §4.)**
- Mark each schema field as `observed` (already in existing artifacts),
  `derived` (recomputed from existing artifacts), or `not_run` (requires
  new compute). Do not let "we could compute this" masquerade as "we have
  this." **(applied in §4.)**
- Narrow v1 to existing local artifacts only. No API adapters, no bisect,
  no blog in v1. **(applied in §10.)**

**Should-fix before submitting:**

- Pair the Qwen2B parenthesize-word hero with a long-prefix / silent case
  (e.g., `gemma4_e4b_base__token_cert_blank_line_wrap_0212` or
  `qwen35_08b__token_cert_blank_line_wrap_0212`, both in
  `recommended_cases.csv` with `silent_logit_lead >= 16`). The `(a)` case
  is good engineering hero but a branch at `t=0`; the paper needs one
  silent/long-prefix example too.
- "Two-tier specificity" must be phrased as a specificity axis
  (specific edit-boundary rescue subset vs broad trajectory-state
  rescue), not two disjoint mechanism classes. 0/82 strict late-only is
  a hard constraint.
- "Causally movable" means activation-patch movable. Do not conflate with
  prefix-replay movable until Phase 3 measures it.
- Forced-prefix kill criteria defined *before* Phase 3:
  - If full deterministic replay of runs A/B is unstable across our
    local Transformers environment: cut replay from core paper claims.
  - If forced-prefix flips branch tokens in <30% of E07 prompt-LCP-rescue
    cases: report it as a negative control, keep the paper
    activation-patching-centered, and demote replay to a BranchTrace
    capability for silent/trajectory-state cases only.
  - If forced-prefix succeeds in >70%: patching and prefix are
    redundant for this subset; we still keep patching for the cases
    where prefix fails, and we note the overlap.
- Add negative-control table to paper: prompt-token-effective edits that
  do not branch (plentiful in E05), plus replay-unstable branch cases
  if any surface in Phase 3.
- Add model/license + data-release sections. Qwen and Gemma redistribution
  terms; what raw vs derived artifacts ship publicly; exact cards released.
- Statistical hygiene: prompt pair is the unit; report clustered CIs
  where AUROC is central; do not inflate `n` to token level.
- Venue framing: if the paper centers causal activation patching +
  interventions, NeurIPS interp workshop likely fits better than
  BlackboxNLP. BlackboxNLP fits better if the paper centers Branch Cards
  as black-box/open-model debugging artifacts. Pick one frame and commit
  before drafting.

**Nice-to-have (explicit backlog, not v1):**

- OpenAI / Anthropic API replay adapters.
- `branchtrace bisect` prompt-delta binary search.
- Standalone `pip install branchtrace` polish.
- Third blog post.

**Framing changes applied to §5, §7:**

- Tool-first means "one working demo artifact before paper drafting,"
  not "harden BranchTrace before writing."
- Phase 3 replay is local-only; API adapters move to backlog.

## 10. Hatch expert strategic read — 2026-05-06

One-shot expert consultation (no web search; reasoning=medium). Verdict:
**ship it, but pivot paper center of gravity.**

### Approvals

- Novelty clears a workshop bar *if framed as a careful empirical/
  methodological note* — not as a big new mechanistic discovery. The
  combination is what is novel: token-certified edits + deterministic
  branch localization + causal residual patching + held-out random
  replication + explicit specificity-axis framing + Branch Card artifact.
- 2 weeks solo is "barely, but yes" — only if scope is frozen now.
- Venue ranking confirmed: **NeurIPS interp workshop first** with the
  empirical/causal framing. BlackboxNLP is the fallback only if the tool
  gets cleaner than expected, not the primary target.

### Paper center of gravity — pivot

**Superseded twice.** Rev 1 (2026-05-06): pivoted from tool-adjacent
framing to 0/82 strict late-only as central bite. Rev 2 (2026-05-07,
after Opus deep-dive on §3 data): the clean finding is a **three-regime
taxonomy**, not a two-class split. 0/82 remains a key control; the
positive claim is the taxonomy.

**Title:** "Three Regimes of Causal Rescue for Token-Certified Prompt
Edits."

**Thesis (write the paper toward this line):**

> Token-certified tiny edits induce branch events that are causally
> localizable, and the locus depends on whether the edit's effect has
> been integrated into generated context. We identify three regimes:
> **edit-boundary rescue** (41/82; prompt-LCP fully rescues and aligned-
> prompt controls fail in the silent subset 5/25 vs 25/25),
> **trajectory-migration rescue** (27/82 silent-divergence cases;
> generated-prefix fully rescues 25/27 while prompt-LCP is degraded but
> alive at median 0.54 and aligned-prompt controls weak), and
> **prompt-accumulation rescue** (14/82 immediate-branch cases; all
> with tokenization-shift at token 0; the last-prompt-position — a
> near-proxy for final-context at branch_t=0 — fully rescues while
> the token-0 edit representation is partial). Strict late-only
> rescue is empirically nonexistent (0/82), so these are not late
> overwrites; they are a continuum of increasingly downstream causal
> handles as the edit integrates into generated state.

**Pre-registered positive test from the taxonomy:** in silent-divergence
cases, we predict generated-prefix full rescue. Observed: 50/52
(EDIT-silent 25/25 + OTHER-silent 25/27). This is the specific second
finding that distinguishes trajectory-migration from mere "broad
trajectory-state overwrite."

**Credibility anchors:** held-out V5 replication (35/40), reverse-
direction controls (21/21), 0/82 strict late-only.

**Claims to kill / not make:**

- "Basin-switch vs cliff-slip" taxonomy. Margin distributions overlap;
  it is not in the data.
- "Most apparent rescues are broad state overwrites." Wrong direction:
  the positive finding is three distinguishable regimes with specific
  signatures.
- "Edit-boundary is the universal causal handle." It is 41/82; name
  the other two regimes honestly.

### Tool framing — demotion

BranchTrace v0.1 is a **Branch Card format + renderer**, not a
"general branch debugging tool." The paper artifact framing is honest;
the product framing currently is not (generating artifacts is the hard
part; tool wraps that).

Minimum "useful to others" threshold:

- documented JSON schema (Pydantic)
- schema validator or example cards
- at least 2-3 self-contained example cards
- one-command static HTML render
- provenance fields (already in spec §4)
- enough input fields that someone can manually populate from their own run

**Stretch goal for usefulness:** minimal local capture path that, given
two prompts and a HF-style local model, produces divergence + top-k +
replay fields. Patch evidence stays optional. This is the line between
"artifact viewer" and "reusable tool." Aim for it in v1 if Phase 3 lands
cleanly; accept viewer-only if it slips.

### Slip-cut order (re-confirmed)

If Phase 3 forced-prefix replay explodes, cut in this order:

1. Cut claims that depend on new replay (paper stays activation-patching-
   centered).
2. Cut broad replay model coverage; keep one hero + one counterexample.
3. Cut polished HTML features.
4. Cut tool-as-product claim (Branch Card format + renderer only).
5. **Do not cut**: the 0/82 late-only limitation, the held-out V5
   replication, or the reverse-direction controls. Those are the paper.

### Minimum viable paper (floor):

- two Branch Cards (hero + long-prefix silent),
- existing 82-case patching table with specificity breakdown,
- held-out V5 replication table,
- E09 branch prediction as supporting/contextual result (not central),
- very clear limitations section with 0/82, backend/dtype, scaffold, and
  selection provenance called out explicitly.

### Secondary stories — expert ranking

1. **(c) 0/82 strict late-only** — strongest; now central.
2. **(d) backend/dtype branch-timing sensitivity** — broadly relevant
   reproducibility hazard. Use as warning box / limitations call-out.
   Too thin to promote to main result without systematization.
3. **(b) Gemma base/instruct contrast** — secondary evidence that
   branch behavior depends on training recipe; easy to overinterpret.
4. **(a) non-monotonic branch timing with size** — provocative but
   fragile (too many confounds). Keep as exploratory footnote only.

### Evidence that would change the verdict (kill conditions for the plan)

- Forced-prefix replay fails on hero cases — cut replay from paper.
- Backend/dtype shifts reverse the main patching conclusions — stop
  and systematize (d) before submitting.
- Branch Cards cannot be rendered from a clean public schema — cut tool
  from paper; leave it as internal infrastructure.

## 11. Opus deep-dive response — 2026-05-07

One-shot Opus 4.7 consultation with direct CSV inspection. Question:
"Is the 41/82 other half worth a deep dive?" Answer: yes, but not as a
new headline. It cleanly splits into two sub-populations with different
mechanistic signatures, yielding a three-regime taxonomy (folded into
§3 and §10 above).

### Follow-up analyses (no new compute)

1. **Pre-registered gen-prefix test.** Report clean: "In silent-divergence
   cases, we predicted generated-prefix full rescue. Observed: 50/52."
2. **OTHER/imm parenthesize audit (completed 2026-05-07).** Result is
   clean but NOT an LCP-definition win. All 30 immediate-branch cases
   have `first_diff_token=0` (tokenization shifts from the very start
   of the prompt), so there is no multi-boundary edit to redefine LCP
   around. Across all 30 immediate cases, the **last prompt position
   fully rescues (30/30)** — regardless of regime. The EDIT/imm vs
   OTHER/imm split is "token-0 edit representation *also* works"
   (EDIT/imm 16/30) vs "only the last-prompt-position handle works"
   (OTHER/imm 14/30). OTHER/imm is therefore best described as a
   **prompt-accumulation** case: the edit's effect becomes a sufficient
   causal handle only after it has propagated through the prompt to
   the final position. Last-prompt-position for immediate branches is
   a near-proxy for final-context (same timestep for branch_t=0).
   Implication: the three-regime taxonomy survives, but OTHER/imm's
   prose should read "prompt-accumulation rescue" rather than
   "prompt-position-shifted rescue." Headline stays 41/82; no
   re-classification win.
3. **Soft edit-integration-time figure.** Bin silent cases by branch_t
   (0-2, 3-5, 6-10, 11-25, 26+); plot prompt-LCP full rate vs gen-prefix
   full rate. Currently prompt-LCP falls 60%→40% while gen-prefix stays
   ≥90%. Small effect, clean visual.
4. **Gemma base concentration in OTHER/silent.** 6/11 Gemma E2B base
   cases are OTHER/silent (vs 5/11 Gemma E2B IT, 2/11 Qwen 4B). One
   paragraph, no scaling law overclaim.

### Framing changes applied

- §3 rewritten around the 2×2 crossing with a regime table.
- §10 thesis replaced with three-regime version.
- Basin-switch / cliff-slip and "broad trajectory-state overwrite" are
  killed.
- 0/82 strict late-only demoted from central finding to control that
  licenses the continuum framing.

## 12. Immediate next step after review

If review lands cleanly: claim Phase 1 as `E11_branchtrace_card` in
`docs/experiment_index.md`, and start `branchtrace/schema.py` +
`loaders/stability_run.py` against the hero case.
