---
name: qa-slides
description: Visually QA the Marp deck in talk/. Renders slides to PNGs, spawns vision subagents to inspect and fix layout issues. Use after any edit to talk/slides.md that touches tables, divs, code blocks, images, or anything layout-adjacent. The main agent never reads slide PNGs — that's the whole point.
---

# Visual QA for the talk deck

Use this full-deck workflow only for deck-wide changes: theme/CSS changes,
slide-number shifts, multi-slide layout rewrites, tables/images across many
slides, or explicit user requests for full-deck QA.

For a small edit to one known slide, do the scoped workflow instead:

1. Render with `uv run --with Pillow python talk/qa_slides.py`.
2. Inspect only the affected `talk/slide_images/slide.NNN.png` image(s).
3. Fix only the affected slide block in `talk/slides.md` if inspection finds a
   real defect.
4. Do not spawn full-deck inspector/fixer agents.
5. Do not edit generated data files or unrelated scripts.

The full-deck main agent is the coordinator. It never reads slide PNGs. It orchestrates two subagents:

1. **Inspector** (read-only) — reads every PNG, reports defects as a structured list.
2. **Fixer** (edits + re-inspects) — receives the defect list, edits `talk/slides.md` minimally to resolve each, re-renders, re-inspects the edited slides, iterates until clean or up to 3 passes.

This two-stage design exists because:
- Vision reads are expensive and pollute main-agent context.
- The main agent knows the deck's substantive intent (what claim a slide is making, what can be compressed vs what can't). The Fixer needs the Inspector's findings routed through that judgment.
- Separation of concerns: Inspector can't accidentally change content; Fixer's edits are bounded by the Inspector's list.

## Protocol

### Step 1 — Render

```bash
cd /Users/davidrose/git/chaos && uv run --with Pillow python talk/qa_slides.py
```

This cleans `talk/slide_images/`, regenerates one PNG per slide, `slides.html`, and `slides.pdf`. No judgment, just rendering.

### Step 2 — Inspector subagent (read-only)

Spawn with `Agent` tool, `subagent_type: general-purpose`. Prompt body:

```
You are doing visual QA on a rendered Marp slide deck. The PNG files are at:

  /Users/davidrose/git/chaos/talk/slide_images/slide.001.png through slide.NNN.png

Read every PNG in numeric order using your Read tool (which has vision).

Defects to flag:
  - Content clipped at the bottom edge (text cut mid-line, incomplete final
    bullet, table rows missing, text overlapping the footer or page number).
  - Content clipped at the right edge (text running off the slide).
  - A column or div that renders empty when the source clearly intended two
    columns of content.
  - Tables with misaligned cells, rows outside the table, or broken widths.
  - Images that failed to load (broken-image placeholder, giant gap).
  - Overlap between text and images.
  - Any other rendering glitch a human would notice instantly.

Do NOT flag:
  - Intentional whitespace.
  - Content that reaches the bottom but is fully readable.
  - Aesthetic preferences.

For each slide, output exactly one line:
  slide.NNN — OK
  slide.NNN — BROKEN: <one-sentence description>

End with:
  Total slides reviewed: N
  Slides needing fixes: M
  List of filenames needing attention.

Under 400 words. Do not read anything other than the PNGs. Do not edit anything.
```

The main agent routes the Inspector's list forward. It should briefly confirm which defects the Fixer will attempt to resolve.

### Step 3 — Fixer subagent (edits + re-inspects)

Spawn a fresh `Agent` (general-purpose). Give it:
- The Inspector's defect list (full verbatim).
- Explicit scope: only touch `talk/slides.md`, only to fix the listed defects.
- Re-render + re-inspect responsibility.

Prompt body:

```
You are fixing visual layout defects in a Marp slide deck at
/Users/davidrose/git/chaos/talk/slides.md. You have a list of broken slides
from a prior Inspector pass:

<PASTE INSPECTOR OUTPUT HERE>

Your job:
  1. For each broken slide, read the relevant block in talk/slides.md
     (slides are separated by --- markers; count them to locate slide N).
  2. Make a minimal edit to fix the defect. Preferred fixes, in order:
       - Tighten prose (shorter sentences, dropped articles, compressed lists).
       - Reduce font size of an over-tall table via HTML <div style="font-size:0.9em">.
       - Resize images (change h:NNN in Marp image syntax).
       - Split a slide only as a last resort.
     Do NOT change the substantive claim of any slide. Do NOT delete bullets
     unless the same content appears on another slide.
  3. After all edits: re-render with
       cd /Users/davidrose/git/chaos && uv run --with Pillow python talk/qa_slides.py
  4. Re-inspect ONLY the slides you edited (and any slides that shifted
     position because you split one). Read those PNGs directly with your
     Read tool.
  5. For any slide still broken, iterate: another minimal edit, re-render,
     re-inspect. Cap at 3 iterations per slide. If still broken after 3,
     report it and stop.

Return a final structured report:
  - For each originally-broken slide: FIXED / STILL BROKEN / SPLIT INTO N SLIDES
  - List of files edited (should be just talk/slides.md).
  - Any slide numbers that shifted and may need re-review.

Under 500 words.
```

### Step 4 — Main agent wrap-up

- If Fixer reports all FIXED: tell the user, summarize what was changed.
- If Fixer reports STILL BROKEN after 3 iterations on any slide: surface the specific slide and defect, ask the user to intervene or accept.
- If slide numbers shifted due to a split: mention it; no need to re-run Inspector unless the user asks.

## What the main agent never does

- Read PNGs in slide_images/.
- Apply layout fixes directly (unless the user overrides).
- Trust any prior "looks fine" claim from a previous session — always re-run the Inspector on the current render.

## Vision backend choice

Use Claude Sonnet vision subagents for both Inspector and Fixer. If the
Claude Code `Agent` tool is available, use it. Otherwise use the surfaced
Claude Sonnet path (`mcp__hatch__hatch_claude` with `model="sonnet"` or
`hatch claude sonnet`) from the repo root.

Do not route this skill through alternate paid vision/model backends unless
the user explicitly requests that backend for the current run.
