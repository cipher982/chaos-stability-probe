#!/usr/bin/env bash
# Rebuild Fig 2 — cropped hero Branch Card screenshot.
#
# Renders cards/qwen35_2b__parenthesize_word_0434.html via headless Chrome,
# then crops to end right after the Patch evidence section's heatmap (before
# the Replay section header begins).
#
# The 2460px crop height was chosen by visual bisection on 2026-05-12;
# if the card template changes the crop may need to be re-tuned.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
CARD_HTML="$REPO/cards/qwen35_2b__parenthesize_word_0434.html"
OUT_PNG="$REPO/experiments/E11_branchtrace_card/figures/branch_card_hero.png"
TMP_FULL="$(mktemp -t card_full.XXXXXX).png"

CROP_HEIGHT=2460
RENDER_HEIGHT=3500
RENDER_WIDTH=1200

if [[ ! -f "$CARD_HTML" ]]; then
  echo "missing card HTML: $CARD_HTML" >&2
  exit 1
fi

"$CHROME" --headless=new --disable-gpu \
  --window-size=${RENDER_WIDTH},${RENDER_HEIGHT} \
  --screenshot="$TMP_FULL" \
  "file://$CARD_HTML" >/dev/null

uv run --directory "$REPO" python - <<PY
from PIL import Image
im = Image.open("$TMP_FULL")
im.crop((0, 0, $RENDER_WIDTH, $CROP_HEIGHT)).save("$OUT_PNG")
print(f"wrote $OUT_PNG (size {im.size[0]}x{$CROP_HEIGHT})")
PY

rm -f "$TMP_FULL"
