#!/usr/bin/env bash
# Render talk/companion_notes.md -> talk/companion_notes.pdf
# Uses pandoc for HTML, headless Chrome for PDF. No LaTeX required.
set -euo pipefail

cd "$(dirname "$0")"

TMPDIR_LOCAL=$(mktemp -d -t companion.XXXX)
CSS="$TMPDIR_LOCAL/style.css"
HTML="$TMPDIR_LOCAL/companion.html"
trap 'rm -rf "$TMPDIR_LOCAL"' EXIT

cat > "$CSS" <<'EOF'
@page { size: Letter; margin: 0.6in 0.75in; }
body { font-family: -apple-system, "Helvetica Neue", sans-serif; font-size: 10.5pt; line-height: 1.45; color: #1a1a1a; max-width: none; }
h1 { font-size: 18pt; margin-top: 0; }
h2 { font-size: 13pt; margin-top: 1.2em; padding-top: 0.4em; border-top: 2px solid #c8402c; page-break-after: avoid; }
h3 { font-size: 11pt; }
p, li { orphans: 3; widows: 3; }
blockquote { border-left: 3px solid #c8402c; padding: 0.1em 0.8em; margin: 0.5em 0; color: #333; background: #faf6f4; }
code { font-size: 9.5pt; background: #f0ede9; padding: 1px 4px; border-radius: 3px; }
pre { font-size: 9pt; background: #f0ede9; padding: 6px 10px; border-radius: 4px; overflow-x: auto; }
em { color: #555; }
strong { color: #000; }
hr { border: none; border-top: 1px dashed #bbb; margin: 1em 0; }
ul { padding-left: 1.4em; }
EOF

pandoc companion_notes.md \
  -o "$HTML" \
  --standalone \
  --metadata title="Companion Notes" \
  -c "$CSS" \
  --embed-resources

"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless --disable-gpu --no-pdf-header-footer \
  --print-to-pdf=companion_notes.pdf \
  "file://$HTML"

echo "✓ talk/companion_notes.pdf rendered"
