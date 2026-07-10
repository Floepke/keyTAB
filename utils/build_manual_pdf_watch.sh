#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MD_FILE="$ROOT_DIR/docs/NL_manual/NL_handleiding.md"
IMG_DIR="$ROOT_DIR/docs/NL_manual/img"
OUT_PDF="$ROOT_DIR/docs/NL_manual/NL_handleiding.pdf"
POLL_SECONDS="${POLL_SECONDS:-2}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not found at $PYTHON_BIN." >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import pypandoc  # noqa: F401
from weasyprint import HTML  # noqa: F401
PY

if [[ $? -ne 0 ]]; then
  echo "Required Python packages are missing. Install pypandoc-binary and weasyprint in the venv first." >&2
  exit 1
fi

hash_tree() {
  {
    find "$IMG_DIR" -maxdepth 1 -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' -o -name '*.webp' -o -name '*.gif' \) -print0 \
      | xargs -0 -r sha256sum \
      | sort
    sha256sum "$MD_FILE"
  } | sha256sum | awk '{print $1}'
}

build_pdf() {
  echo "[manual-pdf] rebuilding $(basename "$OUT_PDF")"
  "$PYTHON_BIN" - "$MD_FILE" "$OUT_PDF" "$ROOT_DIR/docs/NL_manual" <<'PY'
from pathlib import Path
import sys

import pypandoc
from weasyprint import HTML

md_file = Path(sys.argv[1])
out_pdf = Path(sys.argv[2])
base_dir = Path(sys.argv[3])

html = pypandoc.convert_file(str(md_file), 'html', format='md')
page_css = """
@page {
  size: A4;
  margin: 18mm;
}
body {
  font-family: sans-serif;
  font-size: 11pt;
  line-height: 1.35;
}
img {
  max-width: 100%;
  height: auto;
  display: block;
  margin: 0.4em auto;
  border: 1px solid #000;
  box-sizing: border-box;
  border-radius: 5px;
  box-shadow: 0 1.5pt 5pt rgba(0, 0, 0, 0.18);
}
"""
full_html = f"""<!doctype html>
<html>
<head>
  <meta charset='utf-8'>
  <style>{page_css}</style>
</head>
<body>
{html}
</body>
</html>
"""
HTML(string=full_html, base_url=str(base_dir)).write_pdf(str(out_pdf))
PY
}

last_hash=""
current_hash="$(hash_tree)"
build_pdf
last_hash="$current_hash"

echo "[manual-pdf] watching for changes in docs/NL_manual... (poll ${POLL_SECONDS}s)"
while true; do
  sleep "$POLL_SECONDS"
  current_hash="$(hash_tree)"
  if [[ "$current_hash" != "$last_hash" ]]; then
    last_hash="$current_hash"
    build_pdf
  fi
done
