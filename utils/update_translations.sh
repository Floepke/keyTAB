#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
I18N_DIR="$ROOT_DIR/i18n"

mkdir -p "$I18N_DIR"

for lang in nl de; do
  pylupdate6 \
    "$ROOT_DIR" \
    --exclude "*/__pycache__/*" \
    --exclude "*/.git/*" \
    --no-obsolete \
    --ts "$I18N_DIR/keytab_${lang}.ts"

  echo "Updated translation source: $I18N_DIR/keytab_${lang}.ts"
done
