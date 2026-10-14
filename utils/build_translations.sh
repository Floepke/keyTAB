#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
I18N_DIR="$ROOT_DIR/i18n"
LRELEASE_BIN="${PYSIDE6_LRELEASE:-$ROOT_DIR/.venv/bin/pyside6-lrelease}"

if [[ ! -x "$LRELEASE_BIN" ]]; then
  LRELEASE_BIN="$(command -v pyside6-lrelease || true)"
fi
if [[ -z "$LRELEASE_BIN" || ! -x "$LRELEASE_BIN" ]]; then
  echo "Missing pyside6-lrelease. Activate the project virtual environment or set PYSIDE6_LRELEASE."
  exit 1
fi

for TS_FILE in "$I18N_DIR"/keytab_*.ts; do
  [[ -f "$TS_FILE" ]] || continue
  QM_FILE="${TS_FILE%.ts}.qm"
  "$LRELEASE_BIN" "$TS_FILE" -qm "$QM_FILE"
  echo "Built translation binary: $QM_FILE"
done
