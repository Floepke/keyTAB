#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
I18N_DIR="$ROOT_DIR/i18n"
TS_FILE="$I18N_DIR/keytab_nl.ts"
QM_FILE="$I18N_DIR/keytab_nl.qm"
LRELEASE_BIN="${PYSIDE6_LRELEASE:-$ROOT_DIR/.venv/bin/pyside6-lrelease}"

if [[ ! -f "$TS_FILE" ]]; then
  echo "Missing TS file: $TS_FILE"
  echo "Run utils/update_translations.sh first."
  exit 1
fi

if [[ ! -x "$LRELEASE_BIN" ]]; then
  LRELEASE_BIN="$(command -v pyside6-lrelease || true)"
fi
if [[ -z "$LRELEASE_BIN" || ! -x "$LRELEASE_BIN" ]]; then
  echo "Missing pyside6-lrelease. Activate the project virtual environment or set PYSIDE6_LRELEASE."
  exit 1
fi

"$LRELEASE_BIN" "$TS_FILE" -qm "$QM_FILE"

echo "Built translation binary: $QM_FILE"
