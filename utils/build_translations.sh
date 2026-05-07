#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
I18N_DIR="$ROOT_DIR/i18n"
TS_FILE="$I18N_DIR/keytab_nl.ts"
QM_FILE="$I18N_DIR/keytab_nl.qm"

if [[ ! -f "$TS_FILE" ]]; then
  echo "Missing TS file: $TS_FILE"
  echo "Run utils/update_translations.sh first."
  exit 1
fi

pyside6-lrelease "$TS_FILE" -qm "$QM_FILE"

echo "Built translation binary: $QM_FILE"
