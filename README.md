# keyTAB — Klavarskribo Score Engraver

Welcome to **keyTAB**, a passion project for creating, editing, and engraving Klavarskribo scores. keyTAB blends a piano-roll style editor with print-ready engraving so you can compose, arrange, and share music in the vertical, keyboard-centric Klavarskribo notation.

## Highlights
- Klavarskribo-first workflow: vertical staves, per-hand coloring, dot indicators, and snap-to-grid editing tailored for keyboard music.
- Fast engraving: Cairo-based renderer with headers/footers, page counts, and automatic line breaks; analysis snapshot tracks notes, measures, lines, and pages.
- Powerful selection shortcuts: global arrows transpose and time-shift selections; brackets set hand/color; platform-aware undo/redo.
- MIDI import: load `.mid`/`.midi` files using a pure-Python byte-level parser — no external dependencies. Sets title from filename, handles corrupt/truncated files gracefully, and auto-applies quick line breaks (6 measures per line) to paginate instantly.
- MIDI export: writes Standard MIDI Files (format 1) with three tracks — track 0 (tempo/time signature), track 1 (left hand, channel 0), track 2 (right hand, channel 1).
- MusicXML import: load `.musicxml`/`.mxl`/`.xml` files.
- Smart layout tools: quick line break dialog, measure grouping, beam/slur/dynamic/text tools, and configurable snap size.
- Session safety: autosave, undo/redo, recent files, and embedded fonts/icons for consistent output.

## Core Features
- Editing tools for notes, grace notes, beams, slurs, pedal, dynamics, cresc/decresc, text, tempo, repeats, and line breaks.
- Selection operations: transpose by semitone, shift in time by snap units, assign hand (`<` left, `>` right; note color set to `auto`), cut/copy/paste, delete, select-all.
- Layout & style: adjustable zoom (mm per quarter), page margins, stave ranges, color presets, and per-hand coloring that flows into engraving.
- Engraving: multi-page rendering with headers/footers, document info, creation timestamp, and page numbering suitable for print/PDF.
- Info & analysis: title/author/copyright plus live analysis of notes, measures, lines, pages, and grace notes.
- Error reporting: failed imports and engraver errors show a dialog with a full traceback and a **Copy Error Log** button.

## Typical Workflow
1. Create or import: start a new score or load a MIDI or MusicXML file via File → Load.
2. Shape layout: set snap size, apply quick line breaks (e.g., 6 per line), tweak style.
3. Edit music: add notes/beams/slurs/text/tempo; use shortcuts to transpose or time-shift selections.
4. Review & engrave: view pages, verify headers/footers and line breaks; check analysis counts.
5. Export: save `.piano` files, export to MIDI, or print/PDF from the engraved view.

## Selection Shortcuts
- `[` / `]`: set hand to left/right and reset note color to `auto`.
- `←` / `→`: transpose selection ±1 semitone.
- `↑` / `↓`: shift selection in time by one snap unit.
- `Backspace` / `Delete`: remove selection.
- `Ctrl/Cmd+Z`, `Ctrl/Cmd+Shift+Z`: undo/redo.

## MIDI Import Behavior
## MIDI Import
- Pure-Python byte-level parser — no dependency on `mido` or `pretty_midi`.
- Handles corrupt or truncated MIDI files: bad meta-message fields are defaulted rather than crashing.
- Parses tempo changes, time signatures, and note data; maps MIDI pitches to app keys (MIDI pitch − 20).
- Drum channel (MIDI channel 9) is skipped.
- Dangling note-on events with no matching note-off receive a short fallback duration.
- Sets score title from the filename stem.
- Auto-applies quick line breaks in 6-measure groups so pages are ready to inspect immediately.
- On any parse failure the importer shows an error dialog with a copyable traceback rather than silently failing.

## MIDI Export
- Writes a valid Standard MIDI File (format 1, 480 ticks per beat) using pure Python and stdlib only.
- Three tracks: **track 0** — tempo map and time signature; **track 1** — left hand (MIDI channel 0); **track 2** — right hand (MIDI channel 1).
- Grace notes are exported with a 1/8-beat duration.

## Install & Run (Python)
- Python 3.x with PySide6 and Cairo (see `requirements.txt`).
- Create a venv, `pip install -r requirements.txt`, then `python keyTAB.py`.

## Translations (PySide6 / Qt Linguist)
- keyTAB includes a Qt translation workflow with TS/QM files in `i18n/`.
- Set `ui_language` in Preferences to `system`, `en`, or `nl`.
- Update translatable source strings:
	- `bash scripts/update_translations.sh`
- Translate Dutch strings with Qt Linguist:
	- Open `i18n/keytab_nl.ts` in Qt Linguist and fill missing entries.
- Optional: prefill untranslated strings using a translation service (LibreTranslate-compatible API):
	- `python scripts/auto_translate_ts.py i18n/keytab_nl.ts --source en --target nl`
	- Restrict to dialog contexts only (example):
		- `python scripts/auto_translate_ts.py i18n/keytab_nl.ts --source en --target nl --only-context InfoDialog --only-context LineBreakDialog --only-context StyleDialog`
	- If your service requires an API key or custom URL:
		- `python scripts/auto_translate_ts.py i18n/keytab_nl.ts --url https://your-service/translate --api-key YOUR_KEY`
- Build runtime translation file (`.qm`):
	- `bash scripts/build_translations.sh`
- Restart keyTAB to apply language changes.

## Project Status
Active and evolving. Expect iterative improvements and occasional breaking changes while features solidify.

## Contributing
Feedback and PRs are welcome. If you try keyTAB, please report crashes, layout glitches, or engraving edge cases—real scores help drive fixes.
