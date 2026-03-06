# keyTAB — Klavarskribo Score Engraver

Welcome to **keyTAB**, a passion project for creating, editing, and engraving Klavarskribo scores. keyTAB blends a piano-roll style editor with print-ready engraving so you can compose, arrange, and share music in the vertical, keyboard-centric notation.

## Highlights
- Klavarskribo-first workflow: vertical staves, per-hand coloring, dot indicators, and snap-to-grid editing tailored for keyboard music.
- Fast engraving: Cairo-based renderer with headers/footers, page counts, and automatic line breaks; analysis snapshot tracks notes, measures, lines, and pages.
- Powerful selection shortcuts: global arrows transpose and time-shift selections; brackets set hand/color; platform-aware undo/redo.
- MIDI import: load .mid/.midi, set title from filename, and auto-apply quick line breaks (6 measures per line) to paginate instantly.
- Smart layout tools: quick line break dialog, measure grouping, beam/slur/dynamic/text tools, and configurable snap size.
- Session safety: autosave, undo/redo, recent files, and embedded fonts/icons for consistent output.

## Core Features
- Editing tools for notes, grace notes, beams, slurs, pedal, dynamics, cresc/decresc, text, tempo, repeats, and line breaks.
- Selection operations: transpose by semitone, shift in time by snap units, assign hand (`<` left, `>` right; note color set to `auto`), cut/copy/paste, delete, select-all.
- Layout & style: adjustable zoom (mm per quarter), page margins, stave ranges, color presets, and per-hand coloring that flows into engraving.
- Engraving: multi-page rendering with headers/footers, document info, creation timestamp, and page numbering suitable for print/PDF.
- Info & analysis: title/author/copyright plus live analysis of notes, measures, lines, pages, and grace notes.

## Typical Workflow
1. Create or import: start a new score or load MIDI via File → Load.
2. Shape layout: set snap size, apply quick line breaks (e.g., 6 per line), tweak style.
3. Edit music: add notes/beams/slurs/text/tempo; use shortcuts to transpose or time-shift selections.
4. Review & engrave: view pages, verify headers/footers and line breaks; check analysis counts.
5. Export: save `.piano` files or print/PDF from the engraved view.

## Selection Shortcuts
- `[` / `]`: set hand to left/right and reset note color to `auto`.
- `←` / `→`: transpose selection ±1 semitone.
- `↑` / `↓`: shift selection in time by one snap unit.
- `Backspace` / `Delete`: remove selection.
- `Ctrl/Cmd+Z`, `Ctrl/Cmd+Shift+Z`: undo/redo.

## MIDI Import Behavior
- Parses tempo, time signatures, and note data; maps pitches to app keys.
- Sets score title from filename.
- Auto-applies quick line breaks in 6-measure groups so pages are ready to inspect immediately.

## Install & Run (Python)
- Python 3.x with PySide6, Cairo, pretty_midi/mido (see `requirements.txt`).
- Create a venv, `pip install -r requirements.txt`, then run the app entry point (e.g., `python keyTAB.py`).

## Project Status
Active and evolving. Expect iterative improvements and occasional breaking changes while features solidify.

## Contributing
Feedback and PRs are welcome. If you try keyTAB, please report crashes, layout glitches, or engraving edge cases—real scores help drive fixes.
