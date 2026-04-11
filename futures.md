# Futures

Derived from available git commit subjects in this repository.

Note:
- This list is inference-based (not a strict backlog).
- You mentioned repository switching, so older context may be missing.
- Items are grouped by confidence based on repeated commit themes.

## High-confidence futures

- [ ] Add visual regression samples for engraver edge cases (measure numbering, ledger groups, repeats, end barline, continuation dots).
- [ ] Continue collision-system unification so measure numbers, repeat symbols, and other right-side annotations share one placement engine.
- [ ] Expand notation completeness around dynamics/hairpins/slurs/grace notes with more edge-case behavior parity between editor and engraver.
- [ ] Keep improving cross-platform packaging and release flow (macOS app/DMG, Windows builds, AppImage).
- [ ] Improve playback consistency with score semantics (repeats, tempo mapping, marker behavior).
- [ ] Continue translation/i18n coverage and polish (Dutch and update tooling).
- [ ] Keep performance tuning for large scores (editor drawing, hit testing, engraver throughput).
- [ ] Add stronger import/export reliability and compatibility (MusicXML + MIDI roundtrip quality).

## Medium-confidence futures

- [ ] Improve style system UX: safer presets, clearer defaults, and better migration of style fields.
- [ ] Extend notehead/accent/accidental and articulation behavior where rendering and editing still diverge.
- [ ] Add more robust diagnostics/reporting for engraver/runtime errors with actionable UI feedback.
- [ ] Improve line-break and measure distribution tooling for complex pagination scenarios.
- [ ] Expand scripting workflows (automation helpers around quantize/transpose/export pipelines).

## Lower-confidence but plausible futures

- [ ] Add optional CI checks for lint/type/basic render smoke tests before release artifacts are built.
- [ ] Add a small curated "reference score pack" used for regression testing and release verification.
- [ ] Introduce clearer release notes/changelog automation tied to version bumps.

## Candidate milestones

### Milestone A: Engraving stability
- [ ] Finalize remaining right-side collision edge cases.
- [ ] Add reproducible edge-case fixtures.
- [ ] Lock baseline rendering snapshots.

### Milestone B: Release hardening
- [ ] One-command build validation on macOS/Windows/Linux.
- [ ] Packaging smoke tests (font/audio/plugin presence).
- [ ] Release checklist + changelog automation.

### Milestone C: Interop and workflow
- [ ] Improve MusicXML import fidelity.
- [ ] Improve playback correctness for repeats/tempo markers.
- [ ] Streamline line-break/style editing workflow.
