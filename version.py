"""
Single source of truth for keyTAB versioning.

Bump policy (Semantic Versioning):
  MAJOR - breaking changes (incompatible file format, major UI overhaul)
  MINOR - new features, backward-compatible
  PATCH - bug fixes only
"""

__version__ = "1.1.0beta"
APP_NAME    = "keyTAB"

change_log = '''
1.0.0 (2026-04-01)
- initial release.

1.0.1 (2026-04-05)
- fixed repeat symbols and measure numbers in the engraver when ledger lines are present.

1.0.2 (2026-04-13)
- midi export and import is now dependency free (no more mido or pretty_midi).
- fix: midi import can open older Klavarscript midi files without errors.

1.0.3 (2026-04-14)
- copyright default value changed to "© keyTAB {datetime.now().year}".
- physical undo/redo implemented as buttons in the toolbar, in addition to keyboard shortcuts ctrl+z and ctrl+shift+z.
- added "Set as Default Style" file menu item to save the current style as the default for new projects.
- added "Reset Default Style" file menu item to reset the default style to the built-in defaults.

1.0.4 (2026-04-20)
- fixed libfluidsynth dependency issues on Linux AppImage build.
- fixed beam tool sometimes not detecting the correct hand on click time.
- fixed Windows Cairo rendering issue where dynamic symbols (LelandText) could display as missing glyphs in editor/engraver while still appearing correctly in the Qt menu.
- startup now ensures LelandText is registered/installed on Windows similarly to Edwin so Cairo can resolve the font reliably.
- Windows Inno Setup installer now installs both Edwin and LelandText automatically.
- fixed Windows engraver worker crash caused by LelandText font resolution touching Qt font registry APIs inside the spawned worker process.
- changed preferences UI to use radio buttons instead of dropdowns.
- changed the behavior of velocity editing mode: it no longer remembers the toggle state, as that was impractical in regular usage.
- the app restart path is now more stable and robust (on Linux, the second restart previously caused a crash).
- fix: the save_on_exit preference now works properly, and closeEvent is updated to show the correct information to the user.
- stripped double barline on end barline as it looked unnatural and is not the official Klavarskribo convention.
- fix: the app state is now properly saved in the score file and restored when loading a file.

1.0.5 (2026-04-21)
- implemented in engraver: slurs that span line/page breaks are now split and continue on the next line/page.
- implemented in engraver: connected slurs now render an indication at line-break points to show continuity.
- implemented horizontal read mode in the engraver; this probably still requires further fine-tuning.
- fix: the full-screen toggle checkbox now works correctly in the menu on app startup.
- the Fit button now also works as a splitter handle, allowing drag-resize of the editor and print-preview areas; the tooltip is updated to reflect this behavior.
- implemented a miniature piano keyboard in the editor below the end barline, showing the full key range with octave numbers as a visual pitch-layout aid.
- Fit page to window now works correctly when page orientation is landscape.

1.0.6 (2026-05-05)
- expanded horizontal read/orientation support with multiple engraving/layout fixes (including tempo marker rotation/placement and general horizontal mode behavior).
- print view playhead implemented and stabilized for horizontal read mode.
- fit/viewport behavior refined: fit-to-window in landscape corrected and splitter fit handle behavior improved.
- editor now includes a miniature piano keyboard visual aid, plus updated notehead design and dark-mode default color fine-tuning.
- editing refinements: velocity editing fixes and time-shift keeps rhythmic selection rectangle content intact.
- added single-key shortcuts `z`, `x`, `c`, `v` for undo/cut/copy/paste in addition to existing standard shortcuts.
- added middle-mouse drag panning for editor and print view, plus Ctrl/Cmd+scroll zoom in print view.
- translation updates and general polish/fixes across the release cycle.

1.0.7 (2026-06-01)
- fix: the app now restarts correctly on Linux AppImage builds without crashing.
- fix: repeat symbol positioning corrected in horizontal editor mode.
- scrollbar is now at the bottom in horizontal editor mode.
- finetuned Klavarskribo time signature indicator positioning in horizontal editor mode.
- analysis: added avg_frequency, pitch range, most used pitch, and left/right balance metrics.
- new scripts: reverse, mirror pitch, and updated double/half time script.
- improved text rendering in engraver and editor; with text dialog enhancements, you can now form a multiline text rectangle using the built-in multiline widget from Qt.
- fix: note editing no longer triggers the overlap guard unnecessarily when working with triplets/antisymmetric figures.
- print view: playhead focus now correctly targets both X and Y viewport axes.
- fix: MIDI player no longer makes timing/duration mistakes with fine floating-point note values; MIDI player is overall enhanced

1.0.8 (2026-05-11)
- engraver: dynamic symbols now rendered at 45° rotation with rotated polygon background; hairpin connects to rotated symbol bounds.
- engraver: mini piano keyboard added to first system of first page in both vertical and horizontal read directions, with configurable visibility, octave numbering, color, and keyboard height.
- style dialog: new Repeat tab; visibility toggles appear at top of every tab and are mirrored in the Visibility tab.
- engraver: time signature indicator guide line and Klavarskribo numbering Y positions now correctly account for mini piano reserved height.
- engraver: improved viewport culling logic in line and text drawers for more accurate rendering.
- editor: grace note rendering and selection logic improved for better visual contrast and accuracy.
- editor: note continuation dot logic improved for beam drawing.
- editor: time comparison logic refactored across tools and drawers for consistency.
- editor: note editing performance optimizations.
- fix: line break dialog labels updated for horizontal read direction.
- fix: enhance clipboard functionality by tracking start units for better alignment during cut/copy operations.
- fix: on Windows tooltips now correctly display special characters and multiline text without issues.

1.1.0beta (2026-05-11)
- update to 1.1.0: the interface changes and new features are significant enough to warrant a minor version bump.
- ui change: moved the contextual toolbar to the left side of the editor as a second vertical toolbar.
  * the reason: to have more space for tool buttons on smaller screens.
  * the middle vertical toolbar holds button functions that are always available.
  * the contextual toolbar holds tool-specific functions that belong to the active tool.
- [✓] added: Keyboard Shortcut Card under menu 'Help > Keyboard Shortcut Card'.
- [✓] added: Escape exit confirmation dialog to prevent accidental exits.
- [✓] added: dynamic symbol custom rotation option, allowing symbols to be rotated by a custom angle.
- [] added: multiple staves per system support in the engraver, with proper horizontal spacing and alignment.
- editor performance refactor: introduced centralized render caching (`editor/caching.py`) and wired drawers to reuse precomputed data.
- optimized drawing hotspots in `grid`, `arpeggio`, `beam`, `slur`, `tempo`, `crescendo`, `decrescendo`, and `snap` drawers with viewport/time-window culling.
- improved note/beam rendering pipeline by sharing note stem metrics and reducing repeated geometry work.
- style dialog visibility workflow updated: visibility toggles now live only in the Visibility tab and correctly sync on style load/apply.
- general translation/string cleanup and internal drawing/refactor polish across related tools and dialogs.

1.1.1 (2026-06-01)
- changed under the hood: the editor's internal data structure supports multiple staves per system
- rebuilded engraver from the ground up to support multiple staves per system
'''
