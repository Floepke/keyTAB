"""
Single source of truth for keyTAB versioning.

Bump policy (Semantic Versioning):
  MAJOR - breaking changes (incompatible file format, major UI overhaul)
  MINOR - new features, backward-compatible
  PATCH - bug fixes only
"""

__version__ = "1.0.7"
APP_NAME    = "keyTAB"

change_log = '''
1.0.0 (2026-04-01)
- initial release

1.0.1 (2026-04-05)
- repeat symbols and measure numbers in engraver fix when ledger lines are present

1.0.2 (2026-04-13)
- midi export and import is now dependency free (no more mido or pretty_midi)
- fix: midi import can open older Klavarscript midi files without errors.

1.0.3 (2026-04-14)
- copyright default value changed to "© keyTAB {datetime.now().year}"
- physical undo/redo implemented as buttons in the toolbar, in addition to keyboard shortcuts ctrl+z and ctrl+shift+z
- added "Set as Default Style" file menu item to save the current style as the default for new projects
- added "Reset Default Style" file menu item to reset the default style to the built-in defaults

1.0.4 (2026-04-20)
- fixed libfluidsytnh dependency issues on Linux AppImage build
- fixed beam tool sometimes not detecting the correct hand on click time
- fixed Windows Cairo rendering issue where dynamic symbols (LelandText) could display as missing glyphs in editor/engraver while still appearing correctly in the Qt menu
- startup now ensures LelandText is registered/installed on Windows similarly to Edwin so Cairo can resolve the font reliably
- Windows Inno Setup installer now installs both Edwin and LelandText automatically
- fixed Windows engraver worker crash caused by LelandText font resolution touching Qt font registry APIs inside the spawned worker process
- changed preferences UI to use radio buttons instead of dropdowns
- changed the behavior of the velocity editing mode: it now doesn't remember the toggle state as it was unhandy in practical usage
- the restarting app path now is more stable and robust (on linux the second restart made the app crash)
- fix: the preference save_on_exit now works properly + closeEvent is updated to show the correct info to the user
- stripped double barline on end barline as it looked unnatural and is not the official Klavarskribo convention.
- fix: the appstate is now properly saved in the score file and restored on loading a file

1.0.5 (2026-04-21)
- implemented in engraver: slurs that span line/page breaks are now splitted and continue accordingly on the next line/page
- implemented in engraver: slurs that are connected do render a indication of then connected slur at line break points to indicate the connection
- implemented horizontal read mode in engraver, this requires probably further finetuning
- fix: the full screen toggle checkbox works correct now in the menu on app startup
- fit button works now as splitter handle too, allowing to drag it to resize the editor and print-preview areas. The tooltip is updated to reflect this new behavior.
- implemented miniature piano keyboard in editor below the end barline, showing the full range of keys with octave numbers. This is a visual aid for users to understand the pitch layout of the piano-roll stave.
- fit page to window works correctly if page orientation is landscape now

1.0.6 (2026-05-05)
- expanded horizontal read/orientation support with multiple engraving/layout fixes (including tempo marker rotation/placement and general horizontal mode behavior)
- print view playhead implemented and stabilized for horizontal read mode
- fit/viewport behavior refined: fit-to-window in landscape corrected and splitter fit handle behavior improved
- editor now includes miniature piano keyboard visual aid plus updated notehead design and dark-mode default color finetuning
- editing refinements: velocity editing fixes and time-shift keeps rhythmic selection rectangle content intact
- added single-key shortcuts `z`, `x`, `c`, `v` for undo/cut/copy/paste in addition to existing standard shortcuts
- added middle-mouse drag panning for editor and print view, plus Ctrl/Cmd+scroll zoom in print view
- translation updates and general polish/fixes across the release cycle

1.0.7 (2026-06-01)
- fix: the app now restarts correctly on Linux AppImage builds without crashing
- fix: repeat symbol positioning corrected in horizontal editor mode
- scrollbar is now at the bottom in horizontal editor mode
- finetuned Klavarskribo time signature indicator positioning in horizontal editor mode
- analysis: added avg_frequency, pitch range, most used pitch, and left/right balance metrics
- new scripts: reverse, mirror pitch, and updated double/half time script
- improved text rendering in engraver and editor, with text dialog enhancements, you can now form a multiline text rectangle by using the build-in multiline widget from qt
- fix: note editing no longer triggers overlap guard unwantedly when working with triplets/antisymetric figures
- print view: playhead focus now correctly targets both X and Y viewport axes
- fix: midi player no longer makes timing/duration mistakes with fine floating-point note values, midi player overall enhanced

1.0.8 (2026-06-15)
- fix: on some configurations, the selection rectangle didn't detect underlying note time correctly
'''
