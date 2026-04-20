"""
Single source of truth for keyTAB versioning.

Bump policy (Semantic Versioning):
  MAJOR - breaking changes (incompatible file format, major UI overhaul)
  MINOR - new features, backward-compatible
  PATCH - bug fixes only
"""

__version__ = "1.0.4"
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
'''
