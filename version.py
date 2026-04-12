"""
Single source of truth for keyTAB versioning.

Bump policy (Semantic Versioning):
  MAJOR – breaking changes (incompatible file format, major UI overhaul)
  MINOR – new features, backward-compatible
  PATCH – bug fixes only
"""

__version__ = "1.0.2"
APP_NAME    = "keyTAB"

change_log = '''
1.0.0 (2024-06-01)
- initial release

1.0.1 (2024-06-30)
- repeat symbols and measure numbers in engraver fix when ledger lines are present

1.0.2 (2024-07-15)
- midi export and import is now dependency free (no more mido or pretty_midi)
- fix: midi import can open older Klavarscript midi files without errors.
'''