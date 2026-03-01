'''
    Here all constants used in the application are stored.
'''

import os
from pathlib import Path

from utils.tiny_tool import key_class_filter

# Directory in the user's home used for autosaves and error backups
# Expanded once and reused across the app for any non-user-initiated saves.
UTILS_SAVE_DIR: Path = Path(os.path.expanduser('~/.keyTAB'))

# the meaning of time is defined in this constant.
QUARTER_NOTE_UNIT: float = 256.0

# Drawing orders (single sources of truth)
# Each string corresponds to a drawer name registered in editor/drawers/__init__.py
# Update these lists to control layer stacking in the Editor and Engraver.
EDITOR_LAYERING = [
    # layers from background to foreground
    'snap_band',
    'midi_note',
    'grid_line',
    'stave_three_line',
    'stave_two_line',
    'stave_clef_line',
    'text_bg',
    'barline',
    'stem_hand_split',
    'stop_sign',
    'accidental',
    'stem',
    'notehead_white',
    'notehead_black',
    'grace_note_white_outline',
    'grace_note_white_fill',
    'left_dot',
    'chord_connect',
    'grace_note',
    'beam',
    'beam_stem',
    'measure_number',
    'tempo',
    'count_line_handle',
    'count_line',
    'line_break',
    'selection_rect',
    'keyboard_overlay_bg',
    'keyboard_overlay_keys',
    'cursor',
    'playhead',
    'line_break_guide',
    'beam_marker',
    'time_signature',
    'slur',
    'slur_handle',
    'beam_line_right',
    'beam_connect_right',
    'beam_line_left',
    'beam_connect_left',
    'text',
    'text_handle',
]

ENGRAVER_LAYERING = [
    # layers from background to foreground
    'midi_note',
    'grid_line',
    'count_line',
    'measure_number_guide',
    'measure_number',
    'stave',
    'text_bg',
    'barline',
    'beam_stem',
    'stop_sign',
    'stem_hand_split',
    'chord_connect',
    'continuation_dot',
    'stem',
    'beam',
    'notehead_white',
    'notehead_black',
    'left_dot',
    'grace_note_black_outline',
    'grace_note_white_fill',
    'grace_note_black',
    'title',
    'composer',
    'copyright',
    'engrave_test',
    'text',
    'slur',
]

# Keyboard constants
PIANO_KEY_AMOUNT: int = 88

# key collections
BLACK_KEYS: list[int] = key_class_filter('CDFGA')
BE_KEYS: list[int] = key_class_filter('be')
CF_KEYS: list[int] = key_class_filter('cf')

# Editor colors
ACCENT_COLOR_HEX: str = '#3399FF'
NOTATION_COLOR_HEX: str = '#000000'
CURSOR_COLOR_HEX: str = "#000000"
# convert hex to rgbal with alpha 0.3
def hex_to_rgba(hex_color: str, alpha: float = 1) -> tuple[int, int, int, float]:
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (r, g, b, alpha)
ACCENT_COLOR_HEX: tuple[int, int, int, float] = hex_to_rgba(ACCENT_COLOR_HEX)
CURSOR_COLOR: tuple[int, int, int, float] = hex_to_rgba(CURSOR_COLOR_HEX)
NOTATION_COLOR: tuple[int, int, int, float] = hex_to_rgba(NOTATION_COLOR_HEX)

SHORTEST_DURATION: float = 8.0  # shortest note duration in time units (128th) (for playback and rendering)
# Threshold for interpreting very short notes as grace notes on load/import.
# Defaults to SHORTEST_DURATION so one edit can adjust both behaviors.
GRACENOTE_THRESHOLD: float = SHORTEST_DURATION

ENGRAVER_VERSION: str = '1.0'

ENGRAVER_FRACTIONAL_SCALE_CORRECTION: float = 0.675
