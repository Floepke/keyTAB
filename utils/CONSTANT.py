'''
    Here all constants used in the application are stored.
'''

import os
from pathlib import Path

from utils.tiny_tool import key_class_filter
from version import __version__

# Directory in the user's home used for autosaves and error backups
# Expanded once and reused across the app for any non-user-initiated saves.
UTILS_SAVE_DIR: Path = Path(os.path.expanduser('~/.keyTAB'))

# the meaning of time is defined in this constant.
QUARTER_NOTE_UNIT: float = 256.0

# Editor-only drawing setting: snap side-band width measured in semitone distances.
# Used by drawer code only (not serialized in project files).
EDITOR_SIDE_BAND_INSET_SEMITONES: int = 10

# Drawing orders (single sources of truth)
# Each string corresponds to a tag name used in the Editor and Engraver for layering.
# Update these lists to control layer stacking in the Editor and Engraver.
EDITOR_LAYERING = [
    # layers from background to foreground
    'page_background',
    'piano_octave_band',
    'grid_band_stop_line',
    'grid_band',
    'snap_band',
    'midi_note',
    'grid_line',
    'stave_three_line',
    'stave_two_line',
    'stave_clef_line',
    'barline',
    'stem_hand_split',
    'stop_sign',
    'accidental',
    'stem',
    'chord_connect',
    'notehead_white',
    'notehead_black',
    'grace_note_white_outline',
    'grace_note_white_fill',
    'left_dot',
    'grace_note',
    'beam',
    'beam_stem',
    'barline_symbol',
    'barline_symbol_dot',
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
    'slur_handle',
    'slur',
    'beam_line_right',
    'beam_connect_right',
    'beam_line_left',
    'beam_connect_left',
    'text_bg',
    'text',
    'text_handle',
    'dynamic_symbol_bg',
    'dynamic_symbol_text',
    'hairpin_handle',
    'hairpin',
    'pedal_symbol_bg',
    'pedal_symbol',
    'tempo_bg',
    'tempo_text',
]

ENGRAVER_LAYERING = [
    # layers from background to foreground
    'page_background',
    'grid_band',
    'grid_band_marker',
    'midi_note',
    'grid_line',
    'count_line',
    'measure_number_guide',
    'measure_number',
    'barline_symbol',
    'barline_symbol_dot',
    'stave',
    'piano_octave_band',
    'piano_outline',
    'piano_octave_number',
    'piano_black_key',
    'barline',
    'end_barline',
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
    'text_bg',
    'text',
    'tempo_bg',
    'tempo_text',
    'slur',
    'dynamic_symbol_bg',
    'dynamic_symbol_text',
    'hairpin',
    'ts_klavarskribo',
    'ts_classic',
    'pedal_symbol_bg',
    'pedal_symbol',
]

# Keyboard constants
PIANO_KEY_AMOUNT: int = 88

# key collections
BLACK_KEYS: list[int] = key_class_filter('CDFGA')
BE_KEYS: list[int] = key_class_filter('be')
CF_KEYS: list[int] = key_class_filter('cf')

# Editor colors
def hex_to_rgba(hex_color: str, alpha: float = 1) -> tuple[int, int, int, float]:
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (r, g, b, alpha)

SHORTEST_DURATION: float = 1.0  # shortest note duration in time units (128th) (for playback and rendering)
# Threshold for interpreting very short notes as grace notes on load/import.
# Defaults to SHORTEST_DURATION so one edit can adjust both behaviors.
GRACENOTE_THRESHOLD: float = 16.0  # (32th) Default threshold for interpreting very short notes as grace notes on load/import.

ENGRAVER_VERSION: str = __version__

# TODO: this correction i see now is a little silly and not correct in all cases.
ENGRAVER_FRACTIONAL_TEXT_SCALING_CORRECTION: float = 0.675

# Amount of line segments used to sample/draw cubic slurs.
SLUR_SEGMENT_COUNT: int = 100
