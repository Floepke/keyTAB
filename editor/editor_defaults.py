"""
Hardcoded default values for editor rendering.

These values are frozen to current Layout() defaults to ensure editor UI
consistency regardless of file-model Layout settings. The engraver reads
from file Layout to allow custom output, but the editor always uses these
constants for readability and predictability.
"""

# Core styling
SCALE = 0.35
EDITOR_LINE_WIDTH_MM = 0.1  # used as global editor stroke width

# Note appearance
NOTE_STEM_LENGTH_SEMITONE = 7
NOTE_WIDTH_SCALING = 1.0
NOTE_STEM_THICKNESS_MM = 0.8

# Beam appearance
BEAM_THICKNESS_MM = 2.5
BEAM_CORNER_RADIUS_MM = 0.75

# Pedal appearance
PEDAL_SYMBOL_THICKNESS_MM = 1.0
PEDAL_BACKGROUND_PADDING_MM = 1.0

# Text appearance
TEXT_BACKGROUND_PADDING_MM = 0.5

# Slur appearance
SLUR_WIDTH_SIDES_MM = 0.75
SLUR_WIDTH_MIDDLE_MM = 2.0

# Hairpin (crescendo/decrescendo) appearance
HAIRPIN_LINE_WIDTH_MM = 1.0
HAIRPIN_WIDTH_MM = 10.0
HAIRPIN_TEXT_GAP_MM = 0.5
DYNAMIC_SYMBOL_FONT_SIZE_PT = 12.0
DYNAMIC_SYMBOL_BACKGROUND_PADDING_MM = 1.5

# Grid lines
GRID_GRIDLINE_THICKNESS_MM = 1.0

# Stave appearance
STAVE_TWO_LINE_THICKNESS_MM = 0.5
STAVE_THREE_LINE_THICKNESS_MM = 1.1
STAVE_CLEF_LINE_THICKNESS_MM = 0.75

# Dash patterns (hardcoded for editor)
COUNTLINE_DASH_PATTERN = [0, 1.5]  # editor uses simplified dash pattern
TIME_SIGNATURE_GUIDE_THICKNESS_MM = 1.0
TIME_SIGNATURE_DIVIDE_GUIDE_THICKNESS_MM = 2.0

# Measure numbering
MEASURE_NUMBERING_GUIDE_THICKNESS_MM = 0.7

# Grid band
GRID_BAND_START_PHASE = 'dark'  # 'dark' or 'light'

# Font families (from Layout defaults)
FONT_TITLE_FAMILY = 'Edwin'
FONT_MEASURE_NUMBERING_FAMILY = 'Edwin'
FONT_TEXT_FAMILY = 'Edwin'
FONT_TIME_SIGNATURE_CLASSIC_FAMILY = 'Edwin'
FONT_TIME_SIGNATURE_KLAV_FAMILY = 'Edwin'

FONT_TITLE_SIZE_PT = 12
FONT_MEASURE_NUMBERING_SIZE_PT = 20.0
FONT_TEXT_SIZE_PT = 12.0
FONT_TIME_SIGNATURE_CLASSIC_SIZE_PT = 25.0
FONT_TIME_SIGNATURE_KLAV_SIZE_PT = 15.0
