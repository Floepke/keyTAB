from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
from file_model.events.grid_band import GridBand
from file_model.font import Font

@dataclass
class Layout:
    scale: float = 0.35
    page_orientation: Literal['landscape', 'portrait'] = 'portrait'
    read_direction: Literal['horizontal', 'vertical'] = 'vertical'
    page_width_mm: float = 210.0
    page_height_mm: float = 297.0
    page_top_margin_mm: float = 7.5
    page_bottom_margin_mm: float = 5.0
    page_left_margin_mm: float = 5.0
    page_right_margin_mm: float = 5.0
    header_height_mm: float = 15.0
    footer_height_mm: float = 7.5

    black_note_rule: Literal['above_stem', 'below_stem', 'above_stem_if_collision', 'above_stem_if_chord_and_white_note', 'above_stem_if_chord_and_white_note_same_hand'] = 'above_stem'

    # Note appearance
    note_head_visible: bool = True
    note_stem_visible: bool = True
    accidental_visible: bool = True
    chord_connect_visible: bool = True
    note_stop_visible: bool = True
    note_stem_length_semitone: int = 6
    note_stem_thickness_mm: float = 1.1 # Thickness of the stem as well the notehead outline width
    note_stopsign_thickness_mm: float = 1.25
    note_leftdot_visible: bool = False
    note_continuation_dot_visible: bool = True
    note_continuation_dot_size_mm: float = 2.4
    note_midinote_visible: bool = False
    note_midinote_left_color: str = '#ccc'
    note_midinote_right_color: str = '#ccc'
    note_width_scaling: float = 1.1 # Scaling factor for the horizontal size of the noteheads, to make them wider or narrower.

    # Beam appearance
    beam_visible: bool = True
    beam_thickness_mm: float = 2.5
    beam_corner_radius_mm: float = 0.75

    # Grace note appearance
    grace_note_visible: bool = True
    grace_note_outline_width_mm: float = 1.0
    grace_note_scale: float = 0.75

    # Pedal appearance
    pedal_lane_enabled: bool = False
    pedal_lane_width_mm: float = 2.5

    # Text appearance
    text_visible: bool = True
    text_background_padding_mm: float = 0.5

    # Slur appearance
    slur_visible: bool = True
    slur_width_sides_mm: float = 0.5
    slur_width_middle_mm: float = 1.5

    # Hairpin (crescendo / decrescendo) appearance
    hairpin_visible: bool = True
    hairpin_line_width_mm: float = 1.0
    hairpin_width_mm: float = 10.0  # width of the open end of the hairpin in mm
    hairpin_text_gap_mm: float = 0.5  # gap between hairpin and text in mm
    dynamic_symbol_font_size_pt: float = 12.0  # Font size for standalone dynamic symbols
    dynamic_symbol_background_padding_mm: float = 1.5
    dynamic_symbol_visible: bool = True

    # Repeat markers
    repeat_start_visible: bool = True
    repeat_end_visible: bool = True
    double_barline_visible: bool = True
    
    # Measure grouping (prefill for line break tool; not applied automatically)
    measure_grouping: str = ""

    # Count line
    countline_visible: bool = True
    countline_dash_pattern: list[float] = field(default_factory=lambda: [0.0, 3.5])  # Dash pattern for count lines (e.g., [dash_length, gap_length])
    countline_thickness_mm: float = 1.25

    # Grid lines
    stave_visible: bool = True
    barline_visible: bool = True
    grid_line_visible: bool = True
    grid_band_visible: bool = True
    grid_band_track: list[GridBand] = field(default_factory=list) # Grid Band track. Single track for alternating bands.
    grid_barline_thickness_mm: float = 1.25
    grid_gridline_thickness_mm: float = 0.5
    grid_gridline_dash_pattern_mm: list[float] = field(default_factory=lambda: [1.5, 3.0])
    grid_band_color: str = '#ccc'
    grid_band_start_phase: Literal['dark', 'light'] = 'dark'

    # Time signature indicator type (global)
    time_signature_visible: bool = True
    time_signature_indicator_type: Literal['classical', 'klavarskribo', 'classical & klavarskribo'] = 'classical & klavarskribo'
    
    # Time signature indicator lane (left of stave)
    time_signature_indicator_lane_width_mm: float = 30.0
    time_signature_indicator_guide_thickness_mm: float = 0.5
    time_signature_indicator_divide_guide_thickness_mm: float = 1.0
    time_signature_indicator_classic_font: Font = field(default_factory=lambda: Font(
        family="Edwin",
        size_pt=40.0,
        bold=True,
    ))
    time_signature_indicator_klavarskribo_font: Font = field(default_factory=lambda: Font(
        family="Edwin",
        size_pt=25.0,
        bold=True,
    ))
    measure_numbering_guide_thickness_mm: float = 0.7
    measure_numbering_guide_dash_pattern_mm: list[float] = field(default_factory=lambda: [1.0, 2.0])
    # 'system': number at top of each system; 'barline': number at every barline
    measure_numbering_placement: Literal['system', 'barline'] = 'barline'
    measure_numbering_guide_visible: bool = True
    measure_numbers_visible: bool = True
    measure_numbering_font: Font = field(default_factory=lambda: Font(
        family="Edwin",
        size_pt=25.0,
        bold=True,
        italic=True,
    ))

    font_text: Font = field(default_factory=lambda: Font(
        family="Edwin",
        size_pt=12.0,
        bold=False,
        italic=True,
    ))

    # Info fonts
    font_title: Font = field(default_factory=lambda: Font(
        family="Edwin",
        size_pt=25.0,
        bold=True,
    ))
    font_composer: Font = field(default_factory=lambda: Font(
        family="Edwin",
        size_pt=15.0,
        italic=True,
    ))
    font_copyright: Font = field(default_factory=lambda: Font(
        family="Edwin",
        size_pt=12.0,
    ))
    font_arranger: Font = field(default_factory=lambda: Font(
        family="Edwin",
        size_pt=12.0,
    ))
    font_lyricist: Font = field(default_factory=lambda: Font(
        family="Edwin",
        size_pt=12.0,
    ))

    # Stave appearence
    stave_two_line_thickness_mm: float = 0.5
    stave_three_line_thickness_mm: float = 1.1
    stave_clef_line_thickness_mm: float = 0.5
    stave_ledger_line_length_mm: float = 13.0
    stave_clef_line_dash_pattern_mm: list[float] = field(default_factory=lambda: [3.0])  # Dash pattern for clef lines (e.g., [dash_length, gap_length])

LAYOUT_FLOAT_CONFIG: dict[str, dict[str, float]] = {
    'page_width_mm': {'min': 50.0, 'max': 5_000.0, 'step': 0.5},
    'page_height_mm': {'min': 50.0, 'max': 10_000.0, 'step': 0.5},
    'page_top_margin_mm': {'min': 0.0, 'max': 100.0, 'step': 0.05},
    'page_bottom_margin_mm': {'min': 0.0, 'max': 100.0, 'step': 0.05},
    'page_left_margin_mm': {'min': 0.0, 'max': 100.0, 'step': 0.05},
    'page_right_margin_mm': {'min': 0.0, 'max': 100.0, 'step': 0.05},
    'header_height_mm': {'min': 0.0, 'max': 100.0, 'step': 0.05},
    'footer_height_mm': {'min': 0.0, 'max': 100.0, 'step': 0.05},
    'scale': {'min': 0.25, 'max': 1.0, 'step': 0.005},
    'note_stem_length_semitone': {'min': 1.0, 'max': 20.0, 'step': 1.0},
    'note_stem_thickness_mm': {'min': 0.05, 'max': 5.0, 'step': 0.05},
    'note_stopsign_thickness_mm': {'min': 0.05, 'max': 5.0, 'step': 0.05},
    'note_continuation_dot_size_mm': {'min': 0.05, 'max': 10.0, 'step': 0.05},
    'note_width_scaling': {'min': 0.05, 'max': 2.0, 'step': 0.01},
    'beam_thickness_mm': {'min': 0.05, 'max': 5.0, 'step': 0.05},
    'beam_corner_radius_mm': {'min': 0.0, 'max': 5.0, 'step': 0.05},
    'grace_note_outline_width_mm': {'min': 0.05, 'max': 5.0, 'step': 0.05},
    'grace_note_scale': {'min': 0.05, 'max': 1.0, 'step': 0.05},
    'pedal_lane_width_mm': {'min': 0.05, 'max': 20.0, 'step': 0.05},
    'text_background_padding_mm': {'min': 0.0, 'max': 20.0, 'step': 0.05},
    'hairpin_line_width_mm': {'min': 0.05, 'max': 5.0, 'step': 0.05},
    'hairpin_width_mm': {'min': 0.05, 'max': 20.0, 'step': 0.05},
    'hairpin_font_size_pt': {'min': 4.0, 'max': 48.0, 'step': 0.5},
    'hairpin_text_gap_mm': {'min': 0.0, 'max': 20.0, 'step': 0.05},
    'dynamic_symbol_background_padding_mm': {'min': 0.0, 'max': 20.0, 'step': 0.05},
    'slur_width_sides_mm': {'min': 0.05, 'max': 5.0, 'step': 0.05},
    'slur_width_middle_mm': {'min': 0.05, 'max': 5.0, 'step': 0.05},
    'countline_thickness_mm': {'min': 0.05, 'max': 5.0, 'step': 0.05},
    'grid_barline_thickness_mm': {'min': 0.05, 'max': 5.0, 'step': 0.05},
    'grid_gridline_thickness_mm': {'min': 0.05, 'max': 5.0, 'step': 0.05},
    'measure_numbering_guide_thickness_mm': {'min': 0.05, 'max': 5.0, 'step': 0.05},
    'time_signature_indicator_lane_width_mm': {'min': 0.05, 'max': 100.0, 'step': 0.05},
    'time_signature_indicator_guide_thickness_mm': {'min': 0.05, 'max': 5.0, 'step': 0.05},
    'time_signature_indicator_divide_guide_thickness_mm': {'min': 0.05, 'max': 5.0, 'step': 0.05},
    'stave_two_line_thickness_mm': {'min': 0.05, 'max': 5.0, 'step': 0.05},
    'stave_three_line_thickness_mm': {'min': 0.05, 'max': 5.0, 'step': 0.05},
    'stave_clef_line_thickness_mm': {'min': 0.05, 'max': 5.0, 'step': 0.05},
    'stave_ledger_line_length_mm': {'min': 0.05, 'max': 100.0, 'step': 0.05},
}
