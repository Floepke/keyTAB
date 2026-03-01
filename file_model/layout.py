from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class LayoutFont:
    family: str = "Edwin"
    size_pt: float = 12.0
    bold: bool = False
    italic: bool = False
    x_offset: float = 0.0
    y_offset: float = 0.0

    def resolve_family(self) -> str:
        try:
            from fonts import resolve_font_family
            return resolve_font_family(self.family)
        except Exception:
            return self.family

@dataclass
class Layout:
    # Page dimensions and margins
    page_width_mm: float = 210.0
    page_height_mm: float = 297.0
    page_top_margin_mm: float = 5.0
    page_bottom_margin_mm: float = 5.0
    page_left_margin_mm: float = 5.0
    page_right_margin_mm: float = 5.0
    
    # header/footer area settings
    header_height_mm: float = 17.5
    footer_height_mm: float = 10.0

    # Global drawing options
    scale: float = 0.33
    black_note_rule: Literal['above_stem', 'below_stem', 'above_stem_if_collision', 'above_stem_if_chord_and_white_note', 'above_stem_if_chord_and_white_note_same_hand'] = 'above_stem_if_collision'

    # Note appearance
    note_head_visible: bool = True
    note_stem_visible: bool = True
    note_stem_length_semitone: int = 5
    note_stem_thickness_mm: float = 1.0 # Thickness of the stem as well the notehead outline
    note_stopsign_thickness_mm: float = 1.0
    note_leftdot_visible: bool = True
    note_continuation_dot_size_mm: float = 2.5
    note_midinote_visible: bool = True
    note_midinote_left_color: str = '#cccccc'
    note_midinote_right_color: str = '#cccccc'

    # Beam appearance
    beam_visible: bool = True
    beam_thickness_mm: float = 2.5

    # Grace note appearance
    grace_note_visible: bool = True
    grace_note_outline_width_mm: float = 1.5
    grace_note_scale: float = 0.75

    # Pedal appearance
    pedal_lane_enabled: bool = False
    pedal_lane_width_mm: float = 2.5

    # Text appearance
    text_visible: bool = True
    text_background_padding_mm: float = 1.5

    # Slur appearance
    slur_visible: bool = True
    slur_width_sides_mm: float = 0.5
    slur_width_middle_mm: float = 2.5

    # Repeat markers
    repeat_start_visible: bool = True
    repeat_end_visible: bool = True
    
    # Measure grouping (prefill for line break tool; not applied automatically)
    measure_grouping: str = ""

    # Count line
    countline_visible: bool = True
    countline_dash_pattern: list[float] = field(default_factory=lambda: [0.0, 3.0])  # Dash pattern for count lines (e.g., [dash_length, gap_length])
    countline_thickness_mm: float = 1.0

    # Grid lines
    grid_barline_thickness_mm: float = 1.0
    grid_gridline_thickness_mm: float = 0.5
    grid_gridline_dash_pattern_mm: list[float] = field(default_factory=lambda: [5.0])

    # Time signature indicator type (global)
    time_signature_indicator_type: Literal['classical', 'klavarskribo', 'both'] = 'both'
    
    # Time signature indicator lane (left of stave)
    time_signature_indicator_lane_width_mm: float = 10.0
    time_signature_indicator_guide_thickness_mm: float = 0.5
    time_signature_indicator_divide_guide_thickness_mm: float = 1.0
    time_signature_indicator_classic_font: LayoutFont = field(default_factory=lambda: LayoutFont(
        family="Latin Modern Roman",
        size_pt=40.0,
        bold=True,
    ))
    time_signature_indicator_klavarskribo_font: LayoutFont = field(default_factory=lambda: LayoutFont(
        family="Latin Modern Roman",
        size_pt=25.0,
    ))
    measure_numbering_font: LayoutFont = field(default_factory=lambda: LayoutFont(
        family="Latin Modern Roman",
        size_pt=25.0,
        italic=True,
    ))

    font_text: LayoutFont = field(default_factory=lambda: LayoutFont(
        family="Latin Modern Roman",
        size_pt=14.0,
        bold=True,
        italic=True,
    ))

    # Info fonts
    font_title: LayoutFont = field(default_factory=lambda: LayoutFont(
        family="Latin Modern Roman Caps",
        size_pt=25.0,
        bold=True,
    ))
    font_composer: LayoutFont = field(default_factory=lambda: LayoutFont(
        family="Latin Modern Roman Caps",
        size_pt=15.0,
    ))
    font_copyright: LayoutFont = field(default_factory=lambda: LayoutFont(
        family="Latin Modern Roman Caps",
        size_pt=11.0,
        italic=True,
    ))
    font_arranger: LayoutFont = field(default_factory=lambda: LayoutFont(
        family="Latin Modern Roman Caps",
        size_pt=12.0,
    ))
    font_lyricist: LayoutFont = field(default_factory=lambda: LayoutFont(
        family="Latin Modern Roman Caps",
        size_pt=12.0,
    ))

    # Stave appearence
    stave_two_line_thickness_mm: float = 0.5
    stave_three_line_thickness_mm: float = 1.25
    stave_clef_line_dash_pattern_mm: list[float] = field(default_factory=lambda: [4.0])  # Dash pattern for clef lines (e.g., [dash_length, gap_length])


LAYOUT_FLOAT_CONFIG: dict[str, dict[str, float]] = {
    'page_width_mm': {'min': 50.0, 'max': 10000.0, 'step': 0.5},
    'page_height_mm': {'min': 50.0, 'max': 100000.0, 'step': 0.5},
    'page_top_margin_mm': {'min': 0.0, 'max': 100.0, 'step': 0.05},
    'page_bottom_margin_mm': {'min': 0.0, 'max': 100.0, 'step': 0.05},
    'page_left_margin_mm': {'min': 0.0, 'max': 100.0, 'step': 0.05},
    'page_right_margin_mm': {'min': 0.0, 'max': 100.0, 'step': 0.05},
    'header_height_mm': {'min': 0.0, 'max': 100.0, 'step': 0.05},
    'footer_height_mm': {'min': 0.0, 'max': 100.0, 'step': 0.05},
    'scale': {'min': 0.01, 'max': 1.0, 'step': 0.01},
    'note_stem_length_semitone': {'min': 1.0, 'max': 20.0, 'step': 1.0},
    'note_stem_thickness_mm': {'min': 0.05, 'max': 5.0, 'step': 0.05},
    'note_stopsign_thickness_mm': {'min': 0.05, 'max': 5.0, 'step': 0.05},
    'note_continuation_dot_size_mm': {'min': 0.05, 'max': 10.0, 'step': 0.05},
    'beam_thickness_mm': {'min': 0.05, 'max': 5.0, 'step': 0.05},
    'grace_note_outline_width_mm': {'min': 0.05, 'max': 5.0, 'step': 0.05},
    'grace_note_scale': {'min': 0.05, 'max': 1.0, 'step': 0.05},
    'pedal_lane_width_mm': {'min': 0.05, 'max': 20.0, 'step': 0.05},
    'text_background_padding_mm': {'min': 0.0, 'max': 20.0, 'step': 0.1},
    'slur_width_sides_mm': {'min': 0.05, 'max': 5.0, 'step': 0.05},
    'slur_width_middle_mm': {'min': 0.05, 'max': 5.0, 'step': 0.05},
    'countline_thickness_mm': {'min': 0.05, 'max': 5.0, 'step': 0.05},
    'grid_barline_thickness_mm': {'min': 0.05, 'max': 5.0, 'step': 0.05},
    'grid_gridline_thickness_mm': {'min': 0.05, 'max': 5.0, 'step': 0.05},
    'time_signature_indicator_lane_width_mm': {'min': 0.05, 'max': 100.0, 'step': 0.05},
    'time_signature_indicator_guide_thickness_mm': {'min': 0.05, 'max': 5.0, 'step': 0.05},
    'time_signature_indicator_divide_guide_thickness_mm': {'min': 0.05, 'max': 5.0, 'step': 0.05},
    'stave_two_line_thickness_mm': {'min': 0.05, 'max': 5.0, 'step': 0.05},
    'stave_three_line_thickness_mm': {'min': 0.05, 'max': 5.0, 'step': 0.05},
}
