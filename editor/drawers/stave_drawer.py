from __future__ import annotations
from file_model.SCORE import SCORE
from settings_manager import get_preferences
from editor.editor_defaults import SCALE
from ui.style import Style
from ui.widgets.draw_util import DrawUtil
from utils.CONSTANT import BLACK_KEYS, QUARTER_NOTE_UNIT
from utils.tiny_tool import key_class_filter
from typing import TYPE_CHECKING, cast

from utils.CONSTANT import PIANO_KEY_AMOUNT

if TYPE_CHECKING:
    from editor.editor import Editor


class StaveDrawerMixin:
    def draw_stave(self, du: DrawUtil) -> None:
        self = cast("Editor", self)
        score: SCORE = self.current_score()
        layout = getattr(score, 'layout', None)
        if layout is None:
            return
        
        # read editor orientation
        preferences = get_preferences()
        if preferences.get("editor_orientation", 'horizontal') == 'horizontal':
            editor_orientation = 'horizontal'
        else:
            editor_orientation = 'vertical'

        # Piano-roll vertical stave: draw vertical lines per semitone across full height
        semitone_dx = float(self.semitone_dist)
        total_score_time = self._calc_base_grid_list_total_length()
        stave_length_mm = (total_score_time / QUARTER_NOTE_UNIT) * float(getattr(score.app_state, 'zoom_mm_per_quarter', 25.0) or 25.0)
        FGA_keys = key_class_filter('FGA')
        y1 = self.margin
        y2 = self.margin + stave_length_mm

        '''Draw stave lines'''
        for key in range(1, PIANO_KEY_AMOUNT):
            if key in BLACK_KEYS:
                x_pos = self.pitch_to_x(key)
                is_clef_line = key in (41, 43)  # C# and D# around middle C
                is_three_line = key in FGA_keys
                if is_clef_line:
                    width_mm = score.layout.stave_clef_line_thickness_mm * SCALE
                    dash = [2]
                    tag = "stave_clef_line"
                elif is_three_line:
                    width_mm = score.layout.stave_three_line_thickness_mm * SCALE
                    dash = None
                    tag = "stave_three_line"
                else:
                    # two line
                    width_mm = score.layout.stave_two_line_thickness_mm * SCALE
                    dash = None
                    tag = "stave_two_line"
                
                # draw
                du.add_line(
                    x_pos,
                    y1,
                    x_pos,
                    y2,
                    color=self.notation_color,
                    width_mm=width_mm,
                    dash_pattern=dash,
                    id=0,
                    tags=[tag]
                )

        '''Draw piano keyboard below stave'''
        bar_width_mm = max(0.01, float(getattr(self, 'editor_line_width_global', 0.1) or 0.1))
        end_t = float(self._calc_base_grid_list_total_length())
        y_kb = float(self.time_to_mm(end_t))
        key_len_mm = semitone_dx * 4.0
        kb_x1 = float(self.margin) - float(self.semitone_dist)
        kb_x2 = float(self.margin) + float(self.stave_width) + (float(self.semitone_dist * 2) )
        kb_y2 = y_kb + 7.0 * semitone_dx
        black_key_width_mm = max(0.05, semitone_dx / 3.0) * 2.0
        black_key_set = set(key_class_filter('ACDFG'))

        def octave_number(key: int) -> int:
            """Return the American octave number for a keyTAB key index."""
            midi_note = 20 + int(key)
            return (midi_note // 12) - 1

        # Grey fill for alternating octaves (1, 3, 5, 7): C-B spans.
        grey_octave_spans = [
            (4,  15),   # C1–B1
            (28, 39),   # C3–B3
            (52, 63),   # C5–B5
            (76, 87),   # C7–B7
        ]
        _sb = Style.get_named_rgb('snap_band', fallback=(230, 230, 230))
        grey_fill = (float(_sb[0]) / 255.0, float(_sb[1]) / 255.0, float(_sb[2]) / 255.0, 1.0)
        for span_start, span_end in grey_octave_spans:
            gx1 = float(self.pitch_to_x(span_start)) - semitone_dx
            gx2 = float(self.pitch_to_x(span_end)) + semitone_dx
            du.add_rectangle(
                gx1,
                y_kb,
                gx2,
                kb_y2,
                stroke_color=None,
                fill_color=grey_fill,
                id=0,
                tags=["piano_keyboard", "piano_octave_band"],
            )

        for key in black_key_set:
            x_pos = float(self.pitch_to_x(key))
            # draw clef dashed
            if key in (41, 43):  # C# and D# around middle C
                dash = [1, 1.4]
            else:
                dash = None
            if kb_x1 <= x_pos <= kb_x2:
                du.add_line(
                    x_pos,
                    y_kb,
                    x_pos,
                    y_kb + key_len_mm,
                    color=self.notation_color,
                    dash_pattern=dash,
                    width_mm=black_key_width_mm,
                    id=0,
                    tags=["piano_keyboard", "piano_black_key"],
                )

        # Label octave numbers
        octave_label_keys = [1] + [key for key in range(4, PIANO_KEY_AMOUNT + 1, 12)]
        for key in octave_label_keys:
            x_pos = float(self.pitch_to_x(key))
            if kb_x1 <= x_pos <= kb_x2 and editor_orientation == 'vertical':
                du.add_text(
                    x_pos,
                    kb_y2 + 1.0,
                    str(octave_number(key)),
                    family="Edwin",
                    color=self.notation_color,
                    anchor='n',
                    size_pt=10.0,
                    id=0,
                    tags=["piano_keyboard", "piano_octave_number"],
                )
            else: # horizontal orientation
                du.add_text(
                    x_pos,
                    kb_y2 + 2.0,
                    str(octave_number(key)),
                    family="Edwin",
                    color=self.notation_color,
                    anchor='s',
                    size_pt=10.0,
                    id=0,
                    tags=["piano_keyboard", "piano_octave_number"],
                    angle_deg=90.0,
                )

        # Draw piano keyboard outline
        if editor_orientation == 'vertical':
            pass
        else: # horizontal orientation
            kb_x2 += 1.0
        du.add_rectangle(
            kb_x1,
            y_kb,
            kb_x2,
            kb_y2,
            stroke_color=self.notation_color,
            stroke_width_mm=bar_width_mm,
            corner_radius=1.0,
            fill_color=None,
            id=0,
            tags=["piano_keyboard", "piano_outline"],
        )
