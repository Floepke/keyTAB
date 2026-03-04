from __future__ import annotations
from file_model.SCORE import SCORE
from ui.widgets.draw_util import DrawUtil
from utils.CONSTANT import QUARTER_NOTE_UNIT
from utils.tiny_tool import key_class_filter
from typing import TYPE_CHECKING, cast

from utils.CONSTANT import PIANO_KEY_AMOUNT

if TYPE_CHECKING:
    from editor.editor import Editor


class StaveDrawerMixin:
    def draw_stave(self, du: DrawUtil) -> None:
        self = cast("Editor", self)
        score: SCORE = self.current_score()
        layout = score.layout

        # Piano-roll vertical stave: draw vertical lines per semitone across full height
        w_mm, h_mm = du.current_page_size_mm()
        margin = float(self.margin)
        stave_width = float(self.stave_width)
        semitone_dx = float(self.semitone_dist)
        stave_left = margin
        stave_right = w_mm - margin
        total_score_time = self._calc_base_grid_list_total_length()
        stave_length_mm = (total_score_time / QUARTER_NOTE_UNIT) * score.editor.zoom_mm_per_quarter
        y1 = margin
        y2 = margin + stave_length_mm

        clef_dash_raw = list(getattr(layout, 'stave_clef_line_dash_pattern_mm', []) or [])
        clef_dash = clef_dash_raw if clef_dash_raw else None

        for key in range(1, PIANO_KEY_AMOUNT + 1):
            if key in key_class_filter('ACDFG'): # black keys
                x_pos = self.pitch_to_x(key)
                is_clef_line = key in (41, 43)  # C# and D# around middle C
                if getattr(self, 'is_tiny_mode', None) and self.is_tiny_mode() and is_clef_line:
                    continue
                is_three_line = key in key_class_filter('FGA')
                if is_clef_line:
                    width_mm = max(0.05, semitone_dx / 5.0)
                    dash = [2]
                    tag = "stave_clef_line"
                elif is_three_line:
                    width_mm = max(0.05, semitone_dx / 3.0)
                    dash = None
                    tag = "stave_three_line"
                else:
                    # two line
                    width_mm = max(0.05, semitone_dx / 10.0)
                    dash = None
                    tag = "stave_two_line"
                
                # draw
                du.add_line(x_pos, y1, x_pos, y2, color=self.notation_color, width_mm=width_mm,
                            dash_pattern=dash, id=0, tags=[tag])
