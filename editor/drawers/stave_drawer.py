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

        # Piano-roll vertical stave: draw vertical lines per semitone across full height
        semitone_dx = float(self.semitone_dist)
        total_score_time = self._calc_base_grid_list_total_length()
        stave_length_mm = (total_score_time / QUARTER_NOTE_UNIT) * float(getattr(score.app_state, 'zoom_mm_per_quarter', 25.0) or 25.0)
        y1 = self.margin
        y2 = self.margin + stave_length_mm

        for key in range(1, PIANO_KEY_AMOUNT):
            if key in key_class_filter('ACDFG'): # black keys
                x_pos = self.pitch_to_x(key)
                is_clef_line = key in (41, 43)  # C# and D# around middle C
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
