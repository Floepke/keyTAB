'''
Grid and barline drawing mixin for the Editor class.

Handles drawing barlines, measure numbers, and gridlines.
'''

from __future__ import annotations
from typing import TYPE_CHECKING, cast
from file_model.SCORE import SCORE
from ui.widgets.draw_util import DrawUtil

if TYPE_CHECKING:
    from editor.editor import Editor


class GridDrawerMixin:
    '''
        Draws:
            - barlines
            - measure numbers
            - gridlines
            - time signature indicators
            - project title and composer at top-left
    '''
    
    def draw_grid(self, du: DrawUtil) -> None:
        self = cast("Editor", self)
        score: SCORE = self.current_score()

        # draw title and composer at top-left
        title_text = score.info.title
        composer_text = score.info.composer
        title_font = score.layout.font_title
        if title_font is not None and callable(getattr(title_font, 'resolve_family', None)):
            family = str(title_font.resolve_family())
        else:
            family = getattr(title_font, 'family', 'Courier New') if title_font is not None else 'Courier New'
        size_pt = 12
        x_off = 0.0
        y_off = 0.0
        du.add_text(
            1 + x_off,
            1 + y_off,
            f"'{title_text}' by composer: {composer_text}",
            size_pt=size_pt,
            color=self.notation_color,
            id=0,
            tags=["title"],
            anchor='nw',
            family=family,
        )

        # Page metrics (mm)
        width_mm, height_mm = du.current_page_size_mm()
        margin = float(self.margin)
        stave_left_position = margin + self.semitone_dist
        stave_right_position = max(0.0, width_mm - margin) - self.semitone_dist * 2

        # Editor zoom controls vertical mm per quarter note
        zpq = score.editor.zoom_mm_per_quarter

        # --------------- drawing the grid lines, barlines, measure numbers ---------------
        base_grid = score.base_grid
        measure_numbering_cursor = 1
        time_cursor = margin
        meas_font = getattr(score.layout, 'measure_numbering_font', None)
        if meas_font is not None and callable(getattr(meas_font, 'resolve_family', None)):
            meas_family = str(meas_font.resolve_family())
        else:
            meas_family = getattr(meas_font, 'family', 'Courier New') if meas_font is not None else 'Courier New'
        meas_size = 20.0
        
        for bg in base_grid:
            numerator = bg.numerator
            denominator = bg.denominator
            measure_amount = bg.measure_amount
            beat_grouping = bg.beat_grouping

            # General formula: quarters per measure = numerator * (4/denominator)
            quarters_per_measure = float(numerator) * (4.0 / max(1.0, float(denominator)))
            measure_len_mm = quarters_per_measure * zpq

            # Beat length inside this base_grid object
            beat_length = measure_len_mm / numerator

            # Draw horizontal barlines across the stave width for each measure boundary
            color = self.notation_color
            bar_width_mm = 0.25

            for i in range(measure_amount):
                # measure numbers:
                measure_number_str = str(measure_numbering_cursor)
                du.add_text(
                    self.margin + self.stave_width + self.margin - 1.0,
                    time_cursor + 1.0,
                    measure_number_str,
                    size_pt=meas_size,
                    color=color,
                    id=0,
                    tags=["measure_number"],
                    anchor='ne',
                    family=meas_family,
                )
                
                # following the 1 == grid system:
                if len(beat_grouping) == int(numerator):
                    full_group = [int(v) for v in beat_grouping] == list(range(1, int(numerator) + 1))
                    for idx, group in enumerate(beat_grouping, start=1):
                        line_y = time_cursor + (beat_length * (idx - 1))
                        # draw the barline
                        if idx == 1:
                            du.add_line(
                                stave_left_position,
                                line_y,
                                stave_right_position,
                                line_y,
                                color=color,
                                width_mm=bar_width_mm,
                                id=0,
                                tags=["barline"],
                                dash_pattern=None
                            )
                            if full_group:
                                # continue to next beat to draw subgrid lines
                                continue
                            continue

                        # draw subgrid lines: all beats for single full group, or only resets (value == 1)
                        if full_group or int(group) == 1:
                            du.add_line(
                                stave_left_position,
                                line_y,
                                stave_right_position,
                                line_y,
                                color=color,
                                width_mm=bar_width_mm / 2,
                                id=0,
                                tags=["grid_line"],
                                dash_pattern=[2.0, 2.0]
                            )
                else:
                    # Fallback: draw only the barline
                    du.add_line(
                        stave_left_position,
                        time_cursor,
                        stave_right_position,
                        time_cursor,
                        color=color,
                        width_mm=bar_width_mm,
                        id=0,
                        tags=["grid_line"],
                        dash_pattern=None
                    )
                
                measure_numbering_cursor += 1
                time_cursor += measure_len_mm

        # draw the end barline with same style policy
        du.add_line(
            stave_left_position,
            time_cursor,
            stave_right_position,
            time_cursor,
            color=color,
            width_mm=bar_width_mm * 3,
            id=0,
            tags=["end_barline"],
            dash_pattern=None
        )