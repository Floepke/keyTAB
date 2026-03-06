'''
Grid and barline drawing mixin for the Editor class.

Handles drawing barlines, measure numbers, and gridlines.
'''

from __future__ import annotations
from typing import TYPE_CHECKING, cast
from file_model.SCORE import SCORE
from ui.widgets.draw_util import DrawUtil
from utils.CONSTANT import QUARTER_NOTE_UNIT

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
        width_mm, _height_mm = du.current_page_size_mm()
        margin = float(self.margin)
        stave_left_position = margin + self.semitone_dist
        stave_right_position = max(0.0, width_mm - margin) - self.semitone_dist * 2

        # --------------- drawing the grid lines, barlines, measure numbers ---------------
        measure_numbering_cursor = 1
        meas_font = getattr(score.layout, 'measure_numbering_font', None)
        if meas_font is not None and callable(getattr(meas_font, 'resolve_family', None)):
            meas_family = str(meas_font.resolve_family())
        else:
            meas_family = getattr(meas_font, 'family', 'Courier New') if meas_font is not None else 'Courier New'
        meas_size = 20.0
        color = self.notation_color
        bar_width_mm = 0.25

        cache = getattr(self, '_draw_cache', None) or {}
        grid_den_times = list(cache.get('grid_den_times') or [])
        barline_times = list(cache.get('barline_times') or [])

        # Safety fallback when draw cache is unavailable.
        if not barline_times:
            cur_t = 0.0
            for bg in list(getattr(score, 'base_grid', []) or []):
                numer = int(getattr(bg, 'numerator', 4) or 4)
                denom = int(getattr(bg, 'denominator', 4) or 4)
                mcount = int(getattr(bg, 'measure_amount', 1) or 1)
                measure_len_ticks = float(numer) * (4.0 / float(max(1, denom))) * float(QUARTER_NOTE_UNIT)
                beat_len_ticks = measure_len_ticks / max(1, numer)
                positions = list(getattr(bg, 'beat_grouping', []) or [])
                for _ in range(mcount):
                    barline_times.append(float(cur_t))
                    if len(positions) == numer:
                        if positions == [v for v in range(1, numer + 1)]:
                            for idx in range(1, numer + 1):
                                grid_den_times.append(float(cur_t + (idx - 1) * beat_len_ticks))
                        else:
                            for idx, val in enumerate(positions, start=1):
                                if int(val) == 1:
                                    grid_den_times.append(float(cur_t + (idx - 1) * beat_len_ticks))
                    else:
                        grid_den_times.append(float(cur_t))
                    cur_t += measure_len_ticks
            barline_times.append(float(cur_t))
            grid_den_times.append(float(cur_t))

        if barline_times:
            barline_keys = {round(float(t), 6) for t in barline_times}

            # Draw measure numbers at each measure start except final end barline.
            for t in barline_times[:-1]:
                y_mm = float(self.time_to_mm(float(t)))
                du.add_text(
                    self.margin + self.stave_width + self.margin - 1.0,
                    y_mm + 1.0,
                    str(measure_numbering_cursor),
                    size_pt=meas_size,
                    color=color,
                    id=0,
                    tags=["measure_number"],
                    anchor='ne',
                    family=meas_family,
                )
                measure_numbering_cursor += 1

            # Draw subgrid lines from cached grid times, excluding barlines.
            for t in grid_den_times:
                if round(float(t), 6) in barline_keys:
                    continue
                y_mm = float(self.time_to_mm(float(t)))
                du.add_line(
                    stave_left_position,
                    y_mm,
                    stave_right_position,
                    y_mm,
                    color=color,
                    width_mm=bar_width_mm / 2,
                    id=0,
                    tags=["grid_line"],
                    dash_pattern=[2.0, 2.0],
                )

            # Draw regular barlines; draw final end barline thicker.
            for idx, t in enumerate(barline_times):
                y_mm = float(self.time_to_mm(float(t)))
                is_last = idx == (len(barline_times) - 1)
                du.add_line(
                    stave_left_position,
                    y_mm,
                    stave_right_position,
                    y_mm,
                    color=color,
                    width_mm=(bar_width_mm * 3.0) if is_last else bar_width_mm,
                    id=0,
                    tags=["end_barline" if is_last else "barline"],
                    dash_pattern=None,
                )