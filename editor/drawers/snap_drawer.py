from __future__ import annotations
from typing import TYPE_CHECKING, cast

from utils.operator import Operator as OP

from ui.widgets.draw_util import DrawUtil
from utils.CONSTANT import QUARTER_NOTE_UNIT

if TYPE_CHECKING:
    from editor.editor import Editor


class SnapDrawerMixin:
    '''
        Draws:
            - Alternating light/darker snap bands along the vertical timeline the size of the snap.
    '''

    def _side_band_tint_rgba(self, layout, side: str) -> tuple[float, float, float, float]:
        """Return a very light accent-derived tint for snap bands.

        Snap bands are a subtle timing aid and should not reuse grid-band styling.
        """
        _layout = layout  # kept for signature compatibility
        _side = side
        try:
            ar, ag, ab, _aa = tuple(getattr(self, 'accent_color', (0.2, 0.5, 1.0, 1.0)))
            pr, pg, pb, _pa = tuple(getattr(self, 'paper_color', (1.0, 1.0, 1.0, 1.0)))
        except Exception:
            ar, ag, ab = (0.2, 0.5, 1.0)
            pr, pg, pb = (1.0, 1.0, 1.0)
        mix = 0.16  # keep tint very light: mostly paper, slight accent cast
        r = pr * (1.0 - mix) + ar * mix
        g = pg * (1.0 - mix) + ag * mix
        b = pb * (1.0 - mix) + ab * mix
        return (float(r), float(g), float(b), 0.18)

    def draw_snap(self, du: DrawUtil) -> None:
        """Draw alternating light/darker snap bands along the vertical timeline.

        - Pattern resets at each measure start and always begins with a light band.
        - We only draw the darker bands; the light bands are the editor background.
        - Follows SCORE.base_grid and current zoom to convert time → mm.
        """
        self = cast("Editor", self)
        score = self.current_score()
        if score is None:
            return

        # Snap-size side bands are visible in all tools except Grid Band mode.
        active_tool = str(getattr(getattr(self, '_tool', None), 'TOOL_NAME', ''))
        if active_tool == 'grid_band':
            return

        # if the snap size < 8.0 units, skip drawing snap bands for performance
        if self.snap_size_units < 8.0:
            return

        op = OP()

        # Page and layout metrics
        page_w_mm, _page_h_mm = du.current_page_size_mm()
        margin = float(self.margin)
        
        # Draw snap pattern across the full stave width.
        stave_left = float(self.margin + self.semitone_dist)
        stave_right = float(page_w_mm - self.margin - self.semitone_dist * 2.0)
        left_x1 = stave_left
        right_x2 = stave_right
        zpq = float(getattr(score.app_state, 'zoom_mm_per_quarter', 25.0) or 25.0)

        # Snap size in time units → mm
        snap_units: float
        if hasattr(self, "snap_size_selector") and hasattr(self.snap_size_selector, "get_snap_size"):
            snap_units = float(self.snap_size_selector.get_snap_size())
        elif hasattr(self, "snap_size_units"):
            snap_units = float(getattr(self, "snap_size_units"))
        else:
            # Fallback: eighth-note snap
            snap_units = float(QUARTER_NOTE_UNIT) / 2.0
        snap_mm = (snap_units / float(QUARTER_NOTE_UNIT)) * zpq

        layout = getattr(score, 'layout', None)
        left_fill_rgba = self._side_band_tint_rgba(layout, 'left')

        # Walk the base grid (measures) and draw darker rectangles on every other snap step
        time_cursor_mm = margin
        for bg in score.base_grid:
            numerator = int(getattr(bg, 'numerator', 4) or 4)
            denominator = int(getattr(bg, 'denominator', 4) or 4)
            measure_amount = int(getattr(bg, 'measure_amount', 1) or 1)

            quarters_per_measure = float(numerator) * (4.0 / max(1.0, float(denominator)))
            measure_len_mm = quarters_per_measure * zpq

            for _ in range(measure_amount):
                sub_cursor = time_cursor_mm
                measure_end_mm = sub_cursor + measure_len_mm
                # Pattern starts with light segment; index 0 is light (skip), 1 is dark (draw)
                seg_index = 0
                while op.less(sub_cursor, measure_end_mm):
                    h = min(snap_mm, measure_end_mm - sub_cursor)
                    if (seg_index % 2) == 0:
                        du.add_rectangle(
                            left_x1,
                            sub_cursor,
                            right_x2,
                            sub_cursor + h,
                            stroke_color=None,
                            fill_color=left_fill_rgba,
                            id=0,
                            tags=["snap_band"],
                        )
                    seg_index += 1
                    sub_cursor += h

                time_cursor_mm += measure_len_mm