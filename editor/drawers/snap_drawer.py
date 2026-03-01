from __future__ import annotations
from typing import TYPE_CHECKING, cast

from utils.operator import Operator as OP

from ui.widgets.draw_util import DrawUtil
from utils.CONSTANT import QUARTER_NOTE_UNIT

try:
    # Theme-aware editor background color
    from ui.style import Style  # type: ignore
except Exception:  # Fallback if style import fails at runtime
    Style = None  # type: ignore

if TYPE_CHECKING:
    from editor.editor import Editor


class SnapDrawerMixin:
    '''
        Draws:
            - Alternating light/darker snap bands along the vertical timeline the size of the snap.
    '''
    
    def _editor_bg_tint_rgba(self) -> tuple[float, float, float, float]:
        """Return a slightly darker tint than the editor background as RGBA floats.

        Uses the named 'editor' color from Style if available; otherwise falls back
        to a neutral grey.
        """
        qc = Style.get_named_qcolor('editor')
        r = max(0, min(255, qc.red()))
        g = max(0, min(255, qc.green()))
        b = max(0, min(255, qc.blue()))
        # Slightly darker (about -7%)
        dr = int(round(r * 0.92))
        dg = int(round(g * 0.92))
        db = int(round(b * 0.92))
        return (dr / 255.0, dg / 255.0, db / 255.0, 1.0)

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
        
        # if the snap size < 8.0 units, skip drawing snap bands for performance
        if self.snap_size_units < 8.0:
            return

        op = OP()

        # Page and layout metrics
        page_w_mm, _page_h_mm = du.current_page_size_mm()
        margin = float(self.margin)
        
        # Match horizontal span under the stave
        stave_left = self.margin + self.semitone_dist
        stave_right = page_w_mm - self.margin - self.semitone_dist * 2.0
        zpq = float(score.editor.zoom_mm_per_quarter)

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

        fill_rgba = self._editor_bg_tint_rgba()

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
                            stave_left,
                            sub_cursor,
                            stave_right,
                            sub_cursor + h,
                            stroke_color=None,
                            fill_color=fill_rgba,
                            id=0,
                            tags=["snap_band"],
                        )
                    seg_index += 1
                    sub_cursor += h

                time_cursor_mm += measure_len_mm