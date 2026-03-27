from __future__ import annotations
from typing import TYPE_CHECKING, cast

from ui.widgets.draw_util import DrawUtil

if TYPE_CHECKING:
    from editor.editor import Editor


class DoubleBarDrawerMixin:
    def draw_double_bar(self, du: DrawUtil) -> None:
        self = cast("Editor", self)
        score = self.current_score()
        if score is None:
            return
        layout = getattr(score, "layout", None)
        if layout is None or not bool(getattr(layout, "double_bar_visible", True)):
            return

        events = list(getattr(score.events, "double_bar", []) or [])
        if not events:
            return

        top_mm = float(getattr(self, "_view_y_mm_offset", 0.0) or 0.0)
        vp_h_mm = float(getattr(self, "_viewport_h_mm", 0.0) or 0.0)
        bottom_mm = top_mm + vp_h_mm
        semitone_dx = float(self.semitone_dist or 2.5)
        bleed_mm = max(4.0, semitone_dx * 3.0)

        x_left = float(self.margin or 0.0) + float(self.stave_width or 0.0) + max(2.0, semitone_dx * 0.8)
        x_right = x_left + max(4.0, semitone_dx * 2.8)
        gap = max(0.6, semitone_dx * 0.35)
        line_w = max(0.15, semitone_dx * 0.12)

        for ev in events:
            try:
                t = float(getattr(ev, "time", 0.0) or 0.0)
                ev_id = int(getattr(ev, "_id", 0) or 0)
            except Exception:
                continue
            y = float(self.time_to_mm(t))
            if y < (top_mm - bleed_mm) or y > (bottom_mm + bleed_mm):
                continue
            du.add_line(
                x_left,
                y - gap,
                x_right,
                y - gap,
                color=self.notation_color,
                width_mm=line_w,
                id=ev_id,
                tags=["barline_symbol"],
            )
            du.add_line(
                x_left,
                y + gap,
                x_right,
                y + gap,
                color=self.notation_color,
                width_mm=line_w,
                id=ev_id,
                tags=["barline_symbol"],
            )
