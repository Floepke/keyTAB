from __future__ import annotations
from typing import TYPE_CHECKING, cast
from ui.widgets.draw_util import DrawUtil

if TYPE_CHECKING:
    from editor.editor import Editor


class StartRepeatDrawerMixin:
    def draw_start_repeat(self, du: DrawUtil) -> None:
        self = cast("Editor", self)
        score = self.current_score()
        if score is None:
            return
        layout = getattr(score, 'layout', None)
        if layout is None or not bool(getattr(layout, 'repeat_start_visible', True)):
            return

        events = list(getattr(score.events, 'start_repeat', []) or [])
        if not events:
            return

        top_mm = float(getattr(self, '_view_y_mm_offset', 0.0) or 0.0)
        vp_h_mm = float(getattr(self, '_viewport_h_mm', 0.0) or 0.0)
        bottom_mm = top_mm + vp_h_mm
        semitone_dx = float(self.semitone_dist or 2.5)
        bleed_mm = max(4.0, semitone_dx * 3.0)

        x_stave_right = float(self.margin or 0.0) + float(self.stave_width or 0.0)
        ext_len = max(4.0, semitone_dx * 2.4)
        x_left = x_stave_right
        x_right = x_left + ext_len
        thick_w = max(0.2, semitone_dx * 0.24)
        dot_d = max(1.0, semitone_dx * 0.6)
        dot_x1 = x_left + (ext_len / 3.0)
        dot_x2 = x_left + ((2.0 * ext_len) / 3.0)
        dot_y = max(0.8, semitone_dx * 0.55)

        for ev in events:
            try:
                t = float(getattr(ev, 'time', 0.0) or 0.0)
                ev_id = int(getattr(ev, '_id', 0) or 0)
            except Exception:
                continue
            y = float(self.time_to_mm(t))
            if y < (top_mm - bleed_mm) or y > (bottom_mm + bleed_mm):
                continue

            # Vertical timeline orientation: start-repeat is a thicker barline extension
            # on the right side, with two side-by-side dots below it.
            du.add_line(
                x_left,
                y,
                x_right,
                y,
                color=self.notation_color,
                width_mm=thick_w,
                id=ev_id,
                tags=['barline_symbol', 'start_repeat'],
            )
            for dot_x in (dot_x1, dot_x2):
                du.add_oval(
                    dot_x - (dot_d / 2.0),
                    y + dot_y - (dot_d / 2.0),
                    dot_x + (dot_d / 2.0),
                    y + dot_y + (dot_d / 2.0),
                    stroke_color=None,
                    fill_color=self.notation_color,
                    id=ev_id,
                    tags=['barline_symbol_dot', 'start_repeat_dot'],
                )
