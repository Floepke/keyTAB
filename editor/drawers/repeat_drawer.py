from __future__ import annotations
from typing import TYPE_CHECKING, cast
from ui.widgets.draw_util import DrawUtil

if TYPE_CHECKING:
    from editor.editor import Editor


class RepeatDrawerMixin:
    def _draw_repeat(self, du: DrawUtil, kind: str) -> None:
        """Draw start or end repeat symbols.

        kind: 'start' — dots below the horizontal line
              'end'   — dots above the horizontal line
        """
        self = cast("Editor", self)
        score = self.current_score()
        if score is None:
            return
        layout = getattr(score, 'layout', None)
        visibility_key = 'repeat_start_visible' if kind == 'start' else 'repeat_end_visible'
        if layout is None or not bool(getattr(layout, visibility_key, True)):
            return

        event_attr = 'start_repeat' if kind == 'start' else 'end_repeat'
        events = list(getattr(score.events, event_attr, []) or [])
        if not events:
            return

        top_mm = float(getattr(self, '_view_y_mm_offset', 0.0) or 0.0)
        vp_h_mm = float(getattr(self, '_viewport_h_mm', 0.0) or 0.0)
        bottom_mm = top_mm + vp_h_mm
        semitone_dx = float(self.semitone_dist or 2.5)
        bleed_mm = max(4.0, semitone_dx * 3.0)

        x_stave_right = float(self.margin or 0.0) + float(self.stave_width or 0.0)
        ext_len = semitone_dx * 3.0
        x_left = x_stave_right
        x_right = x_left + ext_len
        thick_w = max(0.01, float(getattr(self, 'editor_line_width_global', 0.1) or 0.1))
        dot_d = semitone_dx
        # Dot center is semitone_dist away from the outer edge of the line.
        dot_y = thick_w / 2.0 + semitone_dx + dot_d / 2.0
        # Align dot outer edges with line outer edges.
        dot_x1 = x_left + (dot_d / 2.0)
        dot_x2 = x_right - (dot_d / 2.0)

        line_tag = 'start_repeat' if kind == 'start' else 'end_repeat'
        dot_tag = 'start_repeat_dot' if kind == 'start' else 'end_repeat_dot'
        dot_sign = 1.0 if kind == 'start' else -1.0

        for ev in events:
            try:
                t = float(getattr(ev, 'time', 0.0) or 0.0)
                ev_id = int(getattr(ev, '_id', 0) or 0)
            except Exception:
                continue
            y = float(self.time_to_mm(t))
            if y < (top_mm - bleed_mm) or y > (bottom_mm + bleed_mm):
                continue

            du.add_line(
                x_left,
                y,
                x_right,
                y,
                color=self.notation_color,
                width_mm=thick_w,
                id=ev_id,
                tags=['barline_symbol', line_tag],
            )
            dot_cy = y + dot_sign * dot_y
            for dot_x in (dot_x1, dot_x2):
                du.add_oval(
                    dot_x - (dot_d / 2.0),
                    dot_cy - (dot_d / 2.0),
                    dot_x + (dot_d / 2.0),
                    dot_cy + (dot_d / 2.0),
                    stroke_color=None,
                    fill_color=self.notation_color,
                    id=ev_id,
                    tags=['barline_symbol_dot', dot_tag],
                )

    def draw_start_repeat(self, du: DrawUtil) -> None:
        self._draw_repeat(du, 'start')

    def draw_end_repeat(self, du: DrawUtil) -> None:
        self._draw_repeat(du, 'end')
