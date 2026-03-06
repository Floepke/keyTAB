from __future__ import annotations
from typing import TYPE_CHECKING, cast
from ui.widgets.draw_util import DrawUtil

if TYPE_CHECKING:
    from editor.editor import Editor


class LineBreakDrawerMixin:
    def draw_line_break(self, du: DrawUtil) -> None:
        self = cast("Editor", self)
        tool_name = getattr(getattr(self, "_tool", None), "TOOL_NAME", "")
        if tool_name != "line_break":
            return
        score = self.current_score()
        if score is None:
            return

        events = list(getattr(score.events, 'line_break', []) or [])
        if not events:
            return

        # Viewport culling
        top_mm = float(getattr(self, '_view_y_mm_offset', 0.0) or 0.0)
        vp_h_mm = float(getattr(self, '_viewport_h_mm', 0.0) or 0.0)
        bottom_mm = top_mm + vp_h_mm
        bleed_mm = max(2.0, float(getattr(score.app_state, 'zoom_mm_per_quarter', 25.0)) * 0.25)

        # Layout anchors
        editor_left = 0.0

        # Font setup
        try:
            from fonts import register_font_from_bytes
            font_family = register_font_from_bytes('C059') or 'C059'
        except Exception:
            font_family = 'C059'

        for ev in events:
            try:
                t0 = float(getattr(ev, 'time', 0.0) or 0.0)
                is_page = bool(getattr(ev, 'page_break', False))
            except Exception:
                continue
            y_mm = float(self.time_to_mm(t0))
            if y_mm < (top_mm - bleed_mm) or y_mm > (bottom_mm + bleed_mm):
                continue

            label = 'P' if is_page else 'L'
            # Rectangle sized to text, top aligned at time position
            try:
                _xb, _yb, w_mm, h_mm = du._get_text_extents_mm(label, font_family, 18.0, False, True)
            except Exception:
                w_mm, h_mm = (6.0, 6.0)
            rect_w = max(6.0, float(w_mm) + 4.0)
            rect_h = max(6.0, float(h_mm) + 4.0)
            rect_x1 = editor_left
            rect_x2 = rect_x1 + rect_w
            marker_x = rect_x1 + (rect_w * 0.5)
            rect_y1 = y_mm
            rect_y2 = y_mm + rect_h
            du.add_rectangle(
                rect_x1,
                rect_y1,
                rect_x2,
                rect_y2,
                stroke_color=None,
                stroke_width_mm=0.25,
                fill_color=(0, 0, 0, 1),
                id=int(getattr(ev, '_id', 0) or 0),
                tags=["line_break"],
            )
            du.add_text(
                marker_x,
                y_mm + (rect_h * 0.5),
                label,
                size_pt=18.0,
                color=(1, 1, 1, 1),
                id=int(getattr(ev, '_id', 0) or 0),
                tags=["line_break"],
                anchor='center',
                family=font_family,
                bold=True,
            )
