from __future__ import annotations
from typing import TYPE_CHECKING, cast
from ui.widgets.draw_util import DrawUtil

if TYPE_CHECKING:
    from editor.editor import Editor


class CountLineDrawerMixin:
    def draw_count_line(self, du: DrawUtil) -> None:
        self = cast("Editor", self)
        score = self.current_score()
        if score is None:
            return
        events = list(getattr(score.events, 'count_line', []) or [])
        if not events:
            return

        # Viewport culling
        top_mm = float(getattr(self, '_view_y_mm_offset', 0.0) or 0.0)
        vp_h_mm = float(getattr(self, '_viewport_h_mm', 0.0) or 0.0)
        bottom_mm = top_mm + vp_h_mm
        bleed_mm = max(2.0, float(getattr(score.editor, 'zoom_mm_per_quarter', 25.0)) * 0.25)

        # Handle size scales with semitone spacing
        handle_w = max(2.0, float(self.semitone_dist or 2.5) * 0.85)
        handle_h = max(2.0, float(self.semitone_dist or 2.5) * 0.85)
        active_tool = str(getattr(getattr(self, "_tool", None), "TOOL_NAME", ""))
        show_handles = active_tool == "count_line"

        for ev in events:
            try:
                t0 = float(getattr(ev, 'time', 0.0) or 0.0)
                rp1 = int(getattr(ev, 'rpitch1', 0) or 0)
                rp2 = int(getattr(ev, 'rpitch2', 4) or 4)
            except Exception:
                continue
            y_mm = float(self.time_to_mm(t0))
            if y_mm < (top_mm - bleed_mm) or y_mm > (bottom_mm + bleed_mm):
                continue

            x1 = float(self.relative_c4pitch_to_x(rp1))
            x2 = float(self.relative_c4pitch_to_x(rp2))
            if x2 < x1:
                x1, x2 = x2, x1

            # Dashed horizontal line
            du.add_line(
                x1,
                y_mm,
                x2,
                y_mm,
                color=(0, 0, 0, 1),
                width_mm=0.4,
                dash_pattern=[0, 1.5],
                id=int(getattr(ev, '_id', 0) or 0),
                tags=["count_line"],
            )

            # Handle rectangles at both ends (only in count line tool)
            if show_handles:
                du.add_rectangle(
                    x1 - handle_w * 0.5,
                    y_mm - handle_h * 0.5,
                    x1 + handle_w * 0.5,
                    y_mm + handle_h * 0.5,
                    stroke_color=None,
                    stroke_width_mm=0.0,
                    fill_color=(1.0, 0.4, 0.7, 1.0),
                    id=int(getattr(ev, '_id', 0) or 0),
                    tags=["count_line", "count_line_handle", "count_line_handle_start"],
                )
                du.add_rectangle(
                    x2 - handle_w * 0.5,
                    y_mm - handle_h * 0.5,
                    x2 + handle_w * 0.5,
                    y_mm + handle_h * 0.5,
                    stroke_color=None,
                    stroke_width_mm=0.0,
                    fill_color=(1.0, 0.4, 0.7, 1.0),
                    id=int(getattr(ev, '_id', 0) or 0),
                    tags=["count_line", "count_line_handle", "count_line_handle_end"],
                )
