from __future__ import annotations
from typing import TYPE_CHECKING, cast
from ui.widgets.draw_util import DrawUtil

if TYPE_CHECKING:
    from editor.editor import Editor


class CountLineDrawerMixin:
    def draw_count_line(self, du: DrawUtil) -> None:
        self = cast("Editor", self)
        if getattr(self, 'is_tiny_mode', None) and self.is_tiny_mode():
            return
        score = self.current_score()
        if score is None:
            return
        score_events = self.current_events(score)
        events = list(getattr(score_events, 'count_line', []) or [])
        if not events:
            return

        # Viewport culling
        top_mm = float(getattr(self, '_view_y_mm_offset', 0.0) or 0.0)
        vp_h_mm = float(getattr(self, '_viewport_h_mm', 0.0) or 0.0)
        bottom_mm = top_mm + vp_h_mm
        bleed_mm = max(2.0, float(getattr(score.app_state, 'zoom_mm_per_quarter', 25.0)) * 0.25)

        # Handle size scales with semitone spacing
        handle_w = max(2.0, float(self.semitone_dist or 2.5))
        handle_h = max(2.0, float(self.semitone_dist or 2.5))
        active_tool = str(getattr(getattr(self, "_tool", None), "TOOL_NAME", ""))
        show_handles = active_tool == "count_line"
        handle_red = (.5, 0.0, 0.0, 1.0)

        for ev in events:
            try:
                t0 = float(getattr(ev, 'time', 0.0) or 0.0)
                rp1 = int(getattr(ev, 'rpitch1', 0) or 0)
                rp2_raw = getattr(ev, 'rpitch2', 4)
                rp2 = int(4 if rp2_raw is None else rp2_raw)
            except Exception:
                continue
            y_mm = float(self.time_to_mm(t0))
            if y_mm < (top_mm - bleed_mm) or y_mm > (bottom_mm + bleed_mm):
                continue

            # Keep semantic handle identity stable:
            # - start handle is always rpitch1
            # - end handle is always rpitch2
            x_start = float(self.relative_c4pitch_to_x(rp1))
            x_end = float(self.relative_c4pitch_to_x(rp2))
            x_left = min(x_start, x_end)
            x_right = max(x_start, x_end)

            line_hit_half_h = max(0.2, float(getattr(self, 'editor_line_width_global', 0.5) or 0.5))
            line_hit_x1 = min(x_left + handle_w * .5, x_right - handle_w * .5)
            line_hit_x2 = max(x_left + handle_w * .5, x_right - handle_w * .5)
            self.register_hit_rect(
                'count_line',
                int(getattr(ev, '_id', 0) or 0),
                line_hit_x1,
                y_mm - line_hit_half_h,
                line_hit_x2,
                y_mm + line_hit_half_h,
                part='line',
            )

            # the count line itself
            du.add_line(
                x_left,
                y_mm,
                x_right,
                y_mm,
                color=self.accent_color,
                width_mm=0.4,
                dash_pattern=[0, 1.5],
                id=int(getattr(ev, '_id', 0) or 0),
                tags=["count_line"],
            )

            # Handle rectangles at both ends (only in count line tool)
            if show_handles:
                self.register_hit_rect(
                    'count_line',
                    int(getattr(ev, '_id', 0) or 0),
                    x_start - handle_w * .7,
                    y_mm - handle_h * .7,
                    x_start + handle_w * .7,
                    y_mm + handle_h * .7,
                    part='start',
                )
                du.add_rectangle(
                    x_start - handle_w * .7,
                    y_mm - handle_h * .7,
                    x_start + handle_w * .7,
                    y_mm + handle_h * .7,
                    stroke_color=None,
                    stroke_width_mm=0.0,
                    fill_color=handle_red,
                    id=int(getattr(ev, '_id', 0) or 0),
                    tags=["count_line", "count_line_handle", "count_line_handle_start"],
                )
                self.register_hit_rect(
                    'count_line',
                    int(getattr(ev, '_id', 0) or 0),
                    x_end - handle_w * .7,
                    y_mm - handle_h * .7,
                    x_end + handle_w * .7,
                    y_mm + handle_h * .7,
                    part='end',
                )
                du.add_rectangle(
                    x_end - handle_w * .7,
                    y_mm - handle_h * .7,
                    x_end + handle_w * .7,
                    y_mm + handle_h * .7,
                    stroke_color=None,
                    stroke_width_mm=0.0,
                    fill_color=handle_red,
                    id=int(getattr(ev, '_id', 0) or 0),
                    tags=["count_line", "count_line_handle", "count_line_handle_end"],
                )
