from __future__ import annotations
from typing import TYPE_CHECKING, cast
from ui.widgets.draw_util import DrawUtil

if TYPE_CHECKING:
    from editor.editor import Editor


class DecrescendoDrawerMixin:
    def draw_decrescendo(self, du: DrawUtil) -> None:
        self = cast("Editor", self)
        if getattr(self, 'is_tiny_mode', None) and self.is_tiny_mode():
            return
        score = getattr(self, 'current_score', lambda: None)()
        if score is None:
            return

        layout = getattr(score, 'layout', None)
        if layout is not None and not bool(getattr(layout, 'hairpin_visible', True)):
            return

        events = list(getattr(score.events, 'decrescendo', []) or [])
        if not events:
            return

        style_scale = float(getattr(layout, 'scale', 1.0) or 1.0) if layout is not None else 1.0
        lw = float(getattr(layout, 'hairpin_line_width_mm', 0.5) or 0.5) * style_scale
        spread = float(getattr(layout, 'hairpin_spread_mm', 5.0) or 5.0) * style_scale

        top_mm = float(getattr(self, '_view_y_mm_offset', 0.0) or 0.0)
        vp_h_mm = float(getattr(self, '_viewport_h_mm', 0.0) or 0.0)
        bottom_mm = top_mm + vp_h_mm
        bleed_mm = max(2.0, float(getattr(score.app_state, 'zoom_mm_per_quarter', 25.0) or 25.0) * 0.5)

        is_dynamic_tool = str(getattr(getattr(self, '_tool', None), 'TOOL_NAME', '')) == 'dynamic'
        handle_r = max(1.0, (self.semitone_dist or 2.5) * 0.35)

        page_w, _ = du.current_page_size_mm()

        def clamp_x(val: float) -> float:
            if page_w <= 0:
                return val
            return max(0.0, min(float(val), float(page_w)))

        for ev in events:
            t_start = float(getattr(ev, 'time', 0.0) or 0.0)
            duration = float(getattr(ev, 'duration', 256.0) or 256.0)
            t_end = t_start + duration
            x_rpitch = int(getattr(ev, 'x_rpitch', 0) or 0)
            ev_id = int(getattr(ev, '_id', 0) or 0)

            y_start = float(self.time_to_mm(t_start))
            y_end = float(self.time_to_mm(t_end))

            if y_end < (top_mm - bleed_mm) or y_start > (bottom_mm + bleed_mm):
                continue

            x_mm = clamp_x(float(self.relative_c4pitch_to_x(x_rpitch)))
            half_spread = spread * 0.5

            # Decrescendo: open at top (start), closes toward bottom (end/tip)
            # Left arm: top-left → bottom-point
            du.add_line(
                x_mm - half_spread, y_start, x_mm, y_end,
                color=self.accent_color,
                width_mm=lw,
                line_cap='round',
                id=ev_id,
                tags=['decrescendo'],
            )
            # Right arm: top-right → bottom-point
            du.add_line(
                x_mm + half_spread, y_start, x_mm, y_end,
                color=self.accent_color,
                width_mm=lw,
                line_cap='round',
                id=ev_id,
                tags=['decrescendo'],
            )

            if is_dynamic_tool:
                # Start handle (at the open end / top)
                self.register_hairpin_hit_rect(
                    ev_id, 'decrescendo', 'start',
                    x_mm - handle_r, y_start - handle_r,
                    x_mm + handle_r, y_start + handle_r,
                )
                du.add_rectangle(
                    x_mm - handle_r, y_start - handle_r,
                    x_mm + handle_r, y_start + handle_r,
                    stroke_color=self.accent_color,
                    stroke_width_mm=0.5,
                    fill_color=self.accent_color,
                    id=ev_id,
                    tags=['decrescendo', 'decrescendo_handle'],
                )
                # End handle (at the tip / bottom)
                self.register_hairpin_hit_rect(
                    ev_id, 'decrescendo', 'end',
                    x_mm - handle_r, y_end - handle_r,
                    x_mm + handle_r, y_end + handle_r,
                )
                du.add_rectangle(
                    x_mm - handle_r, y_end - handle_r,
                    x_mm + handle_r, y_end + handle_r,
                    stroke_color=self.accent_color,
                    stroke_width_mm=0.5,
                    fill_color=self.accent_color,
                    id=ev_id,
                    tags=['decrescendo', 'decrescendo_handle'],
                )

