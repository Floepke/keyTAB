from __future__ import annotations
from typing import TYPE_CHECKING, cast
from utils.settings_manager import get_preferences
from ui.widgets.draw_util import DrawUtil

if TYPE_CHECKING:
    from editor.editor import Editor


# Horizontal clearance kept between the tempo indicator lane and the volta lane.
_TEMPO_LANE_CLEARANCE_MM = 1.5


class VoltaDrawerMixin:
    """Draws volta brackets alongside the editor stave.

    The geometry mirrors the tempo indicator bracket, but the volta bracket is
    solid instead of dashed and its number is centered inside it.
    """

    def draw_volta(self, du: DrawUtil) -> None:
        self = cast("Editor", self)
        score = self.current_score()
        if score is None:
            return

        events = list(getattr(score, 'volta', []) or [])
        if not events:
            return

        layout_obj = getattr(score, 'layout', None)

        margin = float(self.margin or 0.0)
        # Use the document page width (pitch axis) regardless of viewport rotation.
        page_w_mm = float(getattr(layout_obj, 'page_width_mm', 210.0) or 210.0)

        top_mm = float(getattr(self, '_view_y_mm_offset', 0.0) or 0.0)
        vp_h_mm = float(getattr(self, '_viewport_h_mm', 0.0) or 0.0)
        bottom_mm = top_mm + vp_h_mm
        bleed_mm = max(2.0, float(getattr(score.app_state, 'zoom_mm_per_quarter', 25.0) or 25.0) * 0.5)

        notation_color = tuple(getattr(self, 'notation_color', (0.0, 0.0, 0.0, 1.0)))

        volta_font = getattr(layout_obj, 'volta_font', None)
        try:
            from fonts import register_font_from_bytes
            family_name = str(getattr(volta_font, 'family', 'Edwin') or 'Edwin')
            font_family = register_font_from_bytes(family_name) or family_name
        except Exception:
            font_family = 'Edwin'

        preferences = get_preferences()
        editor_orientation = 'horizontal' if preferences.get("editor_orientation", 'horizontal') == 'horizontal' else 'vertical'

        scale = float(getattr(layout_obj, 'scale', 1.0) or 1.0)
        font_size_pt = float(getattr(volta_font, 'size_pt', 25.0) or 25.0) * scale
        font_bold = bool(getattr(volta_font, 'bold', True))
        font_italic = bool(getattr(volta_font, 'italic', False))
        configured_thickness = float(getattr(layout_obj, 'volta_thickness_mm', 1.25) or 1.25)
        # Editor strokes use the established half-thickness convention; the
        # editor renderer applies its global stroke multiplier afterwards.
        bracket_stroke = configured_thickness * 0.5
        hook_len = float(getattr(layout_obj, 'volta_hook_length_mm', 4.0) or 4.0) * scale

        x_right = float(page_w_mm) - margin * 0.85
        # Shift the whole lane left of the tempo indicators so the two never overlap.
        lane_bounds = self.tempo_lane_bounds_mm(du) if hasattr(self, 'tempo_lane_bounds_mm') else None
        if lane_bounds is not None:
            # Round line caps make the spine bleed half a stroke past x_right.
            clearance = max(0.5, _TEMPO_LANE_CLEARANCE_MM * scale) + (bracket_stroke * 0.5)
            x_right = min(x_right, float(lane_bounds[0]) - clearance)
        # The hook reaches toward the stave but never off the page.
        x_hook_left = max(margin, float(x_right) - hook_len)

        for vt in events:
            t0 = float(getattr(vt, 'time', 0.0) or 0.0)
            du_ticks = float(getattr(vt, 'duration', 0.0) or 0.0)
            if du_ticks <= 0.0:
                continue
            y0 = float(self.time_to_mm(t0))
            y1 = float(self.time_to_mm(t0 + du_ticks))
            if y1 < y0:
                y0, y1 = y1, y0
            if y1 < (top_mm - bleed_mm) or y0 > (bottom_mm + bleed_mm):
                continue

            # Solid open-left bracket: start hook, spine, end hook.
            du.add_line(x_hook_left, y0, x_right, y0, color=notation_color,
                        width_mm=bracket_stroke, id=0, tags=["volta_bracket"])
            du.add_line(x_right, y0, x_right, y1, color=notation_color,
                        width_mm=bracket_stroke, id=0, tags=["volta_bracket"])
            du.add_line(x_hook_left, y1, x_right, y1, color=notation_color,
                        width_mm=bracket_stroke, id=0, tags=["volta_bracket"])

            self.register_hit_rect(
                'volta',
                int(getattr(vt, '_id', 0) or 0),
                float(x_hook_left),
                min(y0, y1),
                x_right,
                max(y0, y1),
            )

            text = str(int(getattr(vt, 'number', 1) or 1))
            du.add_text(
                (x_hook_left + x_right) * 0.5,
                (y0 + y1) * 0.5,
                text,
                family=font_family,
                size_pt=font_size_pt,
                italic=font_italic,
                bold=font_bold,
                color=notation_color,
                anchor='center',
                id=0,
                tags=["volta_text"],
                hit_rect_mm=None,
                angle_deg=0.0 if editor_orientation == 'vertical' else 90.0,
            )
