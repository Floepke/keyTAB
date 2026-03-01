from __future__ import annotations
import math
from typing import TYPE_CHECKING, cast, Tuple
from ui.widgets.draw_util import DrawUtil
from file_model.layout import LayoutFont

if TYPE_CHECKING:
    from editor.editor import Editor


class TextDrawerMixin:
    def _text_bbox(self, du: DrawUtil, text: str, family: str, size_pt: float, italic: bool, bold: bool, angle_deg: float, padding_mm: float) -> Tuple[float, float, float, list[tuple[float, float]], list[tuple[float, float]]]:
        """Return (width_mm, height_mm, offset_down_mm, rotated_corners, rounded_polygon).

        - width/height are axis-aligned (unrotated) text extents with padding applied.
        - offset_down_mm shifts the center downward so the rotated polygon stays below y=0.
        - rotated_corners are the four axis-aligned corners after rotation (for handles).
        - rounded_polygon is a rotated list of points approximating rounded corners.
        """
        xb, yb, w_mm, h_mm = du._get_text_extents_mm(text, family, size_pt, italic, bold)
        pad = max(0.0, float(padding_mm))
        w_mm += pad * 2.0
        h_mm += pad * 2.0
        hw = w_mm * 0.5
        hh = h_mm * 0.5
        r = min(pad, hw, hh)

        def _rounded_rect_points(hw_val: float, hh_val: float, radius: float) -> list[tuple[float, float]]:
            if radius <= 1e-6:
                return [(-hw_val, -hh_val), (hw_val, -hh_val), (hw_val, hh_val), (-hw_val, hh_val)]
            pts: list[tuple[float, float]] = []
            corner_defs = [
                (-hw_val + radius, -hh_val + radius, 180.0, 270.0),  # top-left
                (hw_val - radius, -hh_val + radius, 270.0, 360.0),   # top-right
                (hw_val - radius, hh_val - radius, 0.0, 90.0),       # bottom-right
                (-hw_val + radius, hh_val - radius, 90.0, 180.0),    # bottom-left
            ]
            step = 15.0
            for cx, cy, start_deg, end_deg in corner_defs:
                deg = start_deg
                while deg < end_deg + 0.01:
                    rad = math.radians(deg)
                    pts.append((cx + radius * math.cos(rad), cy + radius * math.sin(rad)))
                    deg += step
            return pts

        base_poly = _rounded_rect_points(hw, hh, r)
        corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        ang = math.radians(angle_deg)
        sin_a = math.sin(ang)
        cos_a = math.cos(ang)
        rot_corners: list[tuple[float, float]] = []
        rot_poly: list[tuple[float, float]] = []
        min_y = float("inf")
        for (dx, dy) in corners:
            rx = dx * cos_a - dy * sin_a
            ry = dx * sin_a + dy * cos_a
            rot_corners.append((rx, ry))
            if ry < min_y:
                min_y = ry
        for (dx, dy) in base_poly:
            rx = dx * cos_a - dy * sin_a
            ry = dx * sin_a + dy * cos_a
            rot_poly.append((rx, ry))
            if ry < min_y:
                min_y = ry
        offset_down = max(0.0, -min_y)
        return w_mm, h_mm, offset_down, rot_corners, rot_poly

    def draw_text(self, du: DrawUtil) -> None:
        self = cast("Editor", self)
        score = getattr(self, 'current_score', lambda: None)()
        if score is None:
            return

        def _coerce_font(value, default_font):
            if isinstance(value, LayoutFont):
                return value
            if isinstance(value, dict):
                return LayoutFont(
                    family=value.get('family', getattr(default_font, 'family', 'Courier New')),
                    size_pt=float(value.get('size_pt', getattr(default_font, 'size_pt', 12.0) or 12.0)),
                    bold=bool(value.get('bold', getattr(default_font, 'bold', False))),
                    italic=bool(value.get('italic', getattr(default_font, 'italic', False))),
                    x_offset=float(value.get('x_offset', getattr(default_font, 'x_offset', 0.0) or 0.0)),
                    y_offset=float(value.get('y_offset', getattr(default_font, 'y_offset', 0.0) or 0.0)),
                )
            return default_font if isinstance(default_font, LayoutFont) else LayoutFont()

        events = list(getattr(score.events, 'text', []) or [])
        if not events:
            return

        # Viewport culling
        top_mm = float(getattr(self, '_view_y_mm_offset', 0.0) or 0.0)
        vp_h_mm = float(getattr(self, '_viewport_h_mm', 0.0) or 0.0)
        bottom_mm = top_mm + vp_h_mm
        bleed_mm = max(2.0, float(getattr(score.editor, 'zoom_mm_per_quarter', 25.0) or 25.0) * 0.25)

        active_tool = str(getattr(getattr(self, '_tool', None), 'TOOL_NAME', ''))
        show_handles = active_tool == 'text'

        for ev in events:
            t = float(getattr(ev, 'time', 0.0) or 0.0)
            rp = int(getattr(ev, 'x_rpitch', 0) or 0)
            angle = float(getattr(ev, 'rotation', 0.0) or 0.0)
            txt = str(getattr(ev, 'text', ''))
            display_txt = txt if txt.strip() else "(no text set)"
            use_custom = bool(getattr(ev, 'use_custom_font', False))
            font = _coerce_font(getattr(ev, 'font', None), getattr(score.layout, 'font_text', None))
            if (not use_custom) or font is None:
                font = _coerce_font(getattr(score.layout, 'font_text', None), getattr(score.layout, 'font_text', None))
            family = font.resolve_family() if font and hasattr(font, 'resolve_family') else getattr(font, 'family', 'Courier New')
            size_pt = float(getattr(font, 'size_pt', 12.0) or 12.0)
            italic = bool(getattr(font, 'italic', False))
            bold = bool(getattr(font, 'bold', False))
            pad_mm = float(getattr(score.layout, 'text_background_padding_mm', 0.0) or 0.0)
            x_off = float(getattr(ev, 'x_offset_mm', 0.0) or 0.0)
            y_off = float(getattr(ev, 'y_offset_mm', 0.0) or 0.0)

            y_mm = float(self.time_to_mm(t) + y_off)
            if y_mm < (top_mm - bleed_mm) or y_mm > (bottom_mm + bleed_mm):
                continue

            try:
                x_mm = float(self.relative_c4pitch_to_x(rp)) + x_off
            except Exception:
                x_mm = 0.0

            try:
                w_mm, h_mm, offset_down, rot_corners, rot_poly = self._text_bbox(du, display_txt, family, size_pt, italic, bold, angle, pad_mm)
            except Exception:
                continue

            cy = y_mm + offset_down
            # Build rotated polygon in absolute coords
            poly = [(x_mm + dx, cy + dy) for (dx, dy) in rot_poly]
            min_x = min(p[0] for p in poly)
            max_x = max(p[0] for p in poly)
            min_y = min(p[1] for p in poly)
            max_y = max(p[1] for p in poly)

            # White background mask to cover stave behind text
            du.add_polygon(
                poly,
                stroke_color=None,
                fill_color=(1.0, 1.0, 1.0, 1.0),
                id=int(getattr(ev, '_id', 0) or 0),
                tags=["text_bg"],
            )

            # Text itself (center anchor, rotated)
            du.add_text(
                x_mm,
                cy,
                display_txt,
                family=family,
                size_pt=size_pt,
                italic=italic,
                bold=bold,
                color=self.notation_color,
                anchor='center',
                angle_deg=angle,
                id=int(getattr(ev, '_id', 0) or 0),
                tags=["text"],
            )

            try:
                self.register_text_hit_rect(int(getattr(ev, '_id', 0) or 0), min_x, min_y, max_x, max_y, kind='body')
            except Exception:
                pass

            if show_handles:
                # Place handle just beyond the rotated right edge
                handle_gap = max(1.5, (self.semitone_dist or 2.5) * 0.3)
                handle_size = max(2.0, (self.semitone_dist or 2.5) * 0.6)
                rad = w_mm * 0.5 + handle_gap
                ang = math.radians(angle)
                hx = x_mm + rad * math.cos(ang)
                hy = cy + rad * math.sin(ang)
                hx1 = hx - handle_size * 0.5
                hx2 = hx + handle_size * 0.5
                hy1 = hy - handle_size * 0.5
                hy2 = hy + handle_size * 0.5
                du.add_rectangle(
                    hx1,
                    hy1,
                    hx2,
                    hy2,
                    stroke_color=self.accent_color,
                    stroke_width_mm=0.4,
                    fill_color=self.accent_color,
                    id=int(getattr(ev, '_id', 0) or 0),
                    tags=["text", "text_handle"],
                )
                try:
                    self.register_text_hit_rect(int(getattr(ev, '_id', 0) or 0), hx1, hy1, hx2, hy2, kind='handle')
                except Exception:
                    pass
