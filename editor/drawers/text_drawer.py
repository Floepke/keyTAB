from __future__ import annotations
import math
from typing import TYPE_CHECKING, cast, Tuple
from ui.widgets.draw_util import DrawUtil
from ui.style import Style
from file_model.font import Font

if TYPE_CHECKING:
    from editor.editor import Editor


class TextDrawerMixin:
    def _line_alignment_x(self, alignment: str, content_w_mm: float, line_w_mm: float) -> float:
        mode = str(alignment or "left").lower()
        if mode == "center":
            return 0.0
        if mode == "right":
            return (content_w_mm * 0.5) - (line_w_mm * 0.5)
        return (-content_w_mm * 0.5) + (line_w_mm * 0.5)

    def _text_layout(
        self,
        du: DrawUtil,
        text: str,
        family: str,
        size_pt: float,
        italic: bool,
        bold: bool,
    ) -> dict:
        raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
        raw = raw.replace("\\n", "\n").replace("\\t", "\t")
        fallback = "(no text set)"
        paragraph_strs = raw.split("\n") if raw.strip() else [fallback]
        line_entries: list[dict] = []
        max_w = 0.0
        line_h_mm = 0.0
        for para in paragraph_strs:
            measure = para if para.strip() else " "
            _, _, w, h = du._get_text_extents_mm(measure, family, size_pt, italic, bold)
            w_mm = float(max(0.0, w)) if para.strip() else 0.0
            h_mm = float(max(0.1, h))
            line_entries.append({"text": para, "width_mm": w_mm, "height_mm": h_mm})
            if w_mm > max_w:
                max_w = w_mm
            if h_mm > line_h_mm:
                line_h_mm = h_mm
        if not line_entries:
            _, _, w, h = du._get_text_extents_mm(fallback, family, size_pt, italic, bold)
            line_entries = [{"text": fallback, "width_mm": float(max(0.0, w)), "height_mm": float(max(0.1, h))}]
            max_w = float(max(0.0, w))
            line_h_mm = float(max(0.1, h))
        line_y_gap_mm = line_h_mm * 0.1
        line_block_h_mm = line_h_mm + line_y_gap_mm * 2.0
        return {
            "lines": line_entries,
            "line_height_mm": line_h_mm,
            "line_y_gap_mm": line_y_gap_mm,
            "line_block_height_mm": line_block_h_mm,
            "content_width_mm": max_w,
            "content_height_mm": line_block_h_mm * len(line_entries),
            "draw_width_mm": max_w,
        }

    def _text_bbox(self, content_w_mm: float, content_h_mm: float, angle_deg: float, padding_mm: float, width_offset_mm: float) -> Tuple[float, float, float, list[tuple[float, float]], list[tuple[float, float]]]:
        """Return (width_mm, height_mm, offset_down_mm, rotated_corners, rounded_polygon).

        - width/height are axis-aligned (unrotated) text extents with padding applied.
        - offset_down_mm shifts the center downward so the rotated polygon stays below y=0.
        - rotated_corners are the four axis-aligned corners after rotation (for handles).
        - rounded_polygon is a rotated list of points approximating rounded corners.
        """
        pad = max(0.0, float(padding_mm))
        base_w_mm = max(0.0, float(content_w_mm) + (pad * 2.0))
        h_mm = max(0.0, float(content_h_mm) + (pad * 2.0))
        base_hw = base_w_mm * 0.5
        hh = h_mm * 0.5

        # Keep text anchor math based on the original background, then apply
        # width offset only to the final rectangle's right side.
        final_x0 = -base_hw
        final_x1 = base_hw + float(width_offset_mm)
        if final_x1 < final_x0:
            final_x1 = final_x0
        w_mm = max(0.0, final_x1 - final_x0)
        r = min(pad, w_mm * 0.5, h_mm * 0.5)

        def _rounded_rect_points(x0: float, x1: float, hh_val: float, radius: float) -> list[tuple[float, float]]:
            if radius <= 1e-6:
                return [(x0, -hh_val), (x1, -hh_val), (x1, hh_val), (x0, hh_val)]
            pts: list[tuple[float, float]] = []
            corner_defs = [
                (x0 + radius, -hh_val + radius, 180.0, 270.0),  # top-left
                (x1 - radius, -hh_val + radius, 270.0, 360.0),   # top-right
                (x1 - radius, hh_val - radius, 0.0, 90.0),       # bottom-right
                (x0 + radius, hh_val - radius, 90.0, 180.0),    # bottom-left
            ]
            step = 15.0
            for cx, cy, start_deg, end_deg in corner_defs:
                deg = start_deg
                while deg < end_deg + 0.01:
                    rad = math.radians(deg)
                    pts.append((cx + radius * math.cos(rad), cy + radius * math.sin(rad)))
                    deg += step
            return pts

        # Base rectangle (without width offset) defines text position anchoring.
        base_poly = _rounded_rect_points(-base_hw, base_hw, hh, min(pad, base_hw, hh))
        corners = [(final_x0, -hh), (final_x1, -hh), (final_x1, hh), (final_x0, hh)]
        draw_poly = _rounded_rect_points(final_x0, final_x1, hh, r)
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
            if ry < min_y:
                min_y = ry
        for (dx, dy) in draw_poly:
            rx = dx * cos_a - dy * sin_a
            ry = dx * sin_a + dy * cos_a
            rot_poly.append((rx, ry))
        offset_down = max(0.0, -min_y)
        return w_mm, h_mm, offset_down, rot_corners, rot_poly

    def draw_text(self, du: DrawUtil) -> None:
        self = cast("Editor", self)
        score = getattr(self, 'current_score', lambda: None)()
        if score is None:
            return

        def _coerce_font(value, default_font):
            if isinstance(value, Font):
                return value
            if isinstance(value, dict):
                return Font(
                    family=value.get('family', getattr(default_font, 'family', 'Courier New')),
                    size_pt=float(value.get('size_pt', getattr(default_font, 'size_pt', 12.0) or 12.0)),
                    bold=bool(value.get('bold', getattr(default_font, 'bold', False))),
                    italic=bool(value.get('italic', getattr(default_font, 'italic', False))),
                    underline=bool(value.get('underline', getattr(default_font, 'underline', False))),
                    x_offset=float(value.get('x_offset', getattr(default_font, 'x_offset', 0.0) or 0.0)),
                    y_offset=float(value.get('y_offset', getattr(default_font, 'y_offset', 0.0) or 0.0)),
                )
            return default_font if isinstance(default_font, Font) else Font()

        events = list(getattr(score.events, 'text', []) or [])
        if not events:
            return

        # Viewport culling
        top_mm = float(getattr(self, '_view_y_mm_offset', 0.0) or 0.0)
        vp_h_mm = float(getattr(self, '_viewport_h_mm', 0.0) or 0.0)
        bottom_mm = top_mm + vp_h_mm
        bleed_mm = max(2.0, float(getattr(score.app_state, 'zoom_mm_per_quarter', 25.0) or 25.0) * 0.25)

        active_tool = str(getattr(getattr(self, '_tool', None), 'TOOL_NAME', ''))
        show_handles = (active_tool == 'text')

        for ev in events:
            t = float(getattr(ev, 'time', 0.0) or 0.0)
            rp = int(getattr(ev, 'x_rpitch', 0) or 0)
            angle = float(getattr(ev, 'rotation', 0.0) or 0.0)
            txt = str(getattr(ev, 'text', ''))
            use_custom = bool(getattr(ev, 'use_custom_font', False))
            font = _coerce_font(getattr(ev, 'font', None), getattr(score.layout, 'font_text', None))
            if (not use_custom) or font is None:
                font = _coerce_font(getattr(score.layout, 'font_text', None), getattr(score.layout, 'font_text', None))
            family = font.resolve_family() if font and hasattr(font, 'resolve_family') else getattr(font, 'family', 'Courier New')
            size_pt = float(getattr(font, 'size_pt', 12.0) or 12.0)
            italic = bool(getattr(font, 'italic', False))
            bold = bool(getattr(font, 'bold', False))
            underline = bool(getattr(font, 'underline', False))
            pad_mm = float(getattr(score.layout, 'text_background_padding_mm', 0.0) or 0.0)
            width_offset_mm = float(getattr(ev, 'text_background_width_offset_mm', 0.0) or 0.0)
            x_off = float(getattr(ev, 'x_offset_mm', 0.0) or 0.0)
            y_off = float(getattr(ev, 'y_offset_mm', 0.0) or 0.0)
            alignment = str(getattr(ev, 'alignment', 'left') or 'left').lower()

            y_mm = float(self.time_to_mm(t) + y_off)
            if y_mm < (top_mm - bleed_mm) or y_mm > (bottom_mm + bleed_mm):
                continue

            x_mm = float(self.relative_c4pitch_to_x(rp)) + x_off

            layout_info = self._text_layout(du, txt, family, size_pt, italic, bold)
            lines = list(layout_info.get("lines", []))
            content_w_mm = float(layout_info.get("content_width_mm", 0.0) or 0.0)
            line_h_mm = float(layout_info.get("line_height_mm", 0.0) or 0.0)
            line_y_gap_mm = float(layout_info.get("line_y_gap_mm", 0.0) or 0.0)
            line_block_h_mm = float(layout_info.get("line_block_height_mm", line_h_mm) or line_h_mm)
            content_h_mm = float(layout_info.get("content_height_mm", 0.0) or 0.0)

            w_mm, _, offset_down, _, rot_poly = self._text_bbox(
                content_w_mm,
                content_h_mm,
                angle,
                pad_mm,
                width_offset_mm,
            )

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

            ang_rad = math.radians(angle)
            cos_a = math.cos(ang_rad)
            sin_a = math.sin(ang_rad)
            total_h = line_block_h_mm * max(1, len(lines))

            def _to_world(local_x: float, local_y: float) -> tuple[float, float]:
                wx = x_mm + (local_x * cos_a) - (local_y * sin_a)
                wy = cy + (local_x * sin_a) + (local_y * cos_a)
                return wx, wy

            for idx_line, line in enumerate(lines):
                line_text = str(line.get("text", ""))
                line_w_mm = float(line.get("width_mm", 0.0) or 0.0)
                line_y_local = (-total_h * 0.5) + (line_block_h_mm * idx_line) + line_y_gap_mm + (line_h_mm * 0.5)
                line_x_local = self._line_alignment_x(alignment, content_w_mm, line_w_mm)

                if line_text:
                    if alignment == 'right':
                        draw_x, draw_y = _to_world(content_w_mm * 0.5, line_y_local)
                        text_anchor = 'e'
                    else:
                        draw_x, draw_y = _to_world(line_x_local, line_y_local)
                        text_anchor = 'center'
                    du.add_text(
                        draw_x,
                        draw_y,
                        line_text,
                        family=family,
                        size_pt=size_pt,
                        italic=italic,
                        bold=bold,
                        color=self.notation_color,
                        anchor=text_anchor,
                        angle_deg=angle,
                        id=int(getattr(ev, '_id', 0) or 0),
                        tags=["text"],
                    )

                if underline and line_text:
                    xb_mm, yb_mm, ink_w_mm, ink_h_mm = du._get_text_extents_mm(line_text, family, size_pt, italic, bold)
                    ul_y_local = -ink_h_mm / 2.0 - yb_mm + max(0.2, size_pt * 0.025)
                    if alignment == 'right':
                        ul_x1, ul_y1 = _to_world(content_w_mm * 0.5 - ink_w_mm, line_y_local + ul_y_local)
                        ul_x2, ul_y2 = _to_world(content_w_mm * 0.5, line_y_local + ul_y_local)
                    else:
                        half_w = ink_w_mm * 0.5
                        ul_x1, ul_y1 = _to_world(line_x_local - half_w, line_y_local + ul_y_local)
                        ul_x2, ul_y2 = _to_world(line_x_local + half_w, line_y_local + ul_y_local)
                    du.add_line(
                        ul_x1, ul_y1, ul_x2, ul_y2,
                        color=self.notation_color,
                        width_mm=max(0.2, size_pt * (0.04 if bold else 0.02)),
                        tags=["text_underline"],
                        id=int(getattr(ev, '_id', 0) or 0),
                    )
            self.register_hit_rect('text', int(getattr(ev, '_id', 0) or 0), min_x, min_y, max_x, max_y, kind='body')

            if show_handles:
                # Place handle just beyond the rotated right edge
                handle_gap = max(1.5, (self.semitone_dist or 2.5) * 0.3)
                # Match count-line handle geometry: side = 1.4 * max(2.0, semitone_dist)
                handle_size = max(2.0, float(self.semitone_dist or 2.5)) * 1.4
                rad = w_mm * 0.5 + handle_gap
                ang = math.radians(angle)
                hx = x_mm + rad * math.cos(ang)
                hy = cy + rad * math.sin(ang)
                hx1 = hx - handle_size * 0.5
                hx2 = hx + handle_size * 0.5
                hy1 = hy - handle_size * 0.5
                hy2 = hy + handle_size * 0.5
                # Get custom handle color and convert from RGB (0-255) to normalized RGBA (0-1)
                color_rgb = Style.get_named_rgb('accent_color2', (128, 0, 0))
                handle_color = (float(color_rgb[0]) / 255.0, float(color_rgb[1]) / 255.0, float(color_rgb[2]) / 255.0, 1.0)
                du.add_rectangle(
                    hx1,
                    hy1,
                    hx2,
                    hy2,
                    stroke_color=None,
                    stroke_width_mm=0.0,
                    fill_color=handle_color,
                    id=int(getattr(ev, '_id', 0) or 0),
                    tags=["text", "text_handle"],
                )
                self.register_hit_rect('text', int(getattr(ev, '_id', 0) or 0), hx1, hy1, hx2, hy2, kind='handle')
