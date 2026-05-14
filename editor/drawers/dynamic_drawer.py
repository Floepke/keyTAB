from __future__ import annotations
import math
from typing import TYPE_CHECKING, cast
from ui.widgets.draw_util import DrawUtil
from editor.editor_defaults import DYNAMIC_SYMBOL_FONT_SIZE_PT, DYNAMIC_SYMBOL_BACKGROUND_PADDING_MM

if TYPE_CHECKING:
    from editor.editor import Editor
    from file_model.SCORE import SCORE


class DynamicDrawerMixin:
    def draw_dynamic(self, du: DrawUtil) -> None:
        self = cast("Editor", self)
        if getattr(self, 'is_tiny_mode_ultra', None) and self.is_tiny_mode_ultra():
            return
        score: SCORE = getattr(self, 'current_score', lambda: None)()
        if score is None:
            return

        events = list(getattr(score.events, 'dynamic_symbol', []) or [])
        if not events:
            return


        # Use hardcoded editor defaults for consistent sizing
        text_size_pt = float(DYNAMIC_SYMBOL_FONT_SIZE_PT or 12.0)
        dynamic_bg_pad = float(DYNAMIC_SYMBOL_BACKGROUND_PADDING_MM or 1.5)
        paper_color = getattr(self, 'paper_color', (1.0, 1.0, 1.0, 1.0))
        text_color = self.notation_color
        text_family = 'LelandText'
        layout_rotation = float(getattr(getattr(score, 'layout', None), 'dynamic_rotation', 0.0) or 0.0)
        extents_cache: dict[tuple[str, str, float], tuple[float, float, float, float]] = {}

        def _get_extents(symbol_text: str) -> tuple[float, float, float, float]:
            key = (symbol_text, text_family, text_size_pt)
            if key in extents_cache:
                return extents_cache[key]
            try:
                val = du._get_text_extents_mm(symbol_text, text_family, text_size_pt, False, False)
            except Exception:
                val = (0.0, 0.0, max(1.0, (text_size_pt / 72.0) * 25.4), max(1.0, (text_size_pt / 72.0) * 25.4 * 0.8))
            extents_cache[key] = val
            return val

        top_mm = float(getattr(self, '_view_y_mm_offset', 0.0) or 0.0)
        vp_h_mm = float(getattr(self, '_viewport_h_mm', 0.0) or 0.0)
        bottom_mm = top_mm + vp_h_mm
        bleed_mm = max(2.0, float(getattr(score.app_state, 'zoom_mm_per_quarter', 25.0) or 25.0) * 0.5)

        is_dynamic_tool = str(getattr(getattr(self, '_tool', None), 'TOOL_NAME', '')) == 'dynamic'

        page_w, _ = du.current_page_size_mm()

        def clamp_x(val: float) -> float:
            if page_w <= 0:
                return val
            return max(0.0, min(float(val), float(page_w)))

        for ev in events:
            symbol = str(getattr(ev, 'symbol', '') or '')
            if not symbol:
                continue
            raw_rotation = getattr(ev, 'rotation', None)
            text_angle_deg = float(layout_rotation if raw_rotation is None else raw_rotation)

            t = float(getattr(ev, 'time', 0.0) or 0.0)
            y_mm = float(self.time_to_mm(t))
            if y_mm < (top_mm - bleed_mm) or y_mm > (bottom_mm + bleed_mm):
                continue

            x_rpitch = int(getattr(ev, 'x_rpitch', 0) or 0)
            x_mm = clamp_x(float(self.relative_c4pitch_to_x(x_rpitch)))
            ev_id = int(getattr(ev, '_id', 0) or 0)

            xb, yb, w, h = _get_extents(symbol)

            bx = float(x_mm) - (float(xb) + (float(w) * 0.5))
            by = float(y_mm) - (float(yb) + (float(h) * 0.5))
            rx = bx + float(xb)
            ry = by + float(yb)
            # Text is rotated around its center in DrawUtil. Build an axis-aligned
            # bbox from the rotated glyph extents so background/hit-rect match.
            cx = rx + (float(w) * 0.5)
            cy = ry + (float(h) * 0.5)
            hw = float(w) * 0.5
            hh = float(h) * 0.5
            ang = math.radians(text_angle_deg)
            sin_a = math.sin(ang)
            cos_a = math.cos(ang)

            # Build the padded glyph rectangle in local space, then rotate it.
            bg_half_w = hw + dynamic_bg_pad
            bg_half_h = hh + dynamic_bg_pad
            local_corners = [
                (-bg_half_w, -bg_half_h),
                (bg_half_w, -bg_half_h),
                (bg_half_w, bg_half_h),
                (-bg_half_w, bg_half_h),
            ]
            bg_poly = [
                (
                    cx + (lx * cos_a) - (ly * sin_a),
                    cy + (lx * sin_a) + (ly * cos_a),
                )
                for (lx, ly) in local_corners
            ]
            sym_x1 = min(p[0] for p in bg_poly)
            sym_y1 = min(p[1] for p in bg_poly)
            sym_x2 = max(p[0] for p in bg_poly)
            sym_y2 = max(p[1] for p in bg_poly)
            sym_hit_rect_mm = (sym_x1, sym_y1, max(0.0, sym_x2 - sym_x1), max(0.0, sym_y2 - sym_y1))

            du.add_polygon(
                bg_poly,
                stroke_color=None,
                fill_color=paper_color,
                id=ev_id,
                tags=['dynamic_symbol_bg'],
                hit_rect_mm=sym_hit_rect_mm,
            )
            du.add_text(
                bx,
                by,
                symbol,
                family=text_family,
                size_pt=text_size_pt,
                italic=False,
                bold=False,
                color=text_color,
                anchor=None,
                angle_deg=text_angle_deg,
                id=ev_id,
                tags=['dynamic_symbol_text'],
                hit_rect_mm=sym_hit_rect_mm,
            )

            if is_dynamic_tool:
                self.register_hit_rect('dynamic_symbol', ev_id, sym_x1, sym_y1, sym_x2, sym_y2)
