from __future__ import annotations
from typing import TYPE_CHECKING, cast
from ui.widgets.draw_util import DrawUtil
from editor.editor_defaults import SCALE, DYNAMIC_SYMBOL_FONT_SIZE_PT, DYNAMIC_SYMBOL_BACKGROUND_PADDING_MM

if TYPE_CHECKING:
    from editor.editor import Editor


class DynamicDrawerMixin:
    def draw_dynamic(self, du: DrawUtil) -> None:
        self = cast("Editor", self)
        if getattr(self, 'is_tiny_mode_ultra', None) and self.is_tiny_mode_ultra():
            return
        score = getattr(self, 'current_score', lambda: None)()
        if score is None:
            return

        events = list(getattr(score.events, 'dynamic_symbol', []) or [])
        if not events:
            return

        # Use hardcoded editor defaults for dynamic symbol styling (not from file layout)
        text_size_pt = float(DYNAMIC_SYMBOL_FONT_SIZE_PT or 12.0)
        dynamic_bg_pad = float(DYNAMIC_SYMBOL_BACKGROUND_PADDING_MM or 1.5)

        paper_color = getattr(self, 'paper_color', (1.0, 1.0, 1.0, 1.0))
        text_color = self.notation_color
        text_family = 'LelandText'
        text_angle_deg = 90.0

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

            t = float(getattr(ev, 'time', 0.0) or 0.0)
            y_mm = float(self.time_to_mm(t))
            if y_mm < (top_mm - bleed_mm) or y_mm > (bottom_mm + bleed_mm):
                continue

            x_rpitch = int(getattr(ev, 'x_rpitch', 0) or 0)
            x_mm = clamp_x(float(self.relative_c4pitch_to_x(x_rpitch)))
            ev_id = int(getattr(ev, '_id', 0) or 0)

            try:
                xb, yb, w, h = du._get_text_extents_mm(symbol, text_family, text_size_pt, False, False)
            except Exception:
                xb, yb, w, h = 0.0, 0.0, max(1.0, (text_size_pt / 72.0) * 25.4), max(1.0, (text_size_pt / 72.0) * 25.4 * 0.8)

            bx = float(x_mm) - (float(xb) + (float(w) * 0.5))
            by = float(y_mm) - (float(yb) + (float(h) * 0.5))
            rx = bx + float(xb)
            ry = by + float(yb)
            # Text is rotated 90 degrees around its center in DrawUtil. Use a
            # swapped axis-aligned bbox so background and hit-rect stay aligned.
            cx = rx + (float(w) * 0.5)
            cy = ry + (float(h) * 0.5)
            rot_w = float(h)
            rot_h = float(w)
            sym_x1 = cx - (rot_w * 0.5) - dynamic_bg_pad
            sym_y1 = cy - (rot_h * 0.5) - dynamic_bg_pad
            sym_x2 = cx + (rot_w * 0.5) + dynamic_bg_pad
            sym_y2 = cy + (rot_h * 0.5) + dynamic_bg_pad

            du.add_rectangle(
                sym_x1,
                sym_y1,
                sym_x2,
                sym_y2,
                corner_radius=max(0.0, dynamic_bg_pad),
                stroke_color=None,
                fill_color=paper_color,
                id=ev_id,
                tags=['dynamic_symbol_bg_top'],
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
                tags=['dynamic_symbol_text_top'],
            )

            if is_dynamic_tool:
                self.register_hit_rect('dynamic_symbol', ev_id, sym_x1, sym_y1, sym_x2, sym_y2)
