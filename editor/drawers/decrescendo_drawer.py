from __future__ import annotations
from typing import TYPE_CHECKING, cast
from ui.widgets.draw_util import DrawUtil
from editor.editor_defaults import SCALE, HAIRPIN_LINE_WIDTH_MM, HAIRPIN_WIDTH_MM, DYNAMIC_SYMBOL_FONT_SIZE_PT, DYNAMIC_SYMBOL_BACKGROUND_PADDING_MM, HAIRPIN_TEXT_GAP_MM
from utils.CONSTANT import SHORTEST_DURATION
from utils.operator import Operator

if TYPE_CHECKING:
    from editor.editor import Editor


class DecrescendoDrawerMixin:
    def _get_dynamic_symbol_at_position(self, t: float, x_rpitch: int) -> dict | None:
        """
        Check if there's a dynamic symbol at the given time and x_rpitch.
        Returns dict with 'glyph', 'width_mm', 'height_mm' if found, None otherwise.
        """
        self = cast("Editor", self)
        score = self.current_score()
        if score is None:
            return None
        
        layout = getattr(score, 'layout', None)
        dynamic_symbols = list(getattr(score.events, 'dynamic_symbol', []) or [])
        op = Operator(float(SHORTEST_DURATION))
        
        for sym in dynamic_symbols:
            sym_time = float(getattr(sym, 'time', 0.0) or 0.0)
            sym_rpitch = int(getattr(sym, 'x_rpitch', 0) or 0)
            
            # Check if symbol is at same time and x position
            if op.eq(sym_time, t) and sym_rpitch == x_rpitch:
                glyph = str(getattr(sym, 'symbol', '') or '')
                if not glyph:
                    return None
                
                # Calculate glyph dimensions using font metrics
                try:
                    from fonts import register_font_from_bytes
                    import PySide6.QtGui as QtGui
                    
                    leland_family = register_font_from_bytes('LelandText') or 'LelandText'
                    symbol_font = QtGui.QFont(leland_family)
                    font_size_pt = float(getattr(layout, 'dynamic_symbol_font_size_pt', 12.0) or 12.0)
                    symbol_font.setPointSizeF(font_size_pt)
                    
                    metrics = QtGui.QFontMetrics(symbol_font)
                    glyph_w_px = metrics.horizontalAdvance(glyph)
                    glyph_h_px = metrics.boundingRect(glyph).height()
                    
                    # Convert to mm (approximate: 1pt ≈ 0.35mm at screen DPI levels)
                    scale = float(getattr(layout, 'scale', 1.0) or 1.0)
                    px_per_mm = float(getattr(self, '_widget_px_per_mm', 1.0) or 1.0)
                    glyph_w_mm = (glyph_w_px / px_per_mm) * scale if px_per_mm else 3.5
                    glyph_h_mm = (glyph_h_px / px_per_mm) * scale if px_per_mm else 2.5
                    
                    padding = float(getattr(layout, 'dynamic_symbol_background_padding_mm', 2.5) or 2.5)
                    
                    return {
                        'glyph': glyph,
                        'width_mm': glyph_w_mm + (2 * padding),
                        'height_mm': glyph_h_mm + (2 * padding),
                    }
                except Exception:
                    # Fallback if font calculation fails
                    return {
                        'glyph': glyph,
                        'width_mm': 4.0,
                        'height_mm': 3.0,
                    }
        
        return None
    
    def _adjust_hairpin_for_symbols(
        self, 
        t_start: float, 
        t_end: float, 
        x_rpitch: int,
        y_start_draw: float,
        y_end_draw: float,
    ) -> tuple[float, float]:
        """
        Adjust hairpin start/end draw positions to avoid overlapping with dynamic symbols.
        Returns (adjusted_y_start_draw, adjusted_y_end_draw).
        """
        self = cast("Editor", self)
        score = self.current_score()
        
        # Use hardcoded editor default for hairpin text gap (not from file layout)
        gap_mm = float(HAIRPIN_TEXT_GAP_MM or 0.5)
        
        # Check if there's a symbol at the start position
        symbol_at_start = self._get_dynamic_symbol_at_position(t_start, x_rpitch)
        if symbol_at_start is not None:
            # Move the visible start inward so the open end does not sit under the symbol.
            symbol_half_height = symbol_at_start['height_mm'] * 0.5
            y_start_draw += symbol_half_height + gap_mm
        
        # Check if there's a symbol at the end position
        symbol_at_end = self._get_dynamic_symbol_at_position(t_end, x_rpitch)
        if symbol_at_end is not None:
            # Move the tip inward so the decrescendo gets shorter instead of extending through the symbol.
            symbol_half_height = symbol_at_end['height_mm'] * 0.5
            y_end_draw -= symbol_half_height + gap_mm
        
        return y_start_draw, y_end_draw
    
    def draw_decrescendo(self, du: DrawUtil) -> None:
        self = cast("Editor", self)
        if getattr(self, 'is_tiny_mode_ultra', None) and self.is_tiny_mode_ultra():
            return
        score = getattr(self, 'current_score', lambda: None)()
        if score is None:
            return

        events = list(getattr(score.events, 'decrescendo', []) or [])
        if not events:
            return

        # Use hardcoded editor defaults for hairpin styling (not from file layout)
        lw = float(HAIRPIN_LINE_WIDTH_MM or 1.0) * float(SCALE or 1.0)
        spread = float(HAIRPIN_WIDTH_MM or 10.0) * float(SCALE or 1.0)

        top_mm = float(getattr(self, '_view_y_mm_offset', 0.0) or 0.0)
        vp_h_mm = float(getattr(self, '_viewport_h_mm', 0.0) or 0.0)
        bottom_mm = top_mm + vp_h_mm
        bleed_mm = max(2.0, float(getattr(score.app_state, 'zoom_mm_per_quarter', 25.0) or 25.0) * 0.5)

        is_dynamic_tool = str(getattr(getattr(self, '_tool', None), 'TOOL_NAME', '')) == 'dynamic'
        # Match count-line handle geometry: side = 1.4 * max(2.0, semitone_dist)
        handle_r = max(2.0, float(self.semitone_dist or 2.5)) * 0.7

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

            y_start_draw = y_start
            y_end_draw = y_end
            min_span = max(0.8, float(self.semitone_dist or 0.5) * 0.6)
            if (y_end_draw - y_start_draw) < min_span:
                mid = (y_start + y_end) * 0.5
                y_start_draw = mid - (min_span * 0.5)
                y_end_draw = mid + (min_span * 0.5)

            # Adjust hairpin position to avoid overlapping with dynamic symbols
            y_start_draw, y_end_draw = self._adjust_hairpin_for_symbols(
                t_start, t_end, x_rpitch, y_start_draw, y_end_draw
            )

            # Decrescendo: open at top (start), closes toward bottom (end/tip)
            # Left arm: top-left → bottom-point
            du.add_line(
                x_mm - half_spread, y_start_draw, x_mm, y_end_draw,
                color=self.accent_color,
                width_mm=lw,
                line_cap='round',
                id=ev_id,
                tags=['decrescendo'],
            )
            # Right arm: top-right → bottom-point
            du.add_line(
                x_mm + half_spread, y_start_draw, x_mm, y_end_draw,
                color=self.accent_color,
                width_mm=lw,
                line_cap='round',
                id=ev_id,
                tags=['decrescendo'],
            )

            if is_dynamic_tool:
                start_handle_x = x_mm
                end_handle_x = x_mm
                # Handles are edge-anchored to the actual drawn hairpin endpoints.
                # This keeps handle position aligned even when text offsets move y_start_draw/y_end_draw.
                start_anchor_y = y_start_draw
                end_anchor_y = y_end_draw
                start_top = start_anchor_y
                start_bottom = start_anchor_y + (2.0 * handle_r)
                end_top = end_anchor_y - (2.0 * handle_r)
                end_bottom = end_anchor_y

                # Start handle (at the open end / top)
                self.register_hit_rect(
                    'hairpin', ev_id,
                    start_handle_x - handle_r, start_top,
                    start_handle_x + handle_r, start_bottom,
                    htype='decrescendo', handle='start',
                )
                du.add_rectangle(
                    start_handle_x - handle_r, start_top,
                    start_handle_x + handle_r, start_bottom,
                    stroke_color=None,
                    stroke_width_mm=0.0,
                    fill_color=(.5, 0.0, 0.0, 1.0),
                    id=ev_id,
                    tags=['hairpin_handle'],
                )
                # End handle (at the tip / bottom)
                self.register_hit_rect(
                    'hairpin', ev_id,
                    end_handle_x - handle_r, end_top,
                    end_handle_x + handle_r, end_bottom,
                    htype='decrescendo', handle='end',
                )
                du.add_rectangle(
                    end_handle_x - handle_r, end_top,
                    end_handle_x + handle_r, end_bottom,
                    stroke_color=None,
                    stroke_width_mm=0.0,
                    fill_color=(.5, 0.0, 0.0, 1.0),
                    id=ev_id,
                    tags=['hairpin_handle'],
                )

