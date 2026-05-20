from __future__ import annotations
import math
from typing import TYPE_CHECKING, cast
from ui.widgets.draw_util import DrawUtil
from editor.editor_defaults import (
    SCALE,
    HAIRPIN_LINE_WIDTH_MM,
    HAIRPIN_WIDTH_MM,
    DYNAMIC_SYMBOL_FONT_SIZE_PT,
    DYNAMIC_SYMBOL_BACKGROUND_PADDING_MM,
)
from utils.CONSTANT import SHORTEST_DURATION
from utils.operator import Operator

if TYPE_CHECKING:
    from editor.editor import Editor


class CrescendoDrawerMixin:
    def _dynamic_symbol_bounds_lookup(self, du: DrawUtil, score, required_keys: set[tuple[float, int]] | None = None) -> dict[tuple[float, int], dict]:
        lookup: dict[tuple[float, int], dict] = {}
        score_events = self.current_events(score)
        dynamic_symbols = list(getattr(score_events, 'dynamic_symbol', []) or [])
        text_family = 'LelandText'
        text_size_pt = float(DYNAMIC_SYMBOL_FONT_SIZE_PT or 12.0)
        bg_pad = float(DYNAMIC_SYMBOL_BACKGROUND_PADDING_MM or 1.5)
        layout_rotation = float(getattr(getattr(score, 'layout', None), 'dynamic_rotation', 0.0) or 0.0)

        extents_cache: dict[tuple[str, str, float], tuple[float, float, float, float]] = {}

        def _extents(glyph: str) -> tuple[float, float, float, float]:
            key = (glyph, text_family, text_size_pt)
            if key in extents_cache:
                return extents_cache[key]
            try:
                val = du._get_text_extents_mm(glyph, text_family, text_size_pt, False, False)
            except Exception:
                val = (0.0, 0.0, max(1.0, (text_size_pt / 72.0) * 25.4), max(1.0, (text_size_pt / 72.0) * 25.4 * 0.8))
            extents_cache[key] = val
            return val

        for sym in dynamic_symbols:
            sym_time = float(getattr(sym, 'time', 0.0) or 0.0)
            sym_rpitch = int(getattr(sym, 'x_rpitch', 0) or 0)
            key = (round(sym_time, 6), int(sym_rpitch))
            if required_keys is not None and key not in required_keys:
                continue
            glyph = str(getattr(sym, 'symbol', '') or '')
            if not glyph:
                continue

            raw_rotation = getattr(sym, 'rotation', None)
            text_angle_deg = float(layout_rotation if raw_rotation is None else raw_rotation)
            y_mm = float(self.time_to_mm(sym_time))
            xb, yb, w, h = _extents(glyph)
            hw = float(w) * 0.5
            hh = float(h) * 0.5
            bg_half_w = hw + bg_pad
            bg_half_h = hh + bg_pad
            ang = math.radians(text_angle_deg)
            rot_half_h = abs(bg_half_w * math.sin(ang)) + abs(bg_half_h * math.cos(ang))
            x_mm = float(self.relative_c4pitch_to_x(sym_rpitch))
            lookup[key] = {
                'glyph': glyph,
                'x_mm': x_mm,
                'y_mm': y_mm,
                'y_min_mm': y_mm - rot_half_h,
                'y_max_mm': y_mm + rot_half_h,
            }

        return lookup

    def _get_dynamic_symbol_at_position(self, du: DrawUtil, t: float, x_rpitch: int, lookup: dict[tuple[float, int], dict] | None = None) -> dict | None:
        """
        Check if there's a dynamic symbol at the given time and x_rpitch.
        Returns dict with rotated Y bounds if found, None otherwise.
        """
        self = cast("Editor", self)
        score = self.current_score()
        if score is None:
            return None

        local_lookup = lookup if lookup is not None else self._dynamic_symbol_bounds_lookup(du, score)
        key = (round(float(t), 6), int(x_rpitch))
        if key in local_lookup:
            return local_lookup[key]

        # Fallback: thresholded time-match for safety when float rounding differs.
        op = Operator(float(SHORTEST_DURATION))
        for (tk, rp), info in local_lookup.items():
            if int(rp) == int(x_rpitch) and op.eq(float(tk), float(t)):
                return info
        return None
    
    def _adjust_hairpin_for_symbols(
        self,
        du: DrawUtil,
        t_start: float,
        t_end: float,
        x_rpitch: int,
        y_start_draw: float,
        y_end_draw: float,
        lookup: dict[tuple[float, int], dict] | None = None,
    ) -> tuple[float, float]:
        """Adjust hairpin start/end y positions to connect to dynamic symbol bounds (matching engraver)."""
        self = cast("Editor", self)

        symbol_at_start = self._get_dynamic_symbol_at_position(du, t_start, x_rpitch, lookup)
        if symbol_at_start is not None:
            y_start_draw = max(float(y_start_draw), float(symbol_at_start['y_max_mm']))

        symbol_at_end = self._get_dynamic_symbol_at_position(du, t_end, x_rpitch, lookup)
        if symbol_at_end is not None:
            y_end_draw = min(float(y_end_draw), float(symbol_at_end['y_min_mm']))

        return y_start_draw, y_end_draw
    
    def draw_crescendo(self, du: DrawUtil) -> None:
        self = cast("Editor", self)
        if getattr(self, 'is_tiny_mode_ultra', None) and self.is_tiny_mode_ultra():
            return
        score = getattr(self, 'current_score', lambda: None)()
        if score is None:
            return
        score_events = self.current_events(score)

        events = list(getattr(score_events, 'crescendo', []) or [])
        if not events:
            return

        # Use hardcoded editor defaults for hairpin styling (not from file layout)
        lw = float(HAIRPIN_LINE_WIDTH_MM or 1.0) * float(SCALE or 1.0)
        spread = float(HAIRPIN_WIDTH_MM or 10.0) * float(SCALE or 1.0)

        top_mm = float(getattr(self, '_view_y_mm_offset', 0.0) or 0.0)
        vp_h_mm = float(getattr(self, '_viewport_h_mm', 0.0) or 0.0)
        bottom_mm = top_mm + vp_h_mm
        bleed_mm = max(2.0, float(getattr(score.app_state, 'zoom_mm_per_quarter', 25.0) or 25.0) * 0.5)

        visible_events = []
        required_symbol_keys: set[tuple[float, int]] = set()
        for ev in events:
            t_start = float(getattr(ev, 'time', 0.0) or 0.0)
            duration = float(getattr(ev, 'duration', 256.0) or 256.0)
            t_end = t_start + duration
            y_start = float(self.time_to_mm(t_start))
            y_end = float(self.time_to_mm(t_end))
            if y_end < (top_mm - bleed_mm) or y_start > (bottom_mm + bleed_mm):
                continue
            visible_events.append(ev)
            x_rpitch = int(getattr(ev, 'x_rpitch', 0) or 0)
            required_symbol_keys.add((round(float(t_start), 6), int(x_rpitch)))
            required_symbol_keys.add((round(float(t_end), 6), int(x_rpitch)))

        if not visible_events:
            return

        dynamic_lookup = self._dynamic_symbol_bounds_lookup(du, score, required_keys=required_symbol_keys)

        is_dynamic_tool = str(getattr(getattr(self, '_tool', None), 'TOOL_NAME', '')) == 'dynamic'
        # Match count-line handle geometry: side = 1.4 * max(2.0, semitone_dist)
        handle_r = max(2.0, float(self.semitone_dist or 2.5)) * 0.7

        page_w, _ = du.current_page_size_mm()

        def clamp_x(val: float) -> float:
            if page_w <= 0:
                return val
            return max(0.0, min(float(val), float(page_w)))

        for ev in visible_events:
            t_start = float(getattr(ev, 'time', 0.0) or 0.0)
            duration = float(getattr(ev, 'duration', 256.0) or 256.0)
            t_end = t_start + duration
            x_rpitch = int(getattr(ev, 'x_rpitch', 0) or 0)
            ev_id = int(getattr(ev, '_id', 0) or 0)

            y_start = float(self.time_to_mm(t_start))
            y_end = float(self.time_to_mm(t_end))

            x_mm = clamp_x(float(self.relative_c4pitch_to_x(x_rpitch)))
            half_spread = spread * 0.5

            y_start_draw = y_start
            y_end_draw = y_end
            min_span = max(0.8, float(self.semitone_dist or 0.5) * 0.6)
            if (y_end_draw - y_start_draw) < min_span:
                mid = (y_start + y_end) * 0.5
                y_start_draw = mid - (min_span * 0.5)
                y_end_draw = mid + (min_span * 0.5)

            # Adjust hairpin position to connect to dynamic symbol bounds
            y_start_draw, y_end_draw = self._adjust_hairpin_for_symbols(
                du, t_start, t_end, x_rpitch, y_start_draw, y_end_draw, dynamic_lookup
            )

            # Handles at the drawn hairpin endpoints (after symbol adjustment)
            y_start_handle = y_start_draw
            y_end_handle = y_end_draw

            # Crescendo: point at top (start), opens toward bottom (end)
            # Left arm: top-point → bottom-left
            du.add_line(
                x_mm, y_start_draw, x_mm - half_spread, y_end_draw,
                color=self.accent_color,
                width_mm=lw,
                line_cap='round',
                id=ev_id,
                tags=['hairpin'],
            )
            # Right arm: top-point → bottom-right
            du.add_line(
                x_mm, y_start_draw, x_mm + half_spread, y_end_draw,
                color=self.accent_color,
                width_mm=lw,
                line_cap='round',
                id=ev_id,
                tags=['hairpin'],
            )

            if is_dynamic_tool:
                start_handle_x = x_mm
                end_handle_x = x_mm
                # Handles stay at original endpoints and do not follow dynamic-symbol
                # connection offsets.
                start_anchor_y = y_start_handle
                end_anchor_y = y_end_handle
                start_top = start_anchor_y
                start_bottom = start_anchor_y + (2.0 * handle_r)
                end_top = end_anchor_y - (2.0 * handle_r)
                end_bottom = end_anchor_y

                # Start handle (at the tip / top)
                self.register_hit_rect(
                    'hairpin', ev_id,
                    start_handle_x - handle_r, start_top,
                    start_handle_x + handle_r, start_bottom,
                    htype='crescendo', handle='start',
                )
                du.add_rectangle(
                    start_handle_x - handle_r, start_top,
                    start_handle_x + handle_r, start_bottom,
                    stroke_color=None,
                    stroke_width_mm=0.0,
                    fill_color=(0.5, 0.0, 0.0, 0.75),
                    id=ev_id,
                    tags=['hairpin_handle'],
                )
                # End handle (at the open end / bottom)
                self.register_hit_rect(
                    'hairpin', ev_id,
                    end_handle_x - handle_r, end_top,
                    end_handle_x + handle_r, end_bottom,
                    htype='crescendo', handle='end',
                )
                du.add_rectangle(
                    end_handle_x - handle_r, end_top,
                    end_handle_x + handle_r, end_bottom,
                    stroke_color=None,
                    stroke_width_mm=0.0,
                    fill_color=(0.5, 0.0, 0.0, 0.75),
                    id=ev_id,
                    tags=['hairpin_handle'],
                )

