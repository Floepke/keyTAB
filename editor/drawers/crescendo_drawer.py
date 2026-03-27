from __future__ import annotations
from typing import TYPE_CHECKING, cast
from ui.widgets.draw_util import DrawUtil

if TYPE_CHECKING:
    from editor.editor import Editor


class CrescendoDrawerMixin:
    def draw_crescendo(self, du: DrawUtil) -> None:
        self = cast("Editor", self)
        if getattr(self, 'is_tiny_mode', None) and self.is_tiny_mode():
            return
        score = getattr(self, 'current_score', lambda: None)()
        if score is None:
            return

        layout = getattr(score, 'layout', None)
        if layout is not None and not bool(getattr(layout, 'hairpin_visible', True)):
            return

        events = list(getattr(score.events, 'crescendo', []) or [])
        if not events:
            return

        style_scale = float(getattr(layout, 'scale', 1.0) or 1.0) if layout is not None else 1.0
        lw = float(getattr(layout, 'hairpin_line_width_mm', 0.5) or 0.5) * style_scale
        spread = float(getattr(layout, 'hairpin_spread_mm', 5.0) or 5.0) * style_scale
        text_size_pt = float(getattr(layout, 'hairpin_text_size_pt', 12.0) or 12.0)
        text_gap = float(getattr(layout, 'hairpin_text_gap_mm', 1.2) or 1.2) * style_scale
        dynamic_bg_pad = float(
            getattr(
                layout,
                'dynamic_symbol_background_padding_mm',
                getattr(
                    layout,
                    'dynamic_symbol_background_padding',
                    getattr(layout, 'dynamic_background_padding', getattr(layout, 'text_background_padding_mm', 0.5)),
                ),
            ) or 0.0
        ) * style_scale
        paper_color = getattr(self, 'paper_color', (1.0, 1.0, 1.0, 1.0))
        text_color = self.notation_color
        text_family = 'LelandText'

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

            # Text-aware spacing for professional spanner engraving.
            start_text = str(getattr(ev, 'start_text', '') or '')
            end_text = str(getattr(ev, 'end_text', '') or '')

            # Connected spanners share same x and touching times.
            all_hairpins = list(getattr(score.events, 'crescendo', []) or []) + list(getattr(score.events, 'decrescendo', []) or [])
            eps = 1e-6
            end_join_peers = [
                hp for hp in all_hairpins
                if hp is not ev
                and int(getattr(hp, 'x_rpitch', 0) or 0) == x_rpitch
                and abs(float(getattr(hp, 'time', 0.0) or 0.0) - t_end) <= eps
            ]
            start_join_peers = [
                hp for hp in all_hairpins
                if hp is not ev
                and int(getattr(hp, 'x_rpitch', 0) or 0) == x_rpitch
                and abs((float(getattr(hp, 'time', 0.0) or 0.0) + float(getattr(hp, 'duration', 0.0) or 0.0)) - t_start) <= eps
            ]

            def _text_w_mm(txt: str) -> float:
                if not txt:
                    return 0.0
                try:
                    _xb, _yb, w, _h = du._get_text_extents_mm(txt, text_family, text_size_pt, False, False)
                    return float(max(0.0, w))
                except Exception:
                    return max(1.0, (text_size_pt / 72.0) * 25.4)

            def _text_h_mm(txt: str) -> float:
                if not txt:
                    return 0.0
                try:
                    _xb, _yb, _w, h = du._get_text_extents_mm(txt, text_family, text_size_pt, False, False)
                    return float(max(0.0, h))
                except Exception:
                    return max(1.0, (text_size_pt / 72.0) * 25.4 * 0.8)

            def _draw_text_centered_at(xc_mm: float, yc_mm: float, txt: str, ev_tag_id: int, ev_tags: list[str]) -> None:
                if not txt:
                    return
                try:
                    xb, yb, w, h = du._get_text_extents_mm(txt, text_family, text_size_pt, False, False)
                except Exception:
                    xb, yb, w, h = 0.0, 0.0, _text_w_mm(txt), _text_h_mm(txt)
                bx = float(xc_mm) - (float(xb) + (float(w) * 0.5))
                by = float(yc_mm) - (float(yb) + (float(h) * 0.5))

                rx = bx + float(xb)
                ry = by + float(yb)
                du.add_rectangle(
                    rx - dynamic_bg_pad,
                    ry - dynamic_bg_pad,
                    rx + float(w) + dynamic_bg_pad,
                    ry + float(h) + dynamic_bg_pad,
                    corner_radius=max(0.0, dynamic_bg_pad),
                    stroke_color=None,
                    fill_color=paper_color,
                    id=ev_tag_id,
                    tags=['dynamic_symbol_bg_top'],
                )

                du.add_text(
                    bx,
                    by,
                    txt,
                    family=text_family,
                    size_pt=text_size_pt,
                    italic=False,
                    bold=False,
                    color=text_color,
                    anchor=None,
                    id=ev_tag_id,
                    tags=['dynamic_symbol_text_top'],
                )

            start_h = _text_h_mm(start_text)
            end_h = _text_h_mm(end_text)
            start_w = _text_w_mm(start_text)
            end_w = _text_w_mm(end_text)
            peer_start_h = max([_text_h_mm(str(getattr(hp, 'start_text', '') or '')) for hp in end_join_peers] or [0.0])
            peer_end_h = max([_text_h_mm(str(getattr(hp, 'end_text', '') or '')) for hp in start_join_peers] or [0.0])
            peer_start_w = max([_text_w_mm(str(getattr(hp, 'start_text', '') or '')) for hp in end_join_peers] or [0.0])
            peer_end_w = max([_text_w_mm(str(getattr(hp, 'end_text', '') or '')) for hp in start_join_peers] or [0.0])

            start_pad = ((start_h * 0.5) + text_gap) if start_text else 0.0
            end_pad = ((end_h * 0.5) + text_gap) if end_text else 0.0
            if start_join_peers and (start_h > 0.0 or peer_end_h > 0.0):
                start_pad = max(start_pad, text_gap + (max(start_h, peer_end_h) * 0.5))
            if end_join_peers and (end_h > 0.0 or peer_start_h > 0.0):
                end_pad = max(end_pad, text_gap + (max(end_h, peer_start_h) * 0.5))

            y_start_draw = y_start + start_pad
            y_end_draw = y_end - end_pad
            min_span = max(0.8, float(self.semitone_dist or 0.5) * 0.6)
            if (y_end_draw - y_start_draw) < min_span:
                mid = (y_start + y_end) * 0.5
                y_start_draw = mid - (min_span * 0.5)
                y_end_draw = mid + (min_span * 0.5)

            # Crescendo: point at top (start), opens toward bottom (end)
            # Left arm: top-point → bottom-left
            du.add_line(
                x_mm, y_start_draw, x_mm - half_spread, y_end_draw,
                color=self.accent_color,
                width_mm=lw,
                line_cap='round',
                id=ev_id,
                tags=['crescendo'],
            )
            # Right arm: top-point → bottom-right
            du.add_line(
                x_mm, y_start_draw, x_mm + half_spread, y_end_draw,
                color=self.accent_color,
                width_mm=lw,
                line_cap='round',
                id=ev_id,
                tags=['crescendo'],
            )

            if start_text:
                _draw_text_centered_at(
                    x_mm,
                    y_start,
                    start_text,
                    ev_id,
                    ['crescendo_text'],
                )
            if end_text:
                _draw_text_centered_at(
                    x_mm,
                    y_end,
                    end_text,
                    ev_id,
                    ['crescendo_text'],
                )

            if is_dynamic_tool:
                start_handle_x = x_mm
                end_handle_x = x_mm
                start_handle_y = y_start_draw if start_join_peers else y_start
                end_handle_y = y_end_draw if end_join_peers else y_end

                # If this hairpin has symbols, force handles around those symbols
                # regardless of connected joins.
                handle_gap = max(0.2, text_gap * 0.5)
                if start_text:
                    start_handle_y = y_start - ((start_h * 0.5) + dynamic_bg_pad + handle_r + handle_gap)
                if end_text:
                    end_handle_y = y_end + ((end_h * 0.5) + dynamic_bg_pad + handle_r + handle_gap)

                # Start handle (at the tip / top)
                self.register_hairpin_hit_rect(
                    ev_id, 'crescendo', 'start',
                    start_handle_x - handle_r, start_handle_y - handle_r,
                    start_handle_x + handle_r, start_handle_y + handle_r,
                )
                du.add_rectangle(
                    start_handle_x - handle_r, start_handle_y - handle_r,
                    start_handle_x + handle_r, start_handle_y + handle_r,
                    stroke_color=(.5, 0.0, 0.0, 1.0),
                    stroke_width_mm=0.5,
                    fill_color=(.5, 0.0, 0.0, 1.0),
                    id=ev_id,
                    tags=['crescendo', 'crescendo_handle'],
                )
                # End handle (at the open end / bottom)
                self.register_hairpin_hit_rect(
                    ev_id, 'crescendo', 'end',
                    end_handle_x - handle_r, end_handle_y - handle_r,
                    end_handle_x + handle_r, end_handle_y + handle_r,
                )
                du.add_rectangle(
                    end_handle_x - handle_r, end_handle_y - handle_r,
                    end_handle_x + handle_r, end_handle_y + handle_r,
                    stroke_color=(.5, 0.0, 0.0, 1.0),
                    stroke_width_mm=0.5,
                    fill_color=(.5, 0.0, 0.0, 1.0),
                    id=ev_id,
                    tags=['crescendo', 'crescendo_handle'],
                )

