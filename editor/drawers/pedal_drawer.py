from __future__ import annotations
from typing import TYPE_CHECKING, cast
from ui.widgets.draw_util import DrawUtil
from symbol_design.pedal import draw_pedal_symbol
from editor.editor_defaults import SCALE, PEDAL_SYMBOL_THICKNESS_MM, PEDAL_BACKGROUND_PADDING_MM

if TYPE_CHECKING:
    from editor.editor import Editor


class PedalDrawerMixin:
    def draw_pedal(self, du: DrawUtil) -> None:
        self = cast("Editor", self)
        if getattr(self, 'is_tiny_mode_ultra', None) and self.is_tiny_mode_ultra():
            return
        score = getattr(self, 'current_score', lambda: None)()
        if score is None:
            return
        score_events = self.current_events(score)

        events = list(getattr(score_events, 'pedal', []) or [])
        if not events:
            return

        layout = getattr(score, 'layout', None)
        
        # Get top and bottom of current view
        top_mm = float(getattr(self, '_view_y_mm_offset', 0.0) or 0.0)
        vp_h_mm = float(getattr(self, '_viewport_h_mm', 0.0) or 0.0)
        bottom_mm = top_mm + vp_h_mm
        bleed_mm = max(2.0, float(getattr(score.app_state, 'zoom_mm_per_quarter', 25.0) or 25.0) * 0.5)

        page_w, _ = du.current_page_size_mm()

        def clamp_x(val: float) -> float:
            if page_w <= 0:
                return val
            return max(0.0, min(float(val), float(page_w)))

        def time_to_y_mm(time: float) -> float:
            return float(self.time_to_mm(time))

        def rpitch_to_x_mm(rpitch: int) -> float:
            return clamp_x(float(self.relative_c4pitch_to_x(rpitch)))

        # Get semitone_space_mm from editor
        semitone_space_mm = float(getattr(self, 'semitone_dist', 0.5) or 0.5)

        # Get notation color
        notation_color = getattr(self, 'notation_color', (0.0, 0.0, 0.0, 1.0))
        paper_color = getattr(self, 'paper_color', (1.0, 1.0, 1.0, 1.0))

        # Use hardcoded editor defaults for pedal styling (not from file layout)
        pedal_thickness_mm = max(0.05, float(PEDAL_SYMBOL_THICKNESS_MM or 1.0) * float(SCALE or 1.0))

        for ev in events:
            t = float(getattr(ev, 'time', 0.0) or 0.0)
            y_mm = time_to_y_mm(t)
            if y_mm < (top_mm - bleed_mm) or y_mm > (bottom_mm + bleed_mm):
                continue

            ev_id = int(getattr(ev, '_id', 0) or 0)
            rp = int(getattr(ev, 'rpitch', 0) or 0)
            x_mm = rpitch_to_x_mm(rp)
            symbol = str(getattr(ev, 'symbol', '') or '').strip().lower()
            is_invisible = bool(getattr(ev, 'invisible', False))

            pedal_color = notation_color
            if is_invisible:
                # Hidden pedal symbols remain editable in editor, shown as muted gray.
                nr, ng, nb, _na = notation_color
                pr, pg, pb, _pa = paper_color
                pedal_color = (
                    (float(nr) * 0.35) + (float(pr) * 0.65),
                    (float(ng) * 0.35) + (float(pg) * 0.65),
                    (float(nb) * 0.35) + (float(pb) * 0.65),
                    1.0,
                )

            try:
                draw_pedal_symbol(
                    du,
                    ev,
                    time_to_y_mm=time_to_y_mm,
                    rpitch_to_x_mm=rpitch_to_x_mm,
                    color=pedal_color,
                    background_color=paper_color,
                    width_mm=pedal_thickness_mm,
                    semitone_space_mm=semitone_space_mm,
                    layout=layout,
                    id=ev_id,
                    tags=['pedal_symbol'],
                )
            except Exception:
                pass

            # Register a hit rectangle so tools can add/delete pedal symbols by click.
            span = max(0.8, float(semitone_space_mm))
            # Use hardcoded editor default for background padding (not from file layout)
            background_pad = float(PEDAL_BACKGROUND_PADDING_MM or 1.0)
            background_pad = max(0.0, float(background_pad))
            if symbol in ('down_keytab', 'down_klavarskribo'):
                x1 = float(x_mm - (span * 2.0) - background_pad)
                y1r = float(y_mm)
                x2 = float(x_mm + (span * 2.0) + background_pad)
                y2r = float(y_mm + (span * 2.0) + background_pad)
            elif symbol in ('up_keytab', 'up_klavarskribo'):
                x1 = float(x_mm - (span * 2.0) - background_pad)
                y1r = float(y_mm - (span * 2.0) - background_pad)
                x2 = float(x_mm + (span * 2.0) + background_pad)
                y2r = float(y_mm + background_pad)
            elif symbol == 'heel':
                x1 = float(x_mm - (span * 0.25))
                y1r = float(y_mm)
                x2 = float(x_mm + (span * 0.25))
                y2r = float(y_mm + span)
            elif symbol == 'toe':
                x1 = float(x_mm)
                y1r = float(y_mm - (span * 0.25))
                x2 = float(x_mm + span)
                y2r = float(y_mm + (span * 0.25))
            else:
                continue

            if hasattr(self, 'register_hit_rect'):
                self.register_hit_rect('pedal', ev_id, x1, y1r, x2, y2r, symbol=symbol)

