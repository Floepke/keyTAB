from __future__ import annotations
from typing import TYPE_CHECKING, cast
from settings_manager import get_preferences
from ui.widgets.draw_util import DrawUtil

if TYPE_CHECKING:
    from editor.editor import Editor


class TempoDrawerMixin:
    def draw_tempo(self, du: DrawUtil) -> None:
        self = cast("Editor", self)
        tool_name = getattr(getattr(self, "_tool", None), "TOOL_NAME", "")
        score = self.current_score()
        if score is None:
            return

        # Layout anchors
        margin = float(self.margin or 0.0)
        # Use the document page width (pitch axis) regardless of viewport rotation.
        # du.current_page_size_mm() returns the OUTPUT size, which swaps width/height
        # in horizontal mode — so always read from the score layout instead.
        layout_obj = getattr(score, 'layout', None)
        page_w_mm = float(getattr(layout_obj, 'page_width_mm', 210.0) or 210.0)

        # Iterate tempo events
        events = list(getattr(score.events, 'tempo', []) or [])
        if not events:
            return

        notation_color = tuple(getattr(self, 'notation_color', (0.0, 0.0, 0.0, 1.0)))
        paper_color = tuple(getattr(self, 'paper_color', (1.0, 1.0, 1.0, 1.0)))

        # Font setup: mirror engraver tempo typography.
        try:
            from fonts import register_font_from_bytes
            font_family = register_font_from_bytes('Edwin') or 'Edwin'
        except Exception:
            font_family = 'Edwin'

        # read editor orientation
        preferences = get_preferences()
        if preferences.get("editor_orientation", 'horizontal') == 'horizontal':
            editor_orientation = 'horizontal'
        else:
            editor_orientation = 'vertical'
        
        # metrics and styling
        scale = float(getattr(layout_obj, 'scale', 1.0) or 1.0)
        font_size_pt = 32.0 * scale
        font_italic = False
        font_bold = True
        bracket_dash = [0.5, 1]
        bracket_stroke = 0.25
        right_outer_stave_x = float(page_w_mm) - margin - (float(self.semitone_dist or 0.0) * 2.0)

        for tp in events:
            t0 = float(getattr(tp, 'time', 0.0) or 0.0)
            du_ticks = float(getattr(tp, 'duration', 0.0) or 0.0)
            tempo_val = int(getattr(tp, 'tempo', 60) or 60)
            is_invisible = bool(getattr(tp, 'invisible', False))
            draw_color = notation_color
            if is_invisible:
                # Keep hidden tempo markers editable in editor by drawing them muted.
                nr, ng, nb, _na = notation_color
                pr, pg, pb, _pa = paper_color
                draw_color = (
                    (float(nr) * 0.35) + (float(pr) * 0.65),
                    (float(ng) * 0.35) + (float(pg) * 0.65),
                    (float(nb) * 0.35) + (float(pb) * 0.65),
                    1.0,
                )
            if du_ticks <= 0.0:
                continue
            # Positions in mm
            y0 = float(self.time_to_mm(t0))
            y1 = float(self.time_to_mm(t0 + du_ticks))
            if y1 < y0:
                y0, y1 = y1, y0
            text = str(tempo_val)
            y_center = (y0 + y1) * 0.5

            x_right = float(page_w_mm) - margin * 0.5
            x_text_right = x_right - (0.6 * scale)
            _, _, text_w, _ = du._get_text_extents_mm(
                text,
                font_family,
                font_size_pt,
                font_italic,
                font_bold,
            )
            text_w = max(1.0, float(text_w))
            text_left = float(x_text_right - (2.0 * scale) - text_w)

            # Open-left dashed bracket (top/right/bottom) like engraver.
            du.add_line(
                right_outer_stave_x,
                y0,
                x_right,
                y0,
                color=draw_color,
                width_mm=bracket_stroke,
                id=0,
                tags=["tempo_bg"],
                dash_pattern=bracket_dash,
            )
            du.add_line(
                x_right,
                y0,
                x_right,
                y1,
                color=draw_color,
                width_mm=bracket_stroke,
                id=0,
                tags=["tempo_bg"],
                dash_pattern=bracket_dash,
            )
            du.add_line(
                text_left,
                y1,
                x_right,
                y1,
                color=draw_color,
                width_mm=bracket_stroke,
                id=0,
                tags=["tempo_bg"],
                dash_pattern=bracket_dash,
            )
            hit_x_left = min(float(right_outer_stave_x), float(text_left))
            self.register_hit_rect('tempo', int(getattr(tp, '_id', 0) or 0), hit_x_left, min(y0, y1), x_right, max(y0, y1))

            # draw tempo text
            du.add_text(
                x_text_right - 2 * scale if editor_orientation == 'vertical' else x_text_right - 4.0 * scale,
                y_center,
                text,
                family=font_family,
                size_pt=font_size_pt,
                italic=font_italic,
                bold=font_bold,
                color=draw_color,
                anchor='e' if editor_orientation == 'vertical' else 'n',
                id=0,
                tags=["tempo_text"],
                hit_rect_mm=None,
                angle_deg=0.0 if editor_orientation == 'vertical' else 90.0,
            )
