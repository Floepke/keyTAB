from __future__ import annotations

from typing import TYPE_CHECKING, cast

from symbol_design.pedal.down import draw_down_symbol
from symbol_design.pedal.up import draw_up_symbol
from ui.widgets.draw_util import DrawUtil

if TYPE_CHECKING:
    from editor.editor import Editor


class PedalDrawerMixin:
    _PEDAL_WIDTH_SEMITONES = 4.0
    _PEDAL_HEIGHT_SEMITONES = 3.0

    def draw_pedal(self, du: DrawUtil) -> None:
        self = cast("Editor", self)
        if getattr(self, 'is_tiny_mode_ultra', None) and self.is_tiny_mode_ultra():
            return
        score = self.current_score()
        if score is None:
            return
        events = self.current_events(score)
        if events is None:
            return

        layout = score.layout
        painted_thickness = max(0.05, float(getattr(layout, 'pedal_thickness_mm', 1.0) or 1.0))
        line_width_factor = 1.0
        thickness = painted_thickness / (2.0 * line_width_factor)
        semitone_space = max(0.05, float(self.semitone_dist or 0.0))
        width = semitone_space * self._PEDAL_WIDTH_SEMITONES
        height = semitone_space * self._PEDAL_HEIGHT_SEMITONES
        half_width = width * 0.5
        padding = painted_thickness * 0.25

        for pedal in getattr(events, 'pedal', []):
            symbol = str(getattr(pedal, 'symbol', '') or '')
            if symbol not in ('up', 'down'):
                continue
            x_mm = float(self.relative_c4pitch_to_x(int(getattr(pedal, 'rpitch', 0) or 0)))
            y_mm = float(self.time_to_mm(float(getattr(pedal, 'time', 0.0) or 0.0)))
            pedal_id = int(getattr(pedal, '_id', 0) or 0)

            if symbol == 'up':
                draw_up_symbol(
                    du,
                    x_mm=x_mm,
                    y_mm=y_mm,
                    width_mm=width,
                    height_mm=height,
                    thickness_mm=thickness,
                    color=self.notation_color,
                    paper_color=self.paper_color,
                    item_id=pedal_id,
                    tags=['pedal_symbol'],
                )
                y1, y2 = y_mm - height - padding, y_mm + padding
            else:
                draw_down_symbol(
                    du,
                    x_mm=x_mm,
                    y_mm=y_mm,
                    width_mm=width,
                    height_mm=height,
                    thickness_mm=thickness,
                    color=self.notation_color,
                    paper_color=self.paper_color,
                    item_id=pedal_id,
                    tags=['pedal_symbol'],
                )
                y1, y2 = y_mm - padding, y_mm + height + padding

            self.register_hit_rect(
                'pedal',
                pedal_id,
                x_mm - half_width - padding,
                y1,
                x_mm + half_width + padding,
                y2,
            )
