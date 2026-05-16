"""RepeatDrawer: renders repeat symbols and endings."""

from utils.CONSTANT import SHORTEST_DURATION
from utils.operator import Operator
from engraver.helpers import time_to_y


class RepeatDrawer:
    """Draw repeat symbols (start/end repeats, endings)."""
    
    def __init__(self, context):
        self.context = context
        self.du = context.du
        self.layout_data = context.layout_data
        self.notation_color = self.layout_data.get('notation_color')
        self.scale = float(self.layout_data.get('scale', 1.0) or 1.0)
        self.op = Operator(SHORTEST_DURATION)
    
    def draw(self) -> None:
        """Draw all repeat symbols for the current page and line."""
        score = self.context.score or {}
        events = dict(score.get('events', {}) or {})
        start_repeats = list(events.get('start_repeat', []) or [])
        end_repeats = list(events.get('end_repeat', []) or [])
        if not start_repeats and not end_repeats:
            return

        layout = self.layout_data.get('layout', {})
        semitone_mm = float(self.layout_data.get('semitone_mm', 2.0) or 2.0)
        symbol_width = max(3.0, semitone_mm * 4.0)
        symbol_thick_w = max(0.1, float(layout.get('grid_barline_thickness_mm', 0.25) or 0.25) * self.scale)
        symbol_dot_d = max(1.0, semitone_mm * self.scale)
        dot_offset_y = semitone_mm * 0.9
        key_to_x_map = self.layout_data.get('pitch_to_x_for_line', {})

        for line_index, line in enumerate(self.layout_data.get('page_lines', []) or []):
            key_to_x = key_to_x_map.get(line_index) if isinstance(key_to_x_map, dict) else None
            if not callable(key_to_x):
                continue
            right_pitch = int(line.get('natural_bound_right', line.get('bound_right', 88)) or 88)
            grid_right = float(key_to_x(right_pitch))
            lt0 = float(line.get('time_start', 0.0) or 0.0)
            lt1 = float(line.get('time_end', lt0) or lt0)

            def _draw_symbol(ev_t: float, ev_id: int, kind: str):
                if self.op.lt(ev_t, lt0) or self.op.gt(ev_t, lt1):
                    return
                y = float(time_to_y(line, ev_t))
                x_left = float(grid_right + semitone_mm * 1.5)
                x_right = float(x_left + symbol_width)
                self.du.add_line(
                    x_left,
                    y,
                    x_right,
                    y,
                    color=self.notation_color,
                    width_mm=symbol_thick_w,
                    id=ev_id,
                    tags=['barline_symbol', f'{kind}_repeat'],
                )
                d1 = x_left + (symbol_width * 0.25)
                d2 = x_left + (symbol_width * 0.75)
                ydot = y + dot_offset_y if kind == 'start' else y - dot_offset_y
                for dx in (d1, d2):
                    self.du.add_oval(
                        dx - (symbol_dot_d / 2.0),
                        ydot - (symbol_dot_d / 2.0),
                        dx + (symbol_dot_d / 2.0),
                        ydot + (symbol_dot_d / 2.0),
                        stroke_color=None,
                        fill_color=self.notation_color,
                        id=ev_id,
                        tags=['barline_symbol_dot', f'{kind}_repeat_dot'],
                    )

            if bool(layout.get('repeat_start_visible', True)):
                for ev in start_repeats:
                    _draw_symbol(float(ev.get('time', 0.0) or 0.0), int(ev.get('_id', ev.get('id', 0)) or 0), 'start')
            if bool(layout.get('repeat_end_visible', True)):
                for ev in end_repeats:
                    _draw_symbol(float(ev.get('time', 0.0) or 0.0), int(ev.get('_id', ev.get('id', 0)) or 0), 'end')
