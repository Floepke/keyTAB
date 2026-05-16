"""CountLineDrawer: renders count guide lines."""

from utils.CONSTANT import SHORTEST_DURATION
from utils.operator import Operator
from engraver.helpers import scaled_dash_pattern_with_default, time_to_y


class CountLineDrawer:
    """Draw count guide lines."""

    def __init__(self, context):
        self.context = context
        self.du = context.du
        self.layout_data = context.layout_data
        self.notation_color = self.layout_data.get('notation_color')
        self.scale = float(self.layout_data.get('scale', 1.0) or 1.0)
        self.op = Operator(SHORTEST_DURATION)

    def draw(self) -> None:
        """Draw all count lines for the current page and line."""
        score = self.context.score or {}
        events = dict(score.get('events', {}) or {})
        count_lines = list(events.get('count_line', []) or [])
        if not count_lines:
            return

        layout = self.layout_data.get('layout', {})
        if not bool(layout.get('countline_visible', True)):
            return

        dash_pattern = scaled_dash_pattern_with_default(
            layout.get('countline_dash_pattern', [0.0, 3.0]),
            [0.0, 3.0],
            self.scale,
        )
        countline_w = float(layout.get('countline_thickness_mm', 0.5) or 0.5) * self.scale
        semitone_mm = float(self.layout_data.get('semitone_mm', 2.0) or 2.0)
        key_to_x_map = self.layout_data.get('pitch_to_x_for_line', {})

        for line_index, line in enumerate(self.layout_data.get('page_lines', []) or []):
            key_to_x = key_to_x_map.get(line_index) if isinstance(key_to_x_map, dict) else None
            if not callable(key_to_x):
                continue
            base_x_c4 = float(key_to_x(40))
            t0 = float(line.get('time_start', 0.0) or 0.0)
            t1 = float(line.get('time_end', t0) or t0)
            for ev in count_lines:
                ev_t = float(ev.get('time', 0.0) or 0.0)
                if self.op.lt(ev_t, t0) or self.op.gt(ev_t, t1):
                    continue
                rp1 = int(ev.get('rpitch1', 0) or 0)
                rp2 = int(ev.get('rpitch2', 4) or 4)
                x1 = base_x_c4 + (float(rp1) * semitone_mm)
                x2 = base_x_c4 + (float(rp2) * semitone_mm)
                if x2 < x1:
                    x1, x2 = x2, x1
                y = time_to_y(line, ev_t)
                self.du.add_line(
                    x1,
                    y,
                    x2,
                    y,
                    color=self.notation_color,
                    width_mm=countline_w,
                    dash_pattern=dash_pattern,
                    id=int(ev.get('_id', 0) or 0),
                    tags=['count_line'],
                )
