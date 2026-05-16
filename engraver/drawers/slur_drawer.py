"""SlurDrawer: renders slur curves."""

import math

from utils.CONSTANT import SHORTEST_DURATION, SLUR_SEGMENT_COUNT
from utils.operator import Operator
from engraver.helpers import time_to_y


class SlurDrawer:
    """Draw slur curves."""
    
    def __init__(self, context):
        self.context = context
        self.du = context.du
        self.layout_data = context.layout_data
        self.notation_color = self.layout_data.get('notation_color')
        self.scale = float(self.layout_data.get('scale', 1.0) or 1.0)
        self.op = Operator(SHORTEST_DURATION)
    
    def draw(self) -> None:
        """Draw all slurs for the current page and line."""
        score = self.context.score or {}
        events = dict(score.get('events', {}) or {})
        slurs = list(events.get('slur', []) or [])
        if not slurs:
            return
        layout = self.layout_data.get('layout', {})
        if not bool(layout.get('slur_visible', True)):
            return

        side_w = float(layout.get('slur_width_sides_mm', 0.1) or 0.1) * self.scale
        mid_w = float(layout.get('slur_width_middle_mm', 1.5) or 1.5) * self.scale
        n_seg = max(2, int(SLUR_SEGMENT_COUNT))
        semitone_mm = float(self.layout_data.get('semitone_mm', 2.0) or 2.0)
        key_to_x_map = self.layout_data.get('pitch_to_x_for_line', {})

        def _tri_interp(t: float) -> float:
            return max(0.0, 1.0 - abs(2.0 * t - 1.0))

        def _width_at(t: float) -> float:
            return side_w + (mid_w - side_w) * _tri_interp(t)

        def _bezier_point(t: float, p0, p1, p2, p3):
            u = 1.0 - t
            x = (u * u * u * p0[0]) + (3.0 * u * u * t * p1[0]) + (3.0 * u * t * t * p2[0]) + (t * t * t * p3[0])
            y = (u * u * u * p0[1]) + (3.0 * u * u * t * p1[1]) + (3.0 * u * t * t * p2[1]) + (t * t * t * p3[1])
            return (x, y)

        for line_index, line in enumerate(self.layout_data.get('page_lines', []) or []):
            key_to_x = key_to_x_map.get(line_index) if isinstance(key_to_x_map, dict) else None
            if not callable(key_to_x):
                continue
            base_x_c4 = float(key_to_x(40))
            lt0 = float(line.get('time_start', 0.0) or 0.0)
            lt1 = float(line.get('time_end', lt0) or lt0)

            def _rpitch_to_x(rp: float) -> float:
                return base_x_c4 + (float(rp) * semitone_mm)

            for sl in slurs:
                y1_t = float(sl.get('y1_time', 0.0) or 0.0)
                y4_t = float(sl.get('y4_time', 0.0) or 0.0)
                if self.op.lt(y4_t, lt0) or self.op.gt(y1_t, lt1):
                    continue
                p0 = (_rpitch_to_x(float(sl.get('x1_rpitch', 0) or 0)), float(time_to_y(line, y1_t)))
                p1 = (_rpitch_to_x(float(sl.get('x2_rpitch', 0) or 0)), float(time_to_y(line, float(sl.get('y2_time', 0.0) or 0.0))))
                p2 = (_rpitch_to_x(float(sl.get('x3_rpitch', 0) or 0)), float(time_to_y(line, float(sl.get('y3_time', 0.0) or 0.0))))
                p3 = (_rpitch_to_x(float(sl.get('x4_rpitch', 0) or 0)), float(time_to_y(line, y4_t)))

                top = []
                bottom = []
                for i in range(n_seg + 1):
                    t = float(i) / float(n_seg)
                    x, y = _bezier_point(t, p0, p1, p2, p3)
                    if i < n_seg:
                        x2, y2 = _bezier_point(min(1.0, t + (1.0 / n_seg)), p0, p1, p2, p3)
                    else:
                        x2, y2 = x, y
                    dx = x2 - x
                    dy = y2 - y
                    ln = math.hypot(dx, dy)
                    if ln <= 1e-9:
                        nx, ny = 0.0, 1.0
                    else:
                        nx, ny = -dy / ln, dx / ln
                    hw = _width_at(t) * 0.5
                    top.append((x + (nx * hw), y + (ny * hw)))
                    bottom.append((x - (nx * hw), y - (ny * hw)))

                poly = top + list(reversed(bottom))
                self.du.add_polygon(
                    poly,
                    stroke_color=None,
                    fill_color=self.notation_color,
                    id=int(sl.get('_id', sl.get('id', 0)) or 0),
                    tags=['slur'],
                )
