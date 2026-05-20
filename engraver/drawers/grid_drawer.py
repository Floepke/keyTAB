from __future__ import annotations


def grid_drawer(du, pre_calc: dict) -> None:
    line = dict(pre_calc.get('line', {}) or {})
    y0 = float(pre_calc.get('y0', 0.0) or 0.0)
    y1 = float(pre_calc.get('y1', 0.0) or 0.0)
    cy = (y0 + y1) * 0.5
    for stv in list(line.get('staves', []) or []):
        events = dict(stv.get('events_in_line', {}) or {})
        for _ev in list(events.get('grid', []) or []):
            x0 = float(stv.get('note_span_left_x_mm', 0.0) or 0.0)
            x1 = float(stv.get('note_span_right_x_mm', x0) or x0)
            du.add_line(x0, cy, x1, cy, color=(0.0, 0.0, 0.0, 0.35), width_mm=0.12, dash_pattern=[0.7, 0.7], tags=['grid'])
