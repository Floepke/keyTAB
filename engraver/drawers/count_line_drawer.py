from __future__ import annotations


def count_line_drawer(du, pre_calc: dict) -> None:
    line = dict(pre_calc.get('line', {}) or {})
    y0 = float(pre_calc.get('y0', 0.0) or 0.0)
    y1 = float(pre_calc.get('y1', 0.0) or 0.0)
    cy = (y0 + y1) * 0.5
    for stv in list(line.get('staves', []) or []):
        events = dict(stv.get('events_in_line', {}) or {})
        for _ev in list(events.get('count_line', []) or []):
            xl = float(stv.get('note_span_left_x_mm', 0.0) or 0.0)
            xr = float(stv.get('note_span_right_x_mm', xl) or xl)
            du.add_line(xl, cy, xr, cy, color=(0.0, 0.0, 0.0, 0.5), width_mm=0.15, dash_pattern=[1.0, 0.8], tags=['count-line'])
