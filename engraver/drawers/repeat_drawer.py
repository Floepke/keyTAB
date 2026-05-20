from __future__ import annotations


def repeat_drawer(du, pre_calc: dict) -> None:
    line = dict(pre_calc.get('line', {}) or {})
    y0 = float(pre_calc.get('y0', 0.0) or 0.0)
    y1 = float(pre_calc.get('y1', 0.0) or 0.0)
    for stv in list(line.get('staves', []) or []):
        events = dict(stv.get('events_in_line', {}) or {})
        if list(events.get('start_repeat', []) or []):
            x = float(stv.get('note_span_left_x_mm', 0.0) or 0.0)
            du.add_line(x, y0, x, y1, color=(0.0, 0.0, 0.0, 1.0), width_mm=0.35, tags=['start-repeat'])
        if list(events.get('end_repeat', []) or []):
            x = float(stv.get('note_span_right_x_mm', 0.0) or 0.0)
            du.add_line(x, y0, x, y1, color=(0.0, 0.0, 0.0, 1.0), width_mm=0.35, tags=['end-repeat'])
