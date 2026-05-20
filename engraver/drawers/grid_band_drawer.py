from __future__ import annotations


def grid_band_drawer(du, pre_calc: dict) -> None:
    line = dict(pre_calc.get('line', {}) or pre_calc.get('system', {}) or {})
    notation_color = tuple(pre_calc.get('notation_color', (0.0, 0.0, 0.0, 1.0)) or (0.0, 0.0, 0.0, 1.0))
    y0 = float(pre_calc.get('y0', 0.0) or 0.0)
    y1 = float(pre_calc.get('y1', 0.0) or 0.0)
    for stv in list(line.get('staves', []) or []):
        events = dict(stv.get('events_in_line', {}) or {})
        if not list(events.get('grid_band', []) or []):
            continue
        x0 = float(stv.get('note_span_left_x_mm', 0.0) or 0.0)
        x1 = float(stv.get('note_span_right_x_mm', x0) or x0)
        du.add_rectangle(
            x0,
            y0,
            x1,
            y1,
            stroke_color=None,
            fill_color=(notation_color[0], notation_color[1], notation_color[2], 0.03),
            tags=['grid_band'],
        )
