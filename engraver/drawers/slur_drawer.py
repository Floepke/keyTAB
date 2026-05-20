from __future__ import annotations


def slur_drawer(du, pre_calc: dict) -> None:
    line = dict(pre_calc.get('line', {}) or pre_calc.get('system', {}) or {})
    layout = dict(pre_calc.get('layout', {}) or {})
    notation_color = tuple(pre_calc.get('notation_color', (0.0, 0.0, 0.0, 1.0)) or (0.0, 0.0, 0.0, 1.0))
    y0 = float(pre_calc.get('y0', 0.0) or 0.0)
    y1 = float(pre_calc.get('y1', 0.0) or 0.0)
    cy = (y0 + y1) * 0.5
    for stv in list(line.get('staves', []) or []):
        composite_scale = float(stv.get('composite_scale', float(layout.get('scale', 1.0) or 1.0)) or 1.0)
        events = dict(stv.get('events_in_line', {}) or {})
        for _sl in list(events.get('slur', []) or []):
            x0 = float(stv.get('note_span_left_x_mm', 0.0) or 0.0)
            x1 = float(stv.get('note_span_right_x_mm', x0) or x0)
            c = (x0 + x1) * 0.5
            du.add_polyline(
                [(x0, cy), (c, cy - 1.0), (x1, cy)],
                stroke_color=notation_color,
                stroke_width_mm=max(0.05, 0.16 * composite_scale),
                tags=['slur'],
            )
