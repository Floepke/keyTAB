from __future__ import annotations


def arpeggio_drawer(du, pre_calc: dict) -> None:
    line = dict(pre_calc.get('line', {}) or pre_calc.get('system', {}) or {})
    layout = dict(pre_calc.get('layout', {}) or {})
    notation_color = tuple(pre_calc.get('notation_color', (0.0, 0.0, 0.0, 1.0)) or (0.0, 0.0, 0.0, 1.0))
    y0 = float(pre_calc.get('y0', 0.0) or 0.0)
    y1 = float(pre_calc.get('y1', 0.0) or 0.0)
    for stv in list(line.get('staves', []) or []):
        composite_scale = float(stv.get('composite_scale', float(layout.get('scale', 1.0) or 1.0)) or 1.0)
        events = dict(stv.get('events_in_line', {}) or {})
        for arp in list(events.get('arpeggio', []) or []):
            x = float(stv.get('stave_left_x_mm', 0.0) or 0.0)
            du.add_line(
                x,
                y0,
                x,
                y1,
                color=(notation_color[0], notation_color[1], notation_color[2], 0.5),
                width_mm=max(0.05, 0.2 * composite_scale),
                dash_pattern=[0.5, 0.5],
                tags=['chord_connect'],
            )
