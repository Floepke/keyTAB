from __future__ import annotations


def arpeggio_drawer(du, pre_calc: dict) -> None:
    line = dict(pre_calc.get('line', {}) or {})
    y0 = float(pre_calc.get('y0', 0.0) or 0.0)
    y1 = float(pre_calc.get('y1', 0.0) or 0.0)
    for stv in list(line.get('staves', []) or []):
        events = dict(stv.get('events_in_line', {}) or {})
        for arp in list(events.get('arpeggio', []) or []):
            x = float(stv.get('stave_left_x_mm', 0.0) or 0.0)
            du.add_line(x, y0, x, y1, color=(0.0, 0.0, 0.0, 0.5), width_mm=0.2, dash_pattern=[0.5, 0.5], tags=['arpeggio'])
