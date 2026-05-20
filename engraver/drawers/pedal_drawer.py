from __future__ import annotations


def pedal_drawer(du, pre_calc: dict) -> None:
    line = dict(pre_calc.get('line', {}) or pre_calc.get('system', {}) or {})
    notation_color = tuple(pre_calc.get('notation_color', (0.0, 0.0, 0.0, 1.0)) or (0.0, 0.0, 0.0, 1.0))
    y1 = float(pre_calc.get('y1', 0.0) or 0.0)
    for stv in list(line.get('staves', []) or []):
        events = dict(stv.get('events_in_line', {}) or {})
        for _ev in list(events.get('pedal', []) or []):
            x = float(stv.get('note_span_left_x_mm', 0.0) or 0.0)
            du.add_text(x, y1 + 1.5, 'Ped.', family='Edwin', size_pt=9.0, color=notation_color, anchor='w', tags=['pedal_symbol'])
