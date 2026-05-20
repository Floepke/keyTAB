from __future__ import annotations


def time_signature_drawer(du, pre_calc: dict) -> None:
    line = dict(pre_calc.get('line', {}) or {})
    y0 = float(pre_calc.get('y0', 0.0) or 0.0)
    y1 = float(pre_calc.get('y1', 0.0) or 0.0)
    cy = (y0 + y1) * 0.5
    for stv in list(line.get('staves', []) or []):
        events = dict(stv.get('events_in_line', {}) or {})
        for ev in list(events.get('time_signature', []) or []):
            if isinstance(ev, dict):
                numer = ev.get('numerator', 4)
                denom = ev.get('denominator', 4)
            else:
                numer = getattr(ev, 'numerator', 4)
                denom = getattr(ev, 'denominator', 4)
            x = float(stv.get('stave_left_x_mm', 0.0) or 0.0)
            du.add_text(x, cy - 0.8, str(numer), family='Edwin', size_pt=9.0, anchor='center', tags=['time-signature'])
            du.add_text(x, cy + 0.8, str(denom), family='Edwin', size_pt=9.0, anchor='center', tags=['time-signature'])
