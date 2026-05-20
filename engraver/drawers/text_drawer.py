from __future__ import annotations


def text_drawer(du, pre_calc: dict) -> None:
    line = dict(pre_calc.get('line', {}) or pre_calc.get('system', {}) or {})
    notation_color = tuple(pre_calc.get('notation_color', (0.0, 0.0, 0.0, 1.0)) or (0.0, 0.0, 0.0, 1.0))
    y0 = float(pre_calc.get('y0', 0.0) or 0.0)
    for stv in list(line.get('staves', []) or []):
        events = dict(stv.get('events_in_line', {}) or {})
        for ev in list(events.get('text', []) or []):
            txt = str((ev.get('text', None) if isinstance(ev, dict) else None) or '')
            if not txt:
                continue
            x = float(stv.get('note_span_left_x_mm', 0.0) or 0.0)
            du.add_text(x, y0 - 2.5, txt, family='Edwin', size_pt=9.0, color=notation_color, anchor='w', tags=['text'])
