from __future__ import annotations


def grace_note_drawer(du, pre_calc: dict) -> None:
    """Template drawer for grace notes (small noteheads + slash)."""
    line = dict(pre_calc.get('line', {}) or {})
    y0 = float(pre_calc.get('y0', 0.0) or 0.0)
    y1 = float(pre_calc.get('y1', 0.0) or 0.0)
    for stv in list(line.get('staves', []) or []):
        events = dict(stv.get('events_in_line', {}) or {})
        for _gr in list(events.get('grace_note', []) or []):
            x = float(stv.get('note_span_left_x_mm', 0.0) or 0.0)
            y = y0 + ((y1 - y0) * 0.2)
            du.add_oval(x - 0.4, y - 0.25, 0.8, 0.5, stroke_color=(0.0, 0.0, 0.0, 1.0), fill_color=(0.0, 0.0, 0.0, 1.0), tags=['grace-note'])
            du.add_line(x - 0.45, y + 0.35, x + 0.45, y - 0.35, color=(0.0, 0.0, 0.0, 1.0), width_mm=0.12, tags=['grace-note'])
