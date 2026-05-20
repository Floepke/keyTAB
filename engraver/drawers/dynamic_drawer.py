from __future__ import annotations


def _event_get(item, key: str, default=None):
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def dynamic_drawer(du, pre_calc: dict) -> None:
    """Draw dynamic symbols and hairpins (crescendo/decrescendo)."""
    line = dict(pre_calc.get('line', {}) or pre_calc.get('system', {}) or {})
    layout = dict(pre_calc.get('layout', {}) or {})
    notation_color = tuple(pre_calc.get('notation_color', (0.0, 0.0, 0.0, 1.0)) or (0.0, 0.0, 0.0, 1.0))
    y0 = float(pre_calc.get('y0', 0.0) or 0.0)
    y1 = float(pre_calc.get('y1', 0.0) or 0.0)
    base_y = y1 - max(1.2, (y1 - y0) * 0.08)
    t0 = float(line.get('time_start', 0.0) or 0.0)
    t1 = float(line.get('time_end', t0 + 1.0) or (t0 + 1.0))
    dt = max(1e-6, t1 - t0)

    for stv in list(line.get('staves', []) or []):
        composite_scale = float(stv.get('composite_scale', float(layout.get('scale', 1.0) or 1.0)) or 1.0)
        hairpin_w = max(0.05, 0.15 * composite_scale)
        events = dict(stv.get('events_in_line', {}) or {})
        xl = float(stv.get('note_span_left_x_mm', 0.0) or 0.0)
        xr = float(stv.get('note_span_right_x_mm', xl) or xl)

        for dyn in list(events.get('dynamic', []) or []):
            sym = str(_event_get(dyn, 'symbol', _event_get(dyn, 'text', 'mf')) or 'mf')
            t = float(_event_get(dyn, 'time', t0) or t0)
            x = xl + ((max(0.0, min(dt, t - t0)) / dt) * max(0.0, xr - xl))
            du.add_text(x, base_y, sym, family='Edwin', size_pt=11.0, color=notation_color, anchor='center', tags=['dynamic_symbol_text'])

        for cresc in list(events.get('crescendo', []) or []):
            t = float(_event_get(cresc, 'time', t0) or t0)
            d = float(_event_get(cresc, 'duration', dt * 0.25) or (dt * 0.25))
            x1 = xl + ((max(0.0, min(dt, t - t0)) / dt) * max(0.0, xr - xl))
            x2 = xl + ((max(0.0, min(dt, (t + d) - t0)) / dt) * max(0.0, xr - xl))
            du.add_line(x1, base_y, x2, base_y - 0.6, color=notation_color, width_mm=hairpin_w, tags=['hairpin'])
            du.add_line(x1, base_y, x2, base_y + 0.6, color=notation_color, width_mm=hairpin_w, tags=['hairpin'])

        for decresc in list(events.get('decrescendo', []) or []):
            t = float(_event_get(decresc, 'time', t0) or t0)
            d = float(_event_get(decresc, 'duration', dt * 0.25) or (dt * 0.25))
            x1 = xl + ((max(0.0, min(dt, t - t0)) / dt) * max(0.0, xr - xl))
            x2 = xl + ((max(0.0, min(dt, (t + d) - t0)) / dt) * max(0.0, xr - xl))
            du.add_line(x1, base_y - 0.6, x2, base_y, color=notation_color, width_mm=hairpin_w, tags=['hairpin'])
            du.add_line(x1, base_y + 0.6, x2, base_y, color=notation_color, width_mm=hairpin_w, tags=['hairpin'])
