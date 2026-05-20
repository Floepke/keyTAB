from __future__ import annotations

from symbol_design.noteheads import (
    normalize_notehead_literal,
    resolve_notehead_spec,
    sheared_notehead_outline_points,
)
from ui.widgets.draw_util import DrawUtil


def _event_get(item, key: str, default=None):
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _time_to_y(t: float, t0: float, t1: float, y0: float, y1: float) -> float:
    if t1 <= t0:
        return y0
    u = max(0.0, min(1.0, (float(t) - float(t0)) / (float(t1) - float(t0))))
    return float(y0 + (u * (y1 - y0)))


def _pitch_to_x(pitch: int, stv: dict) -> float:
    offsets = dict(stv.get('key_offsets', {}) or {})
    span_low = int(stv.get('note_span_low_key', 1) or 1)
    span_left = float(stv.get('stave_content_span_left_mm', 0.0) or 0.0)
    p = int(max(1, min(88, int(pitch))))
    if not offsets:
        return span_left
    return float(span_left + (float(offsets.get(p, 0.0)) - float(offsets.get(span_low, 0.0))))


def _draw_notehead(
    du: DrawUtil,
    x: float,
    y: float,
    semitone_mm: float,
    hand: str,
    is_up: bool,
    literal: str,
    pitch: int,
    width_scale: float,
    height_scale: float,
    base_tilt: float,
    outline_width_mm: float,
    notation_color: tuple[float, float, float, float],
) -> None:
    normalized = normalize_notehead_literal(literal)
    spec = resolve_notehead_spec(
        {'notehead': normalized, 'pitch': int(pitch), 'hand': str(hand)},
        default_black_above=bool(is_up),
    )
    if str(spec.form) == 'x':
        r = max(0.2, float(semitone_mm) * 0.55)
        du.add_line(x - r, y - r, x + r, y + r, color=notation_color, width_mm=max(0.05, float(outline_width_mm)), tags=['notehead_black'])
        du.add_line(x - r, y + r, x + r, y - r, color=notation_color, width_mm=max(0.05, float(outline_width_mm)), tags=['notehead_black'])
        return

    points = sheared_notehead_outline_points(
        hand=hand,
        is_up=bool(spec.is_up),
        semitone_space_mm=max(0.2, float(semitone_mm)),
        width_scale=max(0.05, float(width_scale)),
        height_scale=max(0.1, float(height_scale)),
        base_tilt=max(-1.0, min(1.0, float(base_tilt))),
        sample_count=48,
    )
    abs_points = [(float(x + px), float(y + py)) for px, py in points]
    filled = bool(spec.filled)
    du.add_polygon(
        abs_points,
        stroke_color=notation_color,
        stroke_width_mm=max(0.05, float(outline_width_mm)),
        fill_color=notation_color if filled else None,
        tags=['notehead_black' if filled else 'notehead_white'],
    )


def note_drawer(du: DrawUtil, pre_calc: dict) -> None:
    """Draw notehead + stem + beam + continuation dot + stop symbol from precalculated payload."""
    system = dict(pre_calc.get('system', {}) or {})
    layout = dict(pre_calc.get('layout', {}) or {})
    y0 = float(pre_calc.get('y0', 0.0) or 0.0)
    y1 = float(pre_calc.get('y1', 0.0) or 0.0)
    t0 = float(system.get('time_start', 0.0) or 0.0)
    t1 = float(system.get('time_end', t0 + 1.0) or (t0 + 1.0))
    notation_color = tuple(pre_calc.get('notation_color', (0.0, 0.0, 0.0, 1.0)) or (0.0, 0.0, 0.0, 1.0))

    width_scale = float(layout.get('note_width_scaling', 1.0) or 1.0)
    height_scale = float(layout.get('notehead_height_scaling', 1.0) or 1.0)
    # geometry.py mirrors sign by hand; using positive base makes L=+tilt, R=-tilt.
    base_tilt = float(layout.get('notehead_tilt', 0.0) or 0.0)

    for stv in list(system.get('staves', []) or []):
        events = dict(stv.get('events_in_line', {}) or {})
        notes = list(events.get('note', []) or [])
        semitone_mm = float(stv.get('semitone_mm', 1.0) or 1.0)
        composite_scale = float(stv.get('composite_scale', 1.0) or 1.0)
        outline_width_mm = float(layout.get('note_stem_thickness_mm', 0.8) or 0.8) * composite_scale
        stem_width_mm = float(stv.get('note_stem_width_mm', outline_width_mm) or outline_width_mm)

        draw_items = list(stv.get('note_draw_items', []) or [])
        if not draw_items:
            # Fallback path for older pre-calc payloads.
            for n in notes:
                pitch = int(_event_get(n, 'pitch', 41) or 41)
                nt = float(_event_get(n, 'time', t0) or t0)
                draw_items.append(
                    {
                        'x_mm': _pitch_to_x(pitch, stv),
                        'y_mm': _time_to_y(nt, t0, t1, y0, y1),
                        'time': nt,
                        'pitch': pitch,
                        'hand': str(_event_get(n, 'hand', 'l') or 'l'),
                        'is_up': bool(_event_get(n, 'is_up', True)),
                        'notehead': str(_event_get(n, 'notehead', 'normal') or 'normal'),
                        'beam': bool(_event_get(n, 'beam', False)),
                        'continuation_dot': bool(_event_get(n, 'continuation_dot', False)),
                        'stop_symbol': bool(_event_get(n, 'stop_symbol', False)),
                    }
                )

        note_positions: list[tuple[float, float, dict]] = []
        for n in draw_items:
            pitch = int(_event_get(n, 'pitch', 41) or 41)
            hand = str(_event_get(n, 'hand', 'l') or 'l')
            is_up = bool(_event_get(n, 'is_up', True))
            notehead_literal = str(_event_get(n, 'notehead', 'normal') or 'normal')
            x = float(_event_get(n, 'x_mm', 0.0) or 0.0)
            nt = float(_event_get(n, 'time', t0) or t0)
            # Keep vertical placement tied to timeline mapping, which is stable
            # across payload schema changes.
            y = _time_to_y(nt, t0, t1, y0, y1)

            _draw_notehead(
                du,
                x,
                y,
                semitone_mm,
                hand,
                is_up,
                notehead_literal,
                pitch,
                width_scale,
                height_scale,
                base_tilt,
                outline_width_mm,
                notation_color,
            )

            if bool(_event_get(n, 'continuation_dot', False)):
                dot = max(0.25, semitone_mm * 0.32)
                du.add_oval(
                    x + (semitone_mm * 0.9) - (dot * 0.5),
                    y - (dot * 0.5),
                    dot,
                    dot,
                    stroke_color=(notation_color[0], notation_color[1], notation_color[2], 0.0),
                    fill_color=notation_color,
                    tags=['continuation_dot'],
                )

            if bool(_event_get(n, 'stop_symbol', False)):
                r = max(0.25, semitone_mm * 0.45)
                stop_w = max(0.05, 0.15 * composite_scale)
                du.add_line(x - r, y - r, x + r, y + r, color=notation_color, width_mm=stop_w, tags=['stop_sign'])
                du.add_line(x - r, y + r, x + r, y - r, color=notation_color, width_mm=stop_w, tags=['stop_sign'])

            note_positions.append((x, y, n))

        for seg in list(stv.get('stem_segments', []) or []):
            seg_time = float(_event_get(seg, 'time', t0) or t0)
            seg_y = _time_to_y(seg_time, t0, t1, y0, y1)
            du.add_line(
                float(seg.get('x1_mm', 0.0) or 0.0),
                float(seg_y),
                float(seg.get('x2_mm', 0.0) or 0.0),
                float(seg_y),
                color=notation_color,
                width_mm=max(0.05, float(stem_width_mm)),
                tags=['stem'],
            )

        # Lightweight beam template: connect adjacent notes marked with beam.
        stem_len = max(1.2, semitone_mm * 2.4)
        for i in range(len(note_positions) - 1):
            x1, y1n, n1 = note_positions[i]
            x2, y2n, n2 = note_positions[i + 1]
            if not bool(_event_get(n1, 'beam', False)) and not bool(_event_get(n2, 'beam', False)):
                continue
            beam_y1 = y1n - stem_len if bool(_event_get(n1, 'is_up', True)) else y1n + stem_len
            beam_y2 = y2n - stem_len if bool(_event_get(n2, 'is_up', True)) else y2n + stem_len
            du.add_line(
                x1 + (semitone_mm * 0.55),
                beam_y1,
                x2 + (semitone_mm * 0.55),
                beam_y2,
                color=notation_color,
                width_mm=max(0.2, semitone_mm * 0.35),
                tags=['beam'],
            )
