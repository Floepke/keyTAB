from __future__ import annotations
import bisect

from symbol_design.noteheads import (
    normalize_notehead_literal,
    resolve_notehead_spec,
    sheared_notehead_outline_points,
)
from ui.widgets.draw_util import DrawUtil
from utils.CONSTANT import BLACK_KEYS, QUARTER_NOTE_UNIT, SHORTEST_DURATION
from utils.operator import Operator


def _event_get(item, key: str, default=None):
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _event_get_float(item, key: str, default: float) -> float:
    val = _event_get(item, key, None)
    if val is None:
        return float(default)
    try:
        return float(val)
    except Exception:
        return float(default)


def _time_to_y(t: float, t0: float, t1: float, y0: float, y1: float) -> float:
    if t1 <= t0:
        return y0
    # Keep timeline anchored at t0, but allow negative times to extrapolate
    # slightly before the start (used for pre-roll grace notes).
    u = min(1.0, (float(t) - float(t0)) / (float(t1) - float(t0)))
    return float(y0 + (u * (y1 - y0)))


def _barline_positions(base_grid: list) -> list[float]:
    pos: list[float] = []
    cur = 0.0
    for bg in list(base_grid or []):
        numer = int(_event_get(bg, 'numerator', 4) or 4)
        denom = int(_event_get(bg, 'denominator', 4) or 4)
        measures = int(_event_get(bg, 'measure_amount', 1) or 1)
        measure_len = float(numer) * (4.0 / float(max(1, denom))) * float(QUARTER_NOTE_UNIT)
        for _ in range(int(max(0, measures))):
            pos.append(float(cur))
            cur += measure_len
    return pos


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
    op = Operator(SHORTEST_DURATION)
    barline_times = _barline_positions(list(pre_calc.get('base_grid', []) or []))

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
                nt = _event_get_float(n, 'time', t0)
                draw_items.append(
                    {
                        'id': int(_event_get(n, '_id', 0) or 0),
                        'x_mm': _pitch_to_x(pitch, stv),
                        'y_mm': _time_to_y(nt, t0, t1, y0, y1),
                        'time': nt,
                        'duration': float(_event_get(n, 'duration', 0.0) or 0.0),
                        'pitch': pitch,
                        'hand': str(_event_get(n, 'hand', 'l') or 'l'),
                        'is_up': bool(_event_get(n, 'is_up', True)),
                        'notehead': str(_event_get(n, 'notehead', 'normal') or 'normal'),
                        'beam': bool(_event_get(n, 'beam', False)),
                        'continuation_dot': bool(_event_get(n, 'continuation_dot', False)),
                        'stop_symbol': bool(_event_get(n, 'stop_symbol', False)),
                    }
                )

        for idx, n in enumerate(draw_items):
            if isinstance(n, dict):
                n.setdefault('_idx', int(idx))

        starts_all = sorted(
            [n for n in draw_items if isinstance(n, dict)],
            key=lambda n: _event_get_float(n, 'time', 0.0),
        )
        starts_all_times = [_event_get_float(n, 'time', 0.0) for n in starts_all]

        starts_by_hand: dict[str, list[dict]] = {'l': [], 'r': []}
        for n in starts_all:
            hk = 'l' if str(_event_get(n, 'hand', 'l') or 'l') == 'l' else 'r'
            starts_by_hand[hk].append(n)
        start_times_by_hand = {
            hk: [_event_get_float(n, 'time', 0.0) for n in arr]
            for hk, arr in starts_by_hand.items()
        }

        ends_by_hand: dict[str, list[dict]] = {'l': [], 'r': []}
        end_times_by_hand: dict[str, list[float]] = {'l': [], 'r': []}
        for hk in ('l', 'r'):
            pairs = []
            for n in starts_by_hand.get(hk, []):
                st = _event_get_float(n, 'time', 0.0)
                en = st + _event_get_float(n, 'duration', 0.0)
                pairs.append((float(en), n))
            pairs.sort(key=lambda p: p[0])
            end_times_by_hand[hk] = [float(p[0]) for p in pairs]
            ends_by_hand[hk] = [p[1] for p in pairs]

        def _starts_at_time(ticks: float) -> list[dict]:
            thr = float(op.threshold)
            lo = bisect.bisect_left(starts_all_times, float(ticks) - thr)
            hi = bisect.bisect_right(starts_all_times, float(ticks) + thr)
            out: list[dict] = []
            for i in range(lo, hi):
                n = starts_all[i]
                if op.eq(_event_get_float(n, 'time', 0.0), float(ticks)):
                    out.append(n)
            return out

        def _events_in_open_interval(hand: str, a: float, b: float, by_end: bool = False) -> list[dict]:
            if not op.lt(float(a), float(b)):
                return []
            hk = 'l' if str(hand or 'l') == 'l' else 'r'
            if by_end:
                times = end_times_by_hand.get(hk, [])
                arr = ends_by_hand.get(hk, [])
            else:
                times = start_times_by_hand.get(hk, [])
                arr = starts_by_hand.get(hk, [])
            thr = float(op.threshold)
            lo = bisect.bisect_left(times, float(a) - thr)
            hi = bisect.bisect_right(times, float(b) + thr)
            out: list[dict] = []
            for i in range(lo, hi):
                tv = float(times[i])
                if op.gt(tv, float(a)) and op.lt(tv, float(b)):
                    out.append(arr[i])
            return out

        def _note_key(n: dict) -> tuple[str, int]:
            raw_id = int(_event_get(n, 'id', _event_get(n, '_id', 0)) or 0)
            if raw_id != 0:
                return ('id', raw_id)
            return ('idx', int(_event_get(n, '_idx', -1) or -1))

        note_positions: list[tuple[float, float, dict]] = []
        for n in draw_items:
            pitch = int(_event_get(n, 'pitch', 41) or 41)
            hand = str(_event_get(n, 'hand', 'l') or 'l')
            is_up = bool(_event_get(n, 'is_up', True))
            notehead_literal = str(_event_get(n, 'notehead', 'normal') or 'normal')
            x = float(_event_get(n, 'x_mm', 0.0) or 0.0)
            nt = float(_event_get(n, 'time', t0) or t0)
            nt = _event_get_float(n, 'time', t0)
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

            if bool(layout.get('note_continuation_dot_visible', True)):
                n_key = _note_key(n)
                hand = str(_event_get(n, 'hand', 'l') or 'l')
                start = _event_get_float(n, 'time', t0)
                end = float(start + _event_get_float(n, 'duration', 0.0))
                dot_pitch = int(_event_get(n, 'pitch', 0) or 0)
                dot_times: list[float] = []

                for other in _events_in_open_interval(hand, start, end, by_end=False):
                    if _note_key(other) != n_key:
                        dot_times.append(_event_get_float(other, 'time', 0.0))
                for other in _events_in_open_interval(hand, start, end, by_end=True):
                    if _note_key(other) != n_key:
                        ot = _event_get_float(other, 'time', 0.0)
                        od = _event_get_float(other, 'duration', 0.0)
                        dot_times.append(float(ot + od))

                for bt in barline_times:
                    if op.gt(float(bt), start) and op.lt(float(bt), end):
                        dot_times.append(float(bt))

                if dot_times:
                    dot_size = max(0.05, float(layout.get('note_continuation_dot_size_mm', 2.5) or 2.5) * composite_scale)
                    notehead_center_offset_y = float(semitone_mm) * float(height_scale)
                    min_collision_gap = max(0.0, float(semitone_mm) * 2.0 - 1e-6)
                    for dt in sorted(set(dot_times)):
                        y_center = float(_time_to_y(float(dt), t0, t1, y0, y1)) + float(notehead_center_offset_y)
                        has_adjacent_start = False
                        for other in _starts_at_time(float(dt)):
                            if _note_key(other) == n_key:
                                continue
                            mp = int(_event_get(other, 'pitch', 0) or 0)
                            if abs(mp - dot_pitch) != 1:
                                continue
                            if mp in BLACK_KEYS and bool(_event_get(other, 'is_up', False)):
                                continue
                            other_x = float(_event_get(other, 'x_mm', x) or x)
                            if abs(other_x - float(x)) >= min_collision_gap:
                                continue
                            has_adjacent_start = True
                            break
                        if has_adjacent_start:
                            y_center += float(semitone_mm) * height_scale + semitone_mm

                        du.add_oval(
                            float(x) - (dot_size * 0.5),
                            float(y_center) - (dot_size * 0.5),
                            float(x) + (dot_size * 0.5),
                            float(y_center) + (dot_size * 0.5),
                            stroke_color=None,
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
