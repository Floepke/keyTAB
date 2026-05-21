from __future__ import annotations

import bisect

from symbol_design.noteheads import Notehead
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
    layout: dict,
    x: float,
    y: float,
    semitone_mm: float,
    note_id: int,
    hand: str,
    is_up: bool,
    literal: str,
    pitch: int,
    outline_width_mm: float,
    notation_color: tuple[float, float, float, float],
    paper_color: tuple[float, float, float, float],
    notehead_cache: dict | None = None,
) -> None:
    note_payload = {
        'notehead': str(literal or 'auto'),
        'pitch': int(pitch),
        'hand': str(hand),
    }
    cache_key = (
        str(note_payload['notehead']),
        int(note_payload['pitch']),
        str(note_payload['hand']),
        bool(is_up),
        round(float(max(0.2, float(semitone_mm))), 6),
        round(float(max(0.05, float(outline_width_mm))), 6),
        int(id(layout)),
    )
    notehead = None
    if isinstance(notehead_cache, dict):
        notehead = notehead_cache.get(cache_key)
    if notehead is None:
        notehead = Notehead.from_note(
            x_mm=0.0,
            y_mm=0.0,
            note=note_payload,
            layout=layout,
            semitone_space_mm=float(max(0.2, float(semitone_mm))),
            notation_color=notation_color,
            paper_color=paper_color,
            default_black_above=bool(is_up),
            outline_width_mm_override=max(0.05, float(outline_width_mm)),
        )
        if isinstance(notehead_cache, dict):
            notehead_cache[cache_key] = notehead
    notehead.x_mm = float(x)
    notehead.y_mm = float(y)
    tag = 'notehead_black' if bool(getattr(notehead, 'filled', False)) else 'notehead_white'
    notehead.draw_notehead(du, item_id=int(note_id), tags=[tag], use_custom_color=False)


def note_drawer(du: DrawUtil, pre_calc: dict) -> None:
    system = dict(pre_calc.get('system', {}) or {})
    layout = dict(pre_calc.get('layout', {}) or {})
    y0 = float(pre_calc.get('y0', 0.0) or 0.0)
    y1 = float(pre_calc.get('y1', 0.0) or 0.0)
    t0 = float(system.get('time_start', 0.0) or 0.0)
    t1 = float(system.get('time_end', t0 + 1.0) or (t0 + 1.0))
    notation_color = tuple(pre_calc.get('notation_color', (0.0, 0.0, 0.0, 1.0)) or (0.0, 0.0, 0.0, 1.0))
    paper_color = tuple(pre_calc.get('paper_color', (1.0, 1.0, 1.0, 1.0)) or (1.0, 1.0, 1.0, 1.0))
    op = Operator(SHORTEST_DURATION)
    barline_times = _barline_positions(list(pre_calc.get('base_grid', []) or []))

    note_head_visible = bool(layout.get('note_head_visible', True))
    note_stem_visible = bool(layout.get('note_stem_visible', True))
    note_stop_visible = bool(layout.get('note_stop_visible', True))
    dot_visible = bool(layout.get('note_continuation_dot_visible', True))

    height_scale = float(layout.get('notehead_height_scaling', 1.0) or 1.0)

    notehead_cache: dict = {}

    for stv in list(system.get('staves', []) or []):
        events = dict(stv.get('events_in_line', {}) or {})
        notes = list(events.get('note', []) or [])
        semitone_mm = float(stv.get('semitone_mm', 1.0) or 1.0)
        composite_scale = float(stv.get('composite_scale', 1.0) or 1.0)
        outline_width_mm = float(layout.get('note_stem_thickness_mm', 0.8) or 0.8) * composite_scale
        stem_width_mm = float(stv.get('note_stem_width_mm', outline_width_mm) or outline_width_mm)

        draw_items = list(stv.get('note_draw_items', []) or [])
        if not draw_items:
            for n in notes:
                pitch = int(_event_get(n, 'pitch', 41) or 41)
                nt = _event_get_float(n, 'time', t0)
                draw_items.append(
                    {
                        'id': int(_event_get(n, '_id', 0) or 0),
                        'x_mm': _pitch_to_x(pitch, stv),
                        'time': nt,
                        'duration': float(_event_get(n, 'duration', 0.0) or 0.0),
                        'pitch': pitch,
                        'hand': str(_event_get(n, 'hand', 'l') or 'l'),
                        'is_up': bool(_event_get(n, 'is_up', True)),
                        'notehead': str(_event_get(n, 'notehead', 'normal') or 'normal'),
                    }
                )

        draw_items = [n for n in draw_items if isinstance(n, dict)]
        for idx, n in enumerate(draw_items):
            n.setdefault('_idx', int(idx))
            n_time = _event_get_float(n, 'time', t0)
            n_duration = _event_get_float(n, 'duration', 0.0)
            n_end = float(n_time + n_duration)
            n_id = int(_event_get(n, 'id', _event_get(n, '_id', 0)) or 0)
            n_idx = int(_event_get(n, '_idx', -1) or -1)
            n['_time'] = float(n_time)
            n['_duration'] = float(n_duration)
            n['_end'] = float(n_end)
            n['_pitch_i'] = int(_event_get(n, 'pitch', 41) or 41)
            n['_hand_norm'] = 'l' if str(_event_get(n, 'hand', 'l') or 'l') == 'l' else 'r'
            n['_is_up_b'] = bool(_event_get(n, 'is_up', True))
            n['_notehead_lit'] = str(_event_get(n, 'notehead', 'normal') or 'normal')
            n['_dot_b'] = bool(_event_get(n, 'continuation_dot', True))
            n['_stop_b'] = bool(_event_get(n, 'stop_symbol', False))
            n['_x'] = float(_event_get(n, 'x_mm', 0.0) or 0.0)
            n['_id_i'] = int(n_id)
            n['_nkey'] = ('id', n_id) if n_id != 0 else ('idx', n_idx)

        starts_all = sorted(draw_items, key=lambda n: float(n.get('_time', 0.0) or 0.0))
        starts_all_times = [float(n.get('_time', 0.0) or 0.0) for n in starts_all]
        starts_exact: dict[float, list[dict]] = {}
        starts_by_hand: dict[str, list[dict]] = {'l': [], 'r': []}
        for n in starts_all:
            tv = float(n.get('_time', 0.0) or 0.0)
            starts_exact.setdefault(tv, []).append(n)
            hk = str(n.get('_hand_norm', 'l') or 'l')
            starts_by_hand[hk].append(n)
        start_times_by_hand = {
            hk: [float(n.get('_time', 0.0) or 0.0) for n in arr]
            for hk, arr in starts_by_hand.items()
        }

        ends_by_hand: dict[str, list[dict]] = {'l': [], 'r': []}
        end_times_by_hand: dict[str, list[float]] = {'l': [], 'r': []}
        if dot_visible:
            for hk in ('l', 'r'):
                pairs: list[tuple[float, dict]] = []
                for n in starts_by_hand.get(hk, []):
                    pairs.append((float(n.get('_end', 0.0) or 0.0), n))
                pairs.sort(key=lambda p: p[0])
                end_times_by_hand[hk] = [float(p[0]) for p in pairs]
                ends_by_hand[hk] = [p[1] for p in pairs]

        def _starts_at_time(ticks: float) -> list[dict]:
            exact = starts_exact.get(float(ticks), None)
            if exact is not None:
                return exact
            thr = float(op.threshold)
            lo = bisect.bisect_left(starts_all_times, float(ticks) - thr)
            hi = bisect.bisect_right(starts_all_times, float(ticks) + thr)
            out: list[dict] = []
            for i in range(lo, hi):
                n = starts_all[i]
                if op.eq(float(n.get('_time', 0.0) or 0.0), float(ticks)):
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

        def _is_followed_by_rest_item(n: dict) -> bool:
            hk = str(n.get('_hand_norm', 'l') or 'l')
            end = float(n.get('_end', 0.0) or 0.0)
            times = start_times_by_hand.get(hk, [])
            arr = starts_by_hand.get(hk, [])
            if not times or not arr:
                return True
            thr = float(op.threshold)
            idx = bisect.bisect_left(times, float(end - thr))
            n_key = n.get('_nkey', ('idx', -1))
            min_delta = None
            for j in range(idx, len(arr)):
                m = arr[j]
                if m.get('_nkey', ('idx', -2)) == n_key:
                    continue
                delta = float(float(m.get('_time', 0.0) or 0.0) - end)
                if delta >= -thr:
                    min_delta = delta
                    break
            if min_delta is None:
                return True
            return op.gt(float(min_delta), 0.0)

        for n in draw_items:
            pitch = int(n.get('_pitch_i', 41) or 41)
            hand = str(n.get('_hand_norm', 'l') or 'l')
            is_up = bool(n.get('_is_up_b', True))
            notehead_literal = str(n.get('_notehead_lit', 'normal') or 'normal')
            x = float(n.get('_x', 0.0) or 0.0)
            nt = float(n.get('_time', t0) or t0)
            y = _time_to_y(nt, t0, t1, y0, y1)
            note_end = float(n.get('_end', nt) or nt)

            if note_head_visible:
                _draw_notehead(
                    du,
                    layout,
                    x,
                    y,
                    semitone_mm,
                    int(n.get('_id_i', 0) or 0),
                    hand,
                    is_up,
                    notehead_literal,
                    pitch,
                    outline_width_mm,
                    notation_color,
                    paper_color,
                    notehead_cache,
                )

            if dot_visible and bool(n.get('_dot_b', True)):
                n_key = n.get('_nkey', ('idx', -1))
                start = float(n.get('_time', t0) or t0)
                end = float(note_end)
                dot_pitch = int(n.get('_pitch_i', 0) or 0)
                dot_times: list[float] = []

                for other in _events_in_open_interval(hand, start, end, by_end=False):
                    if other.get('_nkey', ('idx', -2)) != n_key:
                        dot_times.append(float(other.get('_time', 0.0) or 0.0))
                for other in _events_in_open_interval(hand, start, end, by_end=True):
                    if other.get('_nkey', ('idx', -2)) != n_key:
                        ot = float(other.get('_time', 0.0) or 0.0)
                        od = float(other.get('_duration', 0.0) or 0.0)
                        dot_times.append(float(ot + od))

                bt_lo = bisect.bisect_right(barline_times, float(start))
                bt_hi = bisect.bisect_left(barline_times, float(end))
                if bt_hi > bt_lo:
                    dot_times.extend(float(bt) for bt in barline_times[bt_lo:bt_hi])

                if dot_times:
                    dot_size = max(0.05, float(layout.get('note_continuation_dot_size_mm', 2.5) or 2.5) * composite_scale)
                    notehead_center_offset_y = float(semitone_mm) * float(height_scale)
                    min_collision_gap = max(0.0, float(semitone_mm) * 2.0 - 1e-6)
                    for dt in sorted(set(dot_times)):
                        y_center = float(_time_to_y(float(dt), t0, t1, y0, y1)) + float(notehead_center_offset_y)
                        has_adjacent_start = False
                        for other in _starts_at_time(float(dt)):
                            if other.get('_nkey', ('idx', -2)) == n_key:
                                continue
                            mp = int(other.get('_pitch_i', 0) or 0)
                            if abs(mp - dot_pitch) != 1:
                                continue
                            if mp in BLACK_KEYS and bool(other.get('_is_up_b', False)):
                                continue
                            other_x = float(other.get('_x', x) or x)
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

            if note_stop_visible and bool(n.get('_stop_b', False)) and _is_followed_by_rest_item(n):
                stop_y = float(_time_to_y(note_end, t0, t1, y0, y1))
                stop_w = float(semitone_mm) * 1.8
                stop_stroke_w = max(
                    0.05,
                    float(layout.get('note_stopsign_thickness_mm', 1.0) or 1.0) * float(composite_scale),
                )
                stop_points = [
                    (float(x) - (stop_w * 0.5), float(stop_y) - float(stop_w)),
                    (float(x), float(stop_y)),
                    (float(x) + (stop_w * 0.5), float(stop_y) - float(stop_w)),
                ]
                du.add_polyline(
                    stop_points,
                    stroke_color=notation_color,
                    stroke_width_mm=stop_stroke_w,
                    id=0,
                    tags=['stop_sign'],
                )

        if note_stem_visible:
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

        # Beam rendering is centralized in beam_drawer using prepared beam_groups.
