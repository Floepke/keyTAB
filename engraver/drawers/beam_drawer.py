from __future__ import annotations

from engraver.helpers import group_by_beam_markers
from ui.widgets.draw_util import DrawUtil
from utils.CONSTANT import QUARTER_NOTE_UNIT, SHORTEST_DURATION
from utils.operator import Operator


def _item_get(item, key: str, default=None):
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _item_get_float(item, key: str, default: float) -> float:
    val = _item_get(item, key, None)
    if val is None:
        return float(default)
    try:
        return float(val)
    except Exception:
        return float(default)


def _time_to_y(t: float, t0: float, t1: float, y0: float, y1: float) -> float:
    if t1 <= t0:
        return y0
    u = max(0.0, min(1.0, (float(t) - float(t0)) / (float(t1) - float(t0))))
    return float(y0 + (u * (y1 - y0)))


def _build_barline_times(base_grid: list) -> list[float]:
    barline_times: list[float] = []
    cur_t = 0.0
    for bg in list(base_grid or []):
        numer = int(_item_get(bg, 'numerator', 4) or 4)
        denom = int(_item_get(bg, 'denominator', 4) or 4)
        mcount = int(_item_get(bg, 'measure_amount', 1) or 1)
        measure_len_ticks = float(numer) * (4.0 / float(max(1, denom))) * float(QUARTER_NOTE_UNIT)
        for _ in range(max(0, mcount)):
            barline_times.append(float(cur_t))
            cur_t += measure_len_ticks
    barline_times.append(float(cur_t))
    return barline_times


def prepare_beam_groups_for_stave(stv: dict, system: dict, base_grid: list, layout: dict) -> None:
    """Prepare beam groups/segments once and store them on stave pre-calc dict."""
    t0 = float(system.get('time_start', 0.0) or 0.0)
    t1 = float(system.get('time_end', t0) or t0)
    y0 = float(system.get('y_start_mm', 0.0) or 0.0)
    y1 = float(system.get('y_end_mm', y0) or y0)
    semitone_mm = float(stv.get('semitone_mm', 1.0) or 1.0)

    events = dict(stv.get('events_in_line', {}) or {})
    markers_src = [dict(m or {}) for m in list(events.get('beam', []) or []) if isinstance(m, dict)]
    notes_src_all = [dict(n or {}) for n in list(stv.get('note_draw_items', []) or []) if isinstance(n, dict)]
    for i, n in enumerate(notes_src_all):
        if _item_get(n, 'idx', None) is None:
            n['idx'] = int(_item_get(n, '_idx', i) if _item_get(n, '_idx', None) is not None else i)
    notes_src = [n for n in notes_src_all if bool(_item_get(n, 'beam', False))]
    # Some scores rely on beam markers/windows without setting per-note beam flags.
    # In that case, fallback to all note items so beams still render.
    if not notes_src:
        notes_src = notes_src_all

    by_hand_notes = {'l': [], 'r': []}
    by_hand_markers = {'l': [], 'r': []}

    for n in notes_src:
        hand = 'l' if str(_item_get(n, 'hand', 'l') or 'l') == 'l' else 'r'
        nt = _item_get_float(n, 'time', t0)
        dur = max(0.0, _item_get_float(n, 'duration', 0.0))
        entry = dict(n)
        entry['hand'] = hand
        entry['time'] = float(nt)
        entry['end'] = float(nt + dur)
        by_hand_notes[hand].append(entry)

    for mk in markers_src:
        hand = 'l' if str(_item_get(mk, 'hand', 'l') or 'l') == 'l' else 'r'
        by_hand_markers[hand].append(dict(mk))

    barline_times = _build_barline_times(base_grid)
    op = Operator(SHORTEST_DURATION)
    stem_len_mm = float(layout.get('note_stem_length_semitone', 3.0) or 3.0) * float(semitone_mm)

    beam_groups: list[dict] = []
    for hand in ('r', 'l'):
        groups, windows = group_by_beam_markers(
            by_hand_notes[hand],
            by_hand_markers[hand],
            float(t0),
            float(t1),
            list(base_grid or []),
            barline_times,
        )
        for idx, grp in enumerate(groups):
            if not grp:
                continue
            if idx >= len(windows):
                continue
            tw0, tw1 = windows[idx]
            s_min = None
            s_max = None
            for n in grp:
                nt = _item_get_float(n, 'time', tw0)
                if not (op.ge(float(nt), float(tw0)) and op.lt(float(nt), float(tw1))):
                    continue
                if s_min is None or nt < s_min:
                    s_min = nt
                if s_max is None or nt > s_max:
                    s_max = nt
            if s_min is None or s_max is None or op.eq(float(s_min), float(s_max)):
                continue

            if hand == 'r':
                anchor = max(grp, key=lambda n: int(_item_get(n, 'pitch', 0) or 0))
                x1b = _item_get_float(anchor, 'x_mm', 0.0) + float(stem_len_mm)
                x2b = x1b + float(semitone_mm)
            else:
                anchor = min(grp, key=lambda n: int(_item_get(n, 'pitch', 0) or 0))
                x1b = _item_get_float(anchor, 'x_mm', 0.0) - float(stem_len_mm)
                x2b = x1b - float(semitone_mm)

            yb1 = _time_to_y(float(s_min), t0, t1, y0, y1)
            yb2 = _time_to_y(float(s_max), t0, t1, y0, y1)

            connectors: list[dict] = []
            for n in grp:
                mt = _item_get_float(n, 'time', s_min)
                if not (op.ge(float(mt), float(tw0)) and op.lt(float(mt), float(tw1))):
                    continue
                y_note = _time_to_y(float(mt), t0, t1, y0, y1)
                x_note = _item_get_float(n, 'x_mm', 0.0)
                x_tip = x_note + float(stem_len_mm) if hand == 'r' else x_note - float(stem_len_mm)
                if abs(float(yb2) - float(yb1)) > 1e-9:
                    ratio = (float(y_note) - float(yb1)) / (float(yb2) - float(yb1))
                    x_on_beam = float(x1b) + ratio * (float(x2b) - float(x1b))
                else:
                    x_on_beam = float(x1b)
                connectors.append(
                    {
                        'time': float(mt),
                        'x0_mm': float(min(x_tip, x_on_beam)),
                        'x1_mm': float(max(x_tip, x_on_beam)),
                        'y_mm': float(y_note),
                    }
                )

            beam_groups.append(
                {
                    'hand': hand,
                    'window_start': float(tw0),
                    'window_end': float(tw1),
                    't_start': float(s_min),
                    't_end': float(s_max),
                    'x1_mm': float(x1b),
                    'x2_mm': float(x2b),
                    'y1_mm': float(yb1),
                    'y2_mm': float(yb2),
                    'connect_segments': connectors,
                }
            )

    stv['beam_groups'] = beam_groups


def beam_drawer(du: DrawUtil, pre_calc: dict) -> None:
    """Draw beams and beam connectors from prepared beam_groups payload."""
    system = dict(pre_calc.get('system', {}) or {})
    layout = dict(pre_calc.get('layout', {}) or {})
    if not bool(layout.get('beam_visible', True)):
        return
    notation_color = tuple(pre_calc.get('notation_color', (0.0, 0.0, 0.0, 1.0)) or (0.0, 0.0, 0.0, 1.0))

    for stv in list(system.get('staves', []) or []):
        composite_scale = float(stv.get('composite_scale', 1.0) or 1.0)
        beam_w = max(0.1, float(layout.get('beam_thickness_mm', 1.0) or 1.0) * composite_scale)
        stem_w = max(0.05, float(stv.get('note_stem_width_mm', 0.4) or 0.4))
        for bg in list(stv.get('beam_groups', []) or []):
            du.add_line(
                float(bg.get('x1_mm', 0.0) or 0.0),
                float(bg.get('y1_mm', 0.0) or 0.0),
                float(bg.get('x2_mm', 0.0) or 0.0),
                float(bg.get('y2_mm', 0.0) or 0.0),
                color=notation_color,
                width_mm=float(beam_w),
                tags=['beam'],
            )
            for conn in list(bg.get('connect_segments', []) or []):
                du.add_line(
                    float(conn.get('x0_mm', 0.0) or 0.0),
                    float(conn.get('y_mm', 0.0) or 0.0),
                    float(conn.get('x1_mm', 0.0) or 0.0),
                    float(conn.get('y_mm', 0.0) or 0.0),
                    color=notation_color,
                    width_mm=float(stem_w),
                    tags=['beam_stem'],
                )