from __future__ import annotations

import bisect

from file_model.base_grid import resolve_grid_layer_offsets
from ui.widgets.draw_util import DrawUtil
from utils.CONSTANT import QUARTER_NOTE_UNIT, SHORTEST_DURATION
from utils.operator import Operator


def _item_get(item, key: str, default=None):
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _time_to_y(t: float, t0: float, t1: float, y0: float, y1: float) -> float:
    if t1 <= t0:
        return y0
    u = max(0.0, min(1.0, (float(t) - float(t0)) / (float(t1) - float(t0))))
    return float(y0 + (u * (y1 - y0)))


def _build_grid_times(base_grid: list) -> tuple[list[float], list[float], dict[float, int]]:
    """Return (barline_times, grid_times, measure_start_number_by_time)."""
    barline_times: list[float] = []
    grid_times: list[float] = []
    measure_numbers: dict[float, int] = {}
    cur_t = 0.0
    measure_no = 0
    for bg in list(base_grid or []):
        numer = int(_item_get(bg, 'numerator', 4) or 4)
        denom = int(_item_get(bg, 'denominator', 4) or 4)
        mcount = int(_item_get(bg, 'measure_amount', 1) or 1)
        measure_len_ticks = float(numer) * (4.0 / float(max(1, denom))) * float(QUARTER_NOTE_UNIT)
        positions = list(_item_get(bg, 'beat_grouping', []) or [])
        bar_offsets, grid_offsets = resolve_grid_layer_offsets(positions, numer, denom)
        for _ in range(max(0, mcount)):
            measure_no += 1
            t_measure_start = float(cur_t)
            measure_numbers[round(t_measure_start, 6)] = int(measure_no)
            for off in list(bar_offsets or []):
                barline_times.append(float(cur_t + float(off)))
            for off in list(grid_offsets or []):
                grid_times.append(float(cur_t + float(off)))
            cur_t += measure_len_ticks
    barline_times.append(float(cur_t))
    return barline_times, grid_times, measure_numbers


def grid_drawer(du: DrawUtil, pre_calc: dict) -> None:
    system = dict(pre_calc.get('system', {}) or {})
    layout = dict(pre_calc.get('layout', {}) or {})
    base_grid = list(pre_calc.get('base_grid', []) or [])
    notation_color = tuple(pre_calc.get('notation_color', (0.0, 0.0, 0.0, 1.0)) or (0.0, 0.0, 0.0, 1.0))
    y0 = float(pre_calc.get('y0', 0.0) or 0.0)
    y1 = float(pre_calc.get('y1', 0.0) or 0.0)
    t0 = float(system.get('time_start', 0.0) or 0.0)
    t1 = float(system.get('time_end', t0) or t0)

    # Grid and barlines follow the natural stave width.
    x_left = float(system.get('system_stave_left_mm', pre_calc.get('system_content_left_mm', 0.0)) or 0.0)
    x_right = float(
        x_left
        + float(
            system.get(
                'system_stave_width_mm',
                pre_calc.get('system_content_width_mm', 0.0),
            )
            or 0.0
        )
    )

    op = Operator(SHORTEST_DURATION)

    def _in_system(t: float) -> bool:
        return op.ge(float(t), float(t0)) and op.le(float(t), float(t1))

    staves = list(system.get('staves', []) or [])
    system_composite_scale = max([float(st.get('composite_scale', 1.0) or 1.0) for st in staves] or [1.0])
    system_semitone_mm = max([float(st.get('semitone_mm', 1.0) or 1.0) for st in staves] or [1.0])
    beam_visible = bool(layout.get('beam_visible', True))
    barline_visible = bool(layout.get('barline_visible', True))
    grid_line_visible = bool(layout.get('grid_line_visible', True))
    measure_numbers_visible = bool(layout.get('measure_numbers_visible', True))
    barline_w = max(0.01, float(layout.get('grid_barline_thickness_mm', 0.1) or 0.1) * system_composite_scale)
    grid_w = max(0.01, float(layout.get('grid_gridline_thickness_mm', 0.15) or 0.15) * system_composite_scale)
    grid_dash = [
        max(0.01, float(d) * system_composite_scale)
        for d in list(layout.get('grid_gridline_dash_pattern_mm', [0.8, 0.8]) or [0.8, 0.8])
    ]

    # Collision geometry parameters (ported from editor logic, adapted to pre-calc notes).
    note_head_half_w = float(system_semitone_mm) * float(layout.get('note_width_scaling', 0.75) or 0.75)
    stem_len_mm = float(layout.get('note_stem_length_semitone', 3.0) or 3.0) * float(system_semitone_mm)
    stem_collision_pad = max(0.15, float(layout.get('note_stem_thickness_mm', 0.5) or 0.5) * system_composite_scale)
    head_collision_pad = max(0.15, float(system_semitone_mm) * 0.15)
    barline_symbol_gap_mm = max(0.0, float(system_semitone_mm))
    barline_time_eps = 1e-4
    barline_time_op = Operator(float(SHORTEST_DURATION))

    def _barline_time_eq(a: float, b: float) -> bool:
        return barline_time_op.eq(float(a), float(b))

    def _barline_time_in_range(t: float, a: float, b: float) -> bool:
        lo = float(min(a, b)) - float(barline_time_eps)
        hi = float(max(a, b)) + float(barline_time_eps)
        return barline_time_op.ge(float(t), lo) and barline_time_op.le(float(t), hi)

    barline_times, grid_times, measure_numbers = _build_grid_times(base_grid)
    barline_keys = {round(float(t), 6) for t in barline_times}
    query_tick_values = sorted(
        list(
            dict.fromkeys(
                [round(float(t), 6) for t in list(grid_times or [])]
                + [round(float(t), 6) for t in list(barline_times or [])]
            )
        )
    )
    query_tick_set = set(query_tick_values)

    def _norm_hand(v: str) -> str:
        return 'l' if str(v or 'l') == 'l' else 'r'

    notes_at_tick: dict[float, list[dict]] = {}
    chord_span_at_tick_hand: dict[tuple[float, str], tuple[float, float]] = {}
    beam_groups_view: list[dict] = []
    stem_segments_view: list[dict] = []
    for stv in staves:
        items = [n for n in list(stv.get('note_draw_items', []) or []) if isinstance(n, dict)]
        if not items:
            continue
        for n in items:
            nt = round(float(_item_get(n, 'time', 0.0) or 0.0), 6)
            notes_at_tick.setdefault(nt, []).append(n)
            hand = _norm_hand(str(_item_get(n, 'hand', 'l') or 'l'))
            x_note = float(_item_get(n, 'x_mm', x_left) or x_left)
            chord_key = (nt, hand)
            span = chord_span_at_tick_hand.get(chord_key)
            if span is None:
                chord_span_at_tick_hand[chord_key] = (x_note, x_note)
            else:
                chord_span_at_tick_hand[chord_key] = (min(span[0], x_note), max(span[1], x_note))
        for seg in list(stv.get('stem_segments', []) or []):
            if isinstance(seg, dict):
                stem_segments_view.append(seg)
        for grp in list(stv.get('beam_groups', []) or []):
            if isinstance(grp, dict):
                beam_groups_view.append(grp)

    beam_segments_by_tick: dict[float, list[dict]] = {}
    beam_connect_segments_by_tick: dict[float, list[dict]] = {}
    if query_tick_values and beam_groups_view:
        eps = float(barline_time_eps)
        for seg in beam_groups_view:
            bs0 = float(_item_get(seg, 't_start', 0.0) or 0.0)
            bs1 = float(_item_get(seg, 't_end', 0.0) or 0.0)
            lo_t = float(min(bs0, bs1) - eps)
            hi_t = float(max(bs0, bs1) + eps)
            lo_i = bisect.bisect_left(query_tick_values, lo_t)
            hi_i = bisect.bisect_right(query_tick_values, hi_t)
            for i_tick in range(lo_i, hi_i):
                tk = float(query_tick_values[i_tick])
                beam_segments_by_tick.setdefault(tk, []).append(seg)
            for conn in list(_item_get(seg, 'connect_segments', []) or []):
                if not isinstance(conn, dict):
                    continue
                c_tk = round(float(_item_get(conn, 'time', 0.0) or 0.0), 6)
                if c_tk in query_tick_set:
                    beam_connect_segments_by_tick.setdefault(float(c_tk), []).append(conn)

    def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if not intervals:
            return []
        clipped: list[tuple[float, float]] = []
        for a, b in intervals:
            xa = max(float(x_left), min(float(x_right), float(min(a, b))))
            xb = max(float(x_left), min(float(x_right), float(max(a, b))))
            if xb <= xa:
                continue
            clipped.append((xa, xb))
        if not clipped:
            return []
        clipped.sort(key=lambda it: it[0])
        merged: list[tuple[float, float]] = [clipped[0]]
        for a, b in clipped[1:]:
            la, lb = merged[-1]
            if a <= lb:
                merged[-1] = (la, max(lb, b))
            else:
                merged.append((a, b))
        return merged

    # Keep chord spans only for actual chords (>= 2 notes in same hand/tick).
    chord_counts: dict[tuple[float, str], int] = {}
    for tick_key, tick_notes in notes_at_tick.items():
        for n in tick_notes:
            hand = _norm_hand(str(_item_get(n, 'hand', 'l') or 'l'))
            key = (tick_key, hand)
            chord_counts[key] = int(chord_counts.get(key, 0)) + 1
    chord_span_at_tick_hand = {
        k: v for k, v in chord_span_at_tick_hand.items() if int(chord_counts.get(k, 0)) >= 2
    }

    barline_cut_cache: dict[float, list[tuple[float, float]]] = {}

    def _barline_cut_intervals(ticks: float) -> list[tuple[float, float]]:
        tick_key = round(float(ticks), 6)
        if tick_key in barline_cut_cache:
            return barline_cut_cache[tick_key]

        intervals: list[tuple[float, float]] = []
        for n in notes_at_tick.get(tick_key, []):
            x_note = float(_item_get(n, 'x_mm', x_left) or x_left)
            intervals.append(
                (
                    x_note - note_head_half_w - head_collision_pad - barline_symbol_gap_mm,
                    x_note + note_head_half_w + head_collision_pad + barline_symbol_gap_mm,
                )
            )

        tick_notes = notes_at_tick.get(tick_key, [])

        if beam_visible:
            for n in tick_notes:
                x_note = float(_item_get(n, 'x_mm', x_left) or x_left)
                hand_key = _norm_hand(str(_item_get(n, 'hand', 'l') or 'l'))
                x_tip = x_note - stem_len_mm if hand_key == 'l' else x_note + stem_len_mm
                intervals.append(
                    (
                        min(x_note, x_tip) - stem_collision_pad - barline_symbol_gap_mm,
                        max(x_note, x_tip) + stem_collision_pad + barline_symbol_gap_mm,
                    )
                )

            for hand_key in ('l', 'r'):
                chord_span = chord_span_at_tick_hand.get((tick_key, hand_key))
                if chord_span is not None:
                    x_lo, x_hi = chord_span
                    intervals.append(
                        (
                            x_lo - stem_collision_pad - barline_symbol_gap_mm,
                            x_hi + stem_collision_pad + barline_symbol_gap_mm,
                        )
                    )
        else:
            # Hidden-beam mode: cut around actual visible stem segments.
            # Stems are prepared as per-chord horizontal segments:
            # left hand => lowest chord note - stem_length
            # right hand => highest chord note + stem_length
            stem_half_visible = max(0.05, float(layout.get('note_stem_thickness_mm', 0.5) or 0.5) * system_composite_scale * 0.5)
            for seg in stem_segments_view:
                seg_t = float(_item_get(seg, 'time', 0.0) or 0.0)
                if not _barline_time_eq(float(seg_t), float(ticks)):
                    continue
                sx0 = float(_item_get(seg, 'x1_mm', 0.0) or 0.0)
                sx1 = float(_item_get(seg, 'x2_mm', 0.0) or 0.0)
                intervals.append(
                    (
                        min(sx0, sx1) - stem_half_visible - float(system_semitone_mm * 1.5),
                        max(sx0, sx1) + stem_half_visible + float(system_semitone_mm * 1.5),
                    )
                )

        # Beam cuts are only needed while beams are visible.
        if beam_visible:
            beam_line_half_visible = max(0.05, float(layout.get('beam_thickness_mm', 1.0) or 1.0) * system_composite_scale * 0.5)
            beam_stem_half_visible = max(0.05, float(layout.get('note_stem_thickness_mm', 0.5) or 0.5) * system_composite_scale * 0.5)
            visible_extra_pad = max(0.0, float(system_semitone_mm) * 0.1)
            for seg in beam_segments_by_tick.get(tick_key, []):
                bs0 = float(_item_get(seg, 't_start', 0.0) or 0.0)
                bs1 = float(_item_get(seg, 't_end', 0.0) or 0.0)
                if not _barline_time_in_range(float(ticks), bs0, bs1):
                    continue
                dt = float(bs1 - bs0)
                if abs(dt) <= 1e-9:
                    continue
                ratio = (float(ticks) - bs0) / dt
                seg_x1 = float(_item_get(seg, 'x1_mm', 0.0) or 0.0)
                seg_x2 = float(_item_get(seg, 'x2_mm', 0.0) or 0.0)
                x_on_beam = float(seg_x1) + ratio * (float(seg_x2) - float(seg_x1))
                seg_hand = _norm_hand(str(_item_get(seg, 'hand', 'l') or 'l'))
                if seg_hand == 'r':
                    x_on_beam += float(system_semitone_mm)
                else:
                    x_on_beam -= float(system_semitone_mm)
                beam_pad = float(beam_line_half_visible + visible_extra_pad)
                intervals.append((x_on_beam - beam_pad, x_on_beam + beam_pad))

            for conn in beam_connect_segments_by_tick.get(tick_key, []):
                c_t = float(_item_get(conn, 'time', 0.0) or 0.0)
                if not _barline_time_eq(float(c_t), float(ticks)):
                    continue
                c_x0 = float(_item_get(conn, 'x0_mm', 0.0) or 0.0)
                c_x1 = float(_item_get(conn, 'x1_mm', 0.0) or 0.0)
                conn_pad = float(beam_stem_half_visible + visible_extra_pad)
                intervals.append((c_x0 - conn_pad, c_x1 + conn_pad))

        merged = _merge_intervals(intervals)
        barline_cut_cache[tick_key] = merged
        return merged

    def _draw_line_around_chords(y_mm: float, cuts: list[tuple[float, float]], width_mm: float, tags: list[str], dash_pattern=None) -> None:
        if not cuts:
            du.add_line(
                x_left,
                y_mm,
                x_right,
                y_mm,
                color=notation_color,
                width_mm=width_mm,
                dash_pattern=dash_pattern,
                tags=tags,
            )
            return
        x_cursor = float(x_left)
        min_seg = max(0.05, float(width_mm) * 0.5)
        for c0, c1 in cuts:
            if c0 - x_cursor > min_seg:
                du.add_line(
                    x_cursor,
                    y_mm,
                    c0,
                    y_mm,
                    color=notation_color,
                    width_mm=width_mm,
                    dash_pattern=dash_pattern,
                    tags=tags,
                )
            x_cursor = max(x_cursor, c1)
        if float(x_right) - x_cursor > min_seg:
            du.add_line(
                x_cursor,
                y_mm,
                x_right,
                y_mm,
                color=notation_color,
                width_mm=width_mm,
                dash_pattern=dash_pattern,
                tags=tags,
            )

    # Sub-grid lines, excluding primary barline layer times.
    if grid_line_visible:
        for t in grid_times:
            if round(float(t), 6) in barline_keys:
                continue
            if not _in_system(float(t)):
                continue
            y = _time_to_y(float(t), t0, t1, y0, y1)
            cuts = _barline_cut_intervals(float(t))
            _draw_line_around_chords(float(y), cuts, float(grid_w), ['grid_line'], dash_pattern=grid_dash)

    # Barlines connect the full green content rectangle width.
    if barline_visible:
        for i, t in enumerate(barline_times):
            if not _in_system(float(t)):
                continue
            y = _time_to_y(float(t), t0, t1, y0, y1)
            is_last = i == (len(barline_times) - 1)
            cuts = _barline_cut_intervals(float(t))
            _draw_line_around_chords(
                float(y),
                cuts,
                (float(barline_w) * 2.0) if is_last else float(barline_w),
                ['end_barline' if is_last else 'barline'],
            )

    # Measure numbering at right side of the system for visible starts.
    if measure_numbers_visible:
        for t in barline_times[:-1]:
            if not _in_system(float(t)):
                continue
            # Avoid duplicate labels on shared boundaries: the next system will
            # render the label at its start (same time value).
            if op.eq(float(t), float(t1)):
                continue
            key = round(float(t), 6)
            if key not in measure_numbers:
                continue
            y = _time_to_y(float(t), t0, t1, y0, y1)
            du.add_text(
                x_right + 0.8,
                y + 0.1,
                str(int(measure_numbers[key])),
                family='Edwin',
                size_pt=8.5,
                color=notation_color,
                anchor='w',
                tags=['measure_number'],
            )
