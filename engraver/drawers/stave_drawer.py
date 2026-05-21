from __future__ import annotations

from symbol_design.noteheads import resolve_notehead_spec
from ui.widgets.draw_util import DrawUtil
from utils.CONSTANT import BLACK_KEYS
from utils.tiny_tool import key_class_filter


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
    u = min(1.0, (float(t) - float(t0)) / (float(t1) - float(t0)))
    return float(y0 + (u * (y1 - y0)))


def _build_black_groups() -> list[dict]:
    black_keys_sorted = sorted([int(k) for k in BLACK_KEYS])
    fga_keys = set(key_class_filter('FGA'))
    groups: list[dict] = [{'kind': 'single', 'start': 1, 'end': 3}]
    if not black_keys_sorted:
        return groups
    run = [int(black_keys_sorted[0])]
    run_kind = 'three' if int(black_keys_sorted[0]) in fga_keys else 'two'
    for key in black_keys_sorted[1:]:
        k = int(key)
        k_kind = 'three' if k in fga_keys else 'two'
        if k_kind == run_kind:
            run.append(k)
        else:
            groups.append({'kind': run_kind, 'start': int(min(run)), 'end': int(max(run))})
            run = [k]
            run_kind = k_kind
    groups.append({'kind': run_kind, 'start': int(min(run)), 'end': int(max(run))})
    return groups


def _nearest_group_index_for_pitch(groups: list[dict], pitch: int) -> int:
    p = int(pitch)
    for i, g in enumerate(groups):
        lo = int(g.get('start', p) or p)
        hi = int(g.get('end', p) or p)
        if lo <= p <= hi:
            return int(i)
    if not groups:
        return 0
    best_i = 0
    best_d = 10**9
    for i, g in enumerate(groups):
        lo = int(g.get('start', p) or p)
        hi = int(g.get('end', p) or p)
        d = min(abs(lo - p), abs(hi - p))
        if d < best_d:
            best_d = d
            best_i = i
    return int(best_i)


def stave_drawer(du: DrawUtil, pre_calc: dict) -> None:
    """Draw stave black-key lines from pre-calculated geometry."""
    system = dict(pre_calc.get('system', {}) or {})
    layout = dict(pre_calc.get('layout', {}) or {})
    if not bool(layout.get('stave_visible', True)):
        return
    y0 = float(pre_calc.get('y0', 0.0) or 0.0)
    y1 = float(pre_calc.get('y1', 0.0) or 0.0)
    t0 = float(system.get('time_start', 0.0) or 0.0)
    t1 = float(system.get('time_end', t0 + 1.0) or (t0 + 1.0))
    notation_color = tuple(pre_calc.get('notation_color', (0.0, 0.0, 0.0, 1.0)) or (0.0, 0.0, 0.0, 1.0))
    groups = _build_black_groups()
    fga_keys = set(key_class_filter('FGA'))

    for stv in list(system.get('staves', []) or []):
        stv: dict
        composite_scale = float(stv.get('composite_scale', 1.0) or 1.0)
        semitone_mm = float(stv.get('semitone_mm', 1.0) or 1.0)
        height_scale = float(layout.get('notehead_height_scaling', 1.0) or 1.0)
        stave_low_key = int(stv.get('stave_low_key', 1) or 1)
        stave_high_key = int(stv.get('stave_high_key', 88) or 88)
        key_offsets = dict(stv.get('key_offsets', {}) or {})
        span_low = int(stv.get('note_span_low_key', 1) or 1)
        span_left = float(stv.get('stave_content_span_left_mm', 0.0) or 0.0)
        clef_dash = [
            max(0.01, float(d) * composite_scale)
            for d in list(layout.get('stave_clef_line_dash_pattern_mm', [4.0, 3.0]) or [4.0, 3.0])
        ]
        two_w = max(0.05, float(layout.get('stave_two_line_thickness_mm', 0.5) or 0.5) * composite_scale)
        three_w = max(0.05, float(layout.get('stave_three_line_thickness_mm', 1.1) or 1.1) * composite_scale)
        clef_w = max(0.05, float(layout.get('stave_clef_line_thickness_mm', 0.75) or 0.75) * composite_scale)

        def _key_to_x(key: int) -> float:
            k = int(max(1, min(88, int(key))))
            if not key_offsets:
                return float(span_left)
            return float(span_left + (float(key_offsets.get(k, 0.0)) - float(key_offsets.get(span_low, 0.0))))

        # Draw full stave lines for in-range black keys.
        for ln in list(stv.get('black_lines', []) or []):
            x_mm = float(ln.get('x_mm', 0.0) or 0.0)
            du.add_line(
                x_mm,
                y0,
                x_mm,
                y1,
                color=notation_color,
                width_mm=float(ln.get('width_mm', 0.5) or 0.5),
                dash_pattern=ln.get('dash', None),
                id=0,
                tags=['stave'],
            )

        # Ledger stubs: grouped per hand/time chord list to avoid duplicate draws.
        notehead_h = float(semitone_mm) * 2.0 * float(height_scale)
        y_pad = float(semitone_mm)
        chord_lists = []
        chord_lists.extend(list(stv.get('note_left_chord_list', []) or []))
        chord_lists.extend(list(stv.get('note_right_chord_list', []) or []))

        if chord_lists:
            for chord in chord_lists:
                chord_items = [n for n in list(chord or []) if isinstance(n, dict)]
                if not chord_items:
                    continue
                ledger_notes = [
                    n
                    for n in chord_items
                    if int(_item_get(n, 'pitch', 0) or 0) < stave_low_key
                    or int(_item_get(n, 'pitch', 0) or 0) > stave_high_key
                ]
                if not ledger_notes:
                    continue

                note_tops: list[float] = []
                note_bottoms: list[float] = []
                for n in ledger_notes:
                    n_time = _item_get_float(n, 'time', t0)
                    y_anchor = _time_to_y(float(n_time), t0, t1, y0, y1)
                    # Use the rendered notehead direction, not the raw payload flag.
                    # White auto noteheads are always down; black auto noteheads follow
                    # default_black_above from pre-calc.
                    spec = resolve_notehead_spec(n, default_black_above=bool(_item_get(n, 'is_up', False)))
                    if bool(spec.is_up):
                        note_tops.append(float(y_anchor) - float(notehead_h))
                        note_bottoms.append(float(y_anchor))
                    else:
                        note_tops.append(float(y_anchor))
                        note_bottoms.append(float(y_anchor) + float(notehead_h))

                chord_outer_top = float(min(note_tops))
                chord_outer_bottom = float(max(note_bottoms))
                y_top = float(chord_outer_top - y_pad)
                y_bottom = float(chord_outer_bottom + y_pad)

                keys_to_draw: set[int] = set()
                low_edge_i = _nearest_group_index_for_pitch(groups, stave_low_key)
                high_edge_i = _nearest_group_index_for_pitch(groups, stave_high_key)

                for n in ledger_notes:
                    p = int(_item_get(n, 'pitch', 0) or 0)
                    if stave_low_key <= p <= stave_high_key:
                        continue
                    target_i = _nearest_group_index_for_pitch(groups, p)
                    if p > stave_high_key:
                        i0 = int(min(high_edge_i + 1, target_i))
                        i1 = int(max(high_edge_i + 1, target_i))
                    else:
                        i0 = int(min(target_i, low_edge_i - 1))
                        i1 = int(max(target_i, low_edge_i - 1))
                    for gi in range(i0, i1 + 1):
                        if gi < 0 or gi >= len(groups):
                            continue
                        grp = groups[gi]
                        g_lo = int(grp.get('start', p) or p)
                        g_hi = int(grp.get('end', p) or p)
                        for k in range(g_lo, g_hi + 1):
                            if k in BLACK_KEYS:
                                keys_to_draw.add(int(k))

                for k in sorted(keys_to_draw):
                    if stave_low_key <= k <= stave_high_key:
                        # Already covered by full stave lines.
                        continue
                    if k in (41, 43):
                        w = clef_w
                        dash = clef_dash
                    elif k in fga_keys:
                        w = three_w
                        dash = None
                    else:
                        w = two_w
                        dash = None
                    du.add_line(
                        _key_to_x(k),
                        y_top,
                        _key_to_x(k),
                        y_bottom,
                        color=notation_color,
                        width_mm=float(w),
                        dash_pattern=dash,
                        id=0,
                        tags=['ledger_line'],
                    )
