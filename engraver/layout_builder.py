from __future__ import annotations

import bisect

from file_model.SCORE import SCORE
from file_model.analysis import Analysis
from file_model.base_grid import resolve_grid_layer_offsets
from utils.CONSTANT import BE_KEYS, PIANO_KEY_AMOUNT, QUARTER_NOTE_UNIT, SHORTEST_DURATION
from utils.operator import Operator
from utils.tiny_tool import key_class_filter


def _calculate_text_height_mm(font_size_pt: float, padding_mm: float = 0.5) -> float:
    return max(0.0, float(font_size_pt) * 0.35278 + float(padding_mm))


def _sanitize_range(rng) -> list[int]:
    if not isinstance(rng, list) or len(rng) < 2:
        return [1, PIANO_KEY_AMOUNT]
    lo = int(rng[0])
    hi = int(rng[1])
    lo = max(1, min(PIANO_KEY_AMOUNT, lo))
    hi = max(1, min(PIANO_KEY_AMOUNT, hi))
    if hi < lo:
        lo, hi = hi, lo
    return [lo, hi]


def _build_key_positions(start_key: int, end_key: int, semitone_mm: float) -> dict[int, float]:
    positions: dict[int, float] = {}
    x = 0.0
    prev = None
    for key in range(start_key, end_key + 1):
        if prev is not None and prev in BE_KEYS:
            x += semitone_mm
        x += semitone_mm
        positions[key] = x
        prev = key
    return positions


def _pc_char(key: int) -> str:
    pc = (int(key) - 1) % 12
    if pc in (0, 2, 3, 5, 7, 8, 10):
        return {0: 'a', 2: 'b', 3: 'c', 5: 'd', 7: 'e', 8: 'f', 10: 'g'}[pc]
    return {1: 'A', 4: 'C', 6: 'D', 9: 'F', 11: 'G'}[pc]


def _build_line_groups(line_keys: list[int]) -> list[dict]:
    groups: list[dict] = []
    used: set[int] = set()

    def _next_index(start: int, pc_target: str) -> int | None:
        for j in range(start + 1, len(line_keys)):
            if j in used:
                continue
            if _pc_char(line_keys[j]) == pc_target:
                return j
        return None

    for i, key in enumerate(line_keys):
        if i in used:
            continue
        pc = _pc_char(key)
        if pc == 'C':
            keys = [key]
            j = _next_index(i, 'D')
            if j is not None:
                keys.append(line_keys[j])
                used.add(j)
            used.add(i)
            groups.append({'keys': keys})
        elif pc == 'F':
            keys = [key]
            j = _next_index(i, 'G')
            if j is not None:
                keys.append(line_keys[j])
                used.add(j)
                k = _next_index(j, 'A')
                if k is not None:
                    keys.append(line_keys[k])
                    used.add(k)
            used.add(i)
            groups.append({'keys': keys})

    groups.sort(key=lambda g: g['keys'][0])
    for i, grp in enumerate(groups):
        first = grp['keys'][0]
        last = grp['keys'][-1]
        if i == 0:
            low = 1
        else:
            prev_last = groups[i - 1]['keys'][-1]
            low = int((prev_last + first) // 2) + 1
        if i == len(groups) - 1:
            high = PIANO_KEY_AMOUNT
        else:
            next_first = groups[i + 1]['keys'][0]
            high = int((last + next_first) // 2)
        grp['range_low'] = int(max(1, low))
        grp['range_high'] = int(min(PIANO_KEY_AMOUNT, high))
        if 41 in grp['keys'] and 43 in grp['keys']:
            grp['pattern'] = 'c'
        elif len(grp['keys']) == 2:
            grp['pattern'] = '2'
        else:
            grp['pattern'] = '3'
    return groups


def build_layout_bundle(score: SCORE) -> dict:
    score = score or {}
    layout = dict((score or {}).get('layout', {}) or {})
    events = dict((score or {}).get('events', {}) or {})
    base_grid = list(score.get('base_grid', []) or [])
    line_breaks = list(events.get('line_break', []) or [])
    notes = list(events.get('note', []) or [])

    try:
        layout_scale = float(layout.get('scale', 1.0) or 1.0)
    except Exception:
        layout_scale = 1.0
    if layout_scale <= 0.0:
        layout_scale = 1.0

    page_orientation = str(layout.get('page_orientation', 'portrait') or 'portrait').strip().lower()
    if page_orientation == 'vertical':
        page_orientation = 'portrait'
    elif page_orientation == 'horizontal':
        page_orientation = 'landscape'

    read_direction = str(layout.get('read_direction', 'vertical') or 'vertical').strip().lower()
    horizontal_read_direction = read_direction == 'horizontal'
    landscape_page_orientation = page_orientation == 'landscape'

    raw_page_w = float(layout.get('page_width_mm', 210.0) or 210.0)
    raw_page_h = float(layout.get('page_height_mm', 297.0) or 297.0)
    swap_page_axes = landscape_page_orientation != horizontal_read_direction
    if swap_page_axes:
        page_w = raw_page_h
        page_h = raw_page_w
    else:
        page_w = raw_page_w
        page_h = raw_page_h

    user_page_left = float(layout.get('page_left_margin_mm', 5.0) or 5.0)
    user_page_right = float(layout.get('page_right_margin_mm', 5.0) or 5.0)
    user_page_top = float(layout.get('page_top_margin_mm', 10.0) or 10.0)
    user_page_bottom = float(layout.get('page_bottom_margin_mm', 10.0) or 10.0)

    if horizontal_read_direction:
        page_left = user_page_bottom
        page_right = user_page_top
        page_top = user_page_left
        page_bottom = user_page_right
    else:
        page_left = user_page_left
        page_right = user_page_right
        page_top = user_page_top
        page_bottom = user_page_bottom

    semitone_mm = 2.0 * layout_scale
    line_keys = sorted(key_class_filter('ACDFG'))
    stave_line_groups = _build_line_groups(line_keys)
    if not stave_line_groups:
        stave_line_groups = [{'keys': [41, 43], 'range_low': 1, 'range_high': PIANO_KEY_AMOUNT, 'pattern': 'c'}]

    clef_group_index = 0
    for i, grp in enumerate(stave_line_groups):
        if 41 in grp['keys'] and 43 in grp['keys']:
            clef_group_index = i
            break

    def _group_index_for_key(key: int) -> int:
        for i, grp in enumerate(stave_line_groups):
            if int(grp['range_low']) <= int(key) <= int(grp['range_high']):
                return i
        return 0 if int(key) <= int(stave_line_groups[0]['range_low']) else len(stave_line_groups) - 1

    def _visible_line_groups_for_range(lo: int, hi: int, include_clef: bool = True) -> list[dict]:
        lo = int(max(1, min(PIANO_KEY_AMOUNT, lo)))
        hi = int(max(1, min(PIANO_KEY_AMOUNT, hi)))
        if hi < lo:
            lo, hi = hi, lo
        min_group = _group_index_for_key(lo)
        max_group = _group_index_for_key(hi)
        if include_clef:
            if clef_group_index < min_group:
                min_group = clef_group_index
            if clef_group_index > max_group:
                max_group = clef_group_index
        return [stave_line_groups[gi] for gi in range(min_group, max_group + 1)]

    key_positions = _build_key_positions(1, PIANO_KEY_AMOUNT, semitone_mm)

    def _layout_font_size(key: str, fallback_size: float) -> float:
        raw = layout.get(key, {}) if isinstance(layout, dict) else {}
        if not isinstance(raw, dict):
            raw = {}
        return float(raw.get('size_pt', fallback_size) or fallback_size)

    scale = layout_scale
    title_size = _layout_font_size('font_title', 12.0) * scale
    composer_size = _layout_font_size('font_composer', 10.0) * scale
    footer_size = _layout_font_size('font_copyright', 8.0) * scale
    header_height = max(0.0, _calculate_text_height_mm(title_size, 0.3) + _calculate_text_height_mm(composer_size, 0.3) + 1.0)
    footer_height = max(0.0, _calculate_text_height_mm(footer_size, 0.3) + 1.0)

    if horizontal_read_direction:
        line_axis_left_reserve_default = float(footer_height)
        line_axis_right_reserve_default = 0.0
    else:
        line_axis_left_reserve_default = 0.0
        line_axis_right_reserve_default = 0.0

    def _line_axis_reserves_for_page(page_index: int) -> tuple[float, float]:
        left_reserve = float(line_axis_left_reserve_default)
        right_reserve = float(line_axis_right_reserve_default)
        if horizontal_read_direction and page_index == 0:
            right_reserve += float(header_height)
        return left_reserve, right_reserve

    def _total_score_ticks() -> float:
        total = 0.0
        for bg in base_grid:
            numer = int(bg.get('numerator', 4) or 4)
            denom = int(bg.get('denominator', 4) or 4)
            measures = int(bg.get('measure_amount', 1) or 1)
            measure_len = float(numer) * (4.0 / float(max(1, denom))) * float(QUARTER_NOTE_UNIT)
            total += measure_len * float(max(0, measures))
        return float(total)

    total_ticks = _total_score_ticks()
    if total_ticks <= 0.0:
        total_ticks = float(QUARTER_NOTE_UNIT) * 4.0

    op_time = Operator(SHORTEST_DURATION)
    norm_notes: list[dict] = []
    for idx, n in enumerate(notes):
        if not isinstance(n, dict):
            continue
        n_t = float(n.get('time', 0.0) or 0.0)
        n_d = float(n.get('duration', 0.0) or 0.0)
        n_end = n_t + n_d
        p = int(n.get('pitch', 0) or 0)
        hand_raw = str(n.get('hand', 'l') or 'l')
        hand_key = 'l' if hand_raw == 'l' else 'r'
        norm_notes.append(
            {
                'time': n_t,
                'end': n_end,
                'duration': n_d,
                'pitch': p,
                'hand': hand_key,
                'id': int(n.get('_id', 0) or 0),
                'idx': int(idx),
                'raw': n,
            }
        )

    lines: list[dict] = []
    line_breaks = sorted(line_breaks, key=lambda lb: float(lb.get('time', 0.0) or 0.0))
    for i, lb in enumerate(line_breaks):
        lb_time = float(lb.get('time', 0.0) or 0.0)
        next_time = float(line_breaks[i + 1].get('time', total_ticks) or total_ticks) if i + 1 < len(line_breaks) else total_ticks
        if op_time.lt(next_time, lb_time):
            next_time = lb_time
        margin_mm = list(lb.get('margin_mm', [10.0, 10.0]) or [10.0, 10.0])
        if len(margin_mm) < 2:
            margin_mm = [margin_mm[0] if margin_mm else 10.0, 10.0]
        stave_range = lb.get('stave_range', 'auto')
        if stave_range is True:
            stave_range = 'auto'
        if isinstance(stave_range, list) and len(stave_range) >= 2:
            r0 = int(stave_range[0])
            r1 = int(stave_range[1])
            if (r0 == 0 and r1 == 0) or (r0 == 1 and r1 == 1):
                stave_range = 'auto'
        lines.append(
            {
                'time_start': lb_time,
                'time_end': next_time,
                'margin_left': float(margin_mm[0]),
                'margin_right': float(margin_mm[1]),
                'stave_range': stave_range,
                'page_break': bool(lb.get('page_break', False)),
            }
        )

    for line in lines:
        requested_lo = 1
        if line['stave_range'] == 'auto':
            lo = None
            hi = None
            for item in norm_notes:
                n_t = float(item.get('time', 0.0) or 0.0)
                n_end = float(item.get('end', 0.0) or 0.0)
                p = int(item.get('pitch', 0) or 0)
                if op_time.less(n_t, line['time_end']) and op_time.greater(n_end, line['time_start']):
                    if p < 1 or p > PIANO_KEY_AMOUNT:
                        continue
                    lo = p if lo is None else min(lo, p)
                    hi = p if hi is None else max(hi, p)
            if lo is None or hi is None:
                grp = stave_line_groups[clef_group_index]
                visible_keys = list(grp['keys'])
                bound_left = int(visible_keys[0])
                bound_right = int(visible_keys[-1])
            else:
                groups = _visible_line_groups_for_range(int(lo), int(hi), include_clef=True)
                if not groups:
                    grp = stave_line_groups[clef_group_index]
                    visible_keys = list(grp['keys'])
                    bound_left = int(visible_keys[0])
                    bound_right = int(visible_keys[-1])
                else:
                    visible_keys = []
                    for grp in groups:
                        visible_keys.extend(grp['keys'])
                    bound_left = int(visible_keys[0])
                    bound_right = int(visible_keys[-1])
        else:
            manual = _sanitize_range(line['stave_range'])
            requested_lo = int(manual[0]) if manual else 1
            groups = _visible_line_groups_for_range(manual[0], manual[1], include_clef=False)
            if not groups:
                groups = [stave_line_groups[clef_group_index]]
            visible_keys = []
            for grp in groups:
                visible_keys.extend(grp['keys'])
            bound_left = int(visible_keys[0])
            bound_right = int(visible_keys[-1])

        natural_bound_left = int(bound_left)
        natural_bound_right = int(bound_right)
        low_key_present = bool(bound_left <= 2 or (line['stave_range'] != 'auto' and int(requested_lo) <= 2))
        for item in norm_notes:
            n_t = float(item.get('time', 0.0) or 0.0)
            n_end = float(item.get('end', 0.0) or 0.0)
            p = int(item.get('pitch', 0) or 0)
            if n_t >= line['time_end'] or n_end <= line['time_start']:
                continue
            if p in (1, 2, 3):
                low_key_present = True
                break
        a0_ledger_mode = bool(low_key_present and int(natural_bound_left) > 2)

        ledger_bound_left = int(natural_bound_left)
        ledger_bound_right = int(natural_bound_right)
        if a0_ledger_mode:
            ledger_bound_left = min(2, ledger_bound_left)
        if isinstance(line.get('stave_range'), list):
            _manual_bound_group_low = _group_index_for_key(int(natural_bound_left))
            _manual_bound_group_high = _group_index_for_key(int(natural_bound_right))
            for item in norm_notes:
                n_t = float(item.get('time', 0.0) or 0.0)
                n_end = float(item.get('end', 0.0) or 0.0)
                p = int(item.get('pitch', 0) or 0)
                if n_t >= line['time_end'] or n_end <= line['time_start']:
                    continue
                if p < 1 or p > PIANO_KEY_AMOUNT:
                    continue
                g = _group_index_for_key(p)
                if g < _manual_bound_group_low:
                    for grp in stave_line_groups[g:_manual_bound_group_low]:
                        for k in grp.get('keys', []):
                            if int(k) < ledger_bound_left:
                                ledger_bound_left = int(k)
                elif g > _manual_bound_group_high:
                    for grp in stave_line_groups[_manual_bound_group_high + 1:g + 1]:
                        for k in grp.get('keys', []):
                            if int(k) > ledger_bound_right:
                                ledger_bound_right = int(k)

        min_pos = key_positions.get(bound_left, 0.0)
        max_pos = key_positions.get(bound_right, min_pos)
        stave_width = max(0.0, max_pos - min_pos)
        ledger_min_pos = key_positions.get(ledger_bound_left, min_pos)
        ledger_max_pos = key_positions.get(ledger_bound_right, max_pos)
        ledger_left_overhang = max(0.0, float(min_pos) - float(ledger_min_pos))
        stave_plus_ledger_width = max(stave_width, ledger_max_pos - ledger_min_pos)

        base_margin_left = float(line.get('margin_left', 0.0) or 0.0)
        ts_segments: list[dict[str, float | int | list[int] | bool]] = []
        ts_cursor = 0.0
        for bg in base_grid:
            numer = int(bg.get('numerator', 4) or 4)
            denom = int(bg.get('denominator', 4) or 4)
            measures = int(bg.get('measure_amount', 1) or 1)
            beat_grouping = list(bg.get('beat_grouping', []) or [])
            indicator_enabled = bool(bg.get('indicator_enabled', True))
            if measures <= 0:
                continue
            measure_len = float(numer) * (4.0 / float(max(1, denom))) * float(QUARTER_NOTE_UNIT)
            ts_segments.append(
                {
                    'start': float(ts_cursor),
                    'measure_len': float(measure_len),
                    'numerator': int(numer),
                    'denominator': int(denom),
                    'measure_amount': int(measures),
                    'beat_grouping': beat_grouping,
                    'indicator_enabled': bool(indicator_enabled),
                }
            )
            ts_cursor += measure_len * float(measures)

        ts_lane_width = 0.0
        ts_lane_right_offset = 0.0
        ts_lane_padding_mm = 0.0
        ts_segments_in_line = [
            seg
            for seg in ts_segments
            if bool(seg.get('indicator_enabled', True))
            and op_time.ge(float(seg.get('start', 0.0) or 0.0), float(line['time_start']))
            and op_time.lt(float(seg.get('start', 0.0) or 0.0), float(line['time_end']))
        ]
        if ts_segments_in_line:
            ts_lane_width_raw = layout.get('time_signature_indicator_lane_width_mm', 22.0)
            ts_lane_width = float(ts_lane_width_raw or 22.0) * scale
            ts_lane_padding_mm = 1.0
            min_pitch = None
            for seg in ts_segments_in_line:
                win_start = float(seg.get('start', 0.0) or 0.0)
                win_end = win_start + float(seg.get('measure_len', 0.0) or 0.0)
                for item in norm_notes:
                    n_t = float(item.get('time', 0.0) or 0.0)
                    n_end = float(item.get('end', 0.0) or 0.0)
                    if n_t >= line['time_end'] or n_end <= line['time_start']:
                        continue
                    if n_t < win_end and n_end > win_start:
                        p = int(item.get('pitch', 0) or 0)
                        if 1 <= p <= PIANO_KEY_AMOUNT:
                            min_pitch = p if min_pitch is None else min(min_pitch, p)
            if min_pitch is not None:
                stem_len_units = float(layout.get('note_stem_length_semitone', 3) or 3)
                stem_len_mm = stem_len_units * semitone_mm
                origin = float(key_positions.get(bound_left, 0.0))
                note_offset = float(key_positions.get(min_pitch, origin)) - origin
                offset_left = note_offset - stem_len_mm
                ts_lane_gap_mm = semitone_mm
                ts_lane_right_offset = min(0.0, float(offset_left - ts_lane_gap_mm))

        line['visible_keys'] = list(visible_keys)
        line['low_key_left'] = bool(low_key_present)
        line['a0_ledger_mode'] = bool(a0_ledger_mode)
        line['natural_bound_left'] = int(natural_bound_left)
        line['natural_bound_right'] = int(natural_bound_right)
        line['ledger_bound_left'] = int(ledger_bound_left)
        line['ledger_bound_right'] = int(ledger_bound_right)
        line['range'] = [int(bound_left), int(bound_right)]
        line['stave_width'] = float(stave_width)
        line['stave_plus_ledger_width'] = float(stave_plus_ledger_width)
        line['ledger_left_overhang'] = float(ledger_left_overhang)
        line['base_margin_left'] = float(base_margin_left)
        line['margin_left'] = float(base_margin_left)
        line['ts_lane_width'] = float(ts_lane_width)
        line['ts_lane_right_offset'] = float(ts_lane_right_offset)
        line['ts_lane_padding_mm'] = float(ts_lane_padding_mm)
        line['total_width'] = float(line['margin_left'] + stave_plus_ledger_width + line['margin_right'])

    def _available_width_for_page(page_index: int) -> float:
        left_reserve, right_reserve = _line_axis_reserves_for_page(page_index)
        return max(1e-6, page_w - page_left - page_right - left_reserve - right_reserve)

    pages: list[list[dict]] = []
    cur_page: list[dict] = []
    cur_width = 0.0
    for line in lines:
        cur_available_width = _available_width_for_page(len(pages))
        if line.get('page_break', False):
            if cur_page:
                pages.append(cur_page)
            elif not pages:
                pages.append([])
            cur_page = []
            cur_width = 0.0
            cur_available_width = _available_width_for_page(len(pages))
        if cur_page and (cur_width + float(line['total_width'])) > cur_available_width:
            pages.append(cur_page)
            cur_page = []
            cur_width = 0.0
            cur_available_width = _available_width_for_page(len(pages))
        cur_page.append(line)
        cur_width += float(line['total_width'])
    if cur_page:
        pages.append(cur_page)
    if not pages:
        pages = [[]]

    analysis_snapshot = Analysis.compute(score, lines_count=len(lines), pages_count=len(pages))

    first_system_start = float(lines[0].get('time_start', 0.0) or 0.0) if lines else 0.0
    print_time_map: list[list[dict]] = []
    for page_index, page in enumerate(pages):
        page_lines = list(reversed(page)) if horizontal_read_direction else page
        if horizontal_read_direction:
            y_top = float(page_top)
            y_bottom = float(page_h - page_bottom)
        else:
            header_offset = float(header_height) if page_index == 0 else 0.0
            y_top = float(page_top + header_offset)
            y_bottom = float(page_h - page_bottom - footer_height)
        left_reserve, right_reserve = _line_axis_reserves_for_page(page_index)
        available_width = _available_width_for_page(page_index)
        used_width = sum(float(l['total_width']) for l in page_lines)
        leftover = max(0.0, available_width - used_width)
        gap = leftover / float(len(page_lines) + 1) if page_lines else 0.0
        x_cursor = page_left + left_reserve + gap
        lines_map: list[dict] = []
        for line_index, line in enumerate(page_lines):
            ledger_left_overhang = float(line.get('ledger_left_overhang', 0.0) or 0.0)
            line_x_start = x_cursor + float(line['margin_left']) + ledger_left_overhang
            line_x_end = line_x_start + float(line['stave_width'])
            visual_first_line = (line_index == len(page_lines) - 1) if horizontal_read_direction else (line_index == 0)
            mini_piano_enabled = bool(layout.get('mini_piano_visible', True)) and page_index == 0 and visual_first_line
            mini_piano_height_mm = (7.0 * float(semitone_mm)) if mini_piano_enabled else 0.0
            y2_draw = max(y_top + 1.0, y_bottom - mini_piano_height_mm) if mini_piano_enabled else y_bottom
            if y_bottom <= y_top:
                y_bottom = y_top + 1.0
            if y2_draw <= y_top:
                y2_draw = y_top + 1.0
            line_time_start = float(line.get('time_start', 0.0) or 0.0)
            line_time_end = float(line.get('time_end', 0.0) or 0.0)
            line_span_ticks = max(1e-6, float(line_time_end - line_time_start))
            line_span_mm = max(1e-6, float(y2_draw - y_top))
            is_first_system_line = op_time.eq(line_time_start, first_system_start)
            pre_roll_ticks = 0.0
            if is_first_system_line:
                pre_roll_ticks = max(0.0, line_span_ticks * max(0.0, float(y_top)) / line_span_mm)
            line['y_top'] = float(y_top)
            line['y_bottom'] = float(y2_draw)
            line['mini_piano_visible'] = bool(mini_piano_enabled)
            line['mini_piano_height_mm'] = float(mini_piano_height_mm)
            line['mini_piano_y_top'] = float(y2_draw)
            line['mini_piano_y_bottom'] = float(y_bottom)
            line['line_time_start_render'] = float(line_time_start - pre_roll_ticks)
            lines_map.append(
                {
                    'time_start': float(line.get('time_start', 0.0)),
                    'time_end': float(line.get('time_end', 0.0)),
                    'y_top': float(y_top),
                    'y_bottom': float(y2_draw),
                    'x_start': float(line_x_start),
                    'x_end': float(line_x_end),
                }
            )
            x_cursor += float(line['total_width']) + gap
        print_time_map.append(lines_map)

    return {
        'score': score,
        'layout': layout,
        'page_width_mm': float(page_w),
        'page_height_mm': float(page_h),
        'page_left': float(page_left),
        'page_right': float(page_right),
        'page_top': float(page_top),
        'page_bottom': float(page_bottom),
        'horizontal_read_direction': bool(horizontal_read_direction),
        'semitone_mm': float(semitone_mm),
        'pages': pages,
        'print_time_map': print_time_map,
        'analysis': analysis_snapshot,
    }
