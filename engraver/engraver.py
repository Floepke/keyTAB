from PySide6 import QtCore
from datetime import datetime
import bisect, math
import multiprocessing as mp
import queue
from ui.widgets.draw_util import DrawUtil
from utils.CONSTANT import BE_KEYS, QUARTER_NOTE_UNIT, PIANO_KEY_AMOUNT, SHORTEST_DURATION, hex_to_rgba, BLACK_KEYS, ENGRAVER_FRACTIONAL_SCALE_CORRECTION
from utils.tiny_tool import key_class_filter
from utils.operator import Operator
from file_model.SCORE import SCORE
from file_model.info import Info
from file_model.analysis import Analysis

_MP_CONTEXT = mp.get_context("spawn")

def do_engrave(score: SCORE, du: DrawUtil, pageno: int = 0, pdf_export: bool = False) -> None:
    """Compute a full print layout and draw commands into DrawUtil.

    Problem solved: the engraver must be deterministic and thread-safe.
    It converts the score model into page/line geometry without any Qt
    rendering calls, then records only DrawUtil primitives.
    """
    score: SCORE = score or {}
    meta_data = (score.get('meta_data', {}) or {})
    layout = (score.get('layout', {}) or {})
    info = (score.get('info', {}) or {})
    default_info = Info()
    editor = (score.get('editor', {}) or {})
    events = (score.get('events', {}) or {})
    base_grid = list(score.get('base_grid', []) or [])
    line_breaks = list(events.get('line_break', []) or [])
    notes = list(events.get('note', []) or [])
    grace_notes = list(events.get('grace_note', []) or [])
    count_lines = list(events.get('count_line', []) or [])
    beam_markers = list(events.get('beam', []) or [])
    slurs = list(events.get('slur', []) or [])
    texts = list(events.get('text', []) or [])

    # Problem solved: beam markers are organized per hand for fast grouping later.
    beam_by_hand: dict[str, list[dict]] = {'l': [], 'r': []}
    for b in beam_markers:
        if not isinstance(b, dict):
            continue
        bt = float(b.get('time', 0.0) or 0.0)
        bd = float(b.get('duration', 0.0) or 0.0)
        hand_raw = str(b.get('hand', '<') or '<')
        hand_key = 'l' if hand_raw in ('<', 'l') else 'r'
        beam_by_hand[hand_key].append({'time': bt, 'duration': bd})
    for hk in beam_by_hand:
        beam_by_hand[hk] = sorted(beam_by_hand[hk], key=lambda m: float(m.get('time', 0.0)))

    # Problem solved: normalize notes once to avoid repeated dict parsing in loops.
    norm_notes: list[dict] = []
    notes_by_hand: dict[str, list[dict]] = {'<': [], '>': []}
    starts_by_hand: dict[str, list[float]] = {'<': [], '>': []}
    for idx, n in enumerate(notes):
        if not isinstance(n, dict):
            continue
        n_t = float(n.get('time', 0.0) or 0.0)
        n_d = float(n.get('duration', 0.0) or 0.0)
        n_end = n_t + n_d
        p = int(n.get('pitch', 0) or 0)
        hand_raw = str(n.get('hand', '<') or '<')
        hand_key = '<' if hand_raw in ('<', 'l') else '>'
        item = {
            'time': n_t,
            'end': n_end,
            'duration': n_d,
            'pitch': p,
            'hand': hand_key,
            'id': int(n.get('_id', 0) or 0),
            'idx': int(idx),
            'raw': n,
        }
        norm_notes.append(item)
        notes_by_hand[hand_key].append(item)
        starts_by_hand[hand_key].append(n_t)

    for hk in notes_by_hand:
        notes_by_hand[hk] = sorted(notes_by_hand[hk], key=lambda m: float(m.get('time', 0.0) or 0.0))
    for hk in starts_by_hand:
        starts_by_hand[hk] = sorted(starts_by_hand[hk])

    # Normalize grace notes (time + pitch only)
    norm_grace: list[dict] = []
    for idx, g in enumerate(grace_notes):
        if not isinstance(g, dict):
            continue
        g_t = float(g.get('time', 0.0) or 0.0)
        p = int(g.get('pitch', 0) or 0)
        norm_grace.append({
            'time': g_t,
            'pitch': p,
            'id': int(g.get('_id', 0) or 0),
            'idx': int(idx),
        })
    norm_grace = sorted(norm_grace, key=lambda m: float(m.get('time', 0.0) or 0.0))

    norm_slurs: list[dict] = []
    for idx, s in enumerate(slurs):
        if not isinstance(s, dict):
            continue
        norm_slurs.append({
            'x1_rpitch': int(s.get('x1_rpitch', 0) or 0),
            'y1_time': float(s.get('y1_time', 0.0) or 0.0),
            'x2_rpitch': int(s.get('x2_rpitch', 0) or 0),
            'y2_time': float(s.get('y2_time', 0.0) or 0.0),
            'x3_rpitch': int(s.get('x3_rpitch', 0) or 0),
            'y3_time': float(s.get('y3_time', 0.0) or 0.0),
            'x4_rpitch': int(s.get('x4_rpitch', 0) or 0),
            'y4_time': float(s.get('y4_time', 0.0) or 0.0),
            'id': int(s.get('_id', 0) or 0),
            'idx': int(idx),
        })
    if norm_slurs:
        norm_slurs = sorted(norm_slurs, key=lambda m: float(m.get('y1_time', 0.0) or 0.0))

    norm_texts: list[dict] = []
    for idx, t in enumerate(texts):
        if not isinstance(t, dict):
            continue
        norm_texts.append({
            'time': float(t.get('time', 0.0) or 0.0),
            'x_rpitch': float(t.get('x_rpitch', 0) or 0),
            'rotation': float(t.get('rotation', 0.0) or 0.0),
            'x_offset_mm': float(t.get('x_offset_mm', 0.0) or 0.0),
            'y_offset_mm': float(t.get('y_offset_mm', 0.0) or 0.0),
            'text': str(t.get('text', '') or ''),
            'font': t.get('font', None),
            'use_custom_font': bool(t.get('use_custom_font', False)),
            'id': int(t.get('_id', 0) or 0),
            'idx': int(idx),
        })
    if norm_texts:
        norm_texts = sorted(norm_texts, key=lambda m: float(m.get('time', 0.0) or 0.0))

    # Problem solved: materialize layout values early to keep math predictable.
    page_w = float(layout.get('page_width_mm', 210.0) or 210.0)
    page_h = float(layout.get('page_height_mm', 297.0) or 297.0)
    page_left = float(layout.get('page_left_margin_mm', 5.0) or 5.0)
    page_right = float(layout.get('page_right_margin_mm', 5.0) or 5.0)
    page_top = float(layout.get('page_top_margin_mm', 10.0) or 10.0)
    page_bottom = float(layout.get('page_bottom_margin_mm', 10.0) or 10.0)
    scale = float(layout.get('scale', 1.0) or 1.0)
    stave_two_w = float(layout.get('stave_two_line_thickness_mm', 0.5) or 0.5) * scale
    stave_three_w = float(layout.get('stave_three_line_thickness_mm', 0.5) or 0.5) * scale
    clef_dash = list(layout.get('stave_clef_line_dash_pattern_mm', []) or [])
    if clef_dash:
        clef_dash = [float(v) * scale for v in clef_dash]
    op_time = Operator(SHORTEST_DURATION)
    barline_positions: list[float] = []
    cur_bar = 0.0
    for bg in base_grid:
        numer = int(bg.get('numerator', 4) or 4)
        denom = int(bg.get('denominator', 4) or 4)
        measures = int(bg.get('measure_amount', 1) or 1)
        measure_len = float(numer) * (4.0 / float(max(1, denom))) * float(QUARTER_NOTE_UNIT)
        for _ in range(int(max(0, measures))):
            barline_positions.append(float(cur_bar))
            cur_bar += measure_len

    # Problem solved: precompute time signature segments for lane rendering.
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

    # Problem solved: precompute measure windows to number measures consistently.
    measure_windows: list[dict[str, float | int]] = []
    m_idx = 1
    cur_m = 0.0
    for bg in base_grid:
        numer = int(bg.get('numerator', 4) or 4)
        denom = int(bg.get('denominator', 4) or 4)
        measures = int(bg.get('measure_amount', 1) or 1)
        measure_len = float(numer) * (4.0 / float(max(1, denom))) * float(QUARTER_NOTE_UNIT)
        for _ in range(int(max(0, measures))):
            measure_windows.append({'start': float(cur_m), 'end': float(cur_m + measure_len), 'number': int(m_idx)})
            m_idx += 1
            cur_m += measure_len

    def _normalize_hex_color(value: str | None) -> str | None:
        """Normalize hex color strings and allow special hand markers."""
        if value is None:
            return None
        txt = str(value).strip()
        if not txt:
            return None
        if txt in ('<', '>'):
            return txt
        if not txt.startswith('#'):
            txt = f"#{txt}"
        hex_part = txt[1:]
        if len(hex_part) not in (3, 6, 8):
            return None
        if not all(c in '0123456789abcdefABCDEF' for c in hex_part):
            return None
        if len(hex_part) == 3:
            hex_part = ''.join(c * 2 for c in hex_part)
        if len(hex_part) == 8:
            hex_part = hex_part[:6]
        return f"#{hex_part}"

    def _hex_to_rgba01(hex_color: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
        """Convert a hex color into RGBA floats in the 0..1 range."""
        rgba = hex_to_rgba(hex_color, alpha)
        r, g, b, a = rgba
        return (float(r) / 255.0, float(g) / 255.0, float(b) / 255.0, float(a))

    def _allow_font_registry() -> bool:
        """Return True when it is safe to access QFontDatabase (GUI process only)."""
        return mp.current_process().name == "MainProcess"

    def _resolve_font_family(family: str) -> str:
        """Resolve a font family name with the font registry if available."""
        if not _allow_font_registry():
            return family
        from fonts import resolve_font_family
        return str(resolve_font_family(family))

    def _layout_font(key: str, fallback_family: str, fallback_size: float) -> tuple[str, float, bool, bool]:
        """Fetch a layout font entry from the layout dict with fallback values."""
        raw = layout.get(key, {}) if isinstance(layout, dict) else {}
        if not isinstance(raw, dict):
            raw = {}
        family = str(raw.get('family', fallback_family) or fallback_family)
        if family == 'Edwin' and _allow_font_registry():
            from fonts import register_font_from_bytes
            reg = register_font_from_bytes('Edwin')
            if reg:
                family = str(reg)
        family = _resolve_font_family(family)
        size_pt = float(raw.get('size_pt', fallback_size) or fallback_size)
        bold = bool(raw.get('bold', False))
        italic = bool(raw.get('italic', False))
        return family, size_pt, bold, italic

    def _info_text(key: str, fallback: str) -> str:
        """Fetch info text with a fallback, always returning a string."""
        if isinstance(info, dict):
            raw = info.get(key, fallback)
        else:
            raw = fallback
        if isinstance(raw, dict):
            raw = raw.get('text', fallback)
        return str(raw) if raw is not None else str(fallback)

    def _info_font(key: str, fallback_family: str, fallback_size: float) -> tuple[str, float, bool, bool, float, float]:
        """Fetch info font settings from layout (family, size, style, offsets)."""
        family, size_pt, bold, italic = _layout_font(key, fallback_family, fallback_size)
        raw_font = layout.get(key, {}) if isinstance(layout, dict) else {}
        if not isinstance(raw_font, dict):
            raw_font = {}
        x_off = float(raw_font.get('x_offset', 0.0) or 0.0)
        y_off = float(raw_font.get('y_offset', 0.0) or 0.0)
        return family, size_pt, bold, italic, x_off, y_off

    def _assign_groups(notes_sorted: list[dict], windows: list[tuple[float, float]]) -> list[list[dict]]:
        """Assign notes to time windows by overlap and preserve start-time order.

        Problem solved: beam grouping must be stable even when notes straddle
        a window boundary; this uses overlap tests plus de-duplication.
        """
        if not notes_sorted or not windows:
            return []
        starts = [float(n.get('time', 0.0) or 0.0) for n in notes_sorted]
        ends = [float(n.get('end', 0.0) or 0.0) for n in notes_sorted]
        result: list[list[dict]] = []
        j = 0
        for (t0, t1) in windows:
            j = bisect.bisect_left(starts, float(t0) - float(op_time.threshold), j)
            group: list[dict] = []
            k = j
            while k < len(starts):
                s = starts[k]
                if op_time.ge(s, float(t1) + float(op_time.threshold)):
                    break
                e = ends[k]
                if op_time.gt(e, float(t0)) and op_time.lt(s, float(t1)):
                    group.append(notes_sorted[k])
                k += 1
            b = j - 1
            while b >= 0:
                s = starts[b]
                e = ends[b]
                if op_time.gt(e, float(t0)) and op_time.lt(s, float(t1)):
                    group.append(notes_sorted[b])
                b -= 1
            if group:
                keyed: dict[int, dict] = {}
                for m in group:
                    key_id = int(m.get('idx', m.get('id', 0)) or 0)
                    keyed[key_id] = m
                group = sorted(keyed.values(), key=lambda n: float(n.get('time', 0.0) or 0.0))
            result.append(group)
        return result

    def _build_grid_windows(a: float, b: float) -> list[tuple[float, float]]:
        """Build time windows using base grid beat grouping between a and b.

        Problem solved: derive beam groups from musical beat grouping, not
        from raw timestamps, so the visual grouping matches the score grid.
        """
        windows: list[tuple[float, float]] = []
        cur = 0.0
        for bg in base_grid:
            numer = int(bg.get('numerator', 4) or 4)
            denom = int(bg.get('denominator', 4) or 4)
            measures = int(bg.get('measure_amount', 1) or 1)
            seq = list(bg.get('beat_grouping', []) or [])
            measure_len = float(numer) * (4.0 / float(max(1, denom))) * float(QUARTER_NOTE_UNIT)
            beat_len = measure_len / max(1, int(numer))
            full_group = len(seq) == numer and [int(v) for v in seq] == list(range(1, numer + 1))
            for _ in range(int(measures)):
                m_start = float(cur)
                m_end = float(cur + measure_len)
                if op_time.lt(m_end, float(a)):
                    cur = m_end
                    continue
                if op_time.gt(m_start, float(b)):
                    cur = m_end
                    continue
                if len(seq) != numer:
                    seq = [1]
                if full_group:
                    group_starts = list(range(1, numer + 1))
                else:
                    group_starts = [i for i, v in enumerate(seq, start=1) if int(v) == 1]
                    if not group_starts or group_starts[0] != 1:
                        group_starts = [1] + group_starts
                for gi, s in enumerate(group_starts):
                    e = (group_starts[gi + 1] - 1) if (gi + 1) < len(group_starts) else numer
                    w0 = m_start + (s - 1) * beat_len
                    w1 = m_start + float(e) * beat_len
                    w0 = max(float(a), w0)
                    w1 = min(float(b), w1)
                    if op_time.lt(w0, w1):
                        windows.append((w0, w1))
                cur = m_end
        return windows

    def _build_duration_windows(start: float, end: float, dur: float) -> list[tuple[float, float]]:
        """Build consecutive windows of fixed duration between start and end.

        Problem solved: marker-defined windows can override grid grouping with
        explicit durations, enabling custom beam spans.
        """
        if dur <= 0:
            return [(start, end)]
        windows: list[tuple[float, float]] = []
        t = float(start)
        while op_time.lt(t, float(end)):
            t1 = min(float(end), t + float(dur))
            windows.append((t, t1))
            t = t1
        return windows

    def _process_beam_marker_override(default_windows: list[tuple[float, float]], markers: list[dict]) -> list[tuple[float, float]]:
        """Replace default windows with marker spans where they overlap.

        - Start from time-signature (grid) windows.
        - For each marker, drop any default window that overlaps its span and add the marker span.
        - Non-positive duration markers only remove overlapping defaults.
        """
        if not default_windows:
            return []
        if not markers:
            return default_windows
        windows = sorted(default_windows, key=lambda w: float(w[0]))
        for mk in sorted(markers, key=lambda m: float(m.get('time', 0.0))):
            mt = float(mk.get('time', 0.0) or 0.0)
            dur = float(mk.get('duration', 0.0) or 0.0)
            end = mt + max(0.0, dur)
            filtered: list[tuple[float, float]] = []
            for (w0, w1) in windows:
                # Keep windows that do NOT overlap the marker span
                if op_time.ge(w0, end) or op_time.le(w1, mt):
                    filtered.append((w0, w1))
            if dur > 0.0:
                filtered.append((mt, end))
            windows = sorted(filtered, key=lambda w: float(w[0]))
        return windows

    def _group_by_beam_markers(notes: list[dict], markers: list[dict], start: float, end: float) -> tuple[list[list[dict]], list[tuple[float, float]]]:
        """Split notes into beam groups using grid windows with marker overrides."""
        notes_sorted = sorted(notes, key=lambda n: float(n.get('time', 0.0) or 0.0)) if notes else []
        default_windows = _build_grid_windows(start, end)
        windows = _process_beam_marker_override(default_windows, markers)
        groups = _assign_groups(notes_sorted, windows) if notes_sorted else []
        return groups, windows

    def _black_note_above_stem(item: dict, rule: str, notes: list[dict], op: Operator) -> bool:
        if rule == 'above_stem':
            return True
        p0 = int(item.get('pitch', 0) or 0)
        t0 = float(item.get('time', 0.0) or 0.0)
        idx0 = int(item.get('idx', -1) or -1)
        if rule in ('above_stem_if_collision', 'only_above_stem_if_collision'):
            for m in notes:
                if int(m.get('idx', -2) or -2) == idx0:
                    continue
                if not op.eq(float(m.get('time', 0.0) or 0.0), t0):
                    continue
                if abs(int(m.get('pitch', 0) or 0) - p0) == 1:
                    return True
            return False
        if rule == 'above_stem_if_chord_and_white_note':
            for m in notes:
                if int(m.get('idx', -2) or -2) == idx0:
                    continue
                if not op.eq(float(m.get('time', 0.0) or 0.0), t0):
                    continue
                mp = int(m.get('pitch', 0) or 0)
                if mp not in BLACK_KEYS and mp != p0:
                    return True
            return False
        if rule != 'above_stem_if_chord_and_white_note_same_hand':
            return False
        hand0 = str(item.get('hand', '<') or '<')
        for m in notes:
            if int(m.get('idx', -2) or -2) == idx0:
                continue
            if not op.eq(float(m.get('time', 0.0) or 0.0), t0):
                continue
            if str(m.get('hand', '<') or '<') != hand0:
                continue
            mp = int(m.get('pitch', 0) or 0)
            if mp not in BLACK_KEYS and mp != p0:
                return True
        return False

    def _has_followed_rest(item: dict) -> bool:
        """Return True when a note has no immediate following note in its hand.

        Problem solved: stop-signs should mark a gap in the same hand, not
        simply the end of a note.
        """
        hand_key = str(item.get('hand', '<') or '<')
        hand_list = notes_by_hand.get(hand_key, [])
        starts = starts_by_hand.get(hand_key, [])
        if not hand_list or not starts:
            return True
        end = float(item.get('end', 0.0) or 0.0)
        thr = float(op_time.threshold)
        idx = bisect.bisect_left(starts, float(end - thr))
        min_delta = None
        for j in range(idx, len(hand_list)):
            m = hand_list[j]
            if int(m.get('idx', -1) or -1) == int(item.get('idx', -2) or -2):
                continue
            delta = float(m.get('time', 0.0) or 0.0) - end
            if delta >= -thr:
                min_delta = delta
                break
        if min_delta is None:
            return True
        return op_time.gt(float(min_delta), 0.0)

    # Problem solved: reset DrawUtil pages so the engrave output is fresh.
    du._pages = []
    du._current_index = -1

    def _total_score_ticks() -> float:
        """Compute total score duration in ticks from base grid segments."""
        total = 0.0
        for bg in base_grid:
            numer = int(bg.get('numerator', 4) or 4)
            denom = int(bg.get('denominator', 4) or 4)
            measures = int(bg.get('measure_amount', 1) or 1)
            measure_len = float(numer) * (4.0 / float(max(1, denom))) * float(QUARTER_NOTE_UNIT)
            total += measure_len * float(max(0, measures))
        return float(total)

    def _line_break_defaults() -> dict:
        """Return default line break settings used when none exist."""
        return {
            'time': 0.0,
            'margin_mm': [10.0, 10.0],
            'stave_range': 'auto',
            'page_break': False,
        }

    def _sanitize_range(rng) -> list[int]:
        """Clamp and normalize a stave range to valid piano keys."""
        if not isinstance(rng, list) or len(rng) < 2:
            return [1, PIANO_KEY_AMOUNT]
        lo = int(rng[0])
        hi = int(rng[1])
        lo = max(1, min(PIANO_KEY_AMOUNT, lo))
        hi = max(1, min(PIANO_KEY_AMOUNT, hi))
        if hi < lo:
            lo, hi = hi, lo
        return [lo, hi]

    def _pc_char(key: int) -> str:
        """Map a piano key number to a pitch-class character for grouping."""
        pc = (int(key) - 1) % 12
        if pc in (0, 2, 3, 5, 7, 8, 10):
            return {0: 'a', 2: 'b', 3: 'c', 5: 'd', 7: 'e', 8: 'f', 10: 'g'}[pc]
        return {1: 'A', 4: 'C', 6: 'D', 9: 'F', 11: 'G'}[pc]

    line_keys = sorted(key_class_filter('ACDFG'))

    def _build_line_groups() -> list[dict]:
        """Build clef-related line groups and their key ranges.

        Problem solved: map piano keys into vertical stave groups so ledger
        lines can be shown or suppressed predictably.
        """
        groups: list[dict] = []
        used: set[int] = set()

        def _next_index(start: int, pc_target: str) -> int | None:
            """Find the next unused key index matching a pitch-class target."""
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

        # Sort groups by pitch
        groups.sort(key=lambda g: g['keys'][0])

        # Assign membership ranges based on midpoints between groups
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

    line_groups = _build_line_groups()
    if not line_groups:
        line_groups = [{'keys': [41, 43], 'range_low': 1, 'range_high': PIANO_KEY_AMOUNT, 'pattern': 'c'}]
    clef_group_index = 0
    for i, grp in enumerate(line_groups):
        if 41 in grp['keys'] and 43 in grp['keys']:
            clef_group_index = i
            break

    def _group_index_for_key(key: int) -> int:
        """Return the line group index for a key using precomputed ranges."""
        if not line_groups:
            return 0
        for i, grp in enumerate(line_groups):
            if grp['range_low'] <= key <= grp['range_high']:
                return i
        return 0 if key <= line_groups[0]['range_low'] else len(line_groups) - 1

    def _note_range_for_window(t0: float, t1: float) -> tuple[int | None, int | None]:
        """Find the lowest and highest pitches overlapping a time window.

        Problem solved: auto range must reflect actual notes in the window.
        """
        lo = None
        hi = None
        for n in notes:
            n_t = float(n.get('time', 0.0) or 0.0)
            n_d = float(n.get('duration', 0.0) or 0.0)
            n_end = n_t + n_d
            p = int(n.get('pitch', 0) or 0)
            if op_time.lt(n_t, t1) and op_time.gt(n_end, t0):
                if p < 1 or p > PIANO_KEY_AMOUNT:
                    continue
                lo = p if lo is None else min(lo, p)
                hi = p if hi is None else max(hi, p)
        return lo, hi

    def _visible_line_groups_for_range(lo: int, hi: int, include_clef: bool = True) -> list[dict]:
        """Return line groups that cover a pitch range; optionally include clef group.

        Problem solved: when manual ranges omit the clef, we still allow
        precise, minimal stave groups.
        """
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

        return [line_groups[gi] for gi in range(min_group, max_group + 1)]

    def _auto_line_keys_and_bounds(t0: float, t1: float) -> tuple[list[dict], list[int], int, int, bool, str]:
        """Choose stave keys and bounds automatically for a time window.

        Problem solved: auto range must include the clef group and handle
        empty windows without crashing.
        """
        lo, hi = _note_range_for_window(t0, t1)
        if lo is None or hi is None:
            grp = line_groups[clef_group_index]
            keys = list(grp['keys'])
            return [grp], keys, int(keys[0]), int(keys[-1]), True, grp.get('pattern', 'c')
        groups = _visible_line_groups_for_range(lo, hi, include_clef=True)
        if not groups:
            grp = line_groups[clef_group_index]
            keys = list(grp['keys'])
            return [grp], keys, int(keys[0]), int(keys[-1]), True, grp.get('pattern', 'c')
        keys: list[int] = []
        patterns: list[str] = []
        for grp in groups:
            keys.extend(grp['keys'])
            patterns.append(str(grp.get('pattern', '')))
        return groups, keys, int(keys[0]), int(keys[-1]), False, ' '.join(patterns)

    def _notes_in_window_stats(t0: float, t1: float) -> tuple[int, int | None, int | None]:
        """Return note count and pitch bounds overlapping a time window."""
        count = 0
        lo = None
        hi = None
        for n in notes:
            n_t = float(n.get('time', 0.0) or 0.0)
            n_d = float(n.get('duration', 0.0) or 0.0)
            n_end = n_t + n_d
            p = int(n.get('pitch', 0) or 0)
            if op_time.lt(n_t, t1) and op_time.gt(n_end, t0) and 1 <= p <= PIANO_KEY_AMOUNT:
                count += 1
                lo = p if lo is None else min(lo, p)
                hi = p if hi is None else max(hi, p)
        return count, lo, hi

    def _build_key_positions(start_key: int, end_key: int, semitone_mm: float) -> dict[int, float]:
        """Build x positions for keys, adding extra spacing after B/E.

        Problem solved: klavarskribo spacing needs extra gaps after B and E
        to keep black key groups visually balanced.
        """
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

    total_ticks = _total_score_ticks()
    if total_ticks <= 0.0:
        total_ticks = float(QUARTER_NOTE_UNIT) * 4.0
    if not line_breaks:
        line_breaks = [_line_break_defaults()]

    line_breaks = sorted(line_breaks, key=lambda lb: float(lb.get('time', 0.0) or 0.0))

    # Problem solved: convert line break events into contiguous line windows.
    lines = []
    for i, lb in enumerate(line_breaks):
        lb_time = float(lb.get('time', 0.0) or 0.0)
        next_time = float(line_breaks[i + 1].get('time', total_ticks) or total_ticks) if i + 1 < len(line_breaks) else total_ticks
        if next_time < lb_time:
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
        line = {
            'time_start': lb_time,
            'time_end': next_time,
            'margin_left': float(margin_mm[0]),
            'margin_right': float(margin_mm[1]),
            'stave_range': stave_range,
            'page_break': bool(lb.get('page_break', False)),
        }
        lines.append(line)


    # Problem solved: compute per-line horizontal geometry (margins, ranges).
    semitone_mm = 2.5 * scale
    key_positions = _build_key_positions(1, PIANO_KEY_AMOUNT, semitone_mm)
    for line in lines:
        if line['stave_range'] == 'auto':
            groups, keys, bound_left, bound_right, empty, pattern = _auto_line_keys_and_bounds(line['time_start'], line['time_end'])
            line['visible_keys'] = keys
            line['pattern'] = pattern
            if empty:
                count, lo, hi = _notes_in_window_stats(line['time_start'], line['time_end'])
        else:
            manual = _sanitize_range(line['stave_range'])
            groups = _visible_line_groups_for_range(manual[0], manual[1], include_clef=False)
            if not groups:
                grp = line_groups[clef_group_index]
                groups = [grp]
            keys: list[int] = []
            patterns: list[str] = []
            for grp in groups:
                keys.extend(grp['keys'])
                patterns.append(str(grp.get('pattern', '')))
            bound_left = int(keys[0])
            bound_right = int(keys[-1])
            line['visible_keys'] = keys
            line['pattern'] = ' '.join(patterns)
        # Problem solved: avoid clipping A#0 ledger by forcing left edge to key 2.
        low_key_present = False
        for item in norm_notes:
            n_t = float(item.get('time', 0.0) or 0.0)
            n_end = float(item.get('end', 0.0) or 0.0)
            p = int(item.get('pitch', 0) or 0)
            if op_time.ge(n_t, float(line['time_end'])) or op_time.le(n_end, float(line['time_start'])):
                continue
            if p in (1, 2, 3):
                low_key_present = True
                break
        if low_key_present:
            bound_left = 2
        line['low_key_left'] = bool(low_key_present)
        line['range'] = [int(bound_left), int(bound_right)]
        min_pos = key_positions.get(bound_left, 0.0)
        max_pos = key_positions.get(bound_right, min_pos)
        stave_width = max(0.0, max_pos - min_pos)
        line['stave_width'] = float(stave_width)
        base_margin_left = float(line.get('margin_left', 0.0) or 0.0)
        ts_lane_width = 0.0
        ts_lane_right_offset = 0.0
        ts_lane_padding_mm = 0.0
        # Problem solved: if time-signature indicators would collide with notes,
        # expand left margin to reserve a lane.
        ts_segments_in_line = [
            seg
            for seg in ts_segments
            if bool(seg.get('indicator_enabled', True))
            and op_time.ge(float(seg.get('start', 0.0) or 0.0), float(line['time_start']))
            and op_time.lt(float(seg.get('start', 0.0) or 0.0), float(line['time_end']))
        ]
        if ts_segments_in_line:
            ts_lane_width = float(layout.get('time_signature_indicator_lane_width_mm', 22.0) or 22.0)
            ts_lane_padding_mm = 2.5  # Hard-coded right padding so lane ends before the stave.
            min_pitch = None
            for seg in ts_segments_in_line:
                win_start = float(seg.get('start', 0.0) or 0.0)
                win_end = win_start + float(seg.get('measure_len', 0.0) or 0.0)
                for item in norm_notes:
                    n_t = float(item.get('time', 0.0) or 0.0)
                    n_end = float(item.get('end', 0.0) or 0.0)
                    if op_time.ge(n_t, float(line['time_end'])) or op_time.le(n_end, float(line['time_start'])):
                        continue
                    if op_time.lt(n_t, win_end) and op_time.gt(n_end, win_start):
                        p = int(item.get('pitch', 0) or 0)
                        if 1 <= p <= PIANO_KEY_AMOUNT:
                            min_pitch = p if min_pitch is None else min(min_pitch, p)
            if min_pitch is not None:
                stem_len_units = float(layout.get('note_stem_length_semitone', 3) or 3)
                stem_len_mm = stem_len_units * semitone_mm
                origin = float(key_positions.get(bound_left, 0.0))
                note_offset = float(key_positions.get(min_pitch, origin)) - origin
                offset_left = note_offset - stem_len_mm
                ts_lane_gap_mm = 1.0
                ts_lane_right_offset = min(0.0, float(offset_left - ts_lane_gap_mm))
            extra_left = max(0.0, -ts_lane_right_offset)
            lane_margin = ts_lane_width + ts_lane_padding_mm + extra_left
            line['margin_left'] = max(base_margin_left, lane_margin)
        line['base_margin_left'] = base_margin_left
        line['ts_lane_width'] = ts_lane_width
        line['ts_lane_right_offset'] = ts_lane_right_offset
        line['ts_lane_padding_mm'] = ts_lane_padding_mm
        line['total_width'] = float(line['margin_left'] + stave_width + line['margin_right'])
        line['bound_left'] = int(bound_left)
        line['bound_right'] = int(bound_right)

    # Problem solved: paginate lines to fit available width with explicit breaks.
    available_width = max(1e-6, page_w - page_left - page_right)
    pages: list[list[dict]] = []
    cur_page: list[dict] = []
    cur_width = 0.0
    for line in lines:
        if line.get('page_break', False):
            if cur_page:
                pages.append(cur_page)
            elif not pages:
                pages.append([])
            cur_page = []
            cur_width = 0.0
        if cur_page and (cur_width + float(line['total_width'])) > available_width:
            pages.append(cur_page)
            cur_page = []
            cur_width = 0.0
        cur_page.append(line)
        cur_width += float(line['total_width'])
    if cur_page:
        pages.append(cur_page)

    # Problem solved: render each page with header/footer and justified spacing.
    if not pages:
        pages = [[]]

    analysis_snapshot = Analysis.compute(score, lines_count=len(lines), pages_count=len(pages))
    setattr(du, 'analysis', analysis_snapshot)
    target_page_index = 0
    if not pdf_export and pages:
        try:
            target_page_index = int(pageno)
        except Exception:
            target_page_index = 0
        target_page_index = max(0, min(len(pages) - 1, target_page_index))

    for page_index, page in enumerate(pages):
        du.new_page(page_w, page_h)
        if not pdf_export:
            edge_thickness = .5
            edge_dash = [2]
            edge_color = (0.0, 0.0, 0.0, 1.0)
            du.add_line(
                0.0,
                0.0,
                page_w,
                0.0,
                color=edge_color,
                width_mm=edge_thickness,
                dash_pattern=edge_dash,
                line_cap='round',
                id=0,
                tags=['paper_edge_guide', 'paper_edge_guide_top'],
            )
            du.add_line(
                0.0,
                page_h,
                page_w,
                page_h,
                color=edge_color,
                width_mm=edge_thickness,
                dash_pattern=edge_dash,
                line_cap='round',
                id=0,
                tags=['paper_edge_guide', 'paper_edge_guide_bottom'],
            )
        if not pdf_export and page_index != target_page_index:
            continue
        footer_height = float(layout.get('footer_height_mm', 0.0) or 0.0)
        footer_height = max(0.0, footer_height)
        if page_index == 0:
            title_text = _info_text('title', 'title')
            composer_text = _info_text('composer', 'composer')
            title_family, title_size, title_bold, title_italic, title_x_off, title_y_off = _info_font(
                'font_title',
                'Courier',
                12.0,
            )
            composer_family, composer_size, composer_bold, composer_italic, composer_x_off, composer_y_off = _info_font(
                'font_composer',
                'Courier',
                10.0,
            )
            du.add_text(
                page_left + title_x_off,
                page_top + title_y_off,
                title_text,
                family=title_family,
                size_pt=title_size,
                bold=title_bold,
                italic=title_italic,
                color=(0, 0, 0, 1),
                id=0,
                tags=['title'],
                anchor='nw',
            )
            du.add_text(
                (page_w - page_right) + composer_x_off,
                page_top + composer_y_off,
                composer_text,
                family=composer_family,
                size_pt=composer_size,
                bold=composer_bold,
                italic=composer_italic,
                color=(0, 0, 0, 1),
                id=0,
                tags=['composer'],
                anchor='ne',
            )
        if footer_height > 0.0:
            document_title = _info_text('title', 'title').strip()
            if not document_title:
                document_title = 'title'
            default_copyright = getattr(default_info, 'copyright', f"© all rights reserved {datetime.now().year}")
            footer_text = _info_text('copyright', default_copyright).strip()
            if not footer_text:
                footer_text = default_copyright
            footer_family, footer_size, footer_bold, footer_italic, footer_x_off, footer_y_off = _info_font(
                'font_copyright',
                'Courier',
                8.0,
            )
            du.add_text(
                page_left + footer_x_off,
                (page_h - page_bottom) + footer_y_off,
                f"Page {page_index + 1} of {len(pages)} | {document_title} | {footer_text}",
                family=footer_family,
                size_pt=footer_size,
                bold=footer_bold,
                italic=footer_italic,
                color=(0, 0, 0, 1),
                id=0,
                tags=['copyright'],
                anchor='sw',
            )
            if not pageno:
                # place a default keyTAB credit on the right side of the footer
                creation_timestamp = str(meta_data.get('creation_timestamp', '') or '').strip()
                if not creation_timestamp:
                    creation_timestamp = 'unknown'
                credit_size = max(1.0, float(footer_size) * 0.5)
                du.add_text(
                    page_w - page_right,
                    page_h - page_bottom - (credit_size * 0.45),
                    f"keyTAB piano engraving",
                    family=footer_family,
                    size_pt=credit_size,
                    bold=footer_bold,
                    italic=footer_italic,
                    color=(.6, .6, .6, 1),
                    id=0,
                    tags=['copyright'],
                    anchor='se',
                )
                du.add_text(
                    page_w - page_right,
                    page_h - page_bottom,
                    f"created: {creation_timestamp}",
                    family=footer_family,
                    size_pt=credit_size,
                    bold=footer_bold,
                    italic=footer_italic,
                    color=(.5, .5, .5, 1),
                    id=0,
                    tags=['copyright'],
                    anchor='se',
                )
        if not page:
            continue
        used_width = sum(float(l['total_width']) for l in page)
        leftover = max(0.0, available_width - used_width)
        gap = leftover / float(len(page) + 1)
        x_cursor = page_left + gap
        for line in page:
            line_x_start = x_cursor + float(line['margin_left'])
            line_x_end = line_x_start + float(line['stave_width'])
            header_offset = 0.0
            if page_index == 0:
                header_offset = float(layout.get('header_height_mm', 0.0) or 0.0)
            y1 = page_top + header_offset
            y2 = float(page_h - page_bottom - footer_height)
            if y2 <= y1:
                y2 = y1 + 1.0
            line['y_top'] = y1
            line['y_bottom'] = y2

            bound_left = int(line.get('bound_left', line['range'][0]))
            bound_right = int(line.get('bound_right', line['range'][1]))
            origin = float(key_positions.get(bound_left, 0.0))
            manual_range = isinstance(line.get('stave_range'), list) and len(line.get('stave_range')) >= 2
            bound_group_low = _group_index_for_key(bound_left) if manual_range else None
            bound_group_high = _group_index_for_key(bound_right) if manual_range else None
            ledger_drawn: set[tuple[int, int]] = set()

            def _key_to_x(key: int) -> float:
                # Problem solved: convert key index to page X using line origin.
                return line_x_start + (float(key_positions.get(key, 0.0)) - origin)

            def _time_to_y(ticks: float) -> float:
                # Problem solved: normalize time to line height for vertical layout.
                total = max(1e-6, float(line['time_end'] - line['time_start']))
                rel = (float(ticks) - float(line['time_start'])) / total
                rel = max(0.0, min(1.0, rel))
                return y1 + (y2 - y1) * rel

            def _text_bbox(text_val: str, family: str, size_pt: float, italic: bool, bold: bool, angle_deg: float, padding_mm: float, corner_radius_mm: float) -> tuple[float, float, float, list[tuple[float, float]], list[tuple[float, float]]]:
                xb, yb, w_mm, h_mm = du._get_text_extents_mm(text_val, family, size_pt, italic, bold)
                pad = max(0.0, float(padding_mm))
                w_mm += pad * 2.0
                h_mm += pad * 2.0
                hw = w_mm * 0.5
                hh = h_mm * 0.5
                r = min(max(0.0, float(corner_radius_mm)), hw, hh)

                def _rounded_rect_points(hw_val: float, hh_val: float, radius: float) -> list[tuple[float, float]]:
                    if radius <= 1e-6:
                        return [(-hw_val, -hh_val), (hw_val, -hh_val), (hw_val, hh_val), (-hw_val, hh_val)]
                    pts: list[tuple[float, float]] = []
                    corner_defs = [
                        (-hw_val + radius, -hh_val + radius, 180.0, 270.0),
                        (hw_val - radius, -hh_val + radius, 270.0, 360.0),
                        (hw_val - radius, hh_val - radius, 0.0, 90.0),
                        (-hw_val + radius, hh_val - radius, 90.0, 180.0),
                    ]
                    step = 15.0
                    for cx, cy, start_deg, end_deg in corner_defs:
                        deg = start_deg
                        while deg < end_deg + 0.01:
                            rad_ang = math.radians(deg)
                            pts.append((cx + radius * math.cos(rad_ang), cy + radius * math.sin(rad_ang)))
                            deg += step
                    return pts

                base_poly = _rounded_rect_points(hw, hh, r)
                corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
                ang = math.radians(angle_deg)
                sin_a = math.sin(ang)
                cos_a = math.cos(ang)
                rot_corners: list[tuple[float, float]] = []
                rot_poly: list[tuple[float, float]] = []
                min_y = float('inf')
                for (dx, dy) in corners:
                    rx = dx * cos_a - dy * sin_a
                    ry = dx * sin_a + dy * cos_a
                    rot_corners.append((rx, ry))
                    if ry < min_y:
                        min_y = ry
                for (dx, dy) in base_poly:
                    rx = dx * cos_a - dy * sin_a
                    ry = dx * sin_a + dy * cos_a
                    rot_poly.append((rx, ry))
                    if ry < min_y:
                        min_y = ry
                offset_down = max(0.0, -min_y)
                return w_mm, h_mm, offset_down, rot_corners, rot_poly

            tick_per_mm = (float(line['time_end'] - line['time_start'])) / max(1e-6, (y2 - y1))
            mm_per_quarter = float(QUARTER_NOTE_UNIT) / max(1e-6, tick_per_mm)

            indicator_type = str(layout.get('time_signature_indicator_type', 'classical') or 'classical')
            classic_family, classic_size, classic_bold, classic_italic = _layout_font(
                'time_signature_indicator_classic_font',
                'Edwin',
                35.0,
            )
            klav_family, klav_size, klav_bold, klav_italic = _layout_font(
                'time_signature_indicator_klavarskribo_font',
                'Edwin',
                25.0,
            )
            guide_thickness = float(layout.get('time_signature_indicator_guide_thickness_mm', 0.5) or 0.5) * scale
            divider_thickness = float(layout.get('time_signature_indicator_divide_guide_thickness_mm', 1.0) or 1.0) * scale
            classic_size_pt = classic_size * scale
            klav_size_pt = klav_size * scale

            def _ts_color(enabled: bool) -> tuple[float, float, float, float]:
                return (0.0, 0.0, 0.0, 1.0) if enabled else (0.6, 0.6, 0.6, 1.0)

            def _draw_classical_ts(numerator: int, denominator: int, enabled: bool, y_mm: float) -> None:
                color = _ts_color(enabled)
                x = ts_x_right
                size_pt = classic_size_pt
                du.add_text(
                    x,
                    y_mm - (3.0 * scale),
                    f"{int(numerator)}",
                    size_pt=size_pt,
                    color=color,
                    id=0,
                    tags=["time_signature"],
                    anchor='s',
                    family=classic_family,
                    bold=classic_bold,
                    italic=classic_italic,
                )
                du.add_line(
                    x - (3.0 * scale),
                    y_mm,
                    x + (3.0 * scale),
                    y_mm,
                    color=color,
                    width_mm=divider_thickness,
                    id=0,
                    tags=["time_signature_line"],
                    dash_pattern=None,
                )
                du.add_text(
                    x,
                    y_mm + (3.0 * scale),
                    f"{int(denominator)}",
                    size_pt=size_pt,
                    color=color,
                    id=0,
                    tags=["time_signature"],
                    anchor='n',
                    family=classic_family,
                    bold=classic_bold,
                    italic=classic_italic,
                )

            def _draw_klavars_ts(numerator: int, denominator: int, enabled: bool, y_mm: float, grid_positions: list[int]) -> None:
                color = _ts_color(enabled)
                quarters_per_measure = float(numerator) * (4.0 / max(1.0, float(denominator)))
                measure_len_mm = quarters_per_measure * mm_per_quarter
                beat_len_mm = measure_len_mm / max(1, int(numerator))

                seq = [int(p) for p in (grid_positions or []) if 1 <= int(p) <= 9]
                if len(seq) != int(numerator):
                    seq = list(range(1, int(numerator) + 1))

                guide_half_len = min(ts_col_w * 0.45, 3.0 * scale) if ts_col_w > 0.0 else (3.0 * scale)
                guide_width_mm = guide_thickness
                for k, val in enumerate(seq, start=1):
                    y = y_mm + (k - 1) * beat_len_mm
                    du.add_line(
                        ts_x_right - guide_half_len,
                        y,
                        ts_x_right + guide_half_len,
                        y,
                        color=color,
                        width_mm=guide_width_mm,
                        id=0,
                        tags=["ts_klavars_guide"],
                        dash_pattern=None,
                    )
                du.add_line(
                    ts_x_right - guide_half_len,
                    y_mm + measure_len_mm,
                    ts_x_right + guide_half_len,
                    y_mm + measure_len_mm,
                    color=color,
                    width_mm=guide_width_mm,
                    id=0,
                    tags=["ts_klavars_guide"],
                    dash_pattern=None,
                )

                for k, val in enumerate(seq, start=1):
                    y = y_mm + (k - 1) * beat_len_mm
                    du.add_text(
                        ts_x_mid,
                        y,
                        str(val),
                        size_pt=klav_size_pt,
                        color=color,
                        id=0,
                        tags=["ts_klavars_mid"],
                        anchor='center',
                        family=klav_family,
                        bold=klav_bold,
                        italic=klav_italic,
                    )
                du.add_text(
                    ts_x_mid,
                    y_mm + measure_len_mm,
                    "1",
                        size_pt=klav_size_pt,
                    color=color,
                    id=0,
                    tags=["ts_klavars_mid"],
                    anchor='center',
                        family=klav_family,
                        bold=klav_bold,
                        italic=klav_italic,
                )
                group_starts = [i for i, v in enumerate(seq, start=1) if v == 1]
                if not group_starts or group_starts[0] != 1:
                    group_starts = [1] + group_starts
                for gi, s in enumerate(group_starts, start=1):
                    y = y_mm + (s - 1) * beat_len_mm
                    du.add_text(
                        ts_x_left,
                        y,
                        str(gi),
                        size_pt=klav_size_pt,
                        color=color,
                        id=0,
                        tags=["ts_klavars_left"],
                        anchor='center',
                        family=klav_family,
                        bold=klav_bold,
                        italic=klav_italic,
                    )

            # Problem solved: draw barlines and beat lines from the base grid.
            grid_left = line_x_start
            grid_right = line_x_start + float(line['stave_width'])
            ts_right_margin = max(0.0, 1.5 * scale)
            ts_lane_padding_mm = float(line.get('ts_lane_padding_mm', 0.0) or 0.0)
            ts_lane_width = float(line.get('ts_lane_width', 0.0) or 0.0)
            if ts_lane_width > 0.0:
                ts_lane_right = line_x_start + float(line.get('ts_lane_right_offset', 0.0) or 0.0) - ts_lane_padding_mm
                ts_lane_left = ts_lane_right - ts_lane_width
                ts_left_edge = ts_lane_left
                ts_right_bound = ts_lane_right
            else:
                ts_indicator_width = max(0.0, float(line.get('margin_left', 0.0) or 0.0) - ts_right_margin)
                ts_left_edge = grid_left - ts_right_margin - ts_indicator_width
                ts_right_bound = (grid_left - ts_right_margin) - 5.0
            ts_usable = max(0.0, ts_right_bound - ts_left_edge)
            ts_col_w = ts_usable / 3.0 if ts_usable > 0.0 else 0.0
            ts_x_left = ts_left_edge + (ts_col_w * 0.5)
            ts_x_mid = ts_left_edge + (ts_col_w * 1.5)
            ts_x_right = ts_left_edge + (ts_col_w * 2.5)
            grid_color = (0, 0, 0, 1)
            bar_width_mm = float(layout.get('grid_barline_thickness_mm', 0.25) or 0.25) * scale
            grid_width_mm = float(layout.get('grid_gridline_thickness_mm', 0.15) or 0.15) * scale
            dash_pattern = list(layout.get('grid_gridline_dash_pattern_mm', []) or [])
            if dash_pattern:
                dash_pattern = [float(v) * scale for v in dash_pattern]
            time_cursor = 0.0
            for bg in base_grid:
                numerator = int(bg.get('numerator', 4) or 4)
                denominator = int(bg.get('denominator', 4) or 4)
                measure_amount = int(bg.get('measure_amount', 1) or 1)
                beat_grouping = list(bg.get('beat_grouping', []) or [])
                indicator_enabled = bool(bg.get('indicator_enabled', True))
                if measure_amount <= 0:
                    continue
                measure_len = float(numerator) * (4.0 / float(max(1, denominator))) * float(QUARTER_NOTE_UNIT)
                beat_len = measure_len / max(1, int(numerator))
                if op_time.ge(float(time_cursor), float(line['time_start'])) and op_time.lt(float(time_cursor), float(line['time_end'])) and indicator_enabled:
                    y_ts = _time_to_y(float(time_cursor))
                    if indicator_type == 'classical':
                        _draw_classical_ts(numerator, denominator, indicator_enabled, y_ts)
                    elif indicator_type == 'klavarskribo':
                        _draw_klavars_ts(numerator, denominator, indicator_enabled, y_ts, beat_grouping)
                    elif indicator_type == 'both':
                        _draw_classical_ts(numerator, denominator, indicator_enabled, y_ts)
                        _draw_klavars_ts(numerator, denominator, indicator_enabled, y_ts, beat_grouping)
                for _ in range(measure_amount):
                    if op_time.gt(time_cursor, float(line['time_end'])):
                        break
                    full_group = len(beat_grouping) == int(numerator) and [int(v) for v in beat_grouping] == list(range(1, int(numerator) + 1))
                    for idx in range(int(numerator)):
                        t = float(time_cursor + (beat_len * idx))
                        if op_time.lt(t, float(line['time_start'])) or op_time.gt(t, float(line['time_end'])):
                            continue
                        y = _time_to_y(t)
                        if idx == 0:
                            du.add_line(
                                grid_left,
                                y,
                                grid_right,
                                y,
                                color=grid_color,
                                width_mm=bar_width_mm,
                                id=0,
                                tags=['grid_line'],
                                dash_pattern=None,
                            )
                            if full_group:
                                continue
                        else:
                            group_val = beat_grouping[idx] if idx < len(beat_grouping) else (idx + 1)
                            if full_group or int(group_val) == 1:
                                du.add_line(
                                    grid_left,
                                    y,
                                    grid_right,
                                    y,
                                    color=grid_color,
                                    width_mm=max(0.1, grid_width_mm),
                                    id=0,
                                    tags=['grid_line'],
                                    dash_pattern=dash_pattern or [2.0 * scale, 2.0 * scale],
                                )
                    time_cursor += measure_len
                if op_time.gt(time_cursor, float(line['time_end'])):
                    break

            if op_time.ge(total_ticks, float(line['time_start'])) and op_time.le(total_ticks, float(line['time_end'])):
                y_end_bar = _time_to_y(float(total_ticks))
                du.add_line(
                    grid_left,
                    y_end_bar,
                    grid_right,
                    y_end_bar,
                    color=grid_color,
                    width_mm=bar_width_mm * 2.0,
                    id=0,
                    tags=['grid_line'],
                    dash_pattern=None,
                )

            # Problem solved: render count lines as lightweight guides.
            if bool(layout.get('countline_visible', True)) and count_lines:
                dash_pattern = list(layout.get('countline_dash_pattern', []) or [])
                if dash_pattern:
                    dash_pattern = [float(v) * scale for v in dash_pattern]
                countline_w = float(layout.get('countline_thickness_mm', 0.5) or 0.5) * scale
                for ev in count_lines:
                    t0 = float(ev.get('time', 0.0) or 0.0)
                    p1 = int(ev.get('pitch1', 40) or 40)
                    p2 = int(ev.get('pitch2', 44) or 44)
                    if op_time.lt(t0, float(line['time_start'])) or op_time.gt(t0, float(line['time_end'])):
                        continue
                    x1 = _key_to_x(p1)
                    x2 = _key_to_x(p2)
                    if x2 < x1:
                        x1, x2 = x2, x1
                    y_mm = _time_to_y(t0)
                    du.add_line(
                        x1,
                        y_mm,
                        x2,
                        y_mm,
                        color=(0, 0, 0, 1),
                        width_mm=countline_w,
                        dash_pattern=dash_pattern or [0.0, 1.5 * scale],
                        id=int(ev.get('_id', 0) or 0),
                        tags=['count_line'],
                    )

            # Problem solved: pre-filter notes once per line for later passes.
            line_notes: list[dict] = []
            for item in norm_notes:
                n_t = float(item.get('time', 0.0) or 0.0)
                n_end = float(item.get('end', 0.0) or 0.0)
                p = int(item.get('pitch', 0) or 0)
                if op_time.ge(n_t, float(line['time_end'])) or op_time.le(n_end, float(line['time_start'])):
                    continue
                if p < 1 or p > PIANO_KEY_AMOUNT:
                    continue
                line_notes.append(item)

            # Grace notes: time-only, so check time window and key range.
            line_grace: list[dict] = []
            for item in norm_grace:
                g_t = float(item.get('time', 0.0) or 0.0)
                p = int(item.get('pitch', 0) or 0)
                if op_time.lt(g_t, float(line['time_start'])) or op_time.ge(g_t, float(line['time_end'])):
                    continue
                if p < 1 or p > PIANO_KEY_AMOUNT:
                    continue
                line_grace.append(item)

            line_slurs: list[dict] = []
            if norm_slurs:
                line_start = float(line.get('time_start', 0.0) or 0.0)
                line_end = float(line.get('time_end', 0.0) or 0.0)
                for sl in norm_slurs:
                    anchor_t = float(sl.get('y1_time', 0.0) or 0.0)
                    if op_time.lt(anchor_t, float(line_start)) or op_time.ge(anchor_t, float(line_end)):
                        continue
                    line_slurs.append(sl)

            line_texts: list[dict] = []
            if norm_texts:
                line_start = float(line.get('time_start', 0.0) or 0.0)
                line_end = float(line.get('time_end', 0.0) or 0.0)
                for tx in norm_texts:
                    t_time = float(tx.get('time', 0.0) or 0.0)
                    if op_time.lt(t_time, float(line_start)) or op_time.ge(t_time, float(line_end)):
                        continue
                    line_texts.append(tx)

            notes_by_hand_line: dict[str, list[dict]] = {'l': [], 'r': []}
            for item in line_notes:
                hk = str(item.get('hand', '<') or '<')
                hand_norm = 'l' if hk in ('<', 'l') else 'r'
                notes_by_hand_line[hand_norm].append(item)

            beam_groups_by_hand: dict[str, tuple[list[list[dict]], list[tuple[float, float]]]] = {}
            line_start = float(line.get('time_start', 0.0) or 0.0)
            line_end = float(line.get('time_end', 0.0) or 0.0)

            def _is_line_continuation(note_dict: dict) -> bool:
                # Problem solved: avoid redrawing heads/stems when a note ties
                # across a line break; only continuation dots should appear.
                start_t = float(note_dict.get('time', 0.0) or 0.0)
                end_t = float(note_dict.get('end', 0.0) or 0.0)
                return op_time.gt(float(line_start), start_t) and op_time.gt(end_t, float(line_start))
            
            for hand_norm in ('r', 'l'):
                notes_for_hand = notes_by_hand_line.get(hand_norm, [])
                markers_for_hand = beam_by_hand.get(hand_norm, [])
                groups, windows = _group_by_beam_markers(notes_for_hand, markers_for_hand, line_start, line_end)
                beam_groups_by_hand[hand_norm] = (groups, windows)

            # Problem solved: measure numbers must avoid colliding with notes/beams.
            mn_family, mn_size, mn_bold, mn_italic = _layout_font('measure_numbering_font', 'Edwin', 10.0)
            size_pt = mn_size * scale
            mm_per_pt = 25.4 / 72.0
            text_h_mm = size_pt * mm_per_pt
            measure_pad = 1.5
            stem_len_units = float(layout.get('note_stem_length_semitone', 3) or 3)
            stem_len_mm = stem_len_units * semitone_mm

            def _note_x_range(it: dict) -> tuple[float, float]:
                p = int(it.get('pitch', 0) or 0)
                x = _key_to_x(p)
                w = semitone_mm
                hand_key = str(it.get('hand', '<') or '<')
                beam_ext = semitone_mm
                if hand_key in ('l', '<'):
                    x_min = x - max(w, stem_len_mm + beam_ext)
                    x_max = x + w
                else:
                    x_min = x - w
                    x_max = x + max(w, stem_len_mm + beam_ext)
                return (x_min, x_max)

            def _right_extent(t0: float, t1: float) -> float:
                max_x = grid_right
                for it in line_notes:
                    nt = float(it.get('time', 0.0) or 0.0)
                    ne = float(it.get('end', 0.0) or 0.0)
                    if op_time.ge(nt, float(t1)) or op_time.le(ne, float(t0)):
                        continue
                    _x0, x1 = _note_x_range(it)
                    if x1 > max_x:
                        max_x = x1
                return max_x

            def _beam_group_right_extent(t0: float) -> float | None:
                max_x = None
                for hand_norm, payload in beam_groups_by_hand.items():
                    groups, windows = payload
                    for idx, grp in enumerate(groups):
                        if not grp or idx >= len(windows):
                            continue
                        w0, w1 = windows[idx]
                        if op_time.ge(float(t0), float(w1)) or op_time.lt(float(t0), float(w0)):
                            continue
                        highest = max(grp, key=lambda n: int(n.get('pitch', 0) or 0))
                        p = int(highest.get('pitch', 0) or 0)
                        base_x = _key_to_x(p)
                        if hand_norm == 'r':
                            x = base_x + stem_len_mm
                        else:
                            x = base_x + semitone_mm
                        if max_x is None or x > max_x:
                            max_x = x
                return max_x

            def _collides(x0: float, x1: float, t0: float, t1: float) -> bool:
                for it in line_notes:
                    nt = float(it.get('time', 0.0) or 0.0)
                    ne = float(it.get('end', 0.0) or 0.0)
                    if op_time.ge(nt, float(t1)) or op_time.le(ne, float(t0)):
                        continue
                    nx0, nx1 = _note_x_range(it)
                    if (nx1 >= x0) and (nx0 <= x1):
                        return True
                return False

            for mw in measure_windows:
                m_start = float(mw.get('start', 0.0))
                m_end = float(mw.get('end', 0.0))
                if op_time.ge(m_start, float(line['time_end'])) or op_time.le(m_end, float(line['time_start'])):
                    continue
                num_txt = str(int(mw.get('number', 0) or 0))
                if not num_txt:
                    continue
                text_w_mm = max(1.0, text_h_mm * 0.6 * len(num_txt))
                t0 = m_start
                t1 = min(float(line['time_end']), m_start + (text_h_mm * tick_per_mm))
                y_text = _time_to_y(t0) + 1.0

                # Default outside-right; only move further right on collision
                base_right = grid_right + measure_pad
                beam_right = _beam_group_right_extent(t0)
                needed_right = _right_extent(t0, t1) + measure_pad
                if beam_right is not None:
                    needed_right = max(needed_right, float(beam_right) + measure_pad)
                x_pos = max(base_right, needed_right)
                x0 = x_pos
                x1 = x_pos + text_w_mm
                step = text_w_mm + measure_pad
                tries = 0
                while _collides(x0, x1, t0, t1) and tries < 6:
                    x_pos += step
                    x0 = x_pos
                    x1 = x_pos + text_w_mm
                    tries += 1
                guide_y = _time_to_y(t0)
                du.add_line(
                    grid_right,
                    guide_y,
                    x_pos + text_w_mm,
                    guide_y,
                    color=(0, 0, 0, 1),
                    width_mm=max(0.12, 0.15 * scale),
                    id=0,
                    tags=['measure_number_guide'],
                    dash_pattern=[0.8 * scale, 0.8 * scale],
                )
                du.add_text(
                    x_pos,
                    y_text,
                    num_txt,
                    size_pt=size_pt,
                    color=(0, 0, 0, 1),
                    id=0,
                    tags=['measure_number'],
                    anchor='nw',
                    family=mn_family,
                    bold=mn_bold,
                    italic=mn_italic,
                )

            visible_keys = list(line.get('visible_keys', []))
            if not visible_keys:
                visible_keys = [k for k in range(int(line['range'][0]), int(line['range'][1]) + 1) if k in line_keys]
            # Special-case low register: draw A#0 (key 2) line when keys 1-3 appear.
            low_key_present = bool(line.get('low_key_left', False))
            if low_key_present:
                x_pos = _key_to_x(2)
                width_mm = max(stave_three_w, semitone_mm / 3.0)
                du.add_line(
                    x_pos,
                    y1,
                    x_pos,
                    y2,
                    color=(0, 0, 0, 1),
                    width_mm=width_mm,
                    dash_pattern=None,
                    id=0,
                    tags=['stave'],
                )
            for key in visible_keys:
                if low_key_present and int(key) == 2:
                    continue
                x_pos = _key_to_x(key)
                is_clef_line = key in (41, 43)
                is_three_line = key in key_class_filter('FGA')
                if is_clef_line:
                    width_mm = max(stave_two_w, semitone_mm / 6.0)
                    dash = clef_dash if clef_dash else [2.0 * scale, 2.0 * scale]
                elif is_three_line:
                    width_mm = max(stave_three_w, semitone_mm / 3.0)
                    dash = None
                else:
                    width_mm = max(stave_two_w, semitone_mm / 10.0)
                    dash = None
                du.add_line(x_pos, y1, x_pos, y2, color=(0, 0, 0, 1), width_mm=width_mm, dash_pattern=dash, id=0, tags=['stave'])

            # ---- Beam drawing per line ----
            if bool(layout.get('beam_visible', True)):
                notes_by_hand_line: dict[str, list[dict]] = {'l': [], 'r': []}
                for item in line_notes:
                    hk = str(item.get('hand', '<') or '<')
                    hand_norm = 'l' if hk in ('<', 'l') else 'r'
                    notes_by_hand_line[hand_norm].append(item)

                stem_len_units = float(layout.get('note_stem_length_semitone', 3) or 3)
                layout_stem_len = stem_len_units * semitone_mm
                beam_w = float(layout.get('beam_thickness_mm', 1.0) or 1.0) * scale
                stem_w = float(layout.get('note_stem_thickness_mm', 0.5) or 0.5) * scale
                line_start = float(line.get('time_start', 0.0) or 0.0)
                line_end = float(line.get('time_end', 0.0) or 0.0)

                for hand_norm in ('r', 'l'):
                    notes_for_hand = notes_by_hand_line.get(hand_norm, [])
                    markers_for_hand = beam_by_hand.get(hand_norm, [])
                    groups, windows = _group_by_beam_markers(notes_for_hand, markers_for_hand, line_start, line_end)
                    for idx, grp in enumerate(groups):
                        if not grp:
                            continue
                        t0, t1 = windows[idx] if idx < len(windows) else (line_start, line_end)
                        starts_in = [float(n.get('time', 0.0) or 0.0) for n in grp if op_time.ge(float(n.get('time', 0.0) or 0.0), float(t0)) and op_time.lt(float(n.get('time', 0.0) or 0.0), float(t1))]
                        if not starts_in:
                            continue
                        s_min, s_max = min(starts_in), max(starts_in)
                        if op_time.eq(float(s_min), float(s_max)):
                            continue
                        t_first = min(starts_in)
                        t_last = max(starts_in)
                        if hand_norm == 'r':
                            highest = max(grp, key=lambda n: int(n.get('pitch', 0) or 0))
                            x1 = _key_to_x(int(highest.get('pitch', 0) or 0)) + float(layout_stem_len)
                            x2 = x1 + float(semitone_mm)
                        else:
                            lowest = min(grp, key=lambda n: int(n.get('pitch', 0) or 0))
                            x1 = _key_to_x(int(lowest.get('pitch', 0) or 0)) - float(layout_stem_len)
                            x2 = x1 - float(semitone_mm)
                        yb1 = _time_to_y(float(t_first))
                        yb2 = _time_to_y(float(t_last))
                        du.add_line(
                            x1,
                            yb1,
                            x2,
                            yb2,
                            color=(0, 0, 0, 1),
                            width_mm=max(0.2, beam_w),
                            id=0,
                            tags=['beam'],
                        )
                        for m in grp:
                            mt = float(m.get('time', t_first) or t_first)
                            if not (op_time.ge(mt, float(t0)) and op_time.lt(mt, float(t1))):
                                continue
                            y_note = _time_to_y(float(mt))
                            if hand_norm == 'r':
                                x_tip = _key_to_x(int(m.get('pitch', 0) or 0)) + float(layout_stem_len)
                            else:
                                x_tip = _key_to_x(int(m.get('pitch', 0) or 0)) - float(layout_stem_len)
                            if abs(yb2 - yb1) > 1e-6:
                                t_ratio = (y_note - yb1) / (yb2 - yb1)
                                x_on_beam = x1 + t_ratio * (x2 - x1)
                            else:
                                x_on_beam = x1
                            du.add_line(
                                x_tip,
                                y_note,
                                float(x_on_beam),
                                y_note,
                                color=(0, 0, 0, 1),
                                width_mm=max(0.15, stem_w),
                                id=0,
                                tags=['beam_stem'],
                            )

            line_start = float(line.get('time_start', 0.0) or 0.0)
            line_end = float(line.get('time_end', 0.0) or 0.0)
            black_rule = str(layout.get('black_note_rule', 'below_stem') or 'below_stem')

            # Grace notes: tiny heads anchored so time sits at the top.
            if bool(layout.get('grace_note_visible', True)) and line_grace:
                # grace_note_scale is a relative factor; semitone_mm already includes layout scale.
                g_scale = float(layout.get('grace_note_scale', 0.75) or 0.75)
                # grace_note_outline_width_mm is a mm value; apply global layout scale to stroke width.
                g_outline = float(layout.get('grace_note_outline_width_mm', layout.get('grace_note_outline_width', 0.3)) or 0.3) * scale
                for item in line_grace:
                    g_t = float(item.get('time', 0.0) or 0.0)
                    p = int(item.get('pitch', 0) or 0)
                    x = _key_to_x(p)
                    y_top = _time_to_y(g_t)
                    w = semitone_mm * g_scale
                    if p in BLACK_KEYS:
                        # Black keys: single filled head.
                        du.add_oval(
                            x - w,
                            y_top,
                            x + w,
                            y_top + (w * 2.0),
                            stroke_color=None,
                            stroke_width_mm=0.0,
                            fill_color=(0, 0, 0, 1),
                            id=int(item.get('id', 0) or 0),
                            tags=['grace_note_black'],
                        )
                    else:
                        # White keys: outer dark fill, inner white fill inset by half the outline width.
                        du.add_oval(
                            x - w,
                            y_top,
                            x + w,
                            y_top + (w * 2.0),
                            stroke_color=None,
                            stroke_width_mm=0.0,
                            fill_color=(0, 0, 0, 1),
                            id=int(item.get('id', 0) or 0),
                            tags=['grace_note_black_outline'],
                        )
                        inset = max(0.0, g_outline * 0.5)
                        il = x - w + inset
                        ir = x + w - inset
                        it = y_top + inset
                        ib = y_top + (w * 2.0) - inset
                        if ir <= il:
                            midx = ( (x - w) + (x + w) ) * 0.5
                            il = ir = midx
                        if ib <= it:
                            midy = (y_top + y_top + (w * 2.0)) * 0.5
                            it = ib = midy
                        du.add_oval(
                            il,
                            it,
                            ir,
                            ib,
                            stroke_color=None,
                            stroke_width_mm=0.0,
                            fill_color=(1, 1, 1, 1),
                            id=int(item.get('id', 0) or 0),
                            tags=['grace_note_white_fill'],
                        )

            # Problem solved: render notes after grid, using precomputed positions.
            for item in line_notes:
                n_t = float(item.get('time', 0.0) or 0.0)
                n_end = float(item.get('end', 0.0) or 0.0)
                p = int(item.get('pitch', 0) or 0)
                hand_key = str(item.get('hand', '<') or '<')
                n = item.get('raw', {}) or {}
                x = _key_to_x(p)
                y_start = _time_to_y(n_t)
                y_end = _time_to_y(n_end)
                if y_end < y_start:
                    y_start, y_end = y_end, y_start
                w = semitone_mm
                note_y = y_start
                if p in BLACK_KEYS and _black_note_above_stem(item, black_rule, line_notes, op_time):
                    note_y = y_start - (w * 2.0)
                # Problem solved: draw the note body with per-hand colors or overrides.
                raw_color = n.get('color', None)
                if raw_color in (None, ''):
                    raw_color = n.get('hand', '<')
                midicol = _normalize_hex_color(raw_color)
                if midicol == '<':
                    base = _normalize_hex_color(layout.get('note_midinote_left_color', '#cccccc'))
                elif midicol == '>':
                    base = _normalize_hex_color(layout.get('note_midinote_right_color', '#cccccc'))
                elif midicol:
                    base = midicol
                else:
                    fallback = 'note_midinote_left_color' if hand_key in ('l', '<') else 'note_midinote_right_color'
                    base = _normalize_hex_color(layout.get(fallback, '#cccccc'))
                if not base:
                    base = '#cccccc'
                fill = _hex_to_rgba01(base, 1.0)
                if bool(layout.get('note_midinote_visible', True)):
                    du.add_polygon(
                        [
                            (x, y_start),
                            (x - w, y_start + semitone_mm),
                            (x - w, y_end),
                            (x + w, y_end),
                            (x + w, y_start + semitone_mm),
                        ],
                        stroke_color=None,
                        fill_color=fill,
                        id=int(item.get('id', 0) or 0),
                        tags=['midi_note'],
                    )

                continues_from_prev_line = _is_line_continuation(item)

                on_barline = False
                for bt in barline_positions:
                    if op_time.eq(float(bt), n_t):
                        on_barline = True
                        break
                if on_barline:
                    stem_len_units = float(layout.get('note_stem_length_semitone', 3) or 3)
                    stem_len = stem_len_units * semitone_mm
                    thickness = float(layout.get('grid_barline_thickness_mm', 0.25) or 0.25) * scale + 0.1
                    if hand_key in ('l', '<'):
                        x1 = x
                        x2 = x + (w * 1.5)
                    else:
                        x1 = x
                        x2 = x - (w * 1.5)
                    
                    if not continues_from_prev_line:
                        du.add_line(
                            x1,
                            y_start,
                            x2,
                            y_start,
                            color=(1,1,1,1),
                            width_mm=thickness,
                            line_cap="butt",
                            id=0,
                            tags=['barline_white_space'],
                        )

                # Problem solved: avoid duplicated heads on continuations.
                if not continues_from_prev_line and bool(layout.get('note_head_visible', True)):
                    outline_w = float(layout.get('note_stem_thickness_mm', 0.5) or 0.5) * scale
                    if p in BLACK_KEYS:
                        du.add_oval(
                            x - (w * 0.8),
                            note_y,
                            x + (w * 0.8),
                            note_y + (w * 2.0),
                            stroke_color=(0, 0, 0, 1),
                            stroke_width_mm=0.3,
                            fill_color=(0, 0, 0, 1),
                            id=int(item.get('id', 0) or 0),
                            tags=['notehead_black'],
                        )
                    else:
                        du.add_oval(
                            x - w,
                            note_y,
                            x + w,
                            note_y + (w * 2.0),
                            stroke_color=(0, 0, 0, 1),
                            stroke_width_mm=outline_w,
                            fill_color=(1, 1, 1, 1),
                            id=int(item.get('id', 0) or 0),
                            tags=['notehead_white'],
                        )

                # Problem solved: attach stems only to non-continuation heads.
                if not continues_from_prev_line and bool(layout.get('note_stem_visible', True)):
                    stem_len_units = float(layout.get('note_stem_length_semitone', 3) or 3)
                    stem_len = stem_len_units * semitone_mm
                    stem_w = float(layout.get('note_stem_thickness_mm', 0.5) or 0.5) * scale
                    x2 = x - stem_len if hand_key in ('l', '<') else x + stem_len
                    du.add_line(
                        x,
                        y_start,
                        x2,
                        y_start,
                        color=(0, 0, 0, 1),
                        width_mm=stem_w,
                        id=0,
                        tags=['stem'],
                    )

                # Problem solved: left-hand dot uses inverse fill on black keys.
                if (not continues_from_prev_line) and bool(layout.get('note_leftdot_visible', True)) and hand_key in ('l', '<'):
                    w2 = w * 2.0
                    dot_d = w2 * 0.3
                    cy = note_y + (w2 / 2.0)
                    fill = (1, 1, 1, 1) if p in BLACK_KEYS else (0, 0, 0, 1)
                    du.add_oval(
                        x - (dot_d / 3.0),
                        cy - (dot_d / 3.0),
                        x + (dot_d / 3.0),
                        cy + (dot_d / 3.0),
                        stroke_color=None,
                        fill_color=fill,
                        id=0,
                        tags=['left_dot'],
                    )

                # Problem solved: show ledger lines only when manual ranges
                # would otherwise hide them.
                if manual_range:
                    ledger_groups: list[dict] = []
                    if p < bound_left:
                        g_start = _group_index_for_key(p)
                        g_end = int(bound_group_low or 0) - 1
                        if g_start <= g_end:
                            ledger_groups = line_groups[g_start:g_end + 1]
                    elif p > bound_right:
                        g_start = int(bound_group_high or 0) + 1
                        g_end = _group_index_for_key(p)
                        if g_start <= g_end:
                            ledger_groups = line_groups[g_start:g_end + 1]
                    if ledger_groups:
                        y_center = note_y + w
                        seg_half = w
                        y_seg1 = y_center - seg_half
                        y_seg2 = y_center + seg_half + seg_half + (seg_half / 2.0)
                        for grp in ledger_groups:
                            for key in grp.get('keys', []):
                                x_pos = _key_to_x(int(key))
                                is_clef_line = int(key) in (41, 43)
                                is_three_line = int(key) in key_class_filter('FGA')
                                if is_clef_line:
                                    width_mm = max(stave_two_w, semitone_mm / 6.0)
                                    dash = clef_dash if clef_dash else [2.0 * scale, 2.0 * scale]
                                elif is_three_line:
                                    width_mm = max(stave_three_w, semitone_mm / 3.0)
                                    dash = None
                                else:
                                    width_mm = max(stave_two_w, semitone_mm / 10.0)
                                    dash = None
                                key_sig = (int(key), int(round(y_center * 1000)))
                                if key_sig in ledger_drawn:
                                    continue
                                ledger_drawn.add(key_sig)
                                du.add_line(
                                    x_pos,
                                    y_seg1,
                                    x_pos,
                                    y_seg2,
                                    color=(0, 0, 0, 1),
                                    width_mm=width_mm,
                                    dash_pattern=dash,
                                    id=0,
                                    tags=['stave'],
                                )

                # Problem solved: continuation dots indicate overlapped starts/ends
                # and line crossings for the same hand.
                dot_times: list[float] = []
                for m in line_notes:
                    if int(m.get('idx', -1) or -1) == int(item.get('idx', -2) or -2):
                        continue
                    if str(m.get('hand', '<') or '<') != hand_key:
                        continue
                    s = float(m.get('time', 0.0) or 0.0)
                    e = float(m.get('end', 0.0) or 0.0)
                    if op_time.gt(s, n_t) and op_time.lt(s, n_end):
                        dot_times.append(s)
                    if op_time.gt(e, n_t) and op_time.lt(e, n_end):
                        dot_times.append(e)
                for bt in barline_positions:
                    bt = float(bt)
                    if op_time.eq(bt, float(line_start)) or op_time.eq(bt, float(line_end)):
                        continue
                    if op_time.gt(bt, n_t) and op_time.lt(bt, n_end):
                        dot_times.append(bt)
                if continues_from_prev_line:
                    dot_times.append(float(line_start))
                if dot_times:
                    dot_d = float(layout.get('note_continuation_dot_size_mm', 0.0) or 0.0)
                    if dot_d > 0.0:
                        dot_d *= scale
                    else:
                        dot_d = w * 0.8
                    for t in sorted(set(dot_times)):
                        y_center = _time_to_y(float(t)) + w
                        du.add_oval(
                            x - dot_d / 2.0,
                            y_center - dot_d / 2.0,
                            x + dot_d / 2.0,
                            y_center + dot_d / 2.0,
                            fill_color=(0, 0, 0, 1),
                            stroke_color=None,
                            id=0,
                            tags=['continuation_dot'],
                        )

                # Problem solved: draw a horizontal connector for same-time chords.
                same_time = [
                    m
                    for m in line_notes
                    if str(m.get('hand', '<') or '<') == hand_key
                    and op_time.eq(float(m.get('time', 0.0) or 0.0), n_t)
                    and not _is_line_continuation(m)
                ]
                if len(same_time) >= 2:
                    lowest = min(same_time, key=lambda m: int(m.get('pitch', 0) or 0))
                    highest = max(same_time, key=lambda m: int(m.get('pitch', 0) or 0))
                    if int(lowest.get('id', 0) or 0) == int(item.get('id', 0) or 0):
                        x1 = _key_to_x(int(lowest.get('pitch', 0) or 0))
                        x2 = _key_to_x(int(highest.get('pitch', 0) or 0))
                        du.add_line(
                            x1,
                            y_start,
                            x2,
                            y_start,
                            color=(0, 0, 0, 1),
                            width_mm=float(layout.get('note_stem_thickness_mm', 0.5) or 0.5) * scale,
                            id=0,
                            tags=['chord_connect'],
                        )

                # Problem solved: stop sign marks a rest gap after a note.
                if _has_followed_rest(item):
                    w_stop = w * 1.8
                    points = [
                        (x - w_stop / 2.0, y_end - w_stop),
                        (x, y_end),
                        (x + w_stop / 2.0, y_end - w_stop),
                    ]
                    du.add_polyline(
                        points,
                        stroke_color=(0, 0, 0, 1),
                        stroke_width_mm=float(layout.get('note_stopsign_thickness_mm', 0.4) or 0.4) * scale,
                        id=0,
                        tags=['stop_sign'],
                    )

            def clamp_x(val: float) -> float:
                if page_w <= 0.0:
                    return float(val)
                return max(0.0, min(float(val), float(page_w)))

            base_x_c4 = _key_to_x(40)

            def rpitch_to_x(rp: float) -> float:
                return clamp_x(base_x_c4 + float(rp) * semitone_mm)

            if line_texts:
                default_font = layout.get('font_text', {}) or {}
                pad_mm = float(layout.get('text_background_padding_mm', 0.0) or 0.0) * scale

                def _resolve_font(tx: dict) -> tuple[str, float, bool, bool]:
                    use_custom = bool(tx.get('use_custom_font', False))
                    fnt = tx.get('font', None) if use_custom else None
                    if not isinstance(fnt, dict):
                        fnt = default_font if isinstance(default_font, dict) else {}
                    family = str(fnt.get('family', default_font.get('family', 'Edwin')))
                    size_pt = float(fnt.get('size_pt', default_font.get('size_pt', 12.0)) or 12.0)
                    italic = bool(fnt.get('italic', default_font.get('italic', False)))
                    bold = bool(fnt.get('bold', default_font.get('bold', False)))
                    return family, size_pt, italic, bold

                for tx in line_texts:
                    t_time = float(tx.get('time', 0.0) or 0.0)
                    x_rp = float(tx.get('x_rpitch', 0) or 0)
                    angle = float(tx.get('rotation', 0.0) or 0.0)
                    x_off = float(tx.get('x_offset_mm', 0.0) or 0.0)
                    y_off = float(tx.get('y_offset_mm', 0.0) or 0.0)
                    txt_raw = str(tx.get('text', '') or '')
                    display_txt = txt_raw if txt_raw.strip() else "(no text set)"
                    family, size_pt_raw, italic, bold = _resolve_font(tx)
                    size_pt = float(size_pt_raw) * ENGRAVER_FRACTIONAL_SCALE_CORRECTION * (scale / 0.3333333333333333)
                    y_mm = _time_to_y(t_time) + y_off
                    x_mm = rpitch_to_x(x_rp) + x_off
                    try:
                        w_mm, h_mm, offset_down, rot_corners, rot_poly = _text_bbox(display_txt, family, size_pt, italic, bold, angle, pad_mm, pad_mm)
                    except Exception:
                        continue
                    cy = y_mm + offset_down
                    poly = [(x_mm + dx, cy + dy) for (dx, dy) in rot_poly]
                    du.add_polygon(
                        poly,
                        stroke_color=None,
                        fill_color=(1.0, 1.0, 1.0, 1.0),
                        id=int(tx.get('id', 0) or 0),
                        tags=['text'],
                    )
                    du.add_text(
                        x_mm,
                        cy,
                        display_txt,
                        family=family,
                        size_pt=size_pt,
                        italic=italic,
                        bold=bold,
                        color=(0.0, 0.0, 0.0, 1.0),
                        anchor='center',
                        angle_deg=angle,
                        id=int(tx.get('id', 0) or 0),
                        tags=['text'],
                    )

            if line_slurs:
                side_w = float(layout.get('slur_width_sides_mm', 0.1) or 0.1) * scale
                mid_w = float(layout.get('slur_width_middle_mm', 1.5) or 1.5) * scale
                n_seg = 50

                def tri_interp(t: float) -> float:
                    return max(0.0, 1.0 - abs(2.0 * t - 1.0))

                def width_at(t: float) -> float:
                    return side_w + (mid_w - side_w) * tri_interp(t)

                for sl in line_slurs:
                    x1 = rpitch_to_x(float(sl.get('x1_rpitch', 0) or 0))
                    x2 = rpitch_to_x(float(sl.get('x2_rpitch', 0) or 0))
                    x3 = rpitch_to_x(float(sl.get('x3_rpitch', 0) or 0))
                    x4 = rpitch_to_x(float(sl.get('x4_rpitch', 0) or 0))
                    t1 = float(sl.get('y1_time', 0.0) or 0.0)
                    t2 = float(sl.get('y2_time', 0.0) or 0.0)
                    t3 = float(sl.get('y3_time', 0.0) or 0.0)
                    t4 = float(sl.get('y4_time', 0.0) or 0.0)
                    y1_sl = _time_to_y(t1)
                    y2_sl = _time_to_y(t2)
                    y3_sl = _time_to_y(t3)
                    y4_sl = _time_to_y(t4)

                    pts: list[tuple[float, float]] = []
                    for i in range(n_seg):
                        if n_seg <= 1:
                            t = 0.0
                        else:
                            t = i / float(n_seg - 1)
                        omt = 1.0 - t
                        bx = (
                            omt * omt * omt * x1
                            + 3 * omt * omt * t * x2
                            + 3 * omt * t * t * x3
                            + t * t * t * x4
                        )
                        by = (
                            omt * omt * omt * y1_sl
                            + 3 * omt * omt * t * y2_sl
                            + 3 * omt * t * t * y3_sl
                            + t * t * t * y4_sl
                        )
                        pts.append((bx, by))

                    for i in range(len(pts) - 1):
                        if n_seg <= 1:
                            t_mid = 0.0
                        else:
                            t_mid = (i + 0.5) / float(n_seg - 1)
                        w_slur = width_at(t_mid)
                        x_a, y_a = pts[i]
                        x_b, y_b = pts[i + 1]
                        du.add_line(
                            x_a,
                            y_a,
                            x_b,
                            y_b,
                            color=(0, 0, 0, 1),
                            width_mm=w_slur,
                            id=int(sl.get('id', 0) or 0),
                            tags=['slur'],
                        )

            x_cursor = x_cursor + float(line['total_width']) + gap


    # Ensure a valid current page index
    if du.page_count() > 0:
        if pdf_export:
            du.set_current_page(0)
        else:
            du.set_current_page(target_page_index)


def _engrave_worker(score: dict, request_id: int, pageno: int, out_queue) -> None:
    """Worker entry point to build DrawUtil in a separate process.

    Problem solved: isolate heavy engraving work from the UI thread.
    """
    local_du = DrawUtil()
    do_engrave(score, local_du, pageno=pageno)
    out_queue.put((int(request_id), local_du))


class Engraver(QtCore.QObject):
    """Convenient engraver API ensuring single-run with latest-request semantics.

    - Call engrave(score) to request an engraving.
    - If one is running, stores the latest pending request and runs it next.
    - Skips intermediate requests; never runs two tasks at the same time.
    """

    engraved = QtCore.Signal()

    def __init__(self, draw_util: DrawUtil, parent=None):
        super().__init__(parent)
        self._du = draw_util
        self._mp_ctx = _MP_CONTEXT
        self._result_queue = self._mp_ctx.Queue()
        self._proc: mp.Process | None = None
        self._poll_timer = QtCore.QTimer(self)
        self._poll_timer.setInterval(50)
        self._poll_timer.timeout.connect(self._poll_results)
        self._running: bool = False
        self._pending_score: dict | None = None
        self._pending_pageno: int | None = None
        self._pending_request_id: int | None = None
        self._latest_request_id: int = 0
        self._min_interval_ms: int = 500
        self._last_start_ms: int = -500
        self._elapsed = QtCore.QElapsedTimer()
        self._elapsed.start()
        self._delay_timer = QtCore.QTimer(self)
        self._delay_timer.setSingleShot(True)
        self._delay_timer.timeout.connect(self._maybe_start_pending)
        self.analysis: Analysis | None = None

    def engrave(self, score: dict, pageno: int | None = None) -> None:
        """Request an engraving; coalesce to the most recent request.

        Problem solved: avoid a backlog of obsolete renders during edits.
        """
        if pageno is None:
            try:
                pageno = int(self._du.current_page_index())
            except Exception:
                pageno = 0
        self._latest_request_id += 1
        req_id = int(self._latest_request_id)
        # If currently running, just replace the pending request
        if self._running:
            self._pending_score = dict(score or {})
            self._pending_pageno = int(pageno)
            self._pending_request_id = req_id
            return
        self._pending_score = dict(score or {})
        self._pending_pageno = int(pageno)
        self._pending_request_id = req_id
        self._maybe_start_pending()

    def _maybe_start_pending(self) -> None:
        """Start a pending request if throttling allows it.

        Problem solved: rate-limit engraving so rapid edits do not spawn too
        many processes.
        """
        if self._running:
            return
        if self._pending_score is None:
            return
        if self._pending_pageno is None:
            return
        if self._pending_request_id is None:
            return
        elapsed_ms = int(self._elapsed.elapsed())
        since_last = elapsed_ms - int(self._last_start_ms)
        if since_last >= self._min_interval_ms:
            next_score = self._pending_score
            next_pageno = int(self._pending_pageno)
            next_req_id = int(self._pending_request_id)
            self._pending_score = None
            self._pending_pageno = None
            self._pending_request_id = None
            self._start_task(next_score, next_pageno, next_req_id)
            return
        delay_ms = max(1, int(self._min_interval_ms - since_last))
        if self._delay_timer.isActive():
            self._delay_timer.stop()
        self._delay_timer.start(delay_ms)

    def _start_task(self, score: dict, pageno: int, request_id: int) -> None:
        """Start a new process to engrave the given score.

        Problem solved: terminate stale workers before launching a new one.
        """
        self._running = True
        self._last_start_ms = int(self._elapsed.elapsed())
        if self._proc is not None:
            if self._proc.is_alive():
                self._proc.terminate()
        self._proc = self._mp_ctx.Process(
            target=_engrave_worker,
            args=(score, request_id, pageno, self._result_queue),
            daemon=True,
        )
        self._proc.start()
        if not self._poll_timer.isActive():
            self._poll_timer.start()

    def _poll_results(self) -> None:
        """Drain worker results and advance the queue.

        Problem solved: process can exit without a result; this keeps the
        state machine moving and restarts pending work.
        """
        got_result = False
        while True:
            try:
                req_id, result_du = self._result_queue.get_nowait()
            except queue.Empty:
                break
            got_result = True
            self._on_finished(req_id, result_du)

        if self._proc is not None and not self._proc.is_alive():
            self._proc.join(timeout=0)
            self._proc = None
            if self._running and not got_result:
                self._running = False
                if self._pending_score is not None:
                    self._maybe_start_pending()
            if not self._running:
                self._poll_timer.stop()

    def shutdown(self) -> None:
        """Stop timers and terminate the worker process if it is still running.

        Problem solved: prevent orphan processes on app shutdown.
        """
        if self._poll_timer.isActive():
            self._poll_timer.stop()
        if self._proc is not None:
            if self._proc.is_alive():
                self._proc.terminate()
            self._proc.join(timeout=0.1)
            self._proc = None

    @QtCore.Slot(int, object)
    def _on_finished(self, request_id: int, result_du: DrawUtil) -> None:
        # Called on worker completion; schedule next or emit signal
        self._running = False
        if self._pending_score is not None:
            # Grab and clear the latest pending, then run it
            self._maybe_start_pending()
            return
        # No pending: notify listeners (e.g., to request render)
        if int(request_id) == int(self._latest_request_id):
            self._du._pages = list(result_du._pages)
            self._du._current_index = int(result_du._current_index)
            self.analysis = getattr(result_du, 'analysis', None)
            try:
                self._du.analysis = self.analysis
            except Exception:
                pass
            self.engraved.emit()