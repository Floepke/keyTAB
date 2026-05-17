import bisect
import multiprocessing as mp

from file_model.base_grid import resolve_grid_layer_offsets
from symbol_design.noteheads import normalize_notehead_literal, resolve_notehead_spec
from utils.CONSTANT import BLACK_KEYS, QUARTER_NOTE_UNIT, SHORTEST_DURATION
from utils.operator import Operator


def is_light_paper(rgb_tuple: tuple[int, int, int]) -> bool:
    r = float(rgb_tuple[0]) / 255.0
    g = float(rgb_tuple[1]) / 255.0
    b = float(rgb_tuple[2]) / 255.0
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return lum >= 0.5


def scaled_dash_pattern_with_default(raw_value, fallback_mm: list[float], local_scale: float) -> list[float] | None:
    def _parse_pattern(value) -> list[float]:
        parsed_local: list[float] = []
        try:
            if isinstance(value, str):
                chunks = str(value).replace(',', ' ').split()
                parsed_local = [float(v) for v in chunks]
            elif isinstance(value, (list, tuple)):
                parsed_local = [float(v) for v in value]
            elif value is not None:
                parsed_local = [float(value)]
        except Exception:
            parsed_local = []

        cleaned: list[float] = []
        for v in parsed_local:
            fv = float(v)
            if fv < 0.0:
                continue
            cleaned.append(fv)
        return cleaned

    raw_pattern = _parse_pattern(raw_value)
    fallback_pattern = _parse_pattern(fallback_mm)

    # Explicit user input wins over fallback, even when it means solid.
    selected = raw_pattern if raw_pattern else fallback_pattern
    if not selected:
        selected = [3.0]

    # A single 0 means "solid" in style fields.
    if not any(float(v) > 0.0 for v in selected):
        return None

    return [float(v) * float(local_scale) for v in selected]


def build_grid_band_dark_intervals(markers: list, bars: list[float], total_len: float, starts_dark: bool = True) -> list[tuple[float, float]]:
    op = Operator(SHORTEST_DURATION)
    if op.le(float(total_len), 0.0):
        return []
    if not markers:
        return []

    bar_times = [
        float(b)
        for b in (bars or [])
        if op.ge(float(b), 0.0) and op.le(float(b), float(total_len))
    ]
    bar_times = sorted(list(dict.fromkeys(round(float(v), 6) for v in bar_times)))
    if not bar_times or op.not_equal(float(bar_times[0]), 0.0):
        bar_times = [0.0] + bar_times
    if op.not_equal(float(bar_times[-1]), float(total_len)):
        bar_times.append(float(total_len))

    track: list[tuple[float, float, int]] = []
    for mk in markers:
        try:
            if isinstance(mk, dict):
                mt = float(mk.get('time', 0.0) or 0.0)
                dur = float(mk.get('duration', 0.0) or 0.0)
                mid = int(mk.get('_id', mk.get('id', 0)) or 0)
            else:
                mt = float(getattr(mk, 'time', 0.0) or 0.0)
                dur = float(getattr(mk, 'duration', 0.0) or 0.0)
                mid = int(getattr(mk, '_id', getattr(mk, 'id', 0)) or 0)
        except Exception:
            continue
        if op.lt(dur, 0.0):
            continue
        if op.ge(mt, float(total_len)):
            continue
        track.append((max(0.0, mt), dur, mid))

    if not track:
        return []
    track.sort(key=lambda x: (float(x[0]), int(x[2])))

    segments: list[tuple[float, float, float]] = []
    for i, (start, step, _mid) in enumerate(track):
        end = float(track[i + 1][0]) if (i + 1) < len(track) else float(total_len)
        if op.le(end, start):
            continue
        segments.append((float(start), float(end), float(step)))

    if not segments:
        return []

    out: list[tuple[float, float]] = []
    for bi in range(len(bar_times) - 1):
        bar_start = float(bar_times[bi])
        bar_end = float(bar_times[bi + 1])
        if op.le(bar_end, bar_start):
            continue

        is_dark = bool(starts_dark)
        for seg_start_raw, seg_end_raw, step in segments:
            if op.le(seg_end_raw, bar_start):
                continue
            if op.ge(seg_start_raw, bar_end):
                break

            seg_start = max(float(seg_start_raw), float(bar_start))
            seg_end = min(float(seg_end_raw), float(bar_end))
            if op.le(seg_end, seg_start):
                continue

            if op.le(step, 0.0):
                is_dark = bool(starts_dark)
                continue

            t0 = float(seg_start)
            color = bool(is_dark)
            while op.lt(t0, seg_end):
                t1 = min(float(seg_end), float(t0 + step))
                if color and op.gt(t1, t0):
                    out.append((float(t0), float(t1)))
                t0 = float(t1)
                color = not color

            is_dark = bool(color)

    if not out:
        return []

    out.sort(key=lambda x: (float(x[0]), float(x[1])))
    merged: list[list[float]] = []
    for s, e in out:
        fs = float(s)
        fe = float(e)
        if (not merged) or op.gt(fs, float(merged[-1][1])):
            merged.append([fs, fe])
        elif op.gt(fe, float(merged[-1][1])):
            merged[-1][1] = fe

    return [(a, b) for a, b in merged if op.gt(float(b), float(a))]


def normalize_hex_color(value: str | None) -> str | None:
    if value is None:
        return None
    txt = str(value).strip()
    if not txt:
        return None
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


def allow_font_registry() -> bool:
    return mp.current_process().name == "MainProcess"


def resolve_font_family_name(family: str) -> str:
    if not allow_font_registry():
        return family
    from fonts import resolve_font_family
    return str(resolve_font_family(family))


def note_has_continuation_dot_in_beam_window(note: dict, notes_sorted: list[dict], barlines: list[float], t0: float, t1: float) -> bool:
    op = Operator(SHORTEST_DURATION)
    n_start = float(note.get('time', 0.0) or 0.0)
    n_end = float(note.get('end', n_start + float(note.get('duration', 0.0) or 0.0)) or 0.0)
    if not (op.lt(n_start, float(t0)) and op.gt(n_end, float(t0))):
        return False

    note_idx = int(note.get('idx', note.get('id', 0)) or 0)
    note_hand = str(note.get('hand', 'l') or 'l')

    for other in notes_sorted:
        other_idx = int(other.get('idx', other.get('id', 0)) or 0)
        if other_idx == note_idx:
            continue
        if str(other.get('hand', 'l') or 'l') != note_hand:
            continue
        s = float(other.get('time', 0.0) or 0.0)
        e = float(other.get('end', s + float(other.get('duration', 0.0) or 0.0)) or 0.0)
        if op.gt(s, n_start) and op.lt(s, n_end) and op.ge(s, float(t0)) and op.lt(s, float(t1)):
            return True
        if op.gt(e, n_start) and op.lt(e, n_end) and op.ge(e, float(t0)) and op.lt(e, float(t1)):
            return True

    for bt in barlines:
        bt = float(bt)
        if op.gt(bt, n_start) and op.lt(bt, n_end) and op.ge(bt, float(t0)) and op.lt(bt, float(t1)):
            return True

    return False


def assign_beam_groups(notes_sorted: list[dict], windows: list[tuple[float, float]], barlines: list[float]) -> list[list[dict]]:
    op = Operator(SHORTEST_DURATION)
    if not notes_sorted or not windows:
        return []
    starts = [float(n.get('time', 0.0) or 0.0) for n in notes_sorted]
    ends = [float(n.get('end', 0.0) or 0.0) for n in notes_sorted]
    result: list[list[dict]] = []
    j = 0
    for (t0, t1) in windows:
        j = bisect.bisect_left(starts, float(t0) - float(op.threshold), j)
        group: list[dict] = []
        k = j
        while k < len(starts):
            s = starts[k]
            if op.ge(s, float(t1) + float(op.threshold)):
                break
            e = ends[k]
            if op.gt(e, float(t0)) and op.lt(s, float(t1)):
                group.append(notes_sorted[k])
            k += 1
        b = j - 1
        while b >= 0:
            s = starts[b]
            e = ends[b]
            if op.gt(e, float(t0)) and op.lt(s, float(t1)):
                n = notes_sorted[b]
                if note_has_continuation_dot_in_beam_window(n, notes_sorted, barlines, float(t0), float(t1)):
                    group.append(n)
            b -= 1
        if group:
            keyed: dict[int, dict] = {}
            for m in group:
                key_id = int(m.get('idx', m.get('id', 0)) or 0)
                keyed[key_id] = m
            group = sorted(keyed.values(), key=lambda n: float(n.get('time', 0.0) or 0.0))
        result.append(group)
    return result


def build_grid_windows(base_grid: list[dict], a: float, b: float) -> list[tuple[float, float]]:
    op = Operator(SHORTEST_DURATION)
    windows: list[tuple[float, float]] = []
    cur = 0.0
    for bg in base_grid:
        numer = int(bg.get('numerator', 4) or 4)
        denom = int(bg.get('denominator', 4) or 4)
        measures = int(bg.get('measure_amount', 1) or 1)
        seq = list(bg.get('beat_grouping', []) or [])
        measure_len = float(numer) * (4.0 / float(max(1, denom))) * float(QUARTER_NOTE_UNIT)
        _bar_offsets, grid_offsets = resolve_grid_layer_offsets(seq, numer, denom)
        for _ in range(int(measures)):
            m_start = float(cur)
            m_end = float(cur + measure_len)
            if op.lt(m_end, float(a)):
                cur = m_end
                continue
            if op.gt(m_start, float(b)):
                cur = m_end
                continue
            boundaries = [0.0] + [float(v) for v in grid_offsets if 0.0 < float(v) < measure_len] + [float(measure_len)]
            boundaries = sorted(dict.fromkeys(round(v, 6) for v in boundaries))
            if len(boundaries) < 2:
                boundaries = [0.0, float(measure_len)]
            for idx in range(len(boundaries) - 1):
                w0 = m_start + float(boundaries[idx])
                w1 = m_start + float(boundaries[idx + 1])
                w0 = max(float(a), w0)
                w1 = min(float(b), w1)
                if op.lt(w0, w1):
                    windows.append((w0, w1))
            cur = m_end
    return windows


def process_beam_marker_override(default_windows: list[tuple[float, float]], markers: list[dict]) -> list[tuple[float, float]]:
    op = Operator(SHORTEST_DURATION)
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
            if op.ge(w0, end) or op.le(w1, mt):
                filtered.append((w0, w1))
        if dur > 0.0:
            filtered.append((mt, end))
        windows = sorted(filtered, key=lambda w: float(w[0]))
    return windows


def group_by_beam_markers(notes: list[dict], markers: list[dict], start: float, end: float, base_grid: list[dict], barlines: list[float]) -> tuple[list[list[dict]], list[tuple[float, float]]]:
    notes_sorted = sorted(notes, key=lambda n: float(n.get('time', 0.0) or 0.0)) if notes else []
    default_windows = build_grid_windows(base_grid, start, end)
    windows = process_beam_marker_override(default_windows, markers)
    groups = assign_beam_groups(notes_sorted, windows, barlines) if notes_sorted else []
    return groups, windows


def black_note_above_stem(item: dict, rule: str, notes: list[dict]) -> bool:
    op = Operator(SHORTEST_DURATION)
    
    if rule == 'above_stem':
        return True
    
    p0 = int(item.get('pitch', 0) or 0)
    t0 = float(item.get('time', 0.0) or 0.0)
    idx0 = int(item.get('idx', -1) or -1)
    
    if rule == 'above_stem_if_collision':
        for n in notes:
            if int(n.get('idx', -2) or -2) == idx0:
                continue
            if not op.eq(float(n.get('time', 0.0) or 0.0), t0):
                continue
            if abs(int(n.get('pitch', 0) or 0) - p0) == 1:
                return True
        return False
    
    # NOTE: the original 'above_stem_if_chord_and_white_note' rule 
    # is removed to minimize the available options + it seemed unnecessary.
    if rule == 'above_stem_if_chord_and_white_note':
        rule = 'above_stem_if_chord_and_white_note_same_hand'
    
    if rule != 'above_stem_if_chord_and_white_note_same_hand':
        return False
    hand0 = str(item.get('hand', 'l') or 'l')
    for n in notes:
        if int(n.get('idx', -2) or -2) == idx0:
            continue
        if not op.eq(float(n.get('time', 0.0) or 0.0), t0):
            continue
        if str(n.get('hand', 'l') or 'l') != hand0:
            continue
        np = int(n.get('pitch', 0) or 0)
        if np not in BLACK_KEYS and np != p0:
            return True
    return False


def should_tune_under_stem_black_width(item: dict, rule: str, notes: list[dict]) -> bool:
    op = Operator(SHORTEST_DURATION)
    rule_norm = str(rule or 'below_stem').strip().lower()
    if rule_norm not in ('under_stem', 'below_stem'):
        return False

    raw_note = item.get('raw', item)
    if isinstance(raw_note, dict):
        custom_notehead = normalize_notehead_literal(raw_note.get('notehead', 'auto'))
    else:
        custom_notehead = normalize_notehead_literal(getattr(raw_note, 'notehead', 'auto'))
    if custom_notehead != 'auto':
        custom_spec = resolve_notehead_spec(raw_note, default_black_above=False)
        if bool(getattr(custom_spec, 'is_up', False)):
            return False

    p0 = int(item.get('pitch', 0) or 0)
    if p0 not in BLACK_KEYS:
        return False
    t0 = float(item.get('time', 0.0) or 0.0)
    idx0 = int(item.get('idx', -1) or -1)
    for n in notes:
        if int(n.get('idx', -2) or -2) == idx0:
            continue
        if not op.eq(float(n.get('time', 0.0) or 0.0), t0):
            continue
        if abs(int(n.get('pitch', 0) or 0) - p0) == 1:
            return True
    return False


def time_to_y(line: dict, ticks: float) -> float:
    """Convert time (ticks) to y coordinate for a given line dict."""
    t0 = float(line.get('time_start', 0.0) or 0.0)
    t1 = float(line.get('time_end', t0) or t0)
    y0 = float(line.get('y_top', 0.0) or 0.0)
    y1 = float(line.get('y_bottom', y0) or y0)
    denom = max(1e-6, t1 - t0)
    rel = max(0.0, min(1.0, (float(ticks) - t0) / denom))
    return y0 + ((y1 - y0) * rel)
