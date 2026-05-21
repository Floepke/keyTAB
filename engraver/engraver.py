from PySide6 import QtCore
from datetime import datetime
import bisect, math
import multiprocessing as mp
import traceback
from ui.widgets.draw_util import DrawUtil
from utils.CONSTANT import BE_KEYS, QUARTER_NOTE_UNIT, PIANO_KEY_AMOUNT, SHORTEST_DURATION, hex_to_rgba, BLACK_KEYS, ENGRAVER_FRACTIONAL_TEXT_SCALING_CORRECTION, SLUR_SEGMENT_COUNT
from utils.tiny_tool import key_class_filter
from utils.operator import Operator
from file_model.SCORE import SCORE
from file_model.layout import Layout
from file_model.base_grid import resolve_grid_layer_offsets
from file_model.info import Info
from file_model.analysis import Analysis
from ui.style import Style
from symbol_design.noteheads import (
    Notehead,
    normalize_notehead_literal,
    resolve_notehead_spec,
    sheared_notehead_outline_points,
    support_point_from_outline_points,
    sheared_notehead_support_v,
)
from symbol_design.pedal import draw_pedal_symbol
from file_model.events.note import Note
from engraver.helpers import (
    allow_font_registry as _allow_font_registry,
    black_note_above_stem as _black_note_above_stem,
    build_stem_segments_for_chords as _build_stem_segments_for_chords,
    build_grid_band_dark_intervals as _build_grid_band_dark_intervals,
    group_by_beam_markers as _group_by_beam_markers,
    is_light_paper as _is_light_paper,
    normalize_hand as _normalize_hand,
    normalize_hex_color as _normalize_hex_color,
    resolve_font_family_name as _resolve_font_family,
    scaled_dash_pattern_with_default as _scaled_dash_pattern_with_default,
    should_tune_under_stem_black_width as _should_tune_under_stem_black_width,
)
from engraver.drawers.stave_drawer import stave_drawer
from engraver.drawers.note_drawer import note_drawer
from engraver.drawers.arpeggio_drawer import arpeggio_drawer
from engraver.drawers.count_line_drawer import count_line_drawer
from engraver.drawers.dynamic_drawer import dynamic_drawer
from engraver.drawers.grace_note_drawer import grace_note_drawer
from engraver.drawers.grid_band_drawer import grid_band_drawer
from engraver.drawers.grid_drawer import grid_drawer
from engraver.drawers.pedal_drawer import pedal_drawer
from engraver.drawers.repeat_drawer import repeat_drawer
from engraver.drawers.slur_drawer import slur_drawer
from engraver.drawers.tempo_drawer import tempo_drawer
from engraver.drawers.text_drawer import text_drawer
from engraver.drawers.time_signature_drawer import time_signature_drawer

_MP_CONTEXT = mp.get_context("spawn")

def _grace_layout_no_tilt(layout: dict) -> dict:
    """Return layout dict copy with notehead_tilt set to 0 for grace notes visual contrast."""
    layout_copy = dict(layout)
    layout_copy['notehead_tilt'] = 0.0
    return layout_copy

def do_engrave(score: SCORE, du: DrawUtil, pageno: int = 0, pdf_export: bool = False) -> None:
    """Compute a full print layout and draw commands into DrawUtil.

    Problem solved: the engraver must be deterministic and thread-safe.
    It converts the score model into page/line geometry without any Qt
    rendering calls, then records only DrawUtil primitives.
    """
    score = score or {}
    layout: dict = dict(score.get('layout', {}) or {})
    base_grid: list = list(score.get('base_grid', []) or [])
    staves_raw: list = list(score.get('staves', []) or [])

    scale = float(layout.get('scale', 1.0) or 1.0)
    black_key_set = set(BLACK_KEYS)
    fga_keys = set(key_class_filter('FGA')) # all f# g# a# keys from key 1..88
    be_keys = set(BE_KEYS)
    clef_low_key = 41
    clef_high_key = 43

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

    def _scaled(value: float) -> float:
        return float(value) * float(scale)

    def _page_dimensions() -> tuple[float, float, float, float, float, float]:
        # Page dimensions and page margins are never multiplied by layout.scale.
        orientation = str(layout.get('page_orientation', 'portrait') or 'portrait').strip().lower()
        page_w = float(layout.get('page_width_mm', 210.0) or 210.0)
        page_h = float(layout.get('page_height_mm', 297.0) or 297.0)
        if orientation == 'landscape':
            page_w, page_h = page_h, page_w
        page_left = float(layout.get('page_left_margin_mm', 10.0) or 10.0)
        page_right = float(layout.get('page_right_margin_mm', 10.0) or 10.0)
        page_top = float(layout.get('page_top_margin_mm', 10.0) or 10.0)
        page_bottom = float(layout.get('page_bottom_margin_mm', 10.0) or 10.0)
        return page_w, page_h, page_left, page_right, page_top, page_bottom

    def _collect_enabled_staves() -> list[dict]:
        enabled: list[dict] = []
        for idx, st in enumerate(staves_raw):
            if not isinstance(st, dict):
                continue
            if not bool(st.get('enabled', True)):
                continue
            st_events = st.get('events', None)
            if not isinstance(st_events, dict):
                st_events = {}
            st_scale = float(st.get('scale', 1.0) or 1.0)
            enabled.append({'index': int(idx), 'events': st_events, 'stave_scale': st_scale})
        if not enabled:
            enabled.append({'index': 0, 'events': staves_raw[0].get('events', {}) if staves_raw else {}, 'stave_scale': staves_raw[0].get('scale', 1.0) if staves_raw else 1.0})
        return enabled

    def _total_ticks(enabled_staves: list[dict]) -> float:
        total = 0.0
        for bg in base_grid:
            numer = int(_item_get(bg, 'numerator', 4) or 4)
            denom = int(_item_get(bg, 'denominator', 4) or 4)
            measures = int(_item_get(bg, 'measure_amount', 1) or 1)
            measure_ticks = float(numer) * (4.0 / float(max(1, denom))) * float(QUARTER_NOTE_UNIT)
            total += measure_ticks * float(max(0, measures))
        if total > 0.0:
            return total
        max_end = 0.0
        for st in enabled_staves:
            notes = list((st.get('events', {}) or {}).get('note', []) or [])
            for n in notes:
                t0 = _item_get_float(n, 'time', 0.0)
                dur = _item_get_float(n, 'duration', 0.0)
                max_end = max(max_end, t0 + dur)
        return max(float(QUARTER_NOTE_UNIT), max_end)

    def _normalize_line_breaks(events_dict: dict) -> list[dict]:
        raw = list(events_dict.get('line_break', []) or [])
        out: list[dict] = []
        for lb in raw:
            t = float(_item_get(lb, 'time', 0.0) or 0.0)
            margin_raw = _item_get(lb, 'margin_mm', [5.0, 5.0])
            if isinstance(margin_raw, (list, tuple)) and len(margin_raw) >= 2:
                # Keep raw model values here; per-stave scaling is applied later.
                margin_mm = [float(margin_raw[0] or 0.0), float(margin_raw[1] or 0.0)]
            else:
                margin_mm = [5.0, 5.0]
            sr = _item_get(lb, 'stave_range', 'auto')
            if isinstance(sr, (list, tuple)) and len(sr) >= 2:
                lo = int(sr[0] or 1)
                hi = int(sr[1] or PIANO_KEY_AMOUNT)
                if hi < lo:
                    lo, hi = hi, lo
                stave_range = [max(1, lo), min(PIANO_KEY_AMOUNT, hi)]
            else:
                stave_range = 'auto'
            out.append(
                {
                    'time': float(t),
                    'margin_mm': margin_mm,
                    'stave_range': stave_range,
                    'page_break': bool(_item_get(lb, 'page_break', False)),
                }
            )
        if not out:
            out = [{'time': 0.0, 'margin_mm': [5.0, 5.0], 'stave_range': 'auto', 'page_break': False}]
        out.sort(key=lambda x: float(x.get('time', 0.0) or 0.0))
        if float(out[0].get('time', 0.0) or 0.0) > 0.0:
            out.insert(0, {'time': 0.0, 'margin_mm': [5.0, 5.0], 'stave_range': 'auto', 'page_break': False})
        return out

    def _line_break_for_time(lb_list: list[dict], t: float) -> dict:
        if not lb_list:
            return {'time': 0.0, 'margin_mm': [5.0, 5.0], 'stave_range': 'auto', 'page_break': False}
        active = lb_list[0]
        for lb in lb_list:
            if float(lb.get('time', 0.0) or 0.0) <= float(t):
                active = lb
            else:
                break
        return active

    base_semitone_model_mm = 2.0

    def _build_key_offsets(semitone_step_mm: float) -> dict[int, float]:
        offsets: dict[int, float] = {}
        x_pos = -float(semitone_step_mm)
        for n in range(1, PIANO_KEY_AMOUNT + 1):
            if (n - 1) in be_keys:
                x_pos += float(semitone_step_mm)
            x_pos += float(semitone_step_mm)
            offsets[n] = float(x_pos)
        return offsets

    def _span_width_mm(lo_key: int, hi_key: int, key_offsets_local: dict[int, float], semitone_step_mm: float) -> float:
        lo = max(1, min(PIANO_KEY_AMOUNT, int(lo_key)))
        hi = max(1, min(PIANO_KEY_AMOUNT, int(hi_key)))
        if hi < lo:
            lo, hi = hi, lo
        step = max(0.01, float(semitone_step_mm))
        # Width uses key-to-key distance; adding an extra step here made
        # system content/outer boxes appear exactly one semitone too wide.
        return max(step, float(key_offsets_local[hi] - key_offsets_local[lo]))

    groups: list[dict] = [{'kind': 'single', 'start': 1, 'end': 3}]
    black_keys_sorted = sorted(list(black_key_set))
    if black_keys_sorted:
        run: list[int] = [int(black_keys_sorted[0])]
        run_kind = 'three' if int(black_keys_sorted[0]) in fga_keys else 'two'
        for key in black_keys_sorted[1:]:
            k = int(key)
            k_kind = 'three' if k in fga_keys else 'two'
            # Group symbols follow black-key order by kind (two-line or three-line),
            # not numeric adjacency of absolute key numbers.
            if k_kind == run_kind:
                run.append(k)
            else:
                groups.append({'kind': run_kind, 'start': int(min(run)), 'end': int(max(run))})
                run = [k]
                run_kind = k_kind
        groups.append({'kind': run_kind, 'start': int(min(run)), 'end': int(max(run))})

    black_keys_sorted = sorted([int(k) for k in black_keys_sorted])

    def _nearest_black_floor(key: int) -> int:
        k = max(1, min(PIANO_KEY_AMOUNT, int(key)))
        floor_vals = [bk for bk in black_keys_sorted if bk <= k]
        if floor_vals:
            return int(floor_vals[-1])
        return int(black_keys_sorted[0]) if black_keys_sorted else int(k)

    def _nearest_black_ceil(key: int) -> int:
        k = max(1, min(PIANO_KEY_AMOUNT, int(key)))
        ceil_vals = [bk for bk in black_keys_sorted if bk >= k]
        if ceil_vals:
            return int(ceil_vals[0])
        return int(black_keys_sorted[-1]) if black_keys_sorted else int(k)

    def _nearest_black(key: int) -> int:
        k = max(1, min(PIANO_KEY_AMOUNT, int(key)))
        lo = _nearest_black_floor(k)
        hi = _nearest_black_ceil(k)
        if abs(int(k) - int(lo)) <= abs(int(hi) - int(k)):
            return int(lo)
        return int(hi)

    def _expand_to_group_bounds(lo_key: int, hi_key: int) -> tuple[int, int, list[dict]]:
        lo = max(1, min(PIANO_KEY_AMOUNT, int(lo_key)))
        hi = max(1, min(PIANO_KEY_AMOUNT, int(hi_key)))
        if hi < lo:
            lo, hi = hi, lo
        if black_keys_sorted:
            # Snap each bound to the nearest black-key symbol center to avoid
            # directional one-semitone bias at white-key boundaries.
            lo = _nearest_black(lo)
            hi = _nearest_black(hi)
            if hi < lo:
                lo, hi = hi, lo
        selected = [g for g in groups if not (int(g['end']) < lo or int(g['start']) > hi)]
        if not selected:
            return lo, hi, []
        out_lo = int(min(int(g['start']) for g in selected))
        out_hi = int(max(int(g['end']) for g in selected))
        return out_lo, out_hi, selected

    def _build_lines(enabled_staves: list[dict], total_ticks: float, lb_by_stave: dict[int, list[dict]]) -> list[dict]:
        starts: list[float] = []
        for st in enabled_staves:
            st_idx = int(st.get('index', 0) or 0)
            starts.extend(float(lb.get('time', 0.0) or 0.0) for lb in lb_by_stave.get(st_idx, []))
        starts = sorted(list(dict.fromkeys(starts)))
        if not starts:
            starts = [0.0]
        if starts[0] > 0.0:
            starts.insert(0, 0.0)
        first_idx = int(enabled_staves[0].get('index', 0) or 0)
        first_lbs = lb_by_stave.get(first_idx, [])
        lines: list[dict] = []
        for i, t0 in enumerate(starts):
            t1 = starts[i + 1] if i + 1 < len(starts) else float(total_ticks)
            if t1 <= t0:
                continue
            src = _line_break_for_time(first_lbs, t0)
            lines.append({'start': float(t0), 'end': float(t1), 'page_break': bool(src.get('page_break', False))})
        if not lines:
            lines = [{'start': 0.0, 'end': float(total_ticks), 'page_break': False}]
        return lines

    def _event_in_line(event_type: str, ev, t0: float, t1: float, include_negative_prefix: bool = False) -> bool:
        ev_t = _item_get_float(ev, 'time', 0.0)
        ev_dur = _item_get_float(ev, 'duration', 0.0)
        if include_negative_prefix and ev_t < 0.0:
            return True
        timed_spans = {'note', 'grace_note', 'slur', 'pedal', 'line'}
        if event_type in timed_spans and ev_dur > 0.0:
            ev_end = ev_t + ev_dur
            return not (ev_end <= t0 or ev_t >= t1)
        return t0 <= ev_t < t1

    def _collect_events_for_line(st_events: dict, t0: float, t1: float, include_negative_prefix: bool = False) -> dict:
        event_types = [
            'note', 'arpeggio', 'count_line', 'dynamic', 'grace_note',
            'grid_band', 'grid', 'pedal', 'start_repeat', 'end_repeat',
            'slur', 'tempo', 'text', 'time_signature', 'crescendo', 'decrescendo',
        ]
        out: dict = {}
        for ev_type in event_types:
            items = list(st_events.get(ev_type, []) or [])
            out[ev_type] = [
                ev for ev in items
                if _event_in_line(ev_type, ev, t0, t1, include_negative_prefix=include_negative_prefix)
            ]
        return out

    def _pre_calculate() -> dict:
        page_w, page_h, page_left, page_right, page_top, page_bottom = _page_dimensions()
        enabled_staves = _collect_enabled_staves()
        total_ticks = _total_ticks(enabled_staves)

        lb_by_stave: dict[int, list[dict]] = {}
        for st in enabled_staves:
            st_idx = int(st.get('index', 0) or 0)
            lb_by_stave[st_idx] = _normalize_line_breaks(st.get('events', {}) or {})

        lines_raw = _build_lines(enabled_staves, total_ticks, lb_by_stave)

        y_start = float(page_top)
        y_end = float(page_h - page_bottom)

        measured_systems: list[dict] = []
        first_line_start = float(lines_raw[0].get('start', 0.0) or 0.0) if lines_raw else 0.0
        for si, line in enumerate(lines_raw):
            t0 = float(line.get('start', 0.0) or 0.0)
            t1 = float(line.get('end', t0) or t0)
            include_negative_prefix = bool(si == 0 and first_line_start >= 0.0 and t0 >= 0.0)
            staves_system: list[dict] = []
            system_reserved_width_mm = 0.0

            for st in enabled_staves:
                st_idx = int(st.get('index', 0) or 0)
                st_events = st.get('events', {}) or {}
                stave_scale = float(st.get('stave_scale', 1.0) or 1.0)
                composite_scale = float(scale) * float(stave_scale)
                semitone_mm_stave = float(base_semitone_model_mm) * float(composite_scale)
                key_offsets_stave = _build_key_offsets(semitone_mm_stave)
                lb = _line_break_for_time(lb_by_stave.get(st_idx, []), t0)
                margin_vals = list(lb.get('margin_mm', [5.0, 5.0]) or [5.0, 5.0])
                margin_left = float((margin_vals[0] if len(margin_vals) > 0 else 5.0) * composite_scale)
                margin_right = float((margin_vals[1] if len(margin_vals) > 1 else 5.0) * composite_scale)

                notes = list(st_events.get('note', []) or [])
                pitches: list[int] = []
                for n in notes:
                    nt = _item_get_float(n, 'time', 0.0)
                    nd = _item_get_float(n, 'duration', 0.0)
                    ne = nt + nd
                    if ne <= t0 or nt >= t1:
                        continue
                    p = int(_item_get(n, 'pitch', 0) or 0)
                    if 1 <= p <= PIANO_KEY_AMOUNT:
                        pitches.append(p)

                note_low = int(min(pitches)) if pitches else clef_low_key
                note_high = int(max(pitches)) if pitches else clef_high_key
                raw_range = lb.get('stave_range', 'auto')

                if raw_range == 'auto':
                    mode = 'auto'
                    base_low = min(note_low, clef_low_key)
                    base_high = max(note_high, clef_high_key)
                    manual_range = None
                else:
                    mode = 'manual'
                    sr = list(raw_range if isinstance(raw_range, list) else [1, PIANO_KEY_AMOUNT])
                    if len(sr) < 2:
                        sr = [1, PIANO_KEY_AMOUNT]
                    m_lo = max(1, min(PIANO_KEY_AMOUNT, int(sr[0] or 1)))
                    m_hi = max(1, min(PIANO_KEY_AMOUNT, int(sr[1] or PIANO_KEY_AMOUNT)))
                    if m_hi < m_lo:
                        m_lo, m_hi = m_hi, m_lo
                    manual_range = [int(m_lo), int(m_hi)]
                    # Manual range is forced for stave drawing; do not expand by note range.
                    base_low = int(m_lo)
                    base_high = int(m_hi)

                stave_low, stave_high, stave_groups = _expand_to_group_bounds(base_low, base_high)
                if mode == 'auto':
                    # Keep exact existing auto behavior.
                    span_low, span_high, _span_groups = _expand_to_group_bounds(min(note_low, stave_low), max(note_high, stave_high))
                else:
                    # In manual mode, stave drawing stays forced, but content span
                    # still reserves ledger-line horizontal space equivalent to
                    # what auto mode would reserve for this line.
                    auto_base_low = min(note_low, clef_low_key)
                    auto_base_high = max(note_high, clef_high_key)
                    auto_stave_low, auto_stave_high, _auto_groups = _expand_to_group_bounds(auto_base_low, auto_base_high)
                    auto_span_low, auto_span_high, _auto_span_groups = _expand_to_group_bounds(
                        min(note_low, auto_stave_low),
                        max(note_high, auto_stave_high),
                    )
                    span_low = int(min(stave_low, auto_span_low))
                    span_high = int(max(stave_high, auto_span_high))

                stave_width = _span_width_mm(stave_low, stave_high, key_offsets_stave, semitone_mm_stave)
                stave_content_span_width = _span_width_mm(span_low, span_high, key_offsets_stave, semitone_mm_stave)
                reserve_left_overhang_mm = 0.0
                reserve_right_overhang_mm = 0.0
                total_block = margin_left + reserve_left_overhang_mm + stave_content_span_width + reserve_right_overhang_mm + margin_right
                system_reserved_width_mm += float(total_block)

                staves_system.append(
                    {
                        'stave_index': st_idx,
                        'mode': mode,
                        'manual_range': manual_range,
                        'left_margin_mm': margin_left,
                        'right_margin_mm': margin_right,
                        'note_pitch_low': note_low,
                        'note_pitch_high': note_high,
                        'stave_low_key': stave_low,
                        'stave_high_key': stave_high,
                        'note_span_low_key': span_low,
                        'note_span_high_key': span_high,
                        'stave_width_mm': stave_width,
                        'stave_content_span_width_mm': stave_content_span_width,
                        'reserve_left_overhang_mm': reserve_left_overhang_mm,
                        'reserve_right_overhang_mm': reserve_right_overhang_mm,
                        'group_segments': list(stave_groups),
                        'stave_scale': stave_scale,
                        'composite_scale': composite_scale,
                        'semitone_mm': semitone_mm_stave,
                        'key_offsets': key_offsets_stave,
                        'events_in_line': _collect_events_for_line(
                            st_events,
                            t0,
                            t1,
                            include_negative_prefix=include_negative_prefix,
                        ),
                    }
                )

            measured_systems.append(
                {
                    'system_index': int(si),
                    'time_start': t0,
                    'time_end': t1,
                    'page_break': bool(line.get('page_break', False)),
                    'system_reserved_width_mm': float(system_reserved_width_mm),
                    'staves': staves_system,
                    'y_start_mm': y_start,
                    'y_end_mm': y_end,
                }
            )

        available_w = float(page_w - page_left - page_right)
        pages: list[dict] = []
        current = {'page_index': 0, 'systems': [], 'used_width_mm': 0.0}

        for system in measured_systems:
            if bool(system.get('page_break', False)):
                pages.append(current)
                current = {'page_index': len(pages), 'systems': [], 'used_width_mm': 0.0}

            system_w = float(system.get('system_reserved_width_mm', 0.0) or 0.0)
            if current['systems'] and (float(current['used_width_mm']) + system_w > available_w):
                pages.append(current)
                current = {'page_index': len(pages), 'systems': [], 'used_width_mm': 0.0}

            current['systems'].append(system)
            current['used_width_mm'] = float(current['used_width_mm']) + system_w

        pages.append(current)

        for page in pages:
            used = float(page.get('used_width_mm', 0.0) or 0.0)
            rest = float(available_w - used)
            over = float(max(0.0, -rest))
            rest = float(max(0.0, rest))
            systems_on_page = list(page.get('systems', []) or [])
            system_count = len(systems_on_page)
            # Centering rule after system-level packing:
            # distribute free width over (systems + 1) slots to create equal
            # leading/inter-system/trailing spacing on the page.
            rest_per_slot = (rest / float(system_count + 1)) if system_count > 0 else 0.0

            x_cursor = float(page_left + rest_per_slot)
            for sys in systems_on_page:
                staves_sys = list(sys.get('staves', []) or [])
                system_outer_left_mm = float(x_cursor)
                local_x = float(system_outer_left_mm)
                for stv in staves_sys:
                    ml = float(stv.get('left_margin_mm', 0.0) or 0.0)
                    mr = float(stv.get('right_margin_mm', 0.0) or 0.0)
                    base_span_w = float(stv.get('stave_content_span_width_mm', 0.0) or 0.0)
                    reserve_left = float(stv.get('reserve_left_overhang_mm', 0.0) or 0.0)
                    reserve_right = float(stv.get('reserve_right_overhang_mm', 0.0) or 0.0)
                    span_w = float(base_span_w + reserve_left + reserve_right)
                    span_low = int(stv.get('note_span_low_key', 1) or 1)
                    key_offsets_stave = dict(stv.get('key_offsets', {}) or {})
                    composite_scale = float(stv.get('composite_scale', scale) or scale)

                    span_left = float(local_x + ml - reserve_left)
                    span_right = float(span_left + span_w)

                    def _key_to_x(key: int) -> float:
                        k = max(1, min(PIANO_KEY_AMOUNT, int(key)))
                        if not key_offsets_stave:
                            return float(span_left)
                        return float(span_left + (key_offsets_stave[k] - key_offsets_stave[span_low]))

                    stave_low = int(stv.get('stave_low_key', 1) or 1)
                    stave_high = int(stv.get('stave_high_key', PIANO_KEY_AMOUNT) or PIANO_KEY_AMOUNT)
                    black_lines = []
                    for key in range(stave_low, stave_high + 1):
                        if key not in black_key_set:
                            continue
                        if key in (clef_low_key, clef_high_key):
                            kind = 'clef'
                            width = float(float(layout.get('stave_clef_line_thickness_mm', 0.75) or 0.75) * composite_scale)
                            dash = list(layout.get('stave_clef_line_dash_pattern_mm', [4.0, 3.0]) or [4.0, 3.0])
                            dash = [max(0.01, float(d) * composite_scale) for d in dash]
                        elif key in fga_keys:
                            kind = 'three'
                            width = float(float(layout.get('stave_three_line_thickness_mm', 1.1) or 1.1) * composite_scale)
                            dash = None
                        else:
                            kind = 'two'
                            width = float(float(layout.get('stave_two_line_thickness_mm', 0.5) or 0.5) * composite_scale)
                            dash = None
                        black_lines.append({'key': key, 'x_mm': _key_to_x(key), 'kind': kind, 'width_mm': width, 'dash': dash})

                    stv['stave_content_span_left_mm'] = span_left
                    stv['stave_left_mm'] = _key_to_x(stave_low)
                    stv['black_lines'] = black_lines

                    # Prepare note drawing payload now, so note drawer becomes a thin renderer.
                    events_in_line = dict(stv.get('events_in_line', {}) or {})
                    notes_src = list(events_in_line.get('note', []) or [])
                    note_rule = str(layout.get('black_note_rule', 'above_stem') or 'above_stem')
                    note_refs: list[dict] = []
                    for ni, nraw in enumerate(notes_src):
                        nt = _item_get_float(nraw, 'time', t0)
                        pitch = int(_item_get(nraw, 'pitch', 41) or 41)
                        hand = _normalize_hand(_item_get(nraw, 'hand', 'l'))
                        note_refs.append({'idx': int(ni), 'time': nt, 'pitch': pitch, 'hand': hand, 'raw': nraw})

                    def _time_to_y_sys(ticks: float) -> float:
                        y_start = float(sys.get('y_start_mm', 0.0) or 0.0)
                        y_end = float(sys.get('y_end_mm', y_start) or y_start)
                        denom = max(1e-6, float(t1 - t0))
                        rel = max(0.0, min(1.0, (float(ticks) - float(t0)) / denom))
                        return float(y_start + ((y_end - y_start) * rel))

                    note_draw_items: list[dict] = []
                    for nref in note_refs:
                        nraw = nref.get('raw', {})
                        pitch = int(nref.get('pitch', 41) or 41)
                        nt = _item_get_float(nref, 'time', t0)
                        hand = _normalize_hand(nref.get('hand', 'l'))
                        x_note = _key_to_x(pitch)
                        y_note = _time_to_y_sys(nt)
                        default_black_above = bool(_black_note_above_stem(nref, note_rule, note_refs))
                        note_draw_items.append(
                            {
                                'id': int(_item_get(nraw, '_id', 0) or 0),
                                'x_mm': float(x_note),
                                'y_mm': float(y_note),
                                'time': float(nt),
                                'duration': _item_get_float(nraw, 'duration', 0.0),
                                'pitch': int(pitch),
                                'hand': hand,
                                'is_up': bool(default_black_above),
                                'notehead': str(_item_get(nraw, 'notehead', 'auto') or 'auto'),
                                'beam': bool(_item_get(nraw, 'beam', False)),
                                'continuation_dot': bool(_item_get(nraw, 'continuation_dot', False)),
                                'stop_symbol': bool(_item_get(nraw, 'stop_symbol', False)),
                            }
                        )

                    # Group notes per hand by time-equality (thresholded) so
                    # downstream drawers can treat simultaneous notes as chords.
                    chord_op = Operator(SHORTEST_DURATION)
                    sorted_items = sorted(
                        [dict(it or {}) for it in note_draw_items],
                        key=lambda it: (
                            str(it.get('hand', 'l') or 'l'),
                            _item_get_float(it, 'time', 0.0),
                            int(it.get('pitch', 0) or 0),
                        ),
                    )
                    left_chords: list[tuple[dict, ...]] = []
                    right_chords: list[tuple[dict, ...]] = []
                    i_ch = 0
                    while i_ch < len(sorted_items):
                        cur = sorted_items[i_ch]
                        cur_hand = str(cur.get('hand', 'l') or 'l')
                        cur_t = _item_get_float(cur, 'time', 0.0)
                        bucket: list[dict] = [cur]
                        i_ch += 1
                        while i_ch < len(sorted_items):
                            nxt = sorted_items[i_ch]
                            nxt_hand = str(nxt.get('hand', 'l') or 'l')
                            nxt_t = _item_get_float(nxt, 'time', 0.0)
                            if nxt_hand != cur_hand or not chord_op.eq(float(nxt_t), float(cur_t)):
                                break
                            bucket.append(nxt)
                            i_ch += 1
                        if cur_hand == 'r':
                            right_chords.append(tuple(bucket))
                        else:
                            left_chords.append(tuple(bucket))

                    stem_len_mm = float(layout.get('note_stem_length_semitone', 7.0) or 7.0) * float(semitone_mm_stave)
                    stv['note_draw_items'] = note_draw_items
                    stv['stem_segments'] = _build_stem_segments_for_chords(note_draw_items, stem_len_mm)
                    stv['note_left_chord_list'] = left_chords
                    stv['note_right_chord_list'] = right_chords
                    stv['note_stem_width_mm'] = float(layout.get('note_stem_thickness_mm', 0.8) or 0.8) * float(composite_scale)

                    local_x = float(span_right + mr)

                system_outer_width_mm = float(local_x - system_outer_left_mm)
                if staves_sys:
                    stave_lefts = [float(stv.get('stave_left_mm', system_outer_left_mm)) for stv in staves_sys]
                    stave_rights = [
                        float(stv.get('stave_left_mm', system_outer_left_mm))
                        + float(stv.get('stave_width_mm', 0.0) or 0.0)
                        for stv in staves_sys
                    ]
                    content_lefts = [float(stv.get('stave_content_span_left_mm', system_outer_left_mm)) for stv in staves_sys]
                    content_rights = [
                        float(stv.get('stave_content_span_left_mm', system_outer_left_mm))
                        + float(stv.get('stave_content_span_width_mm', 0.0) or 0.0)
                        for stv in staves_sys
                    ]
                    system_stave_left_mm = float(min(stave_lefts))
                    system_stave_width_mm = float(max(stave_rights) - min(stave_lefts))
                    system_content_left_mm = float(min(content_lefts))
                    system_content_width_mm = float(max(content_rights) - min(content_lefts))
                else:
                    system_stave_left_mm = float(system_outer_left_mm)
                    system_stave_width_mm = float(system_outer_width_mm)
                    system_content_left_mm = float(system_outer_left_mm)
                    system_content_width_mm = float(system_outer_width_mm)

                sys['system_outer_left_mm'] = float(system_outer_left_mm)
                sys['system_outer_width_mm'] = float(system_outer_width_mm)
                sys['system_stave_left_mm'] = float(system_stave_left_mm)
                sys['system_stave_width_mm'] = float(system_stave_width_mm)
                sys['system_content_left_mm'] = float(system_content_left_mm)
                sys['system_content_width_mm'] = float(system_content_width_mm)

                x_cursor = float(system_outer_left_mm + system_outer_width_mm + rest_per_slot)

            page['rest_space_mm'] = rest
            page['over_space_mm'] = over
            page['rest_space_per_slot_mm'] = rest_per_slot

        return {
            'page_width_mm': float(page_w),
            'page_height_mm': float(page_h),
            'page_left_margin_mm': float(page_left),
            'page_right_margin_mm': float(page_right),
            'page_top_margin_mm': float(page_top),
            'page_bottom_margin_mm': float(page_bottom),
            'layout': dict(layout),
            'base_grid': list(base_grid),
            'pages': pages,
        }

    def _draw(precalc: dict) -> None:
        page_w = float(precalc.get('page_width_mm', 210.0) or 210.0)
        page_h = float(precalc.get('page_height_mm', 297.0) or 297.0)
        pages = list(precalc.get('pages', []) or [])
        layout_ctx = dict(precalc.get('layout', {}) or {})
        base_grid_ctx = list(precalc.get('base_grid', []) or [])

        notation_rgb = Style.get_notation_color()
        notation_color = (
            float(notation_rgb[0]) / 255.0,
            float(notation_rgb[1]) / 255.0,
            float(notation_rgb[2]) / 255.0,
            1.0,
        )
        paper_rgb = Style.get_paper_color()
        paper_color = (
            float(paper_rgb[0]) / 255.0,
            float(paper_rgb[1]) / 255.0,
            float(paper_rgb[2]) / 255.0,
            1.0,
        )
        if pdf_export:
            notation_color = (0.0, 0.0, 0.0, 1.0)
            paper_color = (1.0, 1.0, 1.0, 1.0)

        du._pages = []
        du._current_index = -1

        for page in pages:
            du.new_page(page_w, page_h)
            du.add_rectangle(
                0.0,
                0.0,
                float(page_w),
                float(page_h),
                stroke_color=None,
                fill_color=paper_color,
                tags=['paper'],
            )
            for system in list(page.get('systems', []) or []):
                y0 = float(system.get('y_start_mm', 0.0) or 0.0)
                y1 = float(system.get('y_end_mm', 0.0) or 0.0)
                x0 = float(system.get('system_outer_left_mm', 0.0) or 0.0)
                w0 = float(system.get('system_outer_width_mm', 0.0) or 0.0)
                x1 = float(x0 + w0)
                cx0 = float(system.get('system_content_left_mm', x0) or x0)
                cw0 = float(system.get('system_content_width_mm', w0) or w0)
                cx1 = float(cx0 + cw0)

                # Each drawer reads only pre-calculated data and DrawUtil.
                drawer_payload = {
                    'layout': layout_ctx,
                    'base_grid': base_grid_ctx,
                    'notation_color': notation_color,
                    'page': page,
                    'system': system,
                    'y0': y0,
                    'y1': y1,
                    'system_outer_left_mm': x0,
                    'system_outer_width_mm': w0,
                    'system_content_left_mm': cx0,
                    'system_content_width_mm': cw0,
                }
                stave_drawer(du, drawer_payload)
                note_drawer(du, drawer_payload)
                grid_drawer(du, drawer_payload)
                # grid_band_drawer(du, drawer_payload)
                # count_line_drawer(du, drawer_payload)
                # time_signature_drawer(du, drawer_payload)
                # tempo_drawer(du, drawer_payload)
                # text_drawer(du, drawer_payload)
                # dynamic_drawer(du, drawer_payload)
                # pedal_drawer(du, drawer_payload)
                # repeat_drawer(du, drawer_payload)
                # arpeggio_drawer(du, drawer_payload)
                # slur_drawer(du, drawer_payload)
                # grace_note_drawer(du, drawer_payload)

        if du.page_count() > 0:
            try:
                du.set_current_page(max(0, min(int(pageno), du.page_count() - 1)))
            except Exception:
                pass

    pre_calculated = _pre_calculate()
    _draw(pre_calculated)


def _engrave_worker(score: dict, request_id: int, pageno: int, out_conn) -> None:
    """Worker entry point to build DrawUtil in a separate process.

    Problem solved: isolate heavy engraving work from the UI thread.
    """
    try:
        local_du = DrawUtil()
        do_engrave(score, local_du, pageno=pageno)
        out_conn.send(('ok', int(request_id), local_du))
    except Exception as exc:
        try:
            out_conn.send(('error', int(request_id), str(exc), traceback.format_exc()))
        except Exception:
            pass
    finally:
        try:
            out_conn.close()
        except Exception:
            pass


class Engraver(QtCore.QObject):
    """Convenient engraver API ensuring single-run with latest-request semantics.

    - Call engrave(score) to request an engraving.
    - If one is running, stores the latest pending request and runs it next.
    - Skips intermediate requests; never runs two tasks at the same time.
    """

    engraved = QtCore.Signal()
    failed = QtCore.Signal(str, str)

    def __init__(self, draw_util: DrawUtil, parent=None):
        super().__init__(parent)
        self._du: DrawUtil = draw_util
        self._mp_ctx = _MP_CONTEXT
        self._result_recv = None
        self._result_send = None
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

    def _close_result_pipe(self) -> None:
        if self._result_send is not None:
            self._result_send.close()
            self._result_send = None
        if self._result_recv is not None:
            self._result_recv.close()
            self._result_recv = None

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
            self._proc.join(timeout=0.1)
            self._proc = None
        self._close_result_pipe()
        self._result_recv, self._result_send = self._mp_ctx.Pipe(duplex=False)
        self._proc = self._mp_ctx.Process(
            target=_engrave_worker,
            args=(score, request_id, pageno, self._result_send),
            daemon=True,
        )
        self._proc.start()
        if self._result_send is not None:
            try:
                self._result_send.close()
            except Exception:
                pass
            self._result_send = None
        if not self._poll_timer.isActive():
            self._poll_timer.start()

    def _poll_results(self) -> None:
        """Drain worker results and advance the state machine.

        Problem solved: process can exit without a result; this keeps the
        state machine moving and restarts pending work.
        """
        got_result = False
        if self._result_recv is not None:
            try:
                has_result = bool(self._result_recv.poll())
            except (EOFError, OSError):
                has_result = False
            if has_result:
                try:
                    payload = self._result_recv.recv()
                except (EOFError, OSError):
                    pass
                else:
                    got_result = True
                    self._close_result_pipe()
                    kind = str(payload[0]) if isinstance(payload, tuple) and payload else 'ok'
                    if kind == 'ok':
                        _kind, req_id, result_du = payload
                        self._on_finished(req_id, result_du)
                    else:
                        _kind, req_id, error_text, error_details = payload
                        self._on_failed(req_id, str(error_text), str(error_details))

        if self._proc is not None and not self._proc.is_alive():
            self._proc.join(timeout=0.1)
            self._proc = None
            self._close_result_pipe()
            if self._running and not got_result:
                self._running = False
                if self._pending_score is None:
                    self.failed.emit(
                        'Engraving failed',
                        'The engraver worker exited without returning a result.',
                    )
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
        if self._delay_timer.isActive():
            self._delay_timer.stop()
        if self._proc is not None:
            if self._proc.is_alive():
                self._proc.terminate()
            self._proc.join(timeout=0.1)
            self._proc = None
        self._close_result_pipe()
        self._running = False
        self._pending_score = None
        self._pending_pageno = None
        self._pending_request_id = None

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
            self._du.analysis = self.analysis
            self._du.print_time_map = getattr(result_du, 'print_time_map', [])
            self.engraved.emit()

    @QtCore.Slot(int, str, str)
    def _on_failed(self, request_id: int, error_text: str, error_details: str) -> None:
        self._running = False
        if self._pending_score is not None:
            self._maybe_start_pending()
        if int(request_id) == int(self._latest_request_id):
            self.failed.emit(str(error_text or 'Engraving failed'), str(error_details or ''))
