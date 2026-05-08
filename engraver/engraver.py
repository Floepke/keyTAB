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
from symbol_design.noteheads import Notehead, normalize_notehead_literal, resolve_notehead_spec
from symbol_design.pedal import draw_pedal_symbol
from file_model.events.note import Note

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
    events = (score.get('events', {}) or {})
    base_grid = list(score.get('base_grid', []) or [])
    line_breaks = list(events.get('line_break', []) or [])
    notes = list(events.get('note', []) or [])
    grace_notes = list(events.get('grace_note', []) or [])
    count_lines = list(events.get('count_line', []) or [])
    beam_markers = list(events.get('beam', []) or [])
    slurs = list(events.get('slur', []) or [])
    texts = list(events.get('text', []) or [])
    crescendos = list(events.get('crescendo', []) or [])
    decrescendos = list(events.get('decrescendo', []) or [])
    dynamic_symbols = list(events.get('dynamic_symbol', []) or [])
    start_repeats = list(events.get('start_repeat', []) or [])
    end_repeats = list(events.get('end_repeat', []) or [])
    double_bars = list(events.get('double_bar', []) or [])
    tempos = list(events.get('tempo', []) or [])
    pedals = list(events.get('pedal', []) or [])

    # Theme colors
    notation_rgb = Style.get_notation_color()
    paper_rgb = Style.get_paper_color()
    notation_color = (notation_rgb[0] / 255.0, notation_rgb[1] / 255.0, notation_rgb[2] / 255.0, 1.0)
    paper_color = (paper_rgb[0] / 255.0, paper_rgb[1] / 255.0, paper_rgb[2] / 255.0, 1.0)

    if pdf_export:
        # PDF export must stay pure black ink on white paper and preserve raw MIDI colors.
        notation_color = (0.0, 0.0, 0.0, 1.0)
        paper_color = (1.0, 1.0, 1.0, 1.0)

    def _is_light_paper(rgb_tuple: tuple[int, int, int]) -> bool:
        r = float(rgb_tuple[0]) / 255.0
        g = float(rgb_tuple[1]) / 255.0
        b = float(rgb_tuple[2]) / 255.0
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return lum >= 0.5

    def _midi_fill_from_rgb(rgb_tuple: tuple[int, int, int]) -> tuple[float, float, float, float]:
        if pdf_export:
            return (rgb_tuple[0] / 255.0, rgb_tuple[1] / 255.0, rgb_tuple[2] / 255.0, 1.0)
        if _is_light_paper(paper_rgb):
            return (rgb_tuple[0] / 255.0, rgb_tuple[1] / 255.0, rgb_tuple[2] / 255.0, 1.0)
        adjusted = Style.get_contrasting_midi_rgb(rgb_tuple)
        return (adjusted[0] / 255.0, adjusted[1] / 255.0, adjusted[2] / 255.0, 1.0)

    # Problem solved: beam markers are organized per hand for fast grouping later.
    beam_by_hand: dict[str, list[dict]] = {'l': [], 'r': []}
    for b in beam_markers:
        if not isinstance(b, dict):
            continue
        bt = float(b.get('time', 0.0) or 0.0)
        bd = float(b.get('duration', 0.0) or 0.0)
        hand_raw = str(b.get('hand', 'l') or 'l')
        hand_key = 'l' if hand_raw == 'l' else 'r'
        beam_by_hand[hand_key].append({'time': bt, 'duration': bd})
    for hk in beam_by_hand:
        beam_by_hand[hk] = sorted(beam_by_hand[hk], key=lambda m: float(m.get('time', 0.0)))

    # Problem solved: normalize notes once to avoid repeated dict parsing in loops.
    norm_notes: list[dict] = []  # Normalized notes for processing
    notes_by_hand: dict[str, list[dict]] = {'l': [], 'r': []}
    starts_by_hand: dict[str, list[float]] = {'l': [], 'r': []}
    for idx, n in enumerate(notes):
        if not isinstance(n, dict):
            continue
        n_t = float(n.get('time', 0.0) or 0.0)
        n_d = float(n.get('duration', 0.0) or 0.0)
        n_end = n_t + n_d
        p = int(n.get('pitch', 0) or 0)
        hand_raw = str(n.get('hand', 'l') or 'l')
        hand_key = 'l' if hand_raw == 'l' else 'r'
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
            'raw': g,
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

    # Build endpoint map for connected-slur detection.
    # Maps (x_rpitch, y_time_rounded_4dp) → list of slurs sharing that endpoint.
    # A slur is "connected" at an endpoint when ≥2 slurs share the same (x, y) position.
    _slur_ep_map: dict[tuple[int, float], list[dict]] = {}
    for _sl in norm_slurs:
        for _ep_x, _ep_t in (
            (int(_sl['x1_rpitch']), round(float(_sl['y1_time']), 4)),
            (int(_sl['x4_rpitch']), round(float(_sl['y4_time']), 4)),
        ):
            _slur_ep_map.setdefault((_ep_x, _ep_t), []).append(_sl)

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
            'alignment': str(t.get('alignment', 'left') or 'left'),
            'font': t.get('font', None),
            'use_custom_font': bool(t.get('use_custom_font', False)),
            'text_background_width_offset_mm': float(t.get('text_background_width_offset_mm', 0.0) or 0.0),
            'id': int(t.get('_id', 0) or 0),
            'idx': int(idx),
        })
    if norm_texts:
        norm_texts = sorted(norm_texts, key=lambda m: float(m.get('time', 0.0) or 0.0))

    norm_crescendos: list[dict] = []
    for idx, ev in enumerate(crescendos):
        if not isinstance(ev, dict):
            continue
        t0 = float(ev.get('time', 0.0) or 0.0)
        dur = float(ev.get('duration', 0.0) or 0.0)
        norm_crescendos.append({
            'time': t0,
            'duration': dur,
            'end': t0 + dur,
            'x_rpitch': float(ev.get('x_rpitch', 0) or 0),
            'id': int(ev.get('_id', 0) or 0),
            'idx': int(idx),
        })
    if norm_crescendos:
        norm_crescendos = sorted(norm_crescendos, key=lambda m: float(m.get('time', 0.0) or 0.0))

    norm_decrescendos: list[dict] = []
    for idx, ev in enumerate(decrescendos):
        if not isinstance(ev, dict):
            continue
        t0 = float(ev.get('time', 0.0) or 0.0)
        dur = float(ev.get('duration', 0.0) or 0.0)
        norm_decrescendos.append({
            'time': t0,
            'duration': dur,
            'end': t0 + dur,
            'x_rpitch': float(ev.get('x_rpitch', 0) or 0),
            'id': int(ev.get('_id', 0) or 0),
            'idx': int(idx),
        })
    if norm_decrescendos:
        norm_decrescendos = sorted(norm_decrescendos, key=lambda m: float(m.get('time', 0.0) or 0.0))

    norm_dynamic_symbols: list[dict] = []
    for idx, ev in enumerate(dynamic_symbols):
        if not isinstance(ev, dict):
            continue
        norm_dynamic_symbols.append({
            'time': float(ev.get('time', 0.0) or 0.0),
            'x_rpitch': float(ev.get('x_rpitch', 0) or 0),
            'symbol': str(ev.get('symbol', '') or ''),
            'id': int(ev.get('_id', 0) or 0),
            'idx': int(idx),
        })
    if norm_dynamic_symbols:
        norm_dynamic_symbols = sorted(norm_dynamic_symbols, key=lambda m: float(m.get('time', 0.0) or 0.0))

    norm_start_repeats: list[dict] = []
    for idx, ev in enumerate(start_repeats):
        if not isinstance(ev, dict):
            continue
        norm_start_repeats.append({
            'time': float(ev.get('time', 0.0) or 0.0),
            'id': int(ev.get('_id', 0) or 0),
            'idx': int(idx),
        })
    if norm_start_repeats:
        norm_start_repeats = sorted(norm_start_repeats, key=lambda m: float(m.get('time', 0.0) or 0.0))

    norm_end_repeats: list[dict] = []
    for idx, ev in enumerate(end_repeats):
        if not isinstance(ev, dict):
            continue
        norm_end_repeats.append({
            'time': float(ev.get('time', 0.0) or 0.0),
            'id': int(ev.get('_id', 0) or 0),
            'idx': int(idx),
        })
    if norm_end_repeats:
        norm_end_repeats = sorted(norm_end_repeats, key=lambda m: float(m.get('time', 0.0) or 0.0))

    norm_double_bars: list[dict] = []
    for idx, ev in enumerate(double_bars):
        if not isinstance(ev, dict):
            continue
        norm_double_bars.append({
            'time': float(ev.get('time', 0.0) or 0.0),
            'id': int(ev.get('_id', 0) or 0),
            'idx': int(idx),
        })
    if norm_double_bars:
        norm_double_bars = sorted(norm_double_bars, key=lambda m: float(m.get('time', 0.0) or 0.0))

    norm_tempos: list[dict] = []
    for idx, ev in enumerate(tempos):
        if not isinstance(ev, dict):
            continue
        t0 = float(ev.get('time', 0.0) or 0.0)
        dur = float(ev.get('duration', 0.0) or 0.0)
        tempo_val = int(ev.get('tempo', 60) or 60)
        norm_tempos.append({
            'time': t0,
            'duration': dur,
            'end': t0 + dur,
            'tempo': tempo_val,
            'x_offset': float(ev.get('x_offset', 0.0) or 0.0),
            'id': int(ev.get('_id', 0) or 0),
            'idx': int(idx),
            'invisible': bool(ev.get('invisible', False)),
        })
    if norm_tempos:
        norm_tempos = sorted(norm_tempos, key=lambda m: float(m.get('time', 0.0) or 0.0))

    norm_pedals: list[dict] = []
    for idx, ev in enumerate(pedals):
        if not isinstance(ev, dict):
            continue
        type_raw = str(ev.get('type', 'v') or 'v').strip()
        pedal_type = '^' if type_raw == '^' else 'v'
        norm_pedals.append({
            'time': float(ev.get('time', 0.0) or 0.0),
            'type': pedal_type,
            'id': int(ev.get('_id', 0) or 0),
            'idx': int(idx),
        })
    if norm_pedals:
        norm_pedals = sorted(norm_pedals, key=lambda m: float(m.get('time', 0.0) or 0.0))

    pedal_segments: list[dict] = []
    if norm_pedals:
        pedal_down_time: float | None = None
        pedal_down_id: int = 0
        for pe in norm_pedals:
            p_t = float(pe.get('time', 0.0) or 0.0)
            p_type = str(pe.get('type', 'v') or 'v')
            if p_type == 'v':
                if pedal_down_time is None:
                    pedal_down_time = p_t
                    pedal_down_id = int(pe.get('id', 0) or 0)
            else:
                if pedal_down_time is not None and p_t >= pedal_down_time:
                    pedal_segments.append({
                        'start': float(pedal_down_time),
                        'end': float(p_t),
                        'id': int(pedal_down_id),
                    })
                pedal_down_time = None
                pedal_down_id = 0

    # Problem solved: materialize layout values early to keep math predictable.
    page_orientation = str(layout.get('page_orientation', 'portrait') or 'portrait').strip().lower()
    # Backward compatibility with earlier horizontal/vertical orientation values.
    if page_orientation == 'vertical':
        page_orientation = 'portrait'
    elif page_orientation == 'horizontal':
        page_orientation = 'landscape'

    read_direction = str(layout.get('read_direction', 'vertical') or 'vertical').strip().lower()
    horizontal_read_direction = read_direction == 'horizontal'

    landscape_page_orientation = page_orientation == 'landscape'
    raw_page_w = float(layout.get('page_width_mm', 210.0) or 210.0)
    raw_page_h = float(layout.get('page_height_mm', 297.0) or 297.0)
    # In horizontal read mode the page is rotated for presentation, so the
    # drawing-space swap must be inverted to keep final portrait/landscape
    # output matching the selected page orientation.
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

    # In horizontal read mode the page is rotated after drawing.
    # Remap user-facing margins to drawing-space so final output margins match user settings.
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

    header_height = max(0.0, float(layout.get('header_height_mm', 0.0) or 0.0))
    footer_height = max(0.0, float(layout.get('footer_height_mm', 0.0) or 0.0))
    if horizontal_read_direction:
        # Horizontal mode: reserve footer on the left on every page.
        # Header reserve is applied only on page 1 where title/composer are drawn.
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
    scale = float(layout.get('scale', 1.0) or 1.0)
    stave_two_w = float(layout.get('stave_two_line_thickness_mm', 0.5) or 0.5) * scale
    stave_three_w = float(layout.get('stave_three_line_thickness_mm', 0.5) or 0.5) * scale
    stave_clef_w = float(layout.get('stave_clef_line_thickness_mm', 0.5) or 0.5) * scale
    stave_ledger_len = float(layout.get('stave_ledger_line_length_mm', 7.0) or 7.0) * scale

    def _scaled_dash_pattern_with_default(raw_value, fallback_mm: list[float], local_scale: float) -> list[float] | None:
        parsed: list[float] = []
        try:
            if isinstance(raw_value, str):
                tokens = [p.strip() for p in str(raw_value).split(',') if p.strip() != '']
                parsed = [float(v) for v in tokens]
            elif isinstance(raw_value, (list, tuple)):
                parsed = [float(v) for v in raw_value]
            elif raw_value is not None:
                parsed = [float(raw_value)]
        except Exception:
            parsed = []

        valid_mm = [float(v) for v in parsed if float(v) > 0.0]
        if not valid_mm:
            try:
                valid_mm = [float(v) for v in (fallback_mm or []) if float(v) > 0.0]
            except Exception:
                valid_mm = []
        if not valid_mm:
            valid_mm = [3.0]
        return [float(v) * float(local_scale) for v in valid_mm]

    default_clef_dash_mm = list(getattr(Layout(), 'stave_clef_line_dash_pattern_mm', [3.0]) or [3.0])
    default_grid_dash_mm = list(getattr(Layout(), 'grid_gridline_dash_pattern_mm', [2.5, 4.0]) or [2.5, 4.0])
    clef_dash = _scaled_dash_pattern_with_default(
        layout.get('stave_clef_line_dash_pattern_mm', default_clef_dash_mm),
        default_clef_dash_mm,
        scale,
    )

    op_time = Operator(SHORTEST_DURATION)
    barline_positions: list[float] = []
    group_boundary_times: list[float] = []
    cur_bar = 0.0
    for bg in base_grid:
        numer = int(bg.get('numerator', 4) or 4)
        denom = int(bg.get('denominator', 4) or 4)
        measures = int(bg.get('measure_amount', 1) or 1)
        beat_grouping = list(bg.get('beat_grouping', []) or [])
        bar_offsets, grid_offsets = resolve_grid_layer_offsets(beat_grouping, numer, denom)
        measure_len = float(numer) * (4.0 / float(max(1, denom))) * float(QUARTER_NOTE_UNIT)
        inner_group_offsets = sorted(
            list(
                dict.fromkeys(
                    round(float(off), 6)
                    for off in grid_offsets
                    if 0.0 < float(off) < float(measure_len)
                )
            )
        )
        for _ in range(int(max(0, measures))):
            m_start = float(cur_bar)
            m_end = float(cur_bar + measure_len)
            group_boundary_times.append(float(m_start))
            for off in inner_group_offsets:
                group_boundary_times.append(float(m_start + float(off)))
            group_boundary_times.append(float(m_end))
            for off in bar_offsets:
                barline_positions.append(float(cur_bar + float(off)))
            cur_bar += measure_len

    group_boundary_times = sorted(
        list(dict.fromkeys(round(float(t), 6) for t in group_boundary_times))
    )
    all_barlines = sorted(list(dict.fromkeys([0.0] + [float(v) for v in barline_positions] + [float(cur_bar)])))

    # Build grid-band dark intervals with the same rules as editor drawer:
    # - barline resets to dark,
    # - marker start resets phase boundaries,
    # - marker start preserves current color,
    # - marker range truncates at next marker start.
    def _build_grid_band_dark_intervals(markers: list, bars: list[float], total_len: float, starts_dark: bool = True) -> list[tuple[float, float]]:
        op = Operator()
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

                # duration == 0 means stop engraving bands for this segment.
                if op.le(step, 0.0):
                    # OFF marker: next resumed segment should start with configured phase.
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

    # Get grid band track from layout and precompute dark intervals for efficient lookup during engraving.
    grid_bands = list(layout.get('grid_band_track', []) or [])
    grid_band_start_phase = str(layout.get('grid_band_start_phase', 'dark') or 'dark').strip().lower()
    grid_bands_start_dark = bool(grid_band_start_phase != 'light')
    grid_dark_intervals_global = _build_grid_band_dark_intervals(
        grid_bands,
        all_barlines,
        float(cur_bar),
        starts_dark=grid_bands_start_dark,
    )

    # Problem solved: continuation dots count for grid_band pitch sizing by
    # creating synthetic starts at beat-group boundaries crossed by held notes.
    grid_band_starts_by_hand: dict[str, list[float]] = {'l': [], 'r': []}
    grid_band_pitches_by_hand: dict[str, list[int]] = {'l': [], 'r': []}
    for hk in ('l', 'r'):
        hand_notes = notes_by_hand.get(hk, []) or []
        events: list[tuple[float, int]] = []
        for item in hand_notes:
            n_t = float(item.get('time', 0.0) or 0.0)
            n_end = float(item.get('end', 0.0) or 0.0)
            n_pitch = int(item.get('pitch', 0) or 0)
            if n_pitch < 1 or n_pitch > PIANO_KEY_AMOUNT:
                continue

            # Real note start is always an event.
            events.append((float(n_t), int(n_pitch)))

            # Continuation-dot equivalent starts at crossed beat-group boundaries.
            if n_end > n_t and group_boundary_times:
                lo = bisect.bisect_right(group_boundary_times, float(n_t))
                hi = bisect.bisect_left(group_boundary_times, float(n_end))
                for bi in range(lo, hi):
                    events.append((float(group_boundary_times[bi]), int(n_pitch)))

        events.sort(key=lambda it: float(it[0]))
        grid_band_starts_by_hand[hk] = [float(t) for (t, _p) in events]
        grid_band_pitches_by_hand[hk] = [int(p) for (_t, p) in events]

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
        """Normalize hex color strings."""
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

    def _allow_font_registry() -> bool:
        """Return True when it is safe to access QFontDatabase (GUI process only)."""
        return mp.current_process().name == "MainProcess"

    def _resolve_font_family(family: str) -> str:
        """Resolve a font family name with the font registry if available."""
        if not _allow_font_registry():
            return family
        from fonts import resolve_font_family
        return str(resolve_font_family(family))

    def _layout_font(key: str, fallback_family: str, fallback_size: float) -> tuple[str, float, bool, bool, bool]:
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
        underline = bool(raw.get('underline', False))
        return family, size_pt, bold, italic, underline

    def _info_text(key: str, fallback: str) -> str:
        """Fetch info text with a fallback, always returning a string."""
        if isinstance(info, dict):
            raw = info.get(key, fallback)
        else:
            raw = fallback
        if isinstance(raw, dict):
            raw = raw.get('text', fallback)
        return str(raw) if raw is not None else str(fallback)

    def _info_font(key: str, fallback_family: str, fallback_size: float) -> tuple[str, float, bool, bool, bool, float, float]:
        """Fetch info font settings from layout (family, size, style, offsets)."""
        family, size_pt, bold, italic, underline = _layout_font(key, fallback_family, fallback_size)
        raw_font = layout.get(key, {}) if isinstance(layout, dict) else {}
        if not isinstance(raw_font, dict):
            raw_font = {}
        x_off = float(raw_font.get('x_offset', 0.0) or 0.0)
        y_off = float(raw_font.get('y_offset', 0.0) or 0.0)
        return family, size_pt, bold, italic, underline, x_off, y_off

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
            _bar_offsets, grid_offsets = resolve_grid_layer_offsets(seq, numer, denom)
            for _ in range(int(measures)):
                m_start = float(cur)
                m_end = float(cur + measure_len)
                if op_time.lt(m_end, float(a)):
                    cur = m_end
                    continue
                if op_time.gt(m_start, float(b)):
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
                    if op_time.lt(w0, w1):
                        windows.append((w0, w1))
                cur = m_end
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
        if rule == 'above_stem_if_collision':
            for n in notes:
                if int(n.get('idx', -2) or -2) == idx0:
                    continue
                if not op.eq(float(n.get('time', 0.0) or 0.0), t0):
                    continue
                if abs(int(n.get('pitch', 0) or 0) - p0) == 1:
                    return True
            return False
        if rule == 'above_stem_if_chord_and_white_note':
            for n in notes:
                if int(n.get('idx', -2) or -2) == idx0:
                    continue
                if not op.eq(float(n.get('time', 0.0) or 0.0), t0):
                    continue
                np = int(n.get('pitch', 0) or 0)
                if np not in BLACK_KEYS and np != p0:
                    return True
            return False
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

    def _should_tune_under_stem_black_width(item: dict, rule: str, notes: list[dict], op: Operator) -> bool:
        '''if a black note is under the stem and a white note directly next to it
        this method returns True, which means the black note should be tuned to be narrower 
        to form a small second symbol with the white note.'''
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

    def _has_followed_rest(item: dict) -> bool:
        """Return True when a note has no immediate following note in its hand.

        Problem solved: stop-signs should mark a gap in the same hand, not
        simply the end of a note.
        """
        hand_key = str(item.get('hand', 'l') or 'l')
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
    semitone_mm = 2 * scale
    key_positions = _build_key_positions(1, PIANO_KEY_AMOUNT, semitone_mm)
    for line in lines:
        if line['stave_range'] == 'auto':
            groups, keys, bound_left, bound_right, _, pattern = _auto_line_keys_and_bounds(line['time_start'], line['time_end'])
            line['visible_keys'] = keys
            line['pattern'] = pattern
        else:
            manual = _sanitize_range(line['stave_range'])
            requested_lo = int(manual[0]) if manual else 1
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
        natural_bound_left = int(bound_left)  # stave left before any override
        natural_bound_right = int(bound_right)  # stave right before any override
        low_key_present = bool(bound_left <= 2 or line['stave_range'] != 'auto' and int(requested_lo) <= 2)
        for item in norm_notes:
            n_t = float(item.get('time', 0.0) or 0.0)
            n_end = float(item.get('end', 0.0) or 0.0)
            p = int(item.get('pitch', 0) or 0)
            if op_time.ge(n_t, float(line['time_end'])) or op_time.le(n_end, float(line['time_start'])):
                continue
            if p in (1, 2, 3):
                low_key_present = True
                break
        # Ledger mode: notes exist in keys 1-3 but the natural stave range
        # didn't reach down to key 2, so draw short stubs instead of a full line.
        a0_ledger_mode = bool(low_key_present and int(natural_bound_left) > 2)

        # For manual ranges: notes outside the visible stave groups will get ledger
        # stubs. Compute the ledger-extended bounds separately for layout width;
        # the natural stave bounds are kept for origin / barline / grid_left.
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
                if op_time.ge(n_t, float(line['time_end'])) or op_time.le(n_end, float(line['time_start'])):
                    continue
                if p < 1 or p > PIANO_KEY_AMOUNT:
                    continue
                g = _group_index_for_key(p)
                if g < _manual_bound_group_low:
                    for grp in line_groups[g:_manual_bound_group_low]:
                        for k in grp.get('keys', []):
                            if int(k) < ledger_bound_left:
                                ledger_bound_left = int(k)
                elif g > _manual_bound_group_high:
                    for grp in line_groups[_manual_bound_group_high + 1:g + 1]:
                        for k in grp.get('keys', []):
                            if int(k) > ledger_bound_right:
                                ledger_bound_right = int(k)

        # bound_left/bound_right stay at the natural stave range for origin and
        # barline purposes. Ledger extents are tracked separately.
        line['low_key_left'] = bool(low_key_present)
        line['a0_ledger_mode'] = bool(a0_ledger_mode)
        line['natural_bound_left'] = int(natural_bound_left)
        line['natural_bound_right'] = int(natural_bound_right)
        line['ledger_bound_left'] = int(ledger_bound_left)
        line['ledger_bound_right'] = int(ledger_bound_right)
        line['range'] = [int(bound_left), int(bound_right)]
        # stave_width = natural stave span (used for barlines, origin, grid_left/right).
        # stave_plus_ledger_width = full visual span including ledger stubs (used for pagination).
        # ledger_left_overhang = how far ledger stubs extend to the LEFT of the natural stave start.
        # In the drawing pass line_x_start is shifted right by this amount so ledger stubs
        # land inside the allocated column width rather than spilling to the right.
        min_pos = key_positions.get(bound_left, 0.0)
        max_pos = key_positions.get(bound_right, min_pos)
        stave_width = max(0.0, max_pos - min_pos)
        ledger_min_pos = key_positions.get(ledger_bound_left, min_pos)
        ledger_max_pos = key_positions.get(ledger_bound_right, max_pos)
        ledger_left_overhang = max(0.0, float(min_pos) - float(ledger_min_pos))
        stave_plus_ledger_width = max(stave_width, ledger_max_pos - ledger_min_pos)
        line['stave_width'] = float(stave_width)
        line['stave_plus_ledger_width'] = float(stave_plus_ledger_width)
        line['ledger_left_overhang'] = float(ledger_left_overhang)
        base_margin_left = float(line.get('margin_left', 0.0) or 0.0)
        ts_lane_width = 0.0
        ts_lane_right_offset = 0.0
        ts_lane_padding_mm = 0.0
        # Problem solved: if time-signature indicators would collide with notes,
        # allow shifting the indicator left, but do not reserve a lane (margins are user-defined).
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
            ts_lane_padding_mm = 1  # Hard-coded right padding so lane ends before the stave.
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
                ts_lane_gap_mm = semitone_mm  # Minimum gap between indicator lane and notes/beam stems
                ts_lane_right_offset = min(0.0, float(offset_left - ts_lane_gap_mm))
        # Keep user-defined margins: do not expand margin_left for the indicator lane.
        line['margin_left'] = base_margin_left
        line['base_margin_left'] = base_margin_left
        line['ts_lane_width'] = ts_lane_width
        line['ts_lane_right_offset'] = ts_lane_right_offset
        line['ts_lane_padding_mm'] = ts_lane_padding_mm
        line['total_width'] = float(line['margin_left'] + stave_plus_ledger_width + line['margin_right'])
        line['bound_left'] = int(bound_left)
        line['bound_right'] = int(bound_right)

    # Problem solved: paginate lines to fit available width with explicit breaks.
    def _available_width_for_page(page_index: int) -> float:
        left_reserve, right_reserve = _line_axis_reserves_for_page(page_index)
        return max(
            1e-6,
            page_w - page_left - page_right - left_reserve - right_reserve,
        )

    pages: list[list[dict]] = []
    cur_page: list[dict] = []
    cur_width = 0.0
    for line in lines:
        cur_page_index = len(pages)
        cur_available_width = _available_width_for_page(cur_page_index)
        if line.get('page_break', False):
            if cur_page:
                pages.append(cur_page)
            elif not pages:
                pages.append([])
            cur_page = []
            cur_width = 0.0
            cur_page_index = len(pages)
            cur_available_width = _available_width_for_page(cur_page_index)
        if cur_page and (cur_width + float(line['total_width'])) > cur_available_width:
            pages.append(cur_page)
            cur_page = []
            cur_width = 0.0
            cur_page_index = len(pages)
            cur_available_width = _available_width_for_page(cur_page_index)
        cur_page.append(line)
        cur_width += float(line['total_width'])
    if cur_page:
        pages.append(cur_page)

    # Problem solved: render each page with header/footer and justified spacing.
    if not pages:
        pages = [[]]

    analysis_snapshot = Analysis.compute(score, lines_count=len(lines), pages_count=len(pages))
    setattr(du, 'analysis', analysis_snapshot)
    first_system_start = float(lines[0].get('time_start', 0.0) or 0.0) if lines else 0.0
    target_page_index = 0
    if not pdf_export and pages:
        try:
            target_page_index = int(pageno)
        except Exception:
            target_page_index = 0
        target_page_index = max(0, min(len(pages) - 1, target_page_index))

    for page_index, page in enumerate(pages):
        du.new_page(page_w, page_h)
        if horizontal_read_direction:
            du.set_current_page_rotation_deg(-90.0)

        output_page_w = float(page_h) if horizontal_read_direction else float(page_w)
        output_page_h = float(page_w) if horizontal_read_direction else float(page_h)

        def _map_output_to_drawing(x_out: float, y_out: float) -> tuple[float, float]:
            if not horizontal_read_direction:
                return (float(x_out), float(y_out))
            # DrawUtil applies -90° page rotation in horizontal mode.
            return (float(page_w) - float(y_out), float(x_out))

        info_text_angle = 90.0 if horizontal_read_direction else 0.0
        du.add_rectangle(
            0.0,
            0.0,
            page_w,
            page_h,
            stroke_color=None,
            fill_color=paper_color,
            id=0,
            tags=['page_background'],
        )
        if not pdf_export and page_index != target_page_index:
            continue
        if page_index == 0:
            title_text = _info_text('title', 'title')
            composer_text = _info_text('composer', 'composer')
            title_family, title_size, title_bold, title_italic, title_underline, title_x_off, title_y_off = _info_font(
                'font_title',
                'Courier',
                12.0,
            )
            composer_family, composer_size, composer_bold, composer_italic, composer_underline, composer_x_off, composer_y_off = _info_font(
                'font_composer',
                'Courier',
                10.0,
            )
            title_x, title_y = _map_output_to_drawing(
                user_page_left + title_x_off,
                user_page_top + title_y_off,
            )
            du.add_text(
                title_x,
                title_y,
                title_text,
                family=title_family,
                size_pt=title_size,
                bold=title_bold,
                italic=title_italic,
                angle_deg=info_text_angle,
                color=notation_color,
                id=0,
                tags=['title'],
                anchor='nw',
            )
            if title_underline and title_text and not horizontal_read_direction:
                _xb, _yb, _w, _ = du._get_text_extents_mm(title_text, title_family, title_size, title_italic, title_bold)
                _bx = title_x - _xb
                _by = title_y - _yb
                du.add_line(_bx, _by + max(0.2, title_size * 0.025), _bx + _w, _by + max(0.2, title_size * 0.025),
                            color=notation_color, width_mm=max(0.2, title_size * (0.04 if title_bold else 0.02)), tags=['title'])
            composer_x, composer_y = _map_output_to_drawing(
                (output_page_w - user_page_right) + composer_x_off,
                user_page_top + composer_y_off,
            )
            du.add_text(
                composer_x,
                composer_y,
                composer_text,
                family=composer_family,
                size_pt=composer_size,
                bold=composer_bold,
                italic=composer_italic,
                angle_deg=info_text_angle,
                color=notation_color,
                id=0,
                tags=['composer'],
                anchor='ne',
            )
            if composer_underline and composer_text and not horizontal_read_direction:
                _xb, _yb, _w, _ = du._get_text_extents_mm(composer_text, composer_family, composer_size, composer_italic, composer_bold)
                _bx = composer_x - _w - _xb
                _by = composer_y - _yb
                du.add_line(_bx, _by + max(0.2, composer_size * 0.025), _bx + _w, _by + max(0.2, composer_size * 0.025),
                            color=notation_color, width_mm=max(0.2, composer_size * (0.04 if composer_bold else 0.02)), tags=['composer'])
        if footer_height > 0.0:
            document_title = _info_text('title', 'title').strip()
            if not document_title:
                document_title = 'title'
            default_copyright = getattr(default_info, 'copyright', f"© all rights reserved {datetime.now().year}")
            footer_text = _info_text('copyright', default_copyright).strip()
            if not footer_text:
                footer_text = default_copyright
            footer_family, footer_size, footer_bold, footer_italic, footer_underline, footer_x_off, footer_y_off = _info_font(
                'font_copyright',
                'Courier',
                8.0,
            )
            if horizontal_read_direction:
                # Keep drawing-space placement near top-left, but derive it from
                # output bottom-left so final result honors user bottom margin.
                footer_x, footer_y = _map_output_to_drawing(
                    user_page_left + footer_x_off,
                    (output_page_h - user_page_bottom) + footer_y_off,
                )
                footer_anchor = 'sw'
            else:
                footer_x, footer_y = _map_output_to_drawing(
                    user_page_left + footer_x_off,
                    (output_page_h - user_page_bottom) + footer_y_off,
                )
                footer_anchor = None
            _footer_text_full = f"Page {page_index + 1} of {len(pages)} • {document_title} • {footer_text}"
            du.add_text(
                footer_x,
                footer_y,
                _footer_text_full,
                family=footer_family,
                size_pt=footer_size,
                bold=footer_bold,
                italic=footer_italic,
                angle_deg=info_text_angle,
                color=notation_color,
                id=0,
                tags=['copyright'],
                anchor=footer_anchor,
            )
            if footer_underline and _footer_text_full and not horizontal_read_direction:
                _xb, _yb, _w, _ = du._get_text_extents_mm(_footer_text_full, footer_family, footer_size, footer_italic, footer_bold)
                _bx = footer_x
                _by = footer_y
                du.add_line(_bx + _xb, _by + max(0.2, footer_size * 0.025), _bx + _xb + _w, _by + max(0.2, footer_size * 0.025),
                            color=notation_color, width_mm=max(0.2, footer_size * (0.04 if footer_bold else 0.02)), tags=['copyright'])
        if not page:
            continue
        line_axis_left_reserve, line_axis_right_reserve = _line_axis_reserves_for_page(page_index)
        available_width = _available_width_for_page(page_index)
        page_lines = list(reversed(page)) if horizontal_read_direction else page
        used_width = sum(float(l['total_width']) for l in page_lines)
        leftover = max(0.0, available_width - used_width)
        gap = leftover / float(len(page_lines) + 1)
        x_cursor = page_left + line_axis_left_reserve + gap
        for line in page_lines:
            # Shift line_x_start right by the left ledger overhang so left-side
            # ledger stubs land inside the allocated column width.
            _ledger_left_overhang = float(line.get('ledger_left_overhang', 0.0) or 0.0)
            line_x_start = x_cursor + float(line['margin_left']) + _ledger_left_overhang
            line_x_end = line_x_start + float(line['stave_width'])
            if horizontal_read_direction:
                y1 = page_top
                y2 = float(page_h - page_bottom)
            else:
                header_offset = 0.0
                if page_index == 0:
                    header_offset = float(header_height)
                y1 = page_top + header_offset
                y2 = float(page_h - page_bottom - footer_height)
            if y2 <= y1:
                y2 = y1 + 1.0
            line['y_top'] = y1
            line['y_bottom'] = y2

            bound_left = int(line.get('bound_left', line['range'][0]))
            bound_right = int(line.get('bound_right', line['range'][1]))
            # Natural stave bounds (before any ledger expansion) for origin and barline spans.
            natural_bound_left = int(line.get('natural_bound_left', bound_left))
            natural_bound_right = int(line.get('natural_bound_right', bound_right))
            origin = float(key_positions.get(bound_left, 0.0))
            manual_range = isinstance(line.get('stave_range'), list) and len(line.get('stave_range')) >= 2
            # Use natural stave bounds for ledger group comparisons.
            bound_group_low = _group_index_for_key(natural_bound_left) if manual_range else None
            bound_group_high = _group_index_for_key(natural_bound_right) if manual_range else None
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

            def _hand_band_x_span(_hand_key: str, _t0: float, _t1: float) -> tuple[float, float] | None:
                # Single band span from key 10 to key 77.
                x0 = float(_key_to_x(10))
                x1 = float(_key_to_x(77))
                x0 = max(float(grid_left), min(float(grid_right), float(x0)))
                x1 = max(float(grid_left), min(float(grid_right), float(x1)))
                if x1 <= x0:
                    return None
                return (x0, x1)

            def _clip_intervals(intervals: list[tuple[float, float]], t0: float, t1: float) -> list[tuple[float, float]]:
                if not intervals:
                    return []
                out: list[tuple[float, float]] = []
                for a, b in intervals:
                    c0 = max(float(a), float(t0))
                    c1 = min(float(b), float(t1))
                    if c1 > c0:
                        out.append((c0, c1))
                return out

            def _group_window_for_interval(boundaries: list[float], t0: float, t1: float) -> tuple[float, float]:
                # Use interval midpoint to select the beat-group window used for width sizing.
                if len(boundaries) < 2:
                    return (float(t0), float(t1))
                mid = (float(t0) + float(t1)) * 0.5
                idx = bisect.bisect_right(boundaries, float(mid)) - 1
                idx = max(0, min(len(boundaries) - 2, idx))
                g0 = float(boundaries[idx])
                g1 = float(boundaries[idx + 1])
                if g1 <= g0:
                    return (float(t0), float(t1))
                return (g0, g1)

            def _mix_rgba(a: tuple[float, float, float, float], b: tuple[float, float, float, float], wa: float) -> tuple[float, float, float, float]:
                wb = 1.0 - float(wa)
                return (
                    (a[0] * wa) + (b[0] * wb),
                    (a[1] * wa) + (b[1] * wb),
                    (a[2] * wa) + (b[2] * wb),
                    1.0,
                )

            band_hex = _normalize_hex_color(layout.get('grid_band_color', '#cccccc')) or '#cccccc'
            try:
                br, bg, bb, _ = hex_to_rgba(band_hex, 1.0)
            except Exception:
                br, bg, bb = (204, 204, 204)
            band_note_fill = _midi_fill_from_rgb((int(br), int(bg), int(bb)))
            if pdf_export:
                grid_band_tint = band_note_fill
            else:
                grid_band_tint = _mix_rgba(band_note_fill, paper_color, 0.80)

            line_start_ticks = float(line.get('time_start', 0.0) or 0.0)
            line_end_ticks = float(line.get('time_end', 0.0) or 0.0)
            grid_band_visible = bool(layout.get('grid_band_visible', True))
            shared_band_intervals: list[tuple[float, float]] = []
            grid_band_dark_intervals: dict[str, list[tuple[float, float]]] = {'left': shared_band_intervals, 'right': shared_band_intervals}

            def _text_bbox(content_w_mm: float, content_h_mm: float, angle_deg: float, padding_mm: float, corner_radius_mm: float, width_offset_mm: float) -> tuple[float, float, float, list[tuple[float, float]], list[tuple[float, float]]]:
                pad = max(0.0, float(padding_mm))
                base_w_mm = max(0.0, float(content_w_mm) + (pad * 2.0))
                h_mm = max(0.0, float(content_h_mm) + (pad * 2.0))
                base_hw = base_w_mm * 0.5
                hh = h_mm * 0.5
                x0 = -base_hw
                x1 = base_hw + float(width_offset_mm)
                if x1 < x0:
                    x1 = x0
                w_mm = max(0.0, x1 - x0)
                r = min(max(0.0, float(corner_radius_mm)), w_mm * 0.5, hh)

                def _rounded_rect_points(x0_val: float, x1_val: float, hh_val: float, radius: float) -> list[tuple[float, float]]:
                    if radius <= 1e-6:
                        return [(x0_val, -hh_val), (x1_val, -hh_val), (x1_val, hh_val), (x0_val, hh_val)]
                    pts: list[tuple[float, float]] = []
                    corner_defs = [
                        (x0_val + radius, -hh_val + radius, 180.0, 270.0),
                        (x1_val - radius, -hh_val + radius, 270.0, 360.0),
                        (x1_val - radius, hh_val - radius, 0.0, 90.0),
                        (x0_val + radius, hh_val - radius, 90.0, 180.0),
                    ]
                    step = 15.0
                    for cx, cy, start_deg, end_deg in corner_defs:
                        deg = start_deg
                        while deg < end_deg + 0.01:
                            rad_ang = math.radians(deg)
                            pts.append((cx + radius * math.cos(rad_ang), cy + radius * math.sin(rad_ang)))
                            deg += step
                    return pts

                base_poly = _rounded_rect_points(-base_hw, base_hw, hh, min(max(0.0, float(corner_radius_mm)), base_hw, hh))
                draw_poly = _rounded_rect_points(x0, x1, hh, r)
                corners = [(x0, -hh), (x1, -hh), (x1, hh), (x0, hh)]
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
                    if ry < min_y:
                        min_y = ry
                for (dx, dy) in draw_poly:
                    rx = dx * cos_a - dy * sin_a
                    ry = dx * sin_a + dy * cos_a
                    rot_poly.append((rx, ry))
                offset_down = max(0.0, -min_y)
                return w_mm, h_mm, offset_down, rot_corners, rot_poly

            tick_per_mm = (float(line['time_end'] - line['time_start'])) / max(1e-6, (y2 - y1))
            mm_per_quarter = float(QUARTER_NOTE_UNIT) / max(1e-6, tick_per_mm)

            indicator_type = str(layout.get('time_signature_indicator_type', 'classical') or 'classical')
            classic_family, classic_size, classic_bold, classic_italic, _classic_ul = _layout_font(
                'time_signature_indicator_classic_font',
                'Edwin',
                35.0,
            )
            klav_family, klav_size, klav_bold, klav_italic, _klav_ul = _layout_font(
                'time_signature_indicator_klavarskribo_font',
                'Edwin',
                25.0,
            )
            guide_thickness = float(layout.get('time_signature_indicator_guide_thickness_mm', 0.5) or 0.5) * scale
            divider_thickness = float(layout.get('time_signature_indicator_divide_guide_thickness_mm', 1.0) or 1.0) * scale
            classic_size_pt = classic_size * scale
            klav_size_pt = klav_size * scale
            # Shared half-span for classical divider width and klavarskribo guides,
            # also used as vertical offset around the classical divider center.
            ts_indicator_half_span = 3.0 * scale

            def _ts_color(enabled: bool) -> tuple[float, float, float, float]:
                if enabled:
                    return notation_color
                return (
                    max(0.0, notation_color[0] * 0.6 + paper_color[0] * 0.4),
                    max(0.0, notation_color[1] * 0.6 + paper_color[1] * 0.4),
                    max(0.0, notation_color[2] * 0.6 + paper_color[2] * 0.4),
                    1.0,
                )

            def _draw_classical_ts(numerator: int, denominator: int, enabled: bool, y_mm: float) -> None:
                color = _ts_color(enabled)
                x = ts_x_right
                size_pt = classic_size_pt
                classic_angle = 90.0 if horizontal_read_direction else 0.0
                num_txt = f"{int(numerator)}"
                den_txt = f"{int(denominator)}"
                _, _, _, num_h = du._get_text_extents_mm(
                    num_txt,
                    classic_family,
                    size_pt,
                    classic_italic,
                    classic_bold,
                )
                _, _, _, den_h = du._get_text_extents_mm(
                    den_txt,
                    classic_family,
                    size_pt,
                    classic_italic,
                    classic_bold,
                )
                num_center_y = (y_mm - ts_indicator_half_span) - (max(0.0, float(num_h)) * 0.5)
                den_center_y = (y_mm + ts_indicator_half_span) + (max(0.0, float(den_h)) * 0.5)
                du.add_text(
                    x,
                    num_center_y,
                    num_txt,
                    size_pt=size_pt,
                    color=color,
                    id=0,
                    tags=["ts_classic"],
                    anchor='center',
                    family=classic_family,
                    bold=classic_bold,
                    italic=classic_italic,
                    angle_deg=classic_angle,
                )
                du.add_line(
                    x - ts_indicator_half_span,
                    y_mm,
                    x + ts_indicator_half_span,
                    y_mm,
                    color=color,
                    width_mm=divider_thickness,
                    id=0,
                    tags=["ts_classic"],
                    dash_pattern=None,
                )
                du.add_text(
                    x,
                    den_center_y,
                    den_txt,
                    size_pt=size_pt,
                    color=color,
                    id=0,
                    tags=["ts_classic"],
                    anchor='center',
                    family=classic_family,
                    bold=classic_bold,
                    italic=classic_italic,
                    angle_deg=classic_angle,
                )

            def _draw_klavarskribo_ts(numerator: int, denominator: int, enabled: bool, y_mm: float, grid_positions: list[int]) -> None:
                """Match editor time-signature Klavarskribo indicator (three columns)."""
                color = _ts_color(enabled)
                klav_text_angle = 90.0 if horizontal_read_direction else 0.0
                quarters_per_measure = float(numerator) * (4.0 / max(1.0, float(denominator)))
                measure_len_mm = quarters_per_measure * mm_per_quarter
                beat_len_mm = measure_len_mm / max(1, int(numerator))
                measure_len_ticks = float(numerator) * (4.0 / max(1.0, float(denominator))) * float(QUARTER_NOTE_UNIT)
                beat_len_ticks = measure_len_ticks / max(1.0, float(numerator))

                op = Operator(float(SHORTEST_DURATION))
                # Column positions driven by the equally-spaced lane thirds.
                # Right column: guide lines (also aligns with the classical indicator).
                # Middle column: beat numbers.
                # Left column: group numbers.
                x_right = ts_x_right
                x_mid = ts_x_mid
                x_left = ts_x_left

                grid_bar_off, grid_off = resolve_grid_layer_offsets(
                    [float(v) for v in (grid_positions or []) if isinstance(v, (int, float))],
                    int(numerator),
                    int(denominator),
                )
                grid1_positions = sorted(
                    list(
                        dict.fromkeys(
                            [float(v) for v in (grid_bar_off + grid_off) if 0.0 <= float(v) < float(measure_len_ticks)]
                        )
                    )
                )

                beat_positions = [float(k) * float(beat_len_ticks) for k in range(0, int(numerator))]
                beat_has_reset = [any(op.eq(float(bp), float(gp)) for gp in grid1_positions) for bp in beat_positions]
                full_group_mode = bool(beat_has_reset) and all(bool(v) for v in beat_has_reset)

                mid_values: list[int] = []
                group_values: list[int] = []
                group_starts: list[int] = []
                cur_mid = 1
                cur_group = 1
                for k in range(1, int(numerator) + 1):
                    if k == 1:
                        reset_here = True
                    elif full_group_mode:
                        reset_here = False
                    else:
                        reset_here = bool(beat_has_reset[k - 1])

                    if reset_here:
                        cur_mid = 1
                        if k > 1:
                            cur_group += 1
                        group_starts.append(k)
                        group_values.append(cur_group)
                    else:
                        cur_mid += 1
                    mid_values.append(cur_mid)

                # guides
                guide_half_len = ts_indicator_half_span
                guide_width_mm = guide_thickness
                for k in range(1, int(numerator) + 1):
                    y = y_mm + (k - 1) * beat_len_mm
                    du.add_line(
                        x_right - guide_half_len,
                        y,
                        x_right + guide_half_len,
                        y,
                        color=color,
                        width_mm=guide_width_mm,
                        id=0,
                        tags=["ts_klavarskribo"],
                        dash_pattern=None,
                    )
                du.add_line(
                    x_right - guide_half_len,
                    y_mm + measure_len_mm,
                    x_right + guide_half_len,
                    y_mm + measure_len_mm,
                    color=color,
                    width_mm=guide_width_mm,
                    id=0,
                    tags=["ts_klavarskribo"],
                    dash_pattern=None,
                )

                # middle column
                for k, val in enumerate(mid_values, start=1):
                    y = y_mm + (k - 1) * beat_len_mm
                    du.add_text(
                        x_mid,
                        y,
                        str(val),
                        size_pt=klav_size_pt,
                        color=color,
                        id=0,
                        tags=["ts_klavarskribo"],
                        anchor='center',
                        family=klav_family,
                        bold=klav_bold,
                        italic=klav_italic,
                        angle_deg=klav_text_angle,
                    )
                du.add_text(
                    x_mid,
                    y_mm + measure_len_mm,
                    "1",
                    size_pt=klav_size_pt,
                    color=color,
                    id=0,
                    tags=["ts_klavarskribo"],
                    anchor='center',
                    family=klav_family,
                    bold=klav_bold,
                    italic=klav_italic,
                    angle_deg=klav_text_angle,
                )

                # left column
                for gi, s in zip(group_values, group_starts):
                    y = y_mm + (s - 1) * beat_len_mm
                    du.add_text(
                        x_left,
                        y,
                        str(gi),
                        size_pt=klav_size_pt,
                        color=color,
                        id=0,
                        tags=["ts_klavarskribo"],
                        anchor='center',
                        family=klav_family,
                        bold=klav_bold,
                        italic=klav_italic,
                        angle_deg=klav_text_angle,
                    )

            # Problem solved: draw barlines and beat lines from the base grid.
            # grid_left/grid_right always span the natural stave range only.
            # Ledger stubs extend beyond this but are drawn per-note.
            grid_left = _key_to_x(natural_bound_left)
            grid_right = _key_to_x(natural_bound_right)

            line_avg_split_pitch = 43.0
            line_pitches: list[int] = []
            for n_item in norm_notes:
                n_t0 = float(n_item.get('time', 0.0) or 0.0)
                n_t1 = float(n_item.get('end', 0.0) or 0.0)
                n_pitch = int(n_item.get('pitch', 0) or 0)
                if n_pitch < 1 or n_pitch > PIANO_KEY_AMOUNT:
                    continue
                if op_time.ge(n_t0, float(line['time_end'])) or op_time.le(n_t1, float(line['time_start'])):
                    continue
                line_pitches.append(int(n_pitch))
            if line_pitches:
                line_avg_split_pitch = float(sum(line_pitches)) / float(len(line_pitches))
            line_avg_split_pitch = max(1.0, min(float(PIANO_KEY_AMOUNT), float(line_avg_split_pitch)))

            split_lo = int(math.floor(line_avg_split_pitch))
            split_hi = int(math.ceil(line_avg_split_pitch))
            if split_hi <= split_lo:
                grid_band_split_x = _key_to_x(split_lo)
            else:
                x_lo = _key_to_x(split_lo)
                x_hi = _key_to_x(split_hi)
                frac = float(line_avg_split_pitch - float(split_lo))
                grid_band_split_x = float(x_lo + (x_hi - x_lo) * frac)
            grid_band_split_x = max(float(grid_left), min(float(grid_right), float(grid_band_split_x)))
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
            grid_color = notation_color
            bar_width_mm = float(layout.get('grid_barline_thickness_mm', 0.25) or 0.25) * scale
            grid_width_mm = float(layout.get('grid_gridline_thickness_mm', 0.15) or 0.15) * scale
            barline_visible = bool(layout.get('barline_visible', True))
            grid_line_visible = bool(layout.get('grid_line_visible', True))
            dash_pattern = _scaled_dash_pattern_with_default(
                layout.get('grid_gridline_dash_pattern_mm', default_grid_dash_mm),
                default_grid_dash_mm,
                scale,
            )

            # Build collision geometry for constructive barline drawing.
            line_start_ticks_local = float(line.get('time_start', 0.0) or 0.0)
            line_end_ticks_local = float(line.get('time_end', 0.0) or 0.0)
            line_notes_for_barlines: list[dict] = []
            for item in norm_notes:
                n_t = float(item.get('time', 0.0) or 0.0)
                n_end = float(item.get('end', 0.0) or 0.0)
                p = int(item.get('pitch', 0) or 0)
                if op_time.ge(n_t, line_end_ticks_local) or op_time.le(n_end, line_start_ticks_local):
                    continue
                if p < 1 or p > PIANO_KEY_AMOUNT:
                    continue
                line_notes_for_barlines.append(item)

            stem_len_units_for_barlines = float(layout.get('note_stem_length_semitone', 3) or 3)
            stem_len_mm_for_barlines = stem_len_units_for_barlines * semitone_mm
            note_head_half_w = semitone_mm * float(layout.get('note_width_scaling', 0.75) or 0.75)
            stem_collision_pad = max(0.15, float(layout.get('note_stem_thickness_mm', 0.5) or 0.5) * scale)
            head_collision_pad = max(0.15, semitone_mm * 0.15)
            beam_collision_pad = max(0.2, float(layout.get('beam_thickness_mm', 1.0) or 1.0) * scale * 0.7)
            barline_symbol_gap_mm = max(0.0, semitone_mm)

            beam_segments_for_barlines: list[dict[str, float]] = []
            beam_connect_segments_for_barlines: list[dict[str, float]] = []
            for hand_norm in ('r', 'l'):
                notes_for_hand = [
                    n for n in line_notes_for_barlines
                    if ('l' if str(n.get('hand', 'l') or 'l') == 'l' else 'r') == hand_norm
                ]
                markers_for_hand = beam_by_hand.get(hand_norm, [])
                groups, windows = _group_by_beam_markers(notes_for_hand, markers_for_hand, line_start_ticks_local, line_end_ticks_local)
                for idx, grp in enumerate(groups):
                    if not grp or idx >= len(windows):
                        continue
                    t0, t1 = windows[idx]
                    starts_in = [
                        float(n.get('time', 0.0) or 0.0)
                        for n in grp
                        if op_time.ge(float(n.get('time', 0.0) or 0.0), float(t0))
                        and op_time.lt(float(n.get('time', 0.0) or 0.0), float(t1))
                    ]
                    if not starts_in:
                        continue
                    s_min, s_max = min(starts_in), max(starts_in)
                    if op_time.eq(float(s_min), float(s_max)):
                        continue
                    t_first = float(s_min)
                    t_last = float(s_max)
                    if hand_norm == 'r':
                        highest = max(grp, key=lambda n: int(n.get('pitch', 0) or 0))
                        x1b = _key_to_x(int(highest.get('pitch', 0) or 0)) + float(stem_len_mm_for_barlines)
                        x2b = x1b + float(semitone_mm)
                    else:
                        lowest = min(grp, key=lambda n: int(n.get('pitch', 0) or 0))
                        x1b = _key_to_x(int(lowest.get('pitch', 0) or 0)) - float(stem_len_mm_for_barlines)
                        x2b = x1b - float(semitone_mm)
                    yb1 = _time_to_y(float(t_first))
                    yb2 = _time_to_y(float(t_last))
                    beam_segments_for_barlines.append({
                        't_start': float(t_first),
                        't_end': float(t_last),
                        'x1': float(x1b),
                        'x2': float(x2b),
                    })

                    # Beam connection lines (note stem tip to beam line) are part of beam drawing.
                    # Record their x-ranges at each note start so barline cuts include them.
                    for n in grp:
                        mt = float(n.get('time', t_first) or t_first)
                        if not (op_time.ge(mt, float(t0)) and op_time.lt(mt, float(t1))):
                            continue
                        y_note = _time_to_y(float(mt))
                        x_note = _key_to_x(int(n.get('pitch', 0) or 0))
                        if hand_norm == 'r':
                            x_tip = x_note + float(stem_len_mm_for_barlines)
                        else:
                            x_tip = x_note - float(stem_len_mm_for_barlines)
                        if abs(yb2 - yb1) > 1e-9:
                            t_ratio = (y_note - yb1) / (yb2 - yb1)
                            x_on_beam = x1b + t_ratio * (x2b - x1b)
                        else:
                            x_on_beam = x1b
                        beam_connect_segments_for_barlines.append({
                            'time': float(mt),
                            'x0': float(min(x_tip, x_on_beam)),
                            'x1': float(max(x_tip, x_on_beam)),
                            'beam_start': float(t_first),
                        })

            def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
                if not intervals:
                    return []
                clipped: list[tuple[float, float]] = []
                for a, b in intervals:
                    x0 = max(float(grid_left), min(float(grid_right), float(min(a, b))))
                    x1 = max(float(grid_left), min(float(grid_right), float(max(a, b))))
                    if x1 <= x0:
                        continue
                    clipped.append((x0, x1))
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

            def _barline_cut_intervals(ticks: float) -> list[tuple[float, float]]:
                intervals: list[tuple[float, float]] = []
                for item in line_notes_for_barlines:
                    n_t = float(item.get('time', 0.0) or 0.0)
                    if not op_time.eq(n_t, float(ticks)):
                        continue
                    p = int(item.get('pitch', 0) or 0)
                    x_note = _key_to_x(p)
                    intervals.append((
                        x_note - note_head_half_w - head_collision_pad - barline_symbol_gap_mm,
                        x_note + note_head_half_w + head_collision_pad + barline_symbol_gap_mm,
                    ))
                    if bool(layout.get('note_stem_visible', True)):
                        hand_key = str(item.get('hand', 'l') or 'l')
                        x_stem_tip = x_note - stem_len_mm_for_barlines if hand_key == 'l' else x_note + stem_len_mm_for_barlines
                        intervals.append((
                            min(x_note, x_stem_tip) - stem_collision_pad - barline_symbol_gap_mm,
                            max(x_note, x_stem_tip) + stem_collision_pad + barline_symbol_gap_mm,
                        ))

                # Chord connector lines span from lowest to highest pitch in same-hand chords.
                # Without this, the connector line between note heads can cross a barline gap.
                if bool(layout.get('chord_connect_visible', True)):
                    for chord_hand_key in ('l', 'r'):
                        chord_notes_at_tick = [
                            it for it in line_notes_for_barlines
                            if op_time.eq(float(it.get('time', 0.0) or 0.0), float(ticks))
                            and str(it.get('hand', 'l') or 'l') == chord_hand_key
                        ]
                        if len(chord_notes_at_tick) >= 2:
                            pitches_at_tick = [int(n.get('pitch', 0) or 0) for n in chord_notes_at_tick]
                            x_lo = _key_to_x(min(pitches_at_tick))
                            x_hi = _key_to_x(max(pitches_at_tick))
                            intervals.append((
                                x_lo - stem_collision_pad - barline_symbol_gap_mm,
                                x_hi + stem_collision_pad + barline_symbol_gap_mm,
                            ))

                for seg in beam_segments_for_barlines:
                    t0 = float(seg.get('t_start', 0.0) or 0.0)
                    t1 = float(seg.get('t_end', 0.0) or 0.0)
                    if op_time.lt(float(ticks), t0) or op_time.gt(float(ticks), t1):
                        continue
                    dt = t1 - t0
                    if abs(dt) <= 1e-9:
                        continue
                    ratio = (float(ticks) - t0) / dt
                    x_on_beam = float(seg.get('x1', 0.0) or 0.0) + ratio * (float(seg.get('x2', 0.0) or 0.0) - float(seg.get('x1', 0.0) or 0.0))
                    intervals.append((
                        x_on_beam - beam_collision_pad - barline_symbol_gap_mm,
                        x_on_beam + beam_collision_pad + barline_symbol_gap_mm,
                    ))

                for conn in beam_connect_segments_for_barlines:
                    c_t = float(conn.get('time', 0.0) or 0.0)
                    if not op_time.eq(c_t, float(ticks)):
                        continue
                    c_x0 = float(conn.get('x0', 0.0) or 0.0)
                    c_x1 = float(conn.get('x1', 0.0) or 0.0)
                    intervals.append((
                        c_x0 - beam_collision_pad - barline_symbol_gap_mm,
                        c_x1 + beam_collision_pad + barline_symbol_gap_mm,
                    ))

                return _merge_intervals(intervals)

            def _draw_barline_segments(
                yb: float,
                cuts: list[tuple[float, float]],
                width_mm: float,
                tags: list[str],
                item_id: int = 0,
                dash_pattern: list[float] | None = None,
            ) -> None:
                if not cuts:
                    du.add_line(
                        grid_left,
                        yb,
                        grid_right,
                        yb,
                        color=grid_color,
                        width_mm=width_mm,
                        id=item_id,
                        tags=tags,
                        dash_pattern=dash_pattern,
                    )
                    return
                x_cursor_seg = float(grid_left)
                min_seg = max(0.05, width_mm * 0.5)
                for c0, c1 in cuts:
                    if c0 - x_cursor_seg > min_seg:
                        du.add_line(
                            x_cursor_seg,
                            yb,
                            c0,
                            yb,
                            color=grid_color,
                            width_mm=width_mm,
                            id=item_id,
                            tags=tags,
                            dash_pattern=dash_pattern,
                        )
                    x_cursor_seg = max(x_cursor_seg, c1)
                if float(grid_right) - x_cursor_seg > min_seg:
                    du.add_line(
                        x_cursor_seg,
                        yb,
                        grid_right,
                        yb,
                        color=grid_color,
                        width_mm=width_mm,
                        id=item_id,
                        tags=tags,
                        dash_pattern=dash_pattern,
                    )

            def _draw_barline_constructive(ticks: float, width_mm: float, tag: str = 'barline') -> None:
                yb = _time_to_y(float(ticks))
                cuts = _barline_cut_intervals(float(ticks))
                _draw_barline_segments(float(yb), cuts, float(width_mm), [tag], 0)

            def _draw_double_bar_constructive(ticks: float, width_mm: float, gap_mm: float, ev_id: int = 0) -> None:
                yb = _time_to_y(float(ticks))
                cuts = _barline_cut_intervals(float(ticks))
                gap = max(0.1, float(gap_mm))
                tags = ['barline', 'double_barline']
                _draw_barline_segments(float(yb + gap), cuts, float(width_mm), tags, int(ev_id))

            def _draw_gridline_constructive(ticks: float, width_mm: float, dash: list[float] | None) -> None:
                yb = _time_to_y(float(ticks))
                cuts = _barline_cut_intervals(float(ticks))
                _draw_barline_segments(
                    float(yb),
                    cuts,
                    float(width_mm),
                    ['grid_line'],
                    0,
                    dash_pattern=dash,
                )

            '''Draw barlines and grid lines from the base grid, using constructive geometry to cut out collisions with notes and beams.'''
            time_cursor = 0.0
            has_any_barlines = False
            for bg in base_grid:
                numerator = int(bg.get('numerator', 4) or 4)
                denominator = int(bg.get('denominator', 4) or 4)
                measure_amount = int(bg.get('measure_amount', 1) or 1)
                beat_grouping = list(bg.get('beat_grouping', []) or [])
                indicator_enabled = bool(bg.get('indicator_enabled', True))
                bar_offsets, grid_offsets = resolve_grid_layer_offsets(beat_grouping, numerator, denominator)
                if bar_offsets:
                    has_any_barlines = True
                if measure_amount <= 0:
                    continue
                measure_len = float(numerator) * (4.0 / float(max(1, denominator))) * float(QUARTER_NOTE_UNIT)
                if op_time.ge(float(time_cursor), float(line['time_start'])) and op_time.lt(float(time_cursor), float(line['time_end'])) and indicator_enabled and bool(layout.get('time_signature_visible', True)):
                    y_ts = _time_to_y(float(time_cursor))
                    if indicator_type == 'classical':
                        _draw_classical_ts(numerator, denominator, indicator_enabled, y_ts)
                    elif indicator_type == 'klavarskribo':
                        _draw_klavarskribo_ts(numerator, denominator, indicator_enabled, y_ts, beat_grouping)
                    elif indicator_type == 'classical & klavarskribo':
                        _draw_classical_ts(numerator, denominator, indicator_enabled, y_ts)
                        _draw_klavarskribo_ts(numerator, denominator, indicator_enabled, y_ts, beat_grouping)
                for _ in range(measure_amount):
                    if op_time.gt(time_cursor, float(line['time_end'])):
                        break

                    m_start = float(time_cursor)
                    m_end = float(time_cursor + measure_len)
                    ov_start = max(float(line_start_ticks), float(m_start))
                    ov_end = min(float(line_end_ticks), float(m_end))
                    if grid_band_visible and (ov_end > ov_start):
                        group_boundaries = [float(ov_start)]
                        group_boundaries.extend(
                            float(m_start + float(off))
                            for off in grid_offsets
                            if float(ov_start) < float(m_start + float(off)) < float(ov_end)
                        )
                        group_boundaries.append(float(ov_end))
                        group_boundaries = sorted(list(dict.fromkeys(round(float(t), 6) for t in group_boundaries)))

                        dark_intervals = _clip_intervals(grid_dark_intervals_global, float(ov_start), float(ov_end))

                        if dark_intervals:
                            for t0, t1 in dark_intervals:
                                g0, g1 = _group_window_for_interval(group_boundaries, t0, t1)
                                span = _hand_band_x_span('l', g0, g1)
                                if span is None:
                                    continue
                                bx0, bx1 = span
                                y0 = _time_to_y(t0)
                                y1b = _time_to_y(t1)
                                if y1b < y0:
                                    y0, y1b = y1b, y0
                                du.add_rectangle(
                                    bx0,
                                    y0,
                                    bx1,
                                    y1b,
                                    stroke_color=None,
                                    fill_color=grid_band_tint,
                                    id=0,
                                    tags=['grid_band'],
                                )
                                grid_band_dark_intervals['left'].append((t0, t1))

                    for off in bar_offsets:
                        t = float(time_cursor + float(off))
                        if op_time.lt(t, float(line['time_start'])) or op_time.gt(t, float(line['time_end'])):
                            continue
                        if not barline_visible:
                            continue
                        _draw_barline_constructive(t, bar_width_mm, tag='barline')
                    for off in grid_offsets:
                        t = float(time_cursor + float(off))
                        if op_time.lt(t, float(line['time_start'])) or op_time.gt(t, float(line['time_end'])):
                            continue
                        if not grid_line_visible:
                            continue
                        _draw_gridline_constructive(t, max(0.1, grid_width_mm), dash_pattern)
                    time_cursor += measure_len
                if op_time.gt(time_cursor, float(line['time_end'])):
                    break
            
            '''End-barline drawing'''
            if barline_visible and has_any_barlines and op_time.ge(total_ticks, float(line['time_start'])) and op_time.le(total_ticks, float(line['time_end'])):
                # Single final barline (Klavarskribo convention).
                end_thick_w = bar_width_mm * 1.5
                y_end = _time_to_y(float(total_ticks))
                du.add_line(
                    grid_left, 
                    y_end, 
                    grid_right, 
                    y_end, 
                    color=grid_color, 
                    width_mm=end_thick_w * 1.5, 
                    id=0, 
                    tags=['grid_line', 'final_barline'], 
                    dash_pattern=None
                )

            '''Double barlines from events, drawn with the same constructive geometry as regular barlines to avoid collisions.'''
            if barline_visible and bool(layout.get('double_barline_visible', True)) and norm_double_bars:
                double_w_mm = max(0.1, bar_width_mm)
                # Keep visible whitespace between lines after increasing line thickness.
                inner_clear_gap_mm = max(0.35, semitone_mm * 0.5)
                double_gap_mm = max(double_w_mm * 2.0, inner_clear_gap_mm + double_w_mm)
                for ev in norm_double_bars:
                    ev_t = float(ev.get('time', 0.0) or 0.0)
                    if op_time.lt(ev_t, float(line['time_start'])) or op_time.gt(ev_t, float(line['time_end'])):
                        continue
                    _draw_double_bar_constructive(
                        float(ev_t),
                        float(double_w_mm),
                        float(double_gap_mm),
                        int(ev.get('id', 0) or 0),
                    )

            # Problem solved: draw pedal symbols using draw_pedal_symbol for keyboard-aware positioning
            if pedals:
                pedal_thickness_mm = float(layout.get('pedal_symbol_thickness_mm', 0.3) or 0.3) * scale

                def _read_pedal_field(pedal_ev, name: str, default):
                    if isinstance(pedal_ev, dict):
                        return pedal_ev.get(name, default)
                    return getattr(pedal_ev, name, default)
                
                def _pedal_time_to_y(time_val: float) -> float:
                    """Convert time to Y coordinate for this line."""
                    total = max(1e-6, float(line['time_end'] - line['time_start']))
                    rel = (float(time_val) - float(line['time_start'])) / total
                    rel = max(0.0, min(1.0, rel))
                    return y1 + (y2 - y1) * rel

                def _pedal_rpitch_to_x(rpitch_val: int) -> float:
                    """Convert C4-relative semitone offset to page X coordinate."""
                    base_x_c4 = float(_key_to_x(40))
                    return base_x_c4 + (float(rpitch_val) * float(semitone_mm))

                for pedal_ev in pedals:
                    p_t = float(_read_pedal_field(pedal_ev, 'time', 0.0) or 0.0)
                    p_symbol = str(_read_pedal_field(pedal_ev, 'symbol', '') or '')
                    _is_up_symbol = p_symbol in ('up_keytab', 'up_klavarskribo')
                    # up symbols at line_end belong to the ending line only (drawn upward);
                    # skip them at line_start so they don't repeat on the new line.
                    # All other symbols at line_end belong to the next line.
                    if _is_up_symbol:
                        if op_time.le(p_t, float(line['time_start'])) or op_time.gt(p_t, float(line['time_end'])):
                            continue
                    else:
                        if op_time.lt(p_t, float(line['time_start'])) or op_time.ge(p_t, float(line['time_end'])):
                            continue

                    invisible_raw = _read_pedal_field(pedal_ev, 'invisible', False)
                    if isinstance(invisible_raw, str):
                        is_invisible = str(invisible_raw).strip().lower() in ('1', 'true', 'yes', 'on')
                    else:
                        is_invisible = bool(invisible_raw)
                    if is_invisible:
                        continue
                    
                    try:
                        draw_pedal_symbol(
                            du,
                            pedal_ev,
                            time_to_y_mm=_pedal_time_to_y,
                            rpitch_to_x_mm=_pedal_rpitch_to_x,
                            color=notation_color,
                            background_color=paper_color,
                            width_mm=pedal_thickness_mm,
                            semitone_space_mm=semitone_mm,
                            layout=layout,
                            id=int(_read_pedal_field(pedal_ev, '_id', _read_pedal_field(pedal_ev, 'id', 0)) or 0),
                            tags=['pedal_symbol'],
                        )
                    except Exception:
                        pass

            # Problem solved: render count lines as lightweight guides.
            if bool(layout.get('countline_visible', True)) and count_lines:
                dash_pattern_raw = list(layout.get('countline_dash_pattern', []) or [])
                dash_pattern = [float(v) * scale for v in dash_pattern_raw] if dash_pattern_raw else None
                countline_w = float(layout.get('countline_thickness_mm', 0.5) or 0.5) * scale
                base_x_c4 = _key_to_x(40)
                for ev in count_lines:
                    t0 = float(ev.get('time', 0.0) or 0.0)
                    rp1 = int(ev.get('rpitch1', 0) or 0)
                    rp2 = int(ev.get('rpitch2', 4) or 4)
                    if op_time.lt(t0, float(line['time_start'])) or op_time.gt(t0, float(line['time_end'])):
                        continue
                    x1 = base_x_c4 + (float(rp1) * semitone_mm)
                    x2 = base_x_c4 + (float(rp2) * semitone_mm)
                    if x2 < x1:
                        x1, x2 = x2, x1
                    y_mm = _time_to_y(t0)
                    du.add_line(
                        x1,
                        y_mm,
                        x2,
                        y_mm,
                        color=notation_color,
                        width_mm=countline_w,
                        dash_pattern=dash_pattern,
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
            line_slur_continuations: list[dict] = []   # started before, extend into this line
            line_slur_end_indicators: list[dict] = []  # connected slurs starting at line_end
            line_slur_start_indicators: list[dict] = [] # connected slurs ending at line_start
            if norm_slurs:
                line_start = float(line.get('time_start', 0.0) or 0.0)
                line_end = float(line.get('time_end', 0.0) or 0.0)
                for sl in norm_slurs:
                    p1_t = float(sl.get('y1_time', 0.0) or 0.0)
                    p4_t = float(sl.get('y4_time', 0.0) or 0.0)
                    if op_time.ge(p1_t, line_start) and op_time.lt(p1_t, line_end):
                        line_slurs.append(sl)
                    elif op_time.lt(p1_t, line_start) and op_time.gt(p4_t, line_start):
                        line_slur_continuations.append(sl)
                # Connected-slur end-of-line indicators: connected slurs starting at line_end.
                for sl in norm_slurs:
                    p1_t = float(sl.get('y1_time', 0.0) or 0.0)
                    if round(p1_t, 4) != round(line_end, 4):
                        continue
                    p1_ep = (int(sl.get('x1_rpitch', 0) or 0), round(p1_t, 4))
                    if _slur_ep_map.get(p1_ep) and len(_slur_ep_map[p1_ep]) >= 2:
                        line_slur_end_indicators.append(sl)
                # Connected-slur start-of-line indicators: connected slurs ending at line_start.
                for sl in norm_slurs:
                    p4_t = float(sl.get('y4_time', 0.0) or 0.0)
                    if round(p4_t, 4) != round(line_start, 4):
                        continue
                    p4_ep = (int(sl.get('x4_rpitch', 0) or 0), round(p4_t, 4))
                    if _slur_ep_map.get(p4_ep) and len(_slur_ep_map[p4_ep]) >= 2:
                        line_slur_start_indicators.append(sl)

            line_texts: list[dict] = []
            if norm_texts:
                line_start = float(line.get('time_start', 0.0) or 0.0)
                line_end = float(line.get('time_end', 0.0) or 0.0)
                for tx in norm_texts:
                    t_time = float(tx.get('time', 0.0) or 0.0)
                    if op_time.lt(t_time, float(line_start)) or op_time.ge(t_time, float(line_end)):
                        continue
                    line_texts.append(tx)

            line_dynamic_symbols: list[dict] = []
            if norm_dynamic_symbols:
                line_start = float(line.get('time_start', 0.0) or 0.0)
                line_end = float(line.get('time_end', 0.0) or 0.0)
                for ds in norm_dynamic_symbols:
                    t_time = float(ds.get('time', 0.0) or 0.0)
                    if op_time.lt(t_time, float(line_start)) or op_time.ge(t_time, float(line_end)):
                        continue
                    line_dynamic_symbols.append(ds)

            line_crescendos: list[dict] = []
            if norm_crescendos:
                line_start = float(line.get('time_start', 0.0) or 0.0)
                line_end = float(line.get('time_end', 0.0) or 0.0)
                for hp in norm_crescendos:
                    hp_start = float(hp.get('time', 0.0) or 0.0)
                    hp_end = float(hp.get('end', hp_start) or hp_start)
                    if op_time.ge(hp_start, float(line_end)) or op_time.le(hp_end, float(line_start)):
                        continue
                    line_crescendos.append(hp)

            line_decrescendos: list[dict] = []
            if norm_decrescendos:
                line_start = float(line.get('time_start', 0.0) or 0.0)
                line_end = float(line.get('time_end', 0.0) or 0.0)
                for hp in norm_decrescendos:
                    hp_start = float(hp.get('time', 0.0) or 0.0)
                    hp_end = float(hp.get('end', hp_start) or hp_start)
                    if op_time.ge(hp_start, float(line_end)) or op_time.le(hp_end, float(line_start)):
                        continue
                    line_decrescendos.append(hp)

            line_tempos: list[dict] = []
            if norm_tempos:
                line_start = float(line.get('time_start', 0.0) or 0.0)
                line_end = float(line.get('time_end', 0.0) or 0.0)
                for tp in norm_tempos:
                    t_start = float(tp.get('time', 0.0) or 0.0)
                    t_end = float(tp.get('end', t_start) or t_start)
                    if op_time.ge(t_start, float(line_end)) or op_time.le(t_end, float(line_start)):
                        continue
                    line_tempos.append(tp)

            notes_by_hand_line: dict[str, list[dict]] = {'l': [], 'r': []}
            for item in line_notes:
                hk = str(item.get('hand', 'l') or 'l')
                hand_norm = 'l' if hk == 'l' else 'r'
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

            stem_len_units = float(layout.get('note_stem_length_semitone', 3) or 3)
            stem_len_mm = stem_len_units * semitone_mm

            # Pre-compute beam line bounds so measure numbers can test real beam spans.
            beam_thickness_mm = float(layout.get('beam_thickness_mm', 1.0) or 1.0) * scale
            beam_line_bounds: list[dict[str, float]] = []

            def _record_beam_line_bounds() -> None:
                for hand_norm, payload in beam_groups_by_hand.items():
                    groups, windows = payload
                    for idx, grp in enumerate(groups):
                        if not grp or idx >= len(windows):
                            continue
                        t0, t1 = windows[idx]
                        starts_in = [
                            float(n.get('time', 0.0) or 0.0)
                            for n in grp
                            if op_time.ge(float(n.get('time', 0.0) or 0.0), float(t0))
                            and op_time.lt(float(n.get('time', 0.0) or 0.0), float(t1))
                        ]
                        if not starts_in:
                            continue
                        s_min, s_max = min(starts_in), max(starts_in)
                        if op_time.eq(float(s_min), float(s_max)):
                            continue
                        t_first = min(starts_in)
                        t_last = max(starts_in)
                        if hand_norm == 'r':
                            highest = max(grp, key=lambda n: int(n.get('pitch', 0) or 0))
                            x1 = _key_to_x(int(highest.get('pitch', 0) or 0)) + float(stem_len_units * semitone_mm)
                            x2 = x1 + float(semitone_mm)
                        else:
                            lowest = min(grp, key=lambda n: int(n.get('pitch', 0) or 0))
                            x1 = _key_to_x(int(lowest.get('pitch', 0) or 0)) - float(stem_len_units * semitone_mm)
                            x2 = x1 - float(semitone_mm)
                        yb1 = _time_to_y(float(t_first))
                        yb2 = _time_to_y(float(t_last))
                        pad = beam_thickness_mm * 0.5
                        beam_line_bounds.append(
                            {
                                'x_min': min(x1, x2) - pad,
                                'x_max': max(x1, x2) + pad,
                                'y_min': min(yb1, yb2) - pad,
                                'y_max': max(yb1, yb2) + pad,
                            }
                        )

            def _beam_right_at_y(y: float) -> float | None:
                right = None
                for seg in beam_line_bounds:
                    if y < seg['y_min'] or y > seg['y_max']:
                        continue
                    xr = float(seg['x_max'])
                    if right is None or xr > right:
                        right = xr
                return right

            _record_beam_line_bounds()

            # Problem solved: measure numbers must avoid colliding with notes/beams.
            mn_family, mn_size, mn_bold, mn_italic, _mn_ul = _layout_font('measure_numbering_font', 'Edwin', 10.0)
            size_pt = mn_size * scale
            mm_per_pt = 25.4 / 72.0
            text_h_mm = size_pt * mm_per_pt
            measure_pad = 1.5
            measure_symbol_anchor: dict[float, tuple[float, float, float]] = {}
            measure_symbol_guide_right: dict[float, float] = {}
            measure_symbol_default_right_x = float(grid_right + measure_pad * 2.0)
            measure_guide_width_mm = max(
                0.05,
                float(layout.get('measure_numbering_guide_thickness_mm', 1.0) or 1.0) * scale,
            )
            # Allow an explicit measure-numbering guide dash style, with
            # fallback to the global grid line pattern for backward compatibility.
            default_measure_guide_dash_mm = list(
                getattr(Layout(), 'measure_numbering_guide_dash_pattern_mm', default_grid_dash_mm)
                or default_grid_dash_mm
            )
            measure_guide_dash_pattern = _scaled_dash_pattern_with_default(
                layout.get('measure_numbering_guide_dash_pattern_mm', default_measure_guide_dash_mm),
                default_measure_guide_dash_mm,
                scale,
            )

            mn_placement = str(layout.get('measure_numbering_placement', 'system') or 'system')
            mn_guide_visible = layout.get('measure_numbering_guide_visible', True) is not False
            mn_numbers_visible = layout.get('measure_numbers_visible', True) is not False
            tempo_indicator_visible = layout.get('tempo_indicator_visible', True) is not False
            line_time_start = float(line.get('time_start', 0.0) or 0.0)
            black_rule = str(layout.get('black_note_rule', 'below_stem') or 'below_stem')

            def _measure_text_metrics_mm(txt: str) -> tuple[float, float, float, float]:
                # Returns (raw_w, raw_h, effective_w_for_x_collision, effective_h_for_y_time_span)
                _xb, _yb, raw_w, raw_h = du._get_text_extents_mm(txt, mn_family, size_pt, mn_italic, mn_bold)
                raw_w = max(1.0, float(raw_w))
                raw_h = max(0.5, float(raw_h), float(text_h_mm))
                if horizontal_read_direction:
                    # Text is rotated +90 to compensate page rotation in horizontal mode.
                    return raw_w, raw_h, max(1.0, raw_h), max(0.5, raw_w)
                return raw_w, raw_h, raw_w, raw_h

            def _note_x_range(it: dict, include_stem: bool) -> tuple[float, float]:
                p = int(it.get('pitch', 0) or 0)
                x = _key_to_x(p)
                w = semitone_mm
                hand_key = str(it.get('hand', 'l') or 'l')
                beam_ext = semitone_mm if include_stem else 0.0
                if hand_key == 'l':
                    x_min = x - (max(w, stem_len_mm + beam_ext) if include_stem else w)
                    x_max = x + w
                else:
                    x_min = x - w
                    x_max = x + (max(w, stem_len_mm + beam_ext) if include_stem else w)
                return (x_min, x_max)

            def _right_extent(t0: float, t1: float) -> float:
                max_x = grid_right
                for it in line_notes:
                    nt = float(it.get('time', 0.0) or 0.0)
                    ne = float(it.get('end', 0.0) or 0.0)
                    if op_time.ge(nt, float(t1)) or op_time.le(ne, float(t0)):
                        continue
                    near_start = op_time.lt(nt, float(t1)) and op_time.ge(nt, float(t0))
                    _x0, x1 = _note_x_range(it, include_stem=near_start)
                    if x1 > max_x:
                        max_x = x1
                return max_x

            def _y_to_time_unclamped(y_mm: float) -> float:
                return float(line['time_start']) + ((float(y_mm) - float(y1)) * float(tick_per_mm))

            ledger_collision_segments: list[dict[str, float]] = []

            def _collect_ledger_collision_segments() -> None:
                if not bool(layout.get('stave_visible', True)):
                    return

                def _append_segment_right_edge(x_right: float, t_start: float, t_end: float) -> None:
                    if float(x_right) <= float(grid_right):
                        return
                    s = min(float(t_start), float(t_end))
                    e = max(float(t_start), float(t_end))
                    ledger_collision_segments.append({'x_right': float(x_right), 't_start': s, 't_end': e})

                def _ledger_line_width_for_key(key_i: int) -> float:
                    if key_i in (41, 43):
                        return float(stave_clef_w)
                    if key_i in key_class_filter('FGA'):
                        return float(stave_three_w)
                    return float(stave_two_w)

                def _append_manual_ledger_collision_for_pitch_at_y(pitch_value: int, y_center: float) -> None:
                    y_seg1 = float(y_center) - float(semitone_mm)
                    y_draw_start = y_seg1 - float(semitone_mm) * 2.0
                    y_draw_end = y_draw_start + max(0.0, float(stave_ledger_len))
                    t_start = _y_to_time_unclamped(y_draw_start)
                    t_end = _y_to_time_unclamped(y_draw_end)

                    right_edge: float | None = None

                    if int(pitch_value) in (1, 2, 3) and bool(line.get('a0_ledger_mode', False)):
                        x_right = float(_key_to_x(2)) + (_ledger_line_width_for_key(2) * 0.5)
                        right_edge = x_right if right_edge is None else max(float(right_edge), x_right)

                    if manual_range:
                        ledger_groups: list[dict] = []
                        if int(pitch_value) < natural_bound_left:
                            g_start = _group_index_for_key(int(pitch_value))
                            g_end = int(bound_group_low or 0) - 1
                            if g_start <= g_end:
                                ledger_groups = line_groups[g_start:g_end + 1]
                        elif int(pitch_value) > natural_bound_right:
                            g_start = int(bound_group_high or 0) + 1
                            g_end = _group_index_for_key(int(pitch_value))
                            if g_start <= g_end:
                                ledger_groups = line_groups[g_start:g_end + 1]

                        for grp in ledger_groups:
                            for key in grp.get('keys', []):
                                key_i = int(key)
                                x_right = float(_key_to_x(key_i)) + (_ledger_line_width_for_key(key_i) * 0.5)
                                right_edge = x_right if right_edge is None else max(float(right_edge), x_right)

                    if right_edge is not None:
                        _append_segment_right_edge(float(right_edge), t_start, t_end)

                for item in line_notes:
                    p = int(item.get('pitch', 0) or 0)
                    n_t = float(item.get('time', 0.0) or 0.0)
                    n_end = float(item.get('end', 0.0) or 0.0)
                    hand_key = str(item.get('hand', 'l') or 'l')

                    y_start = _time_to_y(n_t)
                    default_black_above = p in BLACK_KEYS and _black_note_above_stem(item, black_rule, line_notes, op_time)
                    spec = resolve_notehead_spec(item.get('raw', {}) or {}, default_black_above=default_black_above)
                    note_y = y_start
                    if bool(getattr(spec, 'is_up', False)):
                        note_y = y_start - (semitone_mm * 2.0)

                    _append_manual_ledger_collision_for_pitch_at_y(int(p), float(note_y + semitone_mm))

                    if not bool(layout.get('note_continuation_dot_visible', True)):
                        continue

                    dot_times: list[float] = []
                    for other in line_notes:
                        if int(other.get('idx', -1) or -1) == int(item.get('idx', -2) or -2):
                            continue
                        if str(other.get('hand', 'l') or 'l') != hand_key:
                            continue
                        s = float(other.get('time', 0.0) or 0.0)
                        e = float(other.get('end', 0.0) or 0.0)
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
                    if _is_line_continuation(item):
                        dot_times.append(float(line_start))

                    if not dot_times:
                        continue

                    dot_x = float(_key_to_x(int(p)))
                    min_collision_gap = max(0.0, float(semitone_mm) * 2.0 - 1e-6)
                    for t in sorted(set(dot_times)):
                        y_center = _time_to_y(float(t)) + float(semitone_mm)

                        has_adjacent_start = False
                        for other in line_notes:
                            if int(other.get('idx', -1) or -1) == int(item.get('idx', -2) or -2):
                                continue
                            if _is_line_continuation(other):
                                continue
                            if not op_time.eq(float(other.get('time', 0.0) or 0.0), float(t)):
                                continue
                            other_pitch = int(other.get('pitch', 0) or 0)
                            if abs(other_pitch - int(p)) == 1:
                                other_black_above = (
                                    other_pitch in BLACK_KEYS
                                    and _black_note_above_stem(other, black_rule, line_notes, op_time)
                                )
                                if other_black_above:
                                    continue
                                other_x = float(_key_to_x(int(other_pitch)))
                                if abs(other_x - dot_x) >= min_collision_gap:
                                    continue
                                has_adjacent_start = True
                                break
                        if has_adjacent_start:
                            y_center += float(semitone_mm) * 2.0

                        _append_manual_ledger_collision_for_pitch_at_y(int(p), float(y_center))

            def _ledger_right_extent(t0: float, t1: float) -> float:
                max_x = grid_right
                for seg in ledger_collision_segments:
                    ls = float(seg.get('t_start', 0.0) or 0.0)
                    le = float(seg.get('t_end', 0.0) or 0.0)
                    if op_time.ge(ls, float(t1)) or op_time.le(le, float(t0)):
                        continue
                    xr = float(seg.get('x_right', grid_right) or grid_right)
                    if xr > max_x:
                        max_x = xr
                return max_x

            _collect_ledger_collision_segments()

            def _collides(x0: float, x1: float, t0: float, t1: float) -> bool:
                for it in line_notes:
                    nt = float(it.get('time', 0.0) or 0.0)
                    ne = float(it.get('end', 0.0) or 0.0)
                    if op_time.ge(nt, float(t1)) or op_time.le(ne, float(t0)):
                        continue
                    near_start = op_time.lt(nt, float(t1)) and op_time.ge(nt, float(t0))
                    nx0, nx1 = _note_x_range(it, include_stem=near_start)
                    if (nx1 >= x0) and (nx0 <= x1):
                        return True
                for seg in ledger_collision_segments:
                    ls = float(seg.get('t_start', 0.0) or 0.0)
                    le = float(seg.get('t_end', 0.0) or 0.0)
                    if op_time.ge(ls, float(t1)) or op_time.le(le, float(t0)):
                        continue
                    xr = float(seg.get('x_right', grid_right) or grid_right)
                    if xr >= x0:
                        return True
                return False

            for mw in measure_windows:
                m_start = float(mw.get('start', 0.0))
                m_end = float(mw.get('end', 0.0))
                if op_time.ge(m_start, float(line['time_end'])) or op_time.le(m_end, float(line['time_start'])):
                    continue
                # Apply measure_numbering_placement filter
                if mn_placement == 'system':
                    if not op_time.eq(m_start, line_time_start):
                        continue
                # 'barline': draw at every measure start (no filter)
                num_txt = str(int(mw.get('number', 0) or 0))
                if not num_txt:
                    continue
                _raw_w, raw_h_mm, text_w_mm, text_h_eff_mm = _measure_text_metrics_mm(num_txt)
                t0 = m_start
                t1 = min(float(line['time_end']), m_start + (text_h_eff_mm * tick_per_mm))
                y_text = _time_to_y(t0) + 1.0

                # Default outside-right; only move further right on collision
                base_right = grid_right + measure_pad
                guide_y = _time_to_y(t0)
                beam_right_candidates = [
                    br for br in (_beam_right_at_y(guide_y), _beam_right_at_y(y_text)) if br is not None
                ]
                beam_right = max(beam_right_candidates) if beam_right_candidates else None
                needed_right = _right_extent(t0, t1) + measure_pad
                needed_right = max(needed_right, _ledger_right_extent(t0, t1) + measure_pad)
                if beam_right is not None:
                    needed_right = max(needed_right, float(beam_right) + measure_pad)
                x_pos_guide = max(base_right, needed_right)
                x_pos = x_pos_guide
                if horizontal_read_direction:
                    # Horizontal read mode: after +90° text rotation, the
                    # effective bbox height maps to drawing-space X progression.
                    # Use unrotated text height for X placement to avoid
                    # digit-width-dependent horizontal jumps (e.g. 9 -> 10).
                    x_pos += float(raw_h_mm)
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
                guide_right = float(x_pos_guide + text_w_mm)
                if mn_guide_visible:
                    du.add_line(
                        grid_right,
                        guide_y,
                        guide_right,
                        guide_y,
                        color=notation_color,
                        width_mm=measure_guide_width_mm,
                        id=0,
                        tags=['measure_number_guide'],
                        dash_pattern=measure_guide_dash_pattern,
                    )
                if mn_numbers_visible:
                    du.add_text(
                        x_pos,
                        y_text,
                        num_txt,
                        size_pt=size_pt,
                        color=notation_color,
                        id=0,
                        tags=['measure_number'],
                        anchor='nw',
                        family=mn_family,
                        bold=mn_bold,
                        italic=mn_italic,
                        angle_deg=90.0 if horizontal_read_direction else 0.0,
                    )
                measure_symbol_anchor[round(float(m_start), 6)] = (
                    float(x_pos),
                    float(text_w_mm),
                    float(y_text),
                )
                if mn_numbers_visible:
                    measure_symbol_guide_right[round(float(m_start), 6)] = float(guide_right)
                measure_symbol_default_right_x = max(
                    float(measure_symbol_default_right_x),
                    float(x_pos_guide + text_w_mm),
                )

            if tempo_indicator_visible and line_tempos:
                try:
                    tempo_font_family = _resolve_font_family('Edwin') or 'Edwin'
                except Exception:
                    tempo_font_family = 'Edwin'

                tempo_font_size_pt = 32.0 * scale
                tempo_dash = [0.5, 1.0]
                tempo_stroke = 0.25
                right_outer_stave_x = float(grid_right)
                tempo_right_pad = max(0.6, 4.0 * scale)

                def _tempo_left_x(tp_time: float) -> float:
                    key = round(float(tp_time), 6)
                    anchored = measure_symbol_anchor.get(key)
                    if anchored is not None:
                        x_num, w_num, _y_num = anchored
                        return float(x_num + w_num)
                    # Mid-measure tempo changes: use the outer measure-number guide end.
                    return float(measure_symbol_default_right_x)

                def _tempo_top_start_x(tp_time: float) -> float:
                    key = round(float(tp_time), 6)
                    # If a measure number exists at this barline position, start the
                    # tempo top line at the outer right edge of the measure guide to
                    # avoid overlapping guide dashes.
                    guide_right = measure_symbol_guide_right.get(key)
                    if guide_right is not None:
                        return float(guide_right)
                    return float(right_outer_stave_x)

                for tp in line_tempos:
                    try:
                        t0 = float(tp.get('time', 0.0) or 0.0)
                        t1 = float(tp.get('end', t0) or t0)
                        tempo_val = int(tp.get('tempo', 60) or 60)
                        tempo_x_offset = float(tp.get('x_offset', 0.0) or 0.0)
                    except Exception:
                        continue
                    if tp.get('invisible', False):
                        continue
                    if op_time.le(t1, t0):
                        continue

                    seg_t0 = max(float(line['time_start']), t0)
                    seg_t1 = min(float(line['time_end']), t1)
                    if op_time.le(seg_t1, seg_t0):
                        continue

                    y0_tempo = _time_to_y(seg_t0)
                    y1_tempo = _time_to_y(seg_t1)
                    if op_time.gt(y0_tempo, y1_tempo):
                        y0_tempo, y1_tempo = y1_tempo, y0_tempo

                    tempo_text = str(tempo_val)
                    _, _, tempo_text_w, tempo_text_h = du._get_text_extents_mm(
                        text=tempo_text,
                        family=tempo_font_family,
                        size_pt=tempo_font_size_pt,
                        italic=False,
                        bold=True,
                    )
                    tempo_text_w = max(1.0, float(tempo_text_w))
                    tempo_text_h = max(0.5, float(tempo_text_h))
                    tempo_text_span_x = float(tempo_text_h) if horizontal_read_direction else float(tempo_text_w)

                    # Keep the left edge of tempo text aligned with the right edge
                    # of the measure-number text; variable text width grows to the right.
                    tempo_text_left = _tempo_left_x(t0) + (tempo_x_offset * scale)
                    tempo_x_left = float(tempo_text_left)
                    tempo_text_center_x = float(tempo_x_left + (tempo_text_span_x * 0.5))
                    tempo_x_right = float(tempo_x_left + tempo_text_span_x + tempo_right_pad)
                    tempo_top_start_x = _tempo_top_start_x(t0)

                    # Open-left dashed bracket (top, right, bottom only).
                    du.add_line(
                        tempo_top_start_x,
                        y0_tempo,
                        tempo_x_right,
                        y0_tempo,
                        color=notation_color,
                        width_mm=tempo_stroke,
                        id=int(tp.get('id', 0) or 0),
                        tags=['tempo_bg'],
                        dash_pattern=tempo_dash,
                    )
                    du.add_line(
                        tempo_x_right,
                        y0_tempo,
                        tempo_x_right,
                        y1_tempo,
                        color=notation_color,
                        width_mm=tempo_stroke,
                        id=int(tp.get('id', 0) or 0),
                        tags=['tempo_bg'],
                        dash_pattern=tempo_dash,
                    )
                    du.add_line(
                        tempo_x_left,
                        y1_tempo,
                        tempo_x_right,
                        y1_tempo,
                        color=notation_color,
                        width_mm=tempo_stroke,
                        id=int(tp.get('id', 0) or 0),
                        tags=['tempo_bg'],
                        dash_pattern=tempo_dash,
                    )

                    y_center_tempo = (y0_tempo + y1_tempo) * 0.5
                    du.add_text(
                        tempo_text_center_x,
                        y_center_tempo,
                        tempo_text,
                        family=tempo_font_family,
                        size_pt=tempo_font_size_pt,
                        italic=False,
                        bold=True,
                        color=notation_color,
                        anchor='center',
                        id=int(tp.get('id', 0) or 0),
                        tags=['tempo_text'],
                        hit_rect_mm=None,
                        angle_deg=90.0 if horizontal_read_direction else 0.0,
                    )

            symbol_width = max(3.0, semitone_mm * 4.0)
            symbol_gap = max(0.6, semitone_mm * 0.35)
            symbol_thick_w = max(0.1, bar_width_mm)
            symbol_dot_d = max(1.0, semitone_mm * 0.6)
            # Minimum clear gap between the outer edge of the horizontal line
            # and the nearest edge of a dot. Increase to space dots further out.
            dot_line_gap = semitone_mm
            # dot center must be at least line_half + gap + dot_radius away from y_rep
            symbol_dot_y = max(
                max(0.8, semitone_mm * 0.55),
                symbol_thick_w / 2.0 + dot_line_gap + symbol_dot_d / 2.0,
            )

            def _symbol_left_x(rep_t: float) -> float:
                key = round(float(rep_t), 6)
                anchored = measure_symbol_anchor.get(key)
                if anchored is not None:
                    x_num, w_num, _y_num = anchored
                    return float(x_num + w_num + symbol_gap)
                return float(measure_symbol_default_right_x + symbol_gap)

            # Set of barline times (measure starts) for detecting non-barline repeat positions.
            _barline_time_set: set[float] = {
                round(float(mw.get('start', 0.0)), 6) for mw in measure_windows
            }

            def _draw_repeat_symbol(rep_t: float, ev_id: int, kind: str) -> None:
                if op_time.lt(rep_t, float(line['time_start'])) or op_time.gt(rep_t, float(line['time_end'])):
                    return
                y_rep = float(_time_to_y(rep_t))
                is_on_barline = round(float(rep_t), 6) in _barline_time_set
                if is_on_barline:
                    x_left = _symbol_left_x(rep_t)
                else:
                    # Beam/note-aware positioning for mid-measure repeats.
                    t0 = rep_t
                    t1 = min(float(line['time_end']), rep_t + (symbol_width * tick_per_mm))
                    base_right = grid_right + measure_pad
                    beam_right_val = _beam_right_at_y(y_rep)
                    needed_right = _right_extent(t0, t1) + measure_pad
                    needed_right = max(needed_right, _ledger_right_extent(t0, t1) + measure_pad)
                    if beam_right_val is not None:
                        needed_right = max(needed_right, float(beam_right_val) + measure_pad)
                    x_left = max(base_right, needed_right)
                    xl, xr = x_left, x_left + symbol_width
                    _step = symbol_width + measure_pad
                    _tries = 0
                    while _collides(xl, xr, t0, t1) and _tries < 16:
                        x_left += _step
                        xl = x_left
                        xr = x_left + symbol_width
                        _tries += 1
                if horizontal_read_direction:
                    # some finetuning to avoid colliding with measure numbers in horizontal read mode
                    if is_on_barline:
                        x_left -= symbol_width/2
                x_right = x_left + symbol_width
                dot_x1 = x_left + (symbol_width * 0.25)
                dot_x2 = x_left + (symbol_width * 0.75)
                line_end_ticks = float(line.get('time_end', 0.0) or 0.0)
                is_prev_system_duplicate = op_time.eq(rep_t, line_end_ticks) and op_time.gt(rep_t, first_system_start)
                if is_prev_system_duplicate or not is_on_barline:
                    du.add_line(
                        grid_right,
                        y_rep,
                        x_right,
                        y_rep,
                        color=notation_color,
                        width_mm=measure_guide_width_mm,
                        id=0,
                        tags=['measure_number_guide'],
                        dash_pattern=measure_guide_dash_pattern,
                    )
                if kind == 'start':
                    du.add_line(
                        x_left,
                        y_rep,
                        x_right,
                        y_rep,
                        color=notation_color,
                        width_mm=symbol_thick_w,
                        id=ev_id,
                        tags=['barline_symbol', 'start_repeat'],
                        dash_pattern=None,
                    )
                    for dot_x in (dot_x1, dot_x2):
                        du.add_oval(
                            dot_x - (symbol_dot_d / 2.0),
                            y_rep + symbol_dot_y - (symbol_dot_d / 2.0),
                            dot_x + (symbol_dot_d / 2.0),
                            y_rep + symbol_dot_y + (symbol_dot_d / 2.0),
                            stroke_color=None,
                            fill_color=notation_color,
                            id=ev_id,
                            tags=['barline_symbol_dot', 'start_repeat_dot'],
                        )
                elif kind == 'end':
                    du.add_line(
                        x_left,
                        y_rep,
                        x_right,
                        y_rep,
                        color=notation_color,
                        width_mm=symbol_thick_w,
                        id=ev_id,
                        tags=['barline_symbol', 'end_repeat'],
                        dash_pattern=None,
                    )
                    for dot_x in (dot_x1, dot_x2):
                        du.add_oval(
                            dot_x - (symbol_dot_d / 2.0),
                            y_rep - symbol_dot_y - (symbol_dot_d / 2.0),
                            dot_x + (symbol_dot_d / 2.0),
                            y_rep - symbol_dot_y + (symbol_dot_d / 2.0),
                            stroke_color=None,
                            fill_color=notation_color,
                            id=ev_id,
                            tags=['barline_symbol_dot', 'end_repeat_dot'],
                        )

            if bool(layout.get('repeat_start_visible', True)) and norm_start_repeats:
                for ev in norm_start_repeats:
                    _draw_repeat_symbol(
                        float(ev.get('time', 0.0) or 0.0),
                        int(ev.get('id', 0) or 0),
                        'start',
                    )
            if bool(layout.get('repeat_end_visible', True)) and norm_end_repeats:
                for ev in norm_end_repeats:
                    _draw_repeat_symbol(
                        float(ev.get('time', 0.0) or 0.0),
                        int(ev.get('id', 0) or 0),
                        'end',
                    )
            if bool(layout.get('stave_visible', True)):
                visible_keys = list(line.get('visible_keys', []))
                if not visible_keys:
                    visible_keys = [k for k in range(int(line['range'][0]), int(line['range'][1]) + 1) if k in line_keys]
                # Special-case low register: draw A#0 (key 2) line when keys 1-3 appear.
                # In ledger mode (manual range that excludes key 2) the full line is
                # suppressed; short ledger stubs are drawn per note instead.
                low_key_present = bool(line.get('low_key_left', False))
                a0_ledger_mode = bool(line.get('a0_ledger_mode', False))
                if low_key_present and not a0_ledger_mode:
                    x_pos = _key_to_x(2)
                    width_mm = stave_three_w
                    du.add_line(
                        x_pos,
                        y1,
                        x_pos,
                        y2,
                        color=notation_color,
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
                        width_mm = stave_clef_w
                        dash = clef_dash
                    elif is_three_line:
                        width_mm = stave_three_w
                        dash = None
                    else:
                        width_mm = stave_two_w
                        dash = None
                    du.add_line(
                        x_pos,
                        y1,
                        x_pos,
                        y2,
                        color=notation_color,
                        width_mm=width_mm,
                        dash_pattern=dash,
                        id=0,
                        tags=['stave']
                    )

            # ---- Beam drawing per line ----
            if bool(layout.get('beam_visible', True)):
                notes_by_hand_line: dict[str, list[dict]] = {'l': [], 'r': []}
                for item in line_notes:
                    hk = str(item.get('hand', 'l') or 'l')
                    hand_norm = 'l' if hk == 'l' else 'r'
                    notes_by_hand_line[hand_norm].append(item)

                stem_len_units = float(layout.get('note_stem_length_semitone', 3) or 3)
                layout_stem_len = stem_len_units * semitone_mm
                beam_w = float(layout.get('beam_thickness_mm', 1.0) or 1.0) * scale
                stem_w = float(layout.get('note_stem_thickness_mm', 0.5) or 0.5) * scale
                line_start = float(line.get('time_start', 0.0) or 0.0)
                line_end = float(line.get('time_end', 0.0) or 0.0)

                beam_half = max(0.1, float(beam_w) * 0.5)
                stem_half = max(0.05, float(stem_w) * 0.5)
                beam_corner_r = max(0.0, float(layout.get('beam_corner_radius_mm', 0.2) or 0.2) * scale)

                def _rounded_polygon(points: list[tuple[float, float]], radius: float, steps: int = 12) -> list[tuple[float, float]]:
                    if len(points) < 3 or radius <= 1e-6:
                        return points

                    def _sub(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
                        return (float(a[0] - b[0]), float(a[1] - b[1]))

                    def _add(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
                        return (float(a[0] + b[0]), float(a[1] + b[1]))

                    def _mul(v: tuple[float, float], s: float) -> tuple[float, float]:
                        return (float(v[0] * s), float(v[1] * s))

                    def _len(v: tuple[float, float]) -> float:
                        return float(math.hypot(v[0], v[1]))

                    def _norm(v: tuple[float, float]) -> tuple[float, float]:
                        lv = _len(v)
                        if lv <= 1e-9:
                            return (0.0, 0.0)
                        return (float(v[0] / lv), float(v[1] / lv))

                    area = 0.0
                    for i in range(len(points)):
                        x1a, y1a = points[i]
                        x2a, y2a = points[(i + 1) % len(points)]
                        area += float(x1a * y2a - x2a * y1a)
                    ccw = area >= 0.0

                    out: list[tuple[float, float]] = []
                    n = len(points)
                    for i in range(n):
                        p_prev = points[(i - 1) % n]
                        p = points[i]
                        p_next = points[(i + 1) % n]

                        v1 = _sub(p_prev, p)
                        v2 = _sub(p_next, p)
                        l1 = _len(v1)
                        l2 = _len(v2)
                        if l1 <= 1e-9 or l2 <= 1e-9:
                            out.append((float(p[0]), float(p[1])))
                            continue

                        u1 = _norm(v1)
                        u2 = _norm(v2)
                        dot = max(-1.0, min(1.0, float(u1[0] * u2[0] + u1[1] * u2[1])))
                        theta = float(math.acos(dot))
                        if theta <= 1e-4 or abs(float(math.pi - theta)) <= 1e-4:
                            out.append((float(p[0]), float(p[1])))
                            continue

                        tan_half = float(math.tan(theta * 0.5))
                        if abs(tan_half) <= 1e-9:
                            out.append((float(p[0]), float(p[1])))
                            continue

                        # Clamp cut distance so tiny beams do not self-overlap.
                        cut = min(float(radius / tan_half), l1 * 0.49, l2 * 0.49)
                        p1 = _add(p, _mul(u1, cut))
                        p2 = _add(p, _mul(u2, cut))

                        bis = _norm(_add(u1, u2))
                        sin_half = float(math.sin(theta * 0.5))
                        if _len(bis) <= 1e-9 or abs(sin_half) <= 1e-9:
                            out.append((float(p1[0]), float(p1[1])))
                            out.append((float(p2[0]), float(p2[1])))
                            continue

                        center = _add(p, _mul(bis, float(radius / sin_half)))
                        a1 = float(math.atan2(float(p1[1] - center[1]), float(p1[0] - center[0])))
                        a2 = float(math.atan2(float(p2[1] - center[1]), float(p2[0] - center[0])))

                        two_pi = float(2.0 * math.pi)
                        if ccw:
                            delta = float((a2 - a1) % two_pi)
                        else:
                            delta = -float((a1 - a2) % two_pi)

                        out.append((float(p1[0]), float(p1[1])))
                        s_count = max(1, int(steps))
                        for s in range(1, s_count):
                            t = float(s) / float(s_count)
                            a = float(a1 + delta * t)
                            out.append((float(center[0] + radius * math.cos(a)), float(center[1] + radius * math.sin(a))))
                        out.append((float(p2[0]), float(p2[1])))

                    return out

                def _draw_beam(x1b: float, y1b: float, x2b: float, y2b: float) -> None:
                    # Baseline (x1b,y1b)->(x2b,y2b) stays exactly the same as the legacy line path.
                    # Build a polygon around that baseline, then optionally round corners.
                    y_start = float(y1b) - stem_half
                    y_end = float(y2b) + stem_half
                    poly = [
                        (float(x1b - beam_half), y_start),
                        (float(x2b - beam_half), y_end),
                        (float(x2b + beam_half), y_end),
                        (float(x1b + beam_half), y_start),
                    ]
                    if beam_corner_r > 1e-6:
                        poly = _rounded_polygon(poly, beam_corner_r, steps=16)
                    du.add_polygon(
                        poly,
                        stroke_color=None,
                        fill_color=notation_color,
                        id=0,
                        tags=['beam'],
                    )

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
                        _draw_beam(float(x1), float(yb1), float(x2), float(yb2))
                        for n in grp:
                            mt = float(n.get('time', t_first) or t_first)
                            if not (op_time.ge(mt, float(t0)) and op_time.lt(mt, float(t1))):
                                continue
                            y_note = _time_to_y(float(mt))
                            if hand_norm == 'r':
                                x_tip = _key_to_x(int(n.get('pitch', 0) or 0)) + float(layout_stem_len)
                            else:
                                x_tip = _key_to_x(int(n.get('pitch', 0) or 0)) - float(layout_stem_len)
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
                                color=notation_color,
                                width_mm=max(0.15, stem_w),
                                id=0,
                                tags=['beam_stem'],
                            )

            line_start = float(line.get('time_start', 0.0) or 0.0)
            line_end = float(line.get('time_end', 0.0) or 0.0)

            def _clip_poly_y(poly: list[tuple[float, float]], y_min: float, y_max: float) -> list[tuple[float, float]]:
                if not poly:
                    return []

                def _clip(points: list[tuple[float, float]], keep_inside, intersect) -> list[tuple[float, float]]:
                    if not points:
                        return []
                    out: list[tuple[float, float]] = []
                    prev = points[-1]
                    prev_in = keep_inside(prev)
                    for cur in points:
                        cur_in = keep_inside(cur)
                        if cur_in:
                            if not prev_in:
                                out.append(intersect(prev, cur))
                            out.append(cur)
                        elif prev_in:
                            out.append(intersect(prev, cur))
                        prev = cur
                        prev_in = cur_in
                    return out

                def _intersect_y(a: tuple[float, float], b: tuple[float, float], yv: float) -> tuple[float, float]:
                    ay = float(a[1])
                    by = float(b[1])
                    if abs(by - ay) <= 1e-9:
                        return (float(b[0]), float(yv))
                    t = (float(yv) - ay) / (by - ay)
                    x = float(a[0]) + t * (float(b[0]) - float(a[0]))
                    return (x, float(yv))

                clipped = _clip(poly, lambda p: float(p[1]) >= float(y_min), lambda a, b: _intersect_y(a, b, y_min))
                clipped = _clip(clipped, lambda p: float(p[1]) <= float(y_max), lambda a, b: _intersect_y(a, b, y_max))
                return clipped

            def _has_adjacent_white_same_hand(note_dict: dict) -> bool:
                p0 = int(note_dict.get('pitch', 0) or 0)
                t0 = float(note_dict.get('time', 0.0) or 0.0)
                h0 = str(note_dict.get('hand', 'l') or 'l')
                idx0 = int(note_dict.get('idx', -1) or -1)
                for m in line_notes:
                    if int(m.get('idx', -2) or -2) == idx0:
                        continue
                    if str(m.get('hand', 'l') or 'l') != h0:
                        continue
                    if not op_time.eq(float(m.get('time', 0.0) or 0.0), t0):
                        continue
                    other_pitch = int(m.get('pitch', 0) or 0)
                    if other_pitch not in BLACK_KEYS and abs(other_pitch - p0) == 1:
                        return True
                return False

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
                    g_raw = item.get('raw', {}) or {}
                    notehead = Notehead.from_note(
                        x_mm=float(x),
                        y_mm=float(y_top),
                        note=g_raw,
                        layout=layout,
                        semitone_space_mm=float(semitone_mm * g_scale),
                        notation_color=notation_color,
                        paper_color=paper_color,
                        default_black_above=False,
                        outline_width_mm_override=float(g_outline),
                    )
                    tag = 'grace_note_black' if bool(getattr(notehead, 'filled', False)) else 'grace_note_white'
                    notehead.draw_notehead(du, item_id=int(item.get('id', 0) or 0), tags=[tag])

            # Problem solved: render notes after grid, using precomputed positions.
            for item in line_notes:
                n_t = float(item.get('time', 0.0) or 0.0)
                n_end = float(item.get('end', 0.0) or 0.0)
                p = int(item.get('pitch', 0) or 0)
                hand_key = str(item.get('hand', 'l') or 'l')
                n = item.get('raw', {}) or {}
                x = _key_to_x(p)
                y_start = _time_to_y(n_t)
                y_end = _time_to_y(n_end)
                if y_end < y_start:
                    y_start, y_end = y_end, y_start
                w = semitone_mm
                default_black_above = p in BLACK_KEYS and _black_note_above_stem(item, black_rule, line_notes, op_time)
                spec = resolve_notehead_spec(n, default_black_above=default_black_above)
                note_y = y_start
                if bool(getattr(spec, 'is_up', False)):
                    note_y = y_start - (w * 2.0)
                # Problem solved: draw the note body with auto-by-hand or explicit hex color.
                raw_color = n.get('color', 'auto')
                color_txt = str(raw_color).strip() if isinstance(raw_color, str) else 'auto'
                fallback = 'note_midinote_left_color' if hand_key == 'l' else 'note_midinote_right_color'
                if color_txt == 'auto':
                    base = _normalize_hex_color(layout.get(fallback, '#cccccc'))
                else:
                    base = _normalize_hex_color(color_txt)
                    if not base:
                        base = _normalize_hex_color(layout.get(fallback, '#cccccc'))
                if not base:
                    base = '#cccccc'
                try:
                    r_i, g_i, b_i, _a = hex_to_rgba(base, 1.0)
                except Exception:
                    r_i, g_i, b_i = (204, 204, 204)
                fill = _midi_fill_from_rgb((int(r_i), int(g_i), int(b_i)))
                if bool(layout.get('note_midinote_visible', True)):
                    midi_poly = [
                        (x, y_start),
                        (x - w, y_start + semitone_mm),
                        (x - w, y_end),
                        (x + w, y_end),
                        (x + w, y_start + semitone_mm),
                    ]
                    du.add_polygon(
                        midi_poly,
                        stroke_color=None,
                        fill_color=fill,
                        id=int(item.get('id', 0) or 0),
                        tags=['midi_note'],
                    )

                continues_from_prev_line = _is_line_continuation(item)
                continues_to_next_line = op_time.lt(n_t, float(line_end)) and op_time.gt(n_end, float(line_end))

                # Problem solved: avoid duplicated heads on continuations.
                if not continues_from_prev_line and bool(layout.get('note_head_visible', True)):
                    is_narrow = _should_tune_under_stem_black_width(item, black_rule, line_notes, op_time)
                    notehead = Notehead.from_note(
                        x_mm=float(x),
                        y_mm=float(y_start),
                        note=n,
                        layout=layout,
                        semitone_space_mm=float(semitone_mm),
                        notation_color=notation_color,
                        paper_color=paper_color,
                        default_black_above=default_black_above,
                        black_note_narrow=is_narrow,
                    )
                    tag = 'notehead_black' if bool(getattr(notehead, 'filled', False)) else 'notehead_white'
                    notehead.draw_notehead(du, item_id=int(item.get('id', 0) or 0), tags=[tag])

                # Problem solved: attach stems only to non-continuation heads.
                if not continues_from_prev_line and bool(layout.get('note_stem_visible', True)):
                    stem_len_units = float(layout.get('note_stem_length_semitone', 3) or 3)
                    stem_len = stem_len_units * semitone_mm
                    stem_w = float(layout.get('note_stem_thickness_mm', 0.5) or 0.5) * scale
                    x2 = x - stem_len if hand_key == 'l' else x + stem_len
                    du.add_line(
                        x,
                        y_start,
                        x2,
                        y_start,
                        color=notation_color,
                        width_mm=stem_w,
                        id=0,
                        tags=['stem'],
                    )

                # Problem solved: accidental guide line points to derived pitch position.
                acc = int(n.get('acc', 0) or 0)
                if acc != 0 and Note.is_valid_accidental(n) and bool(layout.get('accidental_visible', True)):
                    derived_pitch = int(p + acc)
                    x_target = _key_to_x(derived_pitch)
                    note_h = float(semitone_mm * 2.0)
                    is_above_stem = bool(getattr(spec, 'is_up', False))
                    y_anchor = float(y_start - note_h) if is_above_stem else float(y_start + note_h)
                    y_target = float(y_anchor - semitone_mm) if is_above_stem else float(y_anchor + semitone_mm)
                    acc_line_w = max(0.1, float(layout.get('note_stem_thickness_mm', 0.5) or 0.5) * scale * 0.9)
                    du.add_line(
                        float(x),
                        float(y_anchor),
                        float(x_target),
                        float(y_target),
                        color=notation_color,
                        width_mm=acc_line_w,
                        id=0,
                        tags=['accidental_line'],
                    )

                def _draw_manual_ledgers_for_pitch_at_y(pitch_value: int, y_center: float) -> None:
                    if not bool(layout.get('stave_visible', True)):
                        return
                    # A#0 (key 2) sits outside all stave groups; draw its short ledger
                    # stub whenever the note is in the low-register (keys 1-3) and the
                    # natural stave range starts above key 2.
                    if int(pitch_value) in (1, 2, 3) and bool(line.get('a0_ledger_mode', False)):
                        key_sig_a0 = (2, int(round(float(y_center) * 1000)))
                        if key_sig_a0 not in ledger_drawn:
                            ledger_drawn.add(key_sig_a0)
                            y_s1 = float(y_center) - w
                            y_s2 = y_s1 + max(0.0, stave_ledger_len)
                            du.add_line(
                                _key_to_x(2),
                                y_s1 - semitone_mm * 2.0,
                                _key_to_x(2),
                                y_s2 - semitone_mm * 2.0,
                                color=notation_color,
                                width_mm=stave_three_w,
                                dash_pattern=None,
                                id=0,
                                tags=['stave'],
                            )
                        # Fall through: also draw any normal ledger groups below the stave.
                    if not (manual_range and bool(layout.get('stave_visible', True))):
                        return
                    ledger_groups: list[dict] = []
                    if pitch_value < natural_bound_left:
                        g_start = _group_index_for_key(pitch_value)
                        g_end = int(bound_group_low or 0) - 1
                        if g_start <= g_end:
                            ledger_groups = line_groups[g_start:g_end + 1]
                    elif pitch_value > natural_bound_right:
                        g_start = int(bound_group_high or 0) + 1
                        g_end = _group_index_for_key(pitch_value)
                        if g_start <= g_end:
                            ledger_groups = line_groups[g_start:g_end + 1]
                    if not ledger_groups:
                        return
                    y_seg1 = float(y_center) - w
                    y_seg2 = y_seg1 + max(0.0, stave_ledger_len)
                    for grp in ledger_groups:
                        for key in grp.get('keys', []):
                            x_pos = _key_to_x(int(key))
                            is_clef_line = int(key) in (41, 43)
                            is_three_line = int(key) in key_class_filter('FGA')
                            if is_clef_line:
                                width_mm = stave_clef_w
                                dash = clef_dash
                            elif is_three_line:
                                width_mm = stave_three_w
                                dash = None
                            else:
                                width_mm = stave_two_w
                                dash = None
                            key_sig = (int(key), int(round(float(y_center) * 1000)))
                            if key_sig in ledger_drawn:
                                continue
                            ledger_drawn.add(key_sig)
                            du.add_line(
                                x_pos,
                                y_seg1 - semitone_mm * 2.0,
                                x_pos,
                                y_seg2 - semitone_mm * 2.0,
                                color=notation_color,
                                width_mm=width_mm,
                                dash_pattern=dash,
                                id=0,
                                tags=['stave'],
                            )

                # Problem solved: show ledger lines only when manual ranges
                # would otherwise hide them.
                _draw_manual_ledgers_for_pitch_at_y(int(p), float(note_y + w))

                # Problem solved: continuation dots indicate overlapped starts/ends
                # and line crossings for the same hand.
                dot_times: list[float] = []
                for n in line_notes:
                    if int(n.get('idx', -1) or -1) == int(item.get('idx', -2) or -2):
                        continue
                    if str(n.get('hand', 'l') or 'l') != hand_key:
                        continue
                    s = float(n.get('time', 0.0) or 0.0)
                    e = float(n.get('end', 0.0) or 0.0)
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
                # Problem solved: explicitly draw a dot at the stave bottom for any
                # note that continues to the next line, mirroring the continues_from_prev_line
                # behaviour on the previous line and making line crossings always visible.
                if continues_to_next_line:
                    dot_times.append(float(line_end))
                if dot_times and bool(layout.get('note_continuation_dot_visible', True)):
                    dot_d = float(layout.get('note_continuation_dot_size_mm', 0.0) or 0.0)
                    if dot_d > 0.0:
                        dot_d *= scale
                    else:
                        dot_d = w * 0.8
                    dot_pitch = int(item.get('pitch', 0) or 0)
                    # Pre-build a set of double-barline tick positions for fast lookup.
                    _double_bar_ticks: set[float] = {
                        float(ev.get('time', 0.0) or 0.0) for ev in norm_double_bars
                    }
                    dot_x = float(_key_to_x(int(dot_pitch)))
                    min_collision_gap = max(0.0, float(semitone_mm) * 2.0 - 1e-6)
                    for t in sorted(set(dot_times)):
                        y_center = _time_to_y(float(t)) + w
                        # Shift dot down one semitone when it lands on a double barline
                        # so the two vertical lines don't overlap the dot.
                        if any(op_time.eq(float(t), dbt) for dbt in _double_bar_ticks):
                            y_center += float(semitone_mm)

                        # Keep continuation dots legible: if another note starts at
                        # this exact time on adjacent pitch, move the dot forward.
                        has_adjacent_start = False
                        for n in line_notes:
                            if int(n.get('idx', -1) or -1) == int(item.get('idx', -2) or -2):
                                continue
                            if _is_line_continuation(n):
                                continue
                            if not op_time.eq(float(n.get('time', 0.0) or 0.0), float(t)):
                                continue
                            other_pitch = int(n.get('pitch', 0) or 0)
                            if abs(other_pitch - dot_pitch) == 1:
                                # Black adjacent notes rendered above stem do not collide
                                # with the default continuation-dot position.
                                other_black_above = (
                                    other_pitch in BLACK_KEYS
                                    and _black_note_above_stem(n, black_rule, line_notes, op_time)
                                )
                                if other_black_above:
                                    continue
                                other_x = float(_key_to_x(int(other_pitch)))
                                if abs(other_x - dot_x) >= min_collision_gap:
                                    continue
                                has_adjacent_start = True
                                break
                        if has_adjacent_start:
                            y_center += float(semitone_mm) * 2.0

                        _draw_manual_ledgers_for_pitch_at_y(int(dot_pitch), float(y_center))

                        du.add_oval(
                            x - dot_d / 2.0,
                            y_center - dot_d / 2.0,
                            x + dot_d / 2.0,
                            y_center + dot_d / 2.0,
                            fill_color=notation_color,
                            stroke_color=None,
                            id=0,
                            tags=['continuation_dot'],
                        )

                # Problem solved: draw a horizontal connector for same-time chords.
                same_time = [
                    m
                    for m in line_notes
                    if str(m.get('hand', 'l') or 'l') == hand_key
                    and op_time.eq(float(m.get('time', 0.0) or 0.0), n_t)
                    and not _is_line_continuation(m)
                ]
                if len(same_time) >= 2 and bool(layout.get('chord_connect_visible', True)):
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
                            color=notation_color,
                            width_mm=float(layout.get('note_stem_thickness_mm', 0.5) or 0.5) * scale,
                            id=0,
                            tags=['chord_connect'],
                        )

                # Problem solved: stop sign marks a rest gap after a note.
                if (
                    bool(layout.get('note_stop_visible', True))
                    and not continues_to_next_line
                    and _has_followed_rest(item)
                ):
                    w_stop = w * 2.0
                    # Ledger lines start semitone_mm*2 above the outer top edge of
                    # the stop symbol (y_end - w_stop) so they're clearly visible.
                    _draw_manual_ledgers_for_pitch_at_y(int(p), float(y_end - w_stop + w))
                    points = [
                        (x - w_stop / 2.0, y_end - w_stop),
                        (x, y_end),
                        (x + w_stop / 2.0, y_end - w_stop),
                    ]
                    du.add_polyline(
                        points,
                        stroke_color=notation_color,
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

            if bool(layout.get('hairpin_visible', True)) and (line_crescendos or line_decrescendos):
                hairpin_w = float(layout.get('hairpin_line_width_mm', 0.5) or 0.5) * scale
                hairpin_spread = float(layout.get('hairpin_width_mm', 5.0) or 5.0) * scale
                _hg = layout.get('hairpin_text_gap_mm')
                hairpin_gap = float(_hg if _hg is not None else 5.0)
                dynamic_symbol_font_size_pt = float(layout.get('dynamic_symbol_font_size_pt', 12.0) or 12.0)
                dynamic_bg_pad = float(layout.get('dynamic_symbol_background_padding_mm', 0.5) or 0.5) * scale
                dynamic_symbol_angle_deg = 90.0

                def _get_dynamic_symbol_at_position(t: float, x_rpitch: int) -> dict | None:
                    """Get dynamic symbol dimensions at given time and x_rpitch."""
                    for ds in line_dynamic_symbols:
                        ds_time = float(ds.get('time', 0.0) or 0.0)
                        ds_rpitch = int(ds.get('x_rpitch', 0) or 0)
                        
                        # Check if symbol is at same time and x position
                        if abs(ds_time - t) < 0.1 and ds_rpitch == x_rpitch:
                            glyph = str(ds.get('symbol', '') or '')
                            if not glyph:
                                return None
                            
                            # Calculate glyph dimensions
                            try:
                                xb, yb, w, h = du._get_text_extents_mm(glyph, 'LelandText', dynamic_symbol_font_size_pt, False, False)
                            except Exception:
                                # Fallback dimensions
                                w = max(1.0, (dynamic_symbol_font_size_pt / 72.0) * 25.4)
                                h = max(1.0, (dynamic_symbol_font_size_pt / 72.0) * 25.4 * 0.8)
                            
                            return {
                                'glyph': glyph,
                                # Dynamic symbols are rendered rotated by 90°.
                                # Swap extents so spacing/collision follows the
                                # final rendered orientation.
                                'width_mm': h + (2 * dynamic_bg_pad),
                                'height_mm': w + (2 * dynamic_bg_pad),
                            }
                    
                    return None

                def _adjust_hairpin_for_symbols(
                    t_start: float,
                    t_end: float,
                    x_rpitch: int,
                    y0: float,
                    y1: float,
                    is_crescendo: bool,
                    *,
                    adjust_start: bool,
                    adjust_end: bool,
                ) -> tuple[float, float]:
                    """Adjust hairpin y positions to avoid overlapping with dynamic symbols."""
                    start_offset = 0.0
                    end_offset = 0.0

                    if adjust_start:
                        symbol_at_start = _get_dynamic_symbol_at_position(t_start, x_rpitch)
                        if symbol_at_start is not None:
                            start_offset = (symbol_at_start['height_mm'] * 0.5) + hairpin_gap

                    if adjust_end:
                        symbol_at_end = _get_dynamic_symbol_at_position(t_end, x_rpitch)
                        if symbol_at_end is not None:
                            end_offset = (symbol_at_end['height_mm'] * 0.5) + hairpin_gap

                    # Both wedge types need the visible span shortened inward along time.
                    visible_span = max(0.0, y1 - y0)
                    min_visible_span = max(hairpin_w * 2.0, 0.5 * scale)
                    max_inset = max(0.0, visible_span - min_visible_span)
                    requested_inset = start_offset + end_offset
                    if requested_inset > max_inset and requested_inset > 0.0:
                        inset_scale = max_inset / requested_inset
                        start_offset *= inset_scale
                        end_offset *= inset_scale
                    y0 += start_offset
                    y1 -= end_offset

                    return y0, y1

                def _draw_hairpin(hp: dict, is_crescendo: bool) -> None:
                    t_start = float(hp.get('time', 0.0) or 0.0)
                    t_end = float(hp.get('end', t_start) or t_start)
                    if t_end <= t_start:
                        return

                    seg_start = max(t_start, float(line_start))
                    seg_end = min(t_end, float(line_end))
                    if seg_end <= seg_start:
                        return

                    dur = max(1e-6, t_end - t_start)
                    prog0 = max(0.0, min(1.0, (seg_start - t_start) / dur))
                    prog1 = max(0.0, min(1.0, (seg_end - t_start) / dur))

                    x_mm = rpitch_to_x(float(hp.get('x_rpitch', 0.0) or 0.0))
                    y0 = _time_to_y(seg_start)
                    y1 = _time_to_y(seg_end)

                    # Adjust hairpin position to avoid overlapping with dynamic symbols
                    y0, y1 = _adjust_hairpin_for_symbols(
                        t_start,
                        t_end,
                        int(hp.get('x_rpitch', 0) or 0),
                        y0,
                        y1,
                        is_crescendo,
                        adjust_start=abs(seg_start - t_start) <= 1e-6,
                        adjust_end=abs(seg_end - t_end) <= 1e-6,
                    )

                    half_spread = hairpin_spread * 0.5

                    if is_crescendo:
                        half0 = half_spread * prog0
                        half1 = half_spread * prog1
                        tags = ['crescendo']
                    else:
                        half0 = half_spread * (1.0 - prog0)
                        half1 = half_spread * (1.0 - prog1)
                        tags = ['decrescendo']

                    hp_id = int(hp.get('id', 0) or 0)
                    du.add_line(
                        x_mm - half0,
                        y0,
                        x_mm - half1,
                        y1,
                        color=notation_color,
                        width_mm=hairpin_w,
                        line_cap='round',
                        id=hp_id,
                        tags=tags,
                    )
                    du.add_line(
                        x_mm + half0,
                        y0,
                        x_mm + half1,
                        y1,
                        color=notation_color,
                        width_mm=hairpin_w,
                        line_cap='round',
                        id=hp_id,
                        tags=tags,
                    )

                for hp in line_crescendos:
                    _draw_hairpin(hp, True)

                for hp in line_decrescendos:
                    _draw_hairpin(hp, False)

            if bool(layout.get('dynamic_symbol_visible', True)) and line_dynamic_symbols:
                text_size_pt = float(layout.get('dynamic_symbol_font_size_pt', 12.0) or 12.0)
                dynamic_bg_pad = float(
                    layout.get(
                        'dynamic_symbol_background_padding_mm',
                        layout.get('dynamic_symbol_background_padding', layout.get('dynamic_background_padding', layout.get('text_background_padding_mm', 0.5))),
                    ) or 0.0
                ) * scale
                text_family = 'LelandText'
                text_color = notation_color
                dynamic_symbol_angle_deg = 90.0

                for ds in line_dynamic_symbols:
                    symbol = str(ds.get('symbol', '') or '')
                    if not symbol:
                        continue
                    t_time = float(ds.get('time', 0.0) or 0.0)
                    x_mm = rpitch_to_x(float(ds.get('x_rpitch', 0.0) or 0.0))
                    y_mm = _time_to_y(t_time)

                    try:
                        xb, yb, w, h = du._get_text_extents_mm(symbol, text_family, text_size_pt, False, False)
                    except Exception:
                        xb, yb, w, h = 0.0, 0.0, max(1.0, (text_size_pt / 72.0) * 25.4), max(1.0, (text_size_pt / 72.0) * 25.4 * 0.8)

                    bx = float(x_mm) - (float(xb) + (float(w) * 0.5))
                    by = float(y_mm) - (float(yb) + (float(h) * 0.5))
                    rx = bx + float(xb)
                    ry = by + float(yb)
                    # Text rotates around its bbox center in DrawUtil. Use a
                    # swapped axis-aligned bbox for the background.
                    cx = rx + (float(w) * 0.5)
                    cy = ry + (float(h) * 0.5)
                    rot_w = float(h)
                    rot_h = float(w)

                    du.add_rectangle(
                        cx - (rot_w * 0.5) - dynamic_bg_pad,
                        cy - (rot_h * 0.5) - dynamic_bg_pad,
                        cx + (rot_w * 0.5) + dynamic_bg_pad,
                        cy + (rot_h * 0.5) + dynamic_bg_pad,
                        stroke_color=None,
                        fill_color=paper_color,
                        id=int(ds.get('id', 0) or 0),
                        tags=['dynamic_symbol_bg_top'],
                    )
                    du.add_text(
                        bx,
                        by,
                        symbol,
                        family=text_family,
                        size_pt=text_size_pt,
                        italic=False,
                        bold=False,
                        color=text_color,
                        anchor=None,
                        angle_deg=dynamic_symbol_angle_deg,
                        id=int(ds.get('id', 0) or 0),
                        tags=['dynamic_symbol_text_top'],
                    )

            '''Text drawing.'''
            if bool(layout.get('text_visible', True)) and line_texts:
                default_font = layout.get('font_text', {}) or {}
                pad_mm = float(layout.get('text_background_padding_mm', 0.0) or 0.0) * scale

                def _resolve_font(tx: dict) -> tuple[str, float, bool, bool, bool]:
                    use_custom = bool(tx.get('use_custom_font', False))
                    fnt = tx.get('font', None) if use_custom else None
                    if not isinstance(fnt, dict):
                        fnt = default_font if isinstance(default_font, dict) else {}
                    family = str(fnt.get('family', default_font.get('family', 'Edwin')))
                    size_pt = float(fnt.get('size_pt', default_font.get('size_pt', 12.0)) or 12.0)
                    italic = bool(fnt.get('italic', default_font.get('italic', False)))
                    bold = bool(fnt.get('bold', default_font.get('bold', False)))
                    underline = bool(fnt.get('underline', default_font.get('underline', False)))
                    return family, size_pt, italic, bold, underline

                def _prepare_text_layout(
                    txt_raw: str,
                    family: str,
                    size_pt: float,
                    italic: bool,
                    bold: bool,
                ) -> dict:
                    raw = str(txt_raw or '').replace('\r\n', '\n').replace('\r', '\n')
                    raw = raw.replace('\\n', '\n').replace('\\t', '\t')
                    fallback = '(no text set)'
                    paragraph_strs = raw.split('\n') if raw.strip() else [fallback]
                    line_entries: list[dict] = []
                    max_w = 0.0
                    line_h_mm = 0.0
                    for para in paragraph_strs:
                        measure = para if para.strip() else ' '
                        _, _, w, h = du._get_text_extents_mm(measure, family, size_pt, italic, bold)
                        w_mm = float(max(0.0, w)) if para.strip() else 0.0
                        h_mm = float(max(0.1, h))
                        line_entries.append({'text': para, 'width_mm': w_mm, 'height_mm': h_mm})
                        if w_mm > max_w:
                            max_w = w_mm
                        if h_mm > line_h_mm:
                            line_h_mm = h_mm
                    if not line_entries:
                        _, _, w, h = du._get_text_extents_mm(fallback, family, size_pt, italic, bold)
                        line_entries = [{'text': fallback, 'width_mm': float(max(0.0, w)), 'height_mm': float(max(0.1, h))}]
                        max_w = float(max(0.0, w))
                        line_h_mm = float(max(0.1, h))
                    line_y_gap_mm = line_h_mm * 0.1
                    line_block_h_mm = line_h_mm + line_y_gap_mm * 2.0
                    return {
                        'lines': line_entries,
                        'line_height_mm': line_h_mm,
                        'line_y_gap_mm': line_y_gap_mm,
                        'line_block_height_mm': line_block_h_mm,
                        'content_width_mm': max_w,
                        'content_height_mm': line_block_h_mm * len(line_entries),
                        'draw_width_mm': max_w,
                    }

                def _line_alignment_x(alignment: str, content_w_mm: float, line_w_mm: float) -> float:
                    mode = str(alignment or 'left').lower()
                    if mode == 'center':
                        return 0.0
                    if mode == 'right':
                        return (content_w_mm * 0.5) - (line_w_mm * 0.5)
                    return (-content_w_mm * 0.5) + (line_w_mm * 0.5)

                for tx in line_texts:
                    t_time = float(tx.get('time', 0.0) or 0.0)
                    x_rp = float(tx.get('x_rpitch', 0) or 0)
                    angle = float(tx.get('rotation', 0.0) or 0.0)
                    x_off = float(tx.get('x_offset_mm', 0.0) or 0.0)
                    y_off = float(tx.get('y_offset_mm', 0.0) or 0.0)
                    txt_raw = str(tx.get('text', '') or '')
                    alignment = str(tx.get('alignment', 'left') or 'left').lower()
                    family, size_pt_raw, italic, bold, underline = _resolve_font(tx)
                    size_pt = float(size_pt_raw) * ENGRAVER_FRACTIONAL_TEXT_SCALING_CORRECTION * (scale / 0.3333333333333333)
                    width_off_mm = float(tx.get('text_background_width_offset_mm', 0.0) or 0.0)
                    y_mm = _time_to_y(t_time) + y_off
                    x_mm = rpitch_to_x(x_rp) + x_off
                    layout_info = _prepare_text_layout(
                        txt_raw,
                        family,
                        size_pt,
                        italic,
                        bold,
                    )
                    lines = list(layout_info.get('lines', []))
                    content_w_mm = float(layout_info.get('content_width_mm', 0.0) or 0.0)
                    line_h_mm = float(layout_info.get('line_height_mm', 0.0) or 0.0)
                    line_y_gap_mm = float(layout_info.get('line_y_gap_mm', 0.0) or 0.0)
                    line_block_h_mm = float(layout_info.get('line_block_height_mm', line_h_mm) or line_h_mm)
                    content_h_mm = float(layout_info.get('content_height_mm', 0.0) or 0.0)
                    try:
                        w_mm, h_mm, offset_down, rot_corners, rot_poly = _text_bbox(
                            content_w_mm,
                            content_h_mm,
                            angle,
                            pad_mm,
                            pad_mm,
                            width_off_mm,
                        )
                    except Exception:
                        continue
                    cy = y_mm + offset_down
                    poly = [(x_mm + dx, cy + dy) for (dx, dy) in rot_poly]
                    du.add_polygon(
                        poly,
                        stroke_color=None,
                        fill_color=paper_color,
                        id=int(tx.get('id', 0) or 0),
                        tags=['text_bg'],
                    )
                    ang_rad = math.radians(angle)
                    cos_a = math.cos(ang_rad)
                    sin_a = math.sin(ang_rad)
                    total_h = line_block_h_mm * max(1, len(lines))

                    def _to_world(local_x: float, local_y: float) -> tuple[float, float]:
                        wx = x_mm + (local_x * cos_a) - (local_y * sin_a)
                        wy = cy + (local_x * sin_a) + (local_y * cos_a)
                        return wx, wy

                    for idx_line, text_line in enumerate(lines):
                        line_text = str(text_line.get('text', ''))
                        line_w_mm = float(text_line.get('width_mm', 0.0) or 0.0)
                        line_y_local = (-total_h * 0.5) + (line_block_h_mm * idx_line) + line_y_gap_mm + (line_h_mm * 0.5)
                        line_x_local = _line_alignment_x(alignment, content_w_mm, line_w_mm)

                        if line_text:
                            if alignment == 'right':
                                draw_x, draw_y = _to_world(content_w_mm * 0.5, line_y_local)
                                text_anchor = 'e'
                            else:
                                draw_x, draw_y = _to_world(line_x_local, line_y_local)
                                text_anchor = 'center'
                            du.add_text(
                                draw_x,
                                draw_y,
                                line_text,
                                family=family,
                                size_pt=size_pt,
                                italic=italic,
                                bold=bold,
                                color=notation_color,
                                anchor=text_anchor,
                                angle_deg=angle,
                                id=int(tx.get('id', 0) or 0),
                                tags=['text'],
                            )

                        if underline and line_text:
                            xb_mm, yb_mm, ink_w_mm, ink_h_mm = du._get_text_extents_mm(line_text, family, size_pt, italic, bold)
                            ul_y_local = -ink_h_mm / 2.0 - yb_mm + max(0.2, size_pt * 0.025)
                            if alignment == 'right':
                                ul_x1, ul_y1 = _to_world(content_w_mm * 0.5 - ink_w_mm, line_y_local + ul_y_local)
                                ul_x2, ul_y2 = _to_world(content_w_mm * 0.5, line_y_local + ul_y_local)
                            else:
                                half_w = ink_w_mm * 0.5
                                ul_x1, ul_y1 = _to_world(line_x_local - half_w, line_y_local + ul_y_local)
                                ul_x2, ul_y2 = _to_world(line_x_local + half_w, line_y_local + ul_y_local)
                            du.add_line(
                                ul_x1, ul_y1, ul_x2, ul_y2,
                                color=notation_color,
                                width_mm=max(0.2, size_pt * (0.04 if bold else 0.02)),
                                tags=['text_underline'],
                                id=int(tx.get('id', 0) or 0),
                            )

            if bool(layout.get('slur_visible', True)) and (
                    line_slurs or line_slur_continuations
                    or line_slur_end_indicators or line_slur_start_indicators):
                side_w = float(layout.get('slur_width_sides_mm', 0.1) or 0.1) * scale
                mid_w = float(layout.get('slur_width_middle_mm', 1.5) or 1.5) * scale
                n_seg = max(2, int(SLUR_SEGMENT_COUNT))

                def tri_interp(t: float) -> float:
                    return max(0.0, 1.0 - abs(2.0 * t - 1.0))

                def width_at(t: float) -> float:
                    return side_w + (mid_w - side_w) * tri_interp(t)

                def lerp(a: float, b: float, t: float) -> float:
                    return a + (b - a) * t

                def bezier_point(t: float, p0: tuple[float, float], p1: tuple[float, float], p2: tuple[float, float], p3: tuple[float, float]) -> tuple[float, float]:
                    q0x = lerp(p0[0], p1[0], t)
                    q0y = lerp(p0[1], p1[1], t)
                    q1x = lerp(p1[0], p2[0], t)
                    q1y = lerp(p1[1], p2[1], t)
                    q2x = lerp(p2[0], p3[0], t)
                    q2y = lerp(p2[1], p3[1], t)

                    r0x = lerp(q0x, q1x, t)
                    r0y = lerp(q0y, q1y, t)
                    r1x = lerp(q1x, q2x, t)
                    r1y = lerp(q1y, q2y, t)

                    return lerp(r0x, r1x, t), lerp(r0y, r1y, t)

                def _time_to_y_ext(ticks: float) -> float:
                    # Linear (unclamped) time-to-y for indicator rendering outside line bounds.
                    total = max(1e-6, float(line['time_end'] - line['time_start']))
                    rel = (float(ticks) - float(line['time_start'])) / total
                    return y1 + (y2 - y1) * rel

                def _bezier_y_at_t(t_param: float, ctrl_y: tuple) -> float:
                    # Evaluate the cubic Bezier time-coordinate at parameter t_param.
                    cy0, cy1, cy2, cy3 = ctrl_y
                    q0 = cy0 + (cy1 - cy0) * t_param
                    q1 = cy1 + (cy2 - cy1) * t_param
                    q2 = cy2 + (cy3 - cy2) * t_param
                    r0 = q0 + (q1 - q0) * t_param
                    r1 = q1 + (q2 - q1) * t_param
                    return r0 + (r1 - r0) * t_param

                def _find_t_for_time(ctrl_y: tuple, target_time: float) -> float:
                    # Binary search: find t where cubic Bezier time-coord equals target.
                    lo, hi = 0.0, 1.0
                    for _ in range(64):
                        mid = (lo + hi) * 0.5
                        if _bezier_y_at_t(mid, ctrl_y) < target_time:
                            lo = mid
                        else:
                            hi = mid
                    return (lo + hi) * 0.5

                def _casteljau_1d(a: float, b: float, c: float, d: float, t_sp: float):
                    # Split cubic Bezier [a,b,c,d] at t_sp via de Casteljau algorithm.
                    ab = a + (b - a) * t_sp
                    bc = b + (c - b) * t_sp
                    cd = c + (d - c) * t_sp
                    abc = ab + (bc - ab) * t_sp
                    bcd = bc + (cd - bc) * t_sp
                    abcd = abc + (bcd - abc) * t_sp
                    return (a, ab, abc, abcd), (abcd, bcd, cd, d)

                def _split_slur(pxc: tuple, pyc: tuple, t_sp: float):
                    # Split a slur's control points at t_sp; returns first and second halves.
                    pxa, pxb = _casteljau_1d(float(pxc[0]), float(pxc[1]), float(pxc[2]), float(pxc[3]), t_sp)
                    pya, pyb = _casteljau_1d(float(pyc[0]), float(pyc[1]), float(pyc[2]), float(pyc[3]), t_sp)
                    return pxa, pya, pxb, pyb

                def _pts_to_page(pxc: tuple, pyc: tuple, y_fn) -> tuple:
                    # Convert 4 data-space control points to page-space.
                    return (
                        (rpitch_to_x(float(pxc[0])), y_fn(float(pyc[0]))),
                        (rpitch_to_x(float(pxc[1])), y_fn(float(pyc[1]))),
                        (rpitch_to_x(float(pxc[2])), y_fn(float(pyc[2]))),
                        (rpitch_to_x(float(pxc[3])), y_fn(float(pyc[3]))),
                    )

                def _draw_slur_seg(pg0, pg1, pg2, pg3, t_gs: float, t_ge: float, sl_id: int, tags=None) -> None:
                    # Draw a cubic Bezier slur segment with width profile scaled to original t range.
                    pts_sg: list[tuple[float, float]] = []
                    for i in range(n_seg):
                        t_local = i / float(n_seg - 1) if n_seg > 1 else 0.0
                        bx, by = bezier_point(t_local, pg0, pg1, pg2, pg3)
                        pts_sg.append((bx, by))
                    if len(pts_sg) < 2:
                        return
                    left_edge: list[tuple[float, float]] = []
                    right_edge: list[tuple[float, float]] = []
                    last_nx, last_ny = 0.0, 1.0
                    for i, (cx, cy) in enumerate(pts_sg):
                        t_local = i / float(n_seg - 1) if n_seg > 1 else 0.0
                        t_global = t_gs + t_local * (t_ge - t_gs)
                        w_slur = max(0.0, float(width_at(t_global)))
                        half_w = 0.5 * w_slur
                        if i == 0:
                            fwd_x, fwd_y = pts_sg[i + 1]
                            bwd_x, bwd_y = pts_sg[i]
                        elif i == len(pts_sg) - 1:
                            bwd_x, bwd_y = pts_sg[i - 1]
                            fwd_x, fwd_y = pts_sg[i]
                        else:
                            bwd_x, bwd_y = pts_sg[i - 1]
                            fwd_x, fwd_y = pts_sg[i + 1]
                        dx = float(fwd_x) - float(bwd_x)
                        dy = float(fwd_y) - float(bwd_y)
                        dlen = math.hypot(dx, dy)
                        if dlen <= 1e-9:
                            nx, ny = last_nx, last_ny
                        else:
                            nx = -dy / dlen
                            ny = dx / dlen
                            last_nx, last_ny = nx, ny
                        left_edge.append((float(cx) + nx * half_w, float(cy) + ny * half_w))
                        right_edge.append((float(cx) - nx * half_w, float(cy) - ny * half_w))
                    slur_poly = left_edge + list(reversed(right_edge))
                    if len(slur_poly) >= 3:
                        du.add_polygon(
                            slur_poly,
                            stroke_color=None,
                            fill_color=notation_color,
                            id=int(sl_id),
                            tags=tags if tags is not None else ['slur'],
                        )

                _line_t_start = float(line.get('time_start', 0.0) or 0.0)
                _line_t_end = float(line.get('time_end', 0.0) or 0.0)

                # Draw slurs whose p1 starts on this line.
                # If the slur extends past line_end, split exactly at the break.
                for sl in line_slurs:
                    _px = (float(sl.get('x1_rpitch', 0) or 0), float(sl.get('x2_rpitch', 0) or 0),
                           float(sl.get('x3_rpitch', 0) or 0), float(sl.get('x4_rpitch', 0) or 0))
                    _py = (float(sl.get('y1_time', 0.0) or 0.0), float(sl.get('y2_time', 0.0) or 0.0),
                           float(sl.get('y3_time', 0.0) or 0.0), float(sl.get('y4_time', 0.0) or 0.0))
                    _sl_id = int(sl.get('id', 0) or 0)
                    if op_time.gt(_py[3], _line_t_end):
                        # Slur crosses line break: draw only first half up to break.
                        t_cut = _find_t_for_time(_py, _line_t_end)
                        t_cut = max(0.001, min(0.999, t_cut))
                        pxa, pya, _, _ = _split_slur(_px, _py, t_cut)
                        _draw_slur_seg(*_pts_to_page(pxa, pya, _time_to_y), 0.0, t_cut, _sl_id)
                    else:
                        _draw_slur_seg(*_pts_to_page(_px, _py, _time_to_y), 0.0, 1.0, _sl_id)

                # Draw continuation slurs: started on a previous line, extend into this one.
                # Split at line_start; if still extending past line_end, split there too.
                for sl in line_slur_continuations:
                    _px = (float(sl.get('x1_rpitch', 0) or 0), float(sl.get('x2_rpitch', 0) or 0),
                           float(sl.get('x3_rpitch', 0) or 0), float(sl.get('x4_rpitch', 0) or 0))
                    _py = (float(sl.get('y1_time', 0.0) or 0.0), float(sl.get('y2_time', 0.0) or 0.0),
                           float(sl.get('y3_time', 0.0) or 0.0), float(sl.get('y4_time', 0.0) or 0.0))
                    _sl_id = int(sl.get('id', 0) or 0)
                    t_s = _find_t_for_time(_py, _line_t_start)
                    t_s = max(0.0, min(0.999, t_s))
                    _, _, _pxb, _pyb = _split_slur(_px, _py, t_s)
                    t_gs = t_s
                    t_ge = 1.0
                    _pxc, _pyc = _pxb, _pyb
                    if op_time.gt(_pyb[3], _line_t_end):
                        t_e_loc = _find_t_for_time(_pyb, _line_t_end)
                        t_e_loc = max(0.001, min(0.999, t_e_loc))
                        t_ge = t_s + t_e_loc * (1.0 - t_s)
                        _pxc, _pyc, _, _ = _split_slur(_pxb, _pyb, t_e_loc)
                    _draw_slur_seg(*_pts_to_page(_pxc, _pyc, _time_to_y), t_gs, t_ge, _sl_id)

                # End-of-line connected-slur indicators.
                # Show the first semitone_mm*2 of the connected next slur below y2
                # (after the line break barline) so the connection is visible.
                if line_slur_end_indicators:
                    _dur = max(1e-6, _line_t_end - _line_t_start)
                    _mm_pt = (y2 - y1) / _dur
                    _ind_ticks = (semitone_mm * 4.0) / max(1e-9, _mm_pt)
                    for sl in line_slur_end_indicators:
                        _px = (float(sl.get('x1_rpitch', 0) or 0), float(sl.get('x2_rpitch', 0) or 0),
                               float(sl.get('x3_rpitch', 0) or 0), float(sl.get('x4_rpitch', 0) or 0))
                        _py = (float(sl.get('y1_time', 0.0) or 0.0), float(sl.get('y2_time', 0.0) or 0.0),
                               float(sl.get('y3_time', 0.0) or 0.0), float(sl.get('y4_time', 0.0) or 0.0))
                        _sl_id = int(sl.get('id', 0) or 0)
                        t_cut = _find_t_for_time(_py, _line_t_end + _ind_ticks)
                        t_cut = max(0.001, min(1.0, t_cut))
                        pxa, pya, _, _ = _split_slur(_px, _py, t_cut)
                        _draw_slur_seg(*_pts_to_page(pxa, pya, _time_to_y_ext),
                                       0.0, t_cut, _sl_id, tags=['slur', 'slur_indicator'])

                # Start-of-line connected-slur indicators.
                # Show the last semitone_mm*2 of the connected previous slur above y1
                # (before the line start) so the connection is visible.
                if line_slur_start_indicators:
                    _dur = max(1e-6, _line_t_end - _line_t_start)
                    _mm_pt = (y2 - y1) / _dur
                    _ind_ticks = (semitone_mm * 4.0) / max(1e-9, _mm_pt)
                    for sl in line_slur_start_indicators:
                        _px = (float(sl.get('x1_rpitch', 0) or 0), float(sl.get('x2_rpitch', 0) or 0),
                               float(sl.get('x3_rpitch', 0) or 0), float(sl.get('x4_rpitch', 0) or 0))
                        _py = (float(sl.get('y1_time', 0.0) or 0.0), float(sl.get('y2_time', 0.0) or 0.0),
                               float(sl.get('y3_time', 0.0) or 0.0), float(sl.get('y4_time', 0.0) or 0.0))
                        _sl_id = int(sl.get('id', 0) or 0)
                        t_cut = _find_t_for_time(_py, _line_t_start - _ind_ticks)
                        t_cut = max(0.0, min(0.999, t_cut))
                        _, _, _pxb, _pyb = _split_slur(_px, _py, t_cut)
                        _draw_slur_seg(*_pts_to_page(_pxb, _pyb, _time_to_y_ext),
                                       t_cut, 1.0, _sl_id, tags=['slur', 'slur_indicator'])
            x_cursor = x_cursor + float(line['total_width']) + gap


    # Build a compact time-to-page map so the print-view playhead can locate any
    # source-time position without re-running the full engraver.
    # Structure: list[list[dict]] – one outer entry per page, each inner list
    # contains one dict per stave line with time/y/x geometry.
    _ptm: list = []
    for _p_idx, _p_lines in enumerate(pages):
        _page_lines_ord = list(reversed(_p_lines)) if horizontal_read_direction else _p_lines
        # y range for this page
        if horizontal_read_direction:
            _y_top = float(page_top)
            _y_bot = float(page_h - page_bottom)
        else:
            _h_off = float(header_height) if _p_idx == 0 else 0.0
            _y_top = float(page_top + _h_off)
            _y_bot = float(page_h - page_bottom - footer_height)
        # x justification for this page
        _lr, _rr = _line_axis_reserves_for_page(_p_idx)
        _avail = max(1e-6, page_w - page_left - page_right - _lr - _rr)
        _used = sum(float(_l['total_width']) for _l in _page_lines_ord)
        _leftover = max(0.0, _avail - _used)
        _gap = _leftover / float(len(_page_lines_ord) + 1) if _page_lines_ord else 0.0
        _xc = page_left + _lr + _gap
        _lines_map: list = []
        for _line in _page_lines_ord:
            _lo = float(_line.get('ledger_left_overhang', 0.0) or 0.0)
            _lx_s = _xc + float(_line['margin_left']) + _lo
            _lx_e = _lx_s + float(_line.get('stave_width', 0.0) or 0.0)
            _lines_map.append({
                'time_start': float(_line.get('time_start', 0.0)),
                'time_end': float(_line.get('time_end', 0.0)),
                'y_top': float(_y_top),
                'y_bottom': float(_y_bot),
                'x_start': float(_lx_s),
                'x_end': float(_lx_e),
            })
            _xc += float(_line['total_width']) + _gap
        _ptm.append(_lines_map)
    du.print_time_map = _ptm

    # Ensure a valid current page index
    if du.page_count() > 0:
        if pdf_export:
            du.set_current_page(0)
        else:
            du.set_current_page(target_page_index)


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
        self._du = draw_util
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
            try:
                self._du.analysis = self.analysis
            except Exception:
                pass
            try:
                self._du.print_time_map = getattr(result_du, 'print_time_map', [])
            except Exception:
                pass
            self.engraved.emit()

    @QtCore.Slot(int, str, str)
    def _on_failed(self, request_id: int, error_text: str, error_details: str) -> None:
        self._running = False
        if self._pending_score is not None:
            self._maybe_start_pending()
        if int(request_id) == int(self._latest_request_id):
            self.failed.emit(str(error_text or 'Engraving failed'), str(error_details or ''))
