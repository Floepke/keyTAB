'''
Grid and barline drawing mixin for the Editor class.

Handles drawing barlines, measure numbers, and gridlines.
'''

from __future__ import annotations
from typing import TYPE_CHECKING, cast
from file_model.SCORE import SCORE
from file_model.base_grid import resolve_grid_layer_offsets
from ui.widgets.draw_util import DrawUtil
from utils.CONSTANT import QUARTER_NOTE_UNIT
from utils.operator import Operator

if TYPE_CHECKING:
    from editor.editor import Editor


class GridDrawerMixin:
    '''
        Draws:
            - barlines
            - measure numbers
            - gridlines
            - time signature indicators
            - project title and composer at top-left
    '''
    
    def draw_grid(self, du: DrawUtil) -> None:
        self = cast("Editor", self)
        score: SCORE = self.current_score()
        layout = getattr(score, 'layout', None)
        barline_visible = bool(getattr(layout, 'barline_visible', True)) if layout is not None else True
        grid_line_visible = bool(getattr(layout, 'grid_line_visible', True)) if layout is not None else True

        # draw title and composer at top-left
        title_text = score.info.title
        composer_text = score.info.composer
        title_font = score.layout.font_title
        if title_font is not None and callable(getattr(title_font, 'resolve_family', None)):
            family = str(title_font.resolve_family())
        else:
            family = getattr(title_font, 'family', 'Courier New') if title_font is not None else 'Courier New'
        size_pt = 12
        x_off = 0.0
        y_off = 0.0
        du.add_text(
            1 + x_off,
            1 + y_off,
            f"'{title_text}' by composer: {composer_text}",
            size_pt=size_pt,
            color=self.notation_color,
            id=0,
            tags=["title"],
            anchor='nw',
            family=family,
        )

        # Page metrics (mm)
        width_mm, _height_mm = du.current_page_size_mm()
        margin = float(self.margin)
        stave_left_position = margin + self.semitone_dist
        stave_right_position = max(0.0, width_mm - margin) - self.semitone_dist * 2

        # --------------- drawing the grid lines, barlines, measure numbers ---------------
        measure_numbering_cursor = 1
        meas_font = getattr(score.layout, 'measure_numbering_font', None)
        if meas_font is not None and callable(getattr(meas_font, 'resolve_family', None)):
            meas_family = str(meas_font.resolve_family())
        else:
            meas_family = getattr(meas_font, 'family', 'Courier New') if meas_font is not None else 'Courier New'
        meas_size = 20.0
        color = self.notation_color
        style_scale = float(getattr(layout, 'scale', 1.0) or 1.0) if layout is not None else 1.0
        bar_width_mm = max(0.01, float(getattr(self, 'editor_line_width_global', 0.1) or 0.1))
        grid_width_mm = (float(getattr(layout, 'grid_gridline_thickness_mm', 0.15) or 0.15) * style_scale) if layout is not None else 0.15

        cache = getattr(self, '_draw_cache', None) or {}
        grid_den_times = list(cache.get('grid_den_times') or [])
        barline_times = list(cache.get('barline_times') or [])

        # Safety fallback when draw cache is unavailable.
        if not barline_times:
            cur_t = 0.0
            for bg in list(getattr(score, 'base_grid', []) or []):
                numer = int(getattr(bg, 'numerator', 4) or 4)
                denom = int(getattr(bg, 'denominator', 4) or 4)
                mcount = int(getattr(bg, 'measure_amount', 1) or 1)
                measure_len_ticks = float(numer) * (4.0 / float(max(1, denom))) * float(QUARTER_NOTE_UNIT)
                positions = list(getattr(bg, 'beat_grouping', []) or [])
                bar_offsets, grid_offsets = resolve_grid_layer_offsets(positions, numer, denom)
                for _ in range(mcount):
                    for off in bar_offsets:
                        barline_times.append(float(cur_t + float(off)))
                    for off in grid_offsets:
                        grid_den_times.append(float(cur_t + float(off)))
                    cur_t += measure_len_ticks
            if barline_times:
                barline_times.append(float(cur_t))
            if grid_den_times:
                grid_den_times.append(float(cur_t))

        barline_keys = {round(float(t), 6) for t in barline_times}

        # Build collision geometry so barlines are drawn constructively around symbols.
        op = cache.get('op') if isinstance(cache, dict) else None
        if op is None:
            op = Operator(7)
        notes_view = list(cache.get('notes_view') or []) if isinstance(cache, dict) else []
        notes_by_hand_cache = dict(cache.get('notes_by_hand') or {}) if isinstance(cache, dict) else {}
        beam_markers = dict(cache.get('beam_by_hand') or {}) if isinstance(cache, dict) else {}

        note_stem_visible = bool(getattr(layout, 'note_stem_visible', True)) if layout is not None else True
        semitone_mm = float(self.semitone_dist or 0.5)
        stem_len_mm = float(getattr(layout, 'note_stem_length_semitone', 3) or 3) * semitone_mm if layout is not None else (3.0 * semitone_mm)
        note_head_half_w = semitone_mm * float(getattr(layout, 'note_width_scaling', 0.75) or 0.75) if layout is not None else (semitone_mm * 0.75)
        stem_collision_pad = max(0.15, float(getattr(layout, 'note_stem_thickness_mm', 0.5) or 0.5) * style_scale) if layout is not None else 0.5
        head_collision_pad = max(0.15, semitone_mm * 0.15)
        beam_collision_pad = max(0.2, float(getattr(layout, 'beam_thickness_mm', 1.0) or 1.0) * style_scale * 0.7) if layout is not None else 0.2
        barline_symbol_gap_mm = max(0.0, semitone_mm)
        barline_time_eps = 1e-4

        def _barline_time_eq(a: float, b: float) -> bool:
            return abs(float(a) - float(b)) <= float(barline_time_eps)

        def _barline_time_in_range(t: float, t0: float, t1: float) -> bool:
            lo = float(min(t0, t1)) - float(barline_time_eps)
            hi = float(max(t0, t1)) + float(barline_time_eps)
            return lo <= float(t) <= hi

        def _norm_hand_key(v: str) -> str:
            return 'l' if v == 'l' else 'r'

        notes_by_norm: dict[str, list] = {'l': [], 'r': []}
        if notes_by_hand_cache:
            for hand, notes in notes_by_hand_cache.items():
                notes_by_norm[_norm_hand_key(str(hand or 'l'))].extend(list(notes or []))
        else:
            for n in notes_view:
                notes_by_norm[_norm_hand_key(str(getattr(n, 'hand', 'l') or 'l'))].append(n)

        markers_by_norm: dict[str, list] = {'l': [], 'r': []}
        for hand, markers in beam_markers.items():
            markers_by_norm.setdefault(_norm_hand_key(str(hand)), []).extend(list(markers or []))

        def _build_grid_windows(times: list[float]) -> list[tuple[float, float]]:
            if not times:
                return []
            st = sorted(float(t) for t in times)
            windows: list[tuple[float, float]] = []
            for i in range(len(st) - 1):
                a = float(st[i])
                b = float(st[i + 1])
                if b > a:
                    windows.append((a, b))
            return windows

        def _process_beam_marker_override(default_windows: list[tuple[float, float]], markers: list) -> list[tuple[float, float]]:
            windows = sorted(default_windows, key=lambda w: float(w[0]))
            for mk in sorted(markers, key=lambda m: float(getattr(m, 'time', 0.0) or 0.0)):
                mt = float(getattr(mk, 'time', 0.0) or 0.0)
                dur = float(getattr(mk, 'duration', 0.0) or 0.0)
                end = mt + max(0.0, dur)
                filtered: list[tuple[float, float]] = []
                for w0, w1 in windows:
                    if op.ge(float(w0), float(end)) or op.le(float(w1), float(mt)):
                        filtered.append((float(w0), float(w1)))
                if dur > 0.0:
                    filtered.append((float(mt), float(end)))
                windows = sorted(filtered, key=lambda w: float(w[0]))
            return windows

        def _assign_groups(notes: list, windows: list[tuple[float, float]]) -> list[list]:
            groups: list[list] = []
            for w0, w1 in windows:
                grp = []
                for n in notes:
                    nt = float(getattr(n, 'time', 0.0) or 0.0)
                    starts_in = op.ge(float(nt), float(w0)) and op.lt(float(nt), float(w1))
                    if starts_in:
                        grp.append(n)
                groups.append(grp)
            return groups

        beam_segments: list[dict[str, float]] = []
        beam_connect_segments: list[dict[str, float]] = []
        # Use a combined boundary timeline so the first measure start (t=0)
        # is always included even when grid_den_times has only subdivisions.
        beam_time_boundaries = sorted(
            {round(float(t), 6) for t in (list(grid_den_times or []) + list(barline_times or []))}
        )
        grid_windows = _build_grid_windows(beam_time_boundaries)
        for hand_norm in ('r', 'l'):
            notes_hand = notes_by_norm.get(hand_norm, [])
            markers_hand = markers_by_norm.get(hand_norm, [])
            windows = _process_beam_marker_override(grid_windows, markers_hand)
            groups = _assign_groups(notes_hand, windows)
            for idx, grp in enumerate(groups):
                if not grp or idx >= len(windows):
                    continue
                t0, t1 = windows[idx]
                starts_in = [
                    float(getattr(n, 'time', 0.0) or 0.0)
                    for n in grp
                    if op.ge(float(getattr(n, 'time', 0.0) or 0.0), float(t0)) and op.lt(float(getattr(n, 'time', 0.0) or 0.0), float(t1))
                ]
                if not starts_in:
                    continue
                s_min, s_max = min(starts_in), max(starts_in)
                if op.eq(float(s_min), float(s_max)):
                    continue
                t_first = float(s_min)
                t_last = float(s_max)
                if hand_norm == 'r':
                    highest = max(grp, key=lambda n: int(getattr(n, 'pitch', 0) or 0))
                    x1b = float(self.pitch_to_x(int(getattr(highest, 'pitch', 0) or 0))) + float(stem_len_mm)
                    x2b = x1b + float(semitone_mm)
                else:
                    lowest = min(grp, key=lambda n: int(getattr(n, 'pitch', 0) or 0))
                    x1b = float(self.pitch_to_x(int(getattr(lowest, 'pitch', 0) or 0))) - float(stem_len_mm)
                    x2b = x1b - float(semitone_mm)
                yb1 = float(self.time_to_mm(float(t_first)))
                yb2 = float(self.time_to_mm(float(t_last)))
                beam_segments.append({
                    't_start': float(t_first),
                    't_end': float(t_last),
                    'x1': float(x1b),
                    'x2': float(x2b),
                })
                for m in grp:
                    mt = float(getattr(m, 'time', t_first) or t_first)
                    if not (op.ge(float(mt), float(t0)) and op.lt(float(mt), float(t1))):
                        continue
                    y_note = float(self.time_to_mm(float(mt)))
                    x_note = float(self.pitch_to_x(int(getattr(m, 'pitch', 0) or 0)))
                    x_tip = x_note + float(stem_len_mm) if hand_norm == 'r' else x_note - float(stem_len_mm)
                    if abs(float(yb2) - float(yb1)) > 1e-9:
                        ratio = (float(y_note) - float(yb1)) / (float(yb2) - float(yb1))
                        x_on_beam = float(x1b) + ratio * (float(x2b) - float(x1b))
                    else:
                        x_on_beam = float(x1b)
                    beam_connect_segments.append({
                        'time': float(mt),
                        'x0': float(min(x_tip, x_on_beam)),
                        'x1': float(max(x_tip, x_on_beam)),
                        'beam_start': float(t_first),
                    })

        stave_left = float(stave_left_position)
        stave_right = float(stave_right_position)

        def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
            if not intervals:
                return []
            clipped: list[tuple[float, float]] = []
            for a, b in intervals:
                x0 = max(float(stave_left), min(float(stave_right), float(min(a, b))))
                x1 = max(float(stave_left), min(float(stave_right), float(max(a, b))))
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
            for n in notes_view:
                nt = float(getattr(n, 'time', 0.0) or 0.0)
                if not _barline_time_eq(float(nt), float(ticks)):
                    continue
                p = int(getattr(n, 'pitch', 0) or 0)
                x_note = float(self.pitch_to_x(p))
                intervals.append((
                    x_note - note_head_half_w - head_collision_pad - barline_symbol_gap_mm,
                    x_note + note_head_half_w + head_collision_pad + barline_symbol_gap_mm,
                ))
                if note_stem_visible:
                    hand_key = str(getattr(n, 'hand', 'l') or 'l')
                    x_stem_tip = x_note - stem_len_mm if hand_key == 'l' else x_note + stem_len_mm
                    intervals.append((
                        min(x_note, x_stem_tip) - stem_collision_pad - barline_symbol_gap_mm,
                        max(x_note, x_stem_tip) + stem_collision_pad + barline_symbol_gap_mm,
                    ))
            for seg in beam_segments:
                t0 = float(seg.get('t_start', 0.0) or 0.0)
                t1 = float(seg.get('t_end', 0.0) or 0.0)
                if not _barline_time_in_range(float(ticks), float(t0), float(t1)):
                    continue
                dt = float(t1 - t0)
                if abs(dt) <= 1e-9:
                    continue
                ratio = (float(ticks) - float(t0)) / dt
                x_on_beam = float(seg.get('x1', 0.0) or 0.0) + ratio * (float(seg.get('x2', 0.0) or 0.0) - float(seg.get('x1', 0.0) or 0.0))
                intervals.append((
                    x_on_beam - beam_collision_pad - barline_symbol_gap_mm,
                    x_on_beam + beam_collision_pad + barline_symbol_gap_mm,
                ))
            for conn in beam_connect_segments:
                c_t = float(conn.get('time', 0.0) or 0.0)
                if not _barline_time_eq(float(c_t), float(ticks)):
                    continue
                c_x0 = float(conn.get('x0', 0.0) or 0.0)
                c_x1 = float(conn.get('x1', 0.0) or 0.0)
                intervals.append((
                    c_x0 - beam_collision_pad - barline_symbol_gap_mm,
                    c_x1 + beam_collision_pad + barline_symbol_gap_mm,
                ))
            return _merge_intervals(intervals)

        def _draw_barline_segments(y_mm: float, cuts: list[tuple[float, float]], width_mm: float, tags: list[str], item_id: int = 0) -> None:
            if not cuts:
                du.add_line(
                    stave_left,
                    y_mm,
                    stave_right,
                    y_mm,
                    color=color,
                    width_mm=width_mm,
                    id=item_id,
                    tags=tags,
                    dash_pattern=None,
                )
                return
            x_cursor_seg = float(stave_left)
            min_seg = max(0.05, float(width_mm) * 0.5)
            for c0, c1 in cuts:
                if c0 - x_cursor_seg > min_seg:
                    du.add_line(
                        x_cursor_seg,
                        y_mm,
                        c0,
                        y_mm,
                        color=color,
                        width_mm=width_mm,
                        id=item_id,
                        tags=tags,
                        dash_pattern=None,
                    )
                x_cursor_seg = max(x_cursor_seg, c1)
            if float(stave_right) - x_cursor_seg > min_seg:
                du.add_line(
                    x_cursor_seg,
                    y_mm,
                    stave_right,
                    y_mm,
                    color=color,
                    width_mm=width_mm,
                    id=item_id,
                    tags=tags,
                    dash_pattern=None,
                )

        def _draw_barline_constructive(ticks: float, width_mm: float, tags: list[str]) -> None:
            y_mm = float(self.time_to_mm(float(ticks)))
            cuts = _barline_cut_intervals(float(ticks))
            _draw_barline_segments(float(y_mm), cuts, float(width_mm), tags, 0)

        def _draw_double_bar_constructive(ticks: float, width_mm: float, gap_mm: float, ev_id: int) -> None:
            y_mm = float(self.time_to_mm(float(ticks)))
            cuts = _barline_cut_intervals(float(ticks))
            gap = max(0.1, float(gap_mm))
            tags = ["barline", "double_barline"]
            _draw_barline_segments(float(y_mm + gap), cuts, float(width_mm), tags, int(ev_id))

        # Draw measure numbers at each measure start except final end barline.
        for t in barline_times[:-1]:
            y_mm = float(self.time_to_mm(float(t)))
            du.add_text(
                self.margin + self.stave_width + self.margin - 1.0,
                y_mm + 1.0,
                str(measure_numbering_cursor),
                size_pt=meas_size,
                color=color,
                id=0,
                tags=["measure_number"],
                anchor='ne',
                family=meas_family,
            )
            measure_numbering_cursor += 1

        # Draw subgrid lines from cached grid times, excluding barline layer times.
        if grid_line_visible:
            for t in grid_den_times:
                if round(float(t), 6) in barline_keys:
                    continue
                y_mm = float(self.time_to_mm(float(t)))
                du.add_line(
                    stave_left_position,
                    y_mm,
                    stave_right_position,
                    y_mm,
                    color=color,
                    width_mm=grid_width_mm,
                    id=0,
                    tags=["grid_line"],
                    dash_pattern=[2.0, 2.0],
                )

        # Draw regular barlines; draw final end barline thicker.
        if barline_visible:
            for idx, t in enumerate(barline_times):
                is_last = idx == (len(barline_times) - 1)
                _draw_barline_constructive(
                    float(t),
                    bar_width_mm,
                    (["barline", "end_barline"] if is_last else ["barline"]),
                )

        if barline_visible and bool(getattr(layout, 'double_barline_visible', True)) and layout is not None:
            double_events = list(getattr(score.events, 'double_bar', []) or [])
            if double_events:
                top_mm = float(getattr(self, '_view_y_mm_offset', 0.0) or 0.0)
                vp_h_mm = float(getattr(self, '_viewport_h_mm', 0.0) or 0.0)
                bottom_mm = top_mm + vp_h_mm
                bleed_mm = max(4.0, semitone_mm * 3.0)
                double_width_mm = max(0.1, bar_width_mm)
                # Keep visible whitespace between lines after increasing line thickness.
                inner_clear_gap_mm = max(0.35, semitone_mm * 0.5)
                double_gap_mm = max(double_width_mm * 2.0, inner_clear_gap_mm + double_width_mm)
                for ev in double_events:
                    try:
                        t_ev = float(getattr(ev, 'time', 0.0) or 0.0)
                        ev_id = int(getattr(ev, '_id', 0) or 0)
                    except Exception:
                        continue
                    y_ev = float(self.time_to_mm(float(t_ev)))
                    if y_ev < (top_mm - bleed_mm) or y_ev > (bottom_mm + bleed_mm):
                        continue
                    _draw_double_bar_constructive(float(t_ev), float(double_width_mm), float(double_gap_mm), int(ev_id))