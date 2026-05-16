"""NoteheadDrawer: renders noteheads and accidental symbols."""

import bisect
import math

from utils.CONSTANT import SHORTEST_DURATION, PIANO_KEY_AMOUNT, BLACK_KEYS, QUARTER_NOTE_UNIT
from utils.operator import Operator
from symbol_design.noteheads import resolve_notehead_spec, sheared_notehead_outline_points
from engraver.helpers import black_note_above_stem, time_to_y, group_by_beam_markers
from file_model.base_grid import resolve_grid_layer_offsets

class NoteheadDrawer:
    """Draw noteheads and accidental symbols."""

    def __init__(self, context):
        self.context = context
        self.du = context.du
        self.layout_data = context.layout_data
        self.notation_color = self.layout_data.get('notation_color')
        self.scale = self.layout_data.get('scale', 1.0)
        self.op = Operator(SHORTEST_DURATION)
        self.paper_color = self.layout_data.get('paper_color', (1.0, 1.0, 1.0, 1.0))

    def draw(self) -> None:
        """Draw all noteheads for the current page and line (no stems/ledger yet)."""
        page_lines = self.layout_data.get('page_lines', [])
        if not page_lines:
            return

        score = self.context.score or {}
        events = dict(score.get('events', {}) or {})
        score_notes = list(events.get('note', []) or [])
        base_grid = list(score.get('base_grid', []) or [])
        norm_double_bars = list(events.get('double_bar', []) or [])
        beam_markers = list(events.get('beam', []) or [])

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
            beam_by_hand[hk] = sorted(beam_by_hand[hk], key=lambda m: float(m.get('time', 0.0) or 0.0))

        notes_by_hand: dict[str, list[dict]] = {'l': [], 'r': []}
        starts_by_hand: dict[str, list[float]] = {'l': [], 'r': []}
        for idx, n in enumerate(score_notes):
            if not isinstance(n, dict):
                continue
            n_t = float(n.get('time', 0.0) or 0.0)
            n_d = float(n.get('duration', 0.0) or 0.0)
            n_end = n_t + n_d
            hand_raw = str(n.get('hand', 'l') or 'l')
            hand_key = 'l' if hand_raw == 'l' else 'r'
            item = {
                'time': n_t,
                'end': n_end,
                'duration': n_d,
                'pitch': int(n.get('pitch', 0) or 0),
                'hand': hand_key,
                'id': int(n.get('_id', 0) or 0),
                'idx': int(idx),
                'raw': n,
            }
            notes_by_hand[hand_key].append(item)
            starts_by_hand[hand_key].append(float(n_t))
        for hk in ('l', 'r'):
            notes_by_hand[hk] = sorted(notes_by_hand[hk], key=lambda m: float(m.get('time', 0.0) or 0.0))
            starts_by_hand[hk] = sorted(starts_by_hand[hk])

        op = self.op

        def _has_followed_rest(item: dict) -> bool:
            hand_key = str(item.get('hand', 'l') or 'l')
            hand_list = notes_by_hand.get(hand_key, [])
            starts = starts_by_hand.get(hand_key, [])
            if not hand_list or not starts:
                return True
            end = float(item.get('end', 0.0) or 0.0)
            thr = float(op.threshold)
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
            return op.gt(float(min_delta), 0.0)

        barline_positions: list[float] = []
        cur_bar = 0.0
        for bg in base_grid:
            numer = int(bg.get('numerator', 4) or 4)
            denom = int(bg.get('denominator', 4) or 4)
            measures = int(bg.get('measure_amount', 1) or 1)
            beat_grouping = list(bg.get('beat_grouping', []) or [])
            bar_offsets, _grid_offsets = resolve_grid_layer_offsets(beat_grouping, numer, denom)
            measure_len = float(numer) * (4.0 / float(max(1, denom))) * float(QUARTER_NOTE_UNIT)
            for _ in range(int(max(0, measures))):
                for off in bar_offsets:
                    barline_positions.append(float(cur_bar + float(off)))
                cur_bar += measure_len
        all_barlines = sorted(list(dict.fromkeys([0.0] + [float(v) for v in barline_positions] + [float(cur_bar)])))
        double_bar_ticks: set[float] = {
            float(ev.get('time', 0.0) or 0.0) for ev in norm_double_bars if isinstance(ev, dict)
        }

        key_to_x_map = self.layout_data.get('pitch_to_x_for_line', {})
        time_to_y_map = self.layout_data.get('time_to_y_for_line', {})
        for line_index, line in enumerate(page_lines):
            notes = list(line.get('notes', []) or [])
            if not notes:
                continue
            key_to_x = key_to_x_map.get(line_index)
            if not callable(key_to_x):
                continue
            layout = self.layout_data.get('layout', {})
            semitone_mm = float(self.layout_data.get('semitone_mm', 2.0) or 2.0)
            note_width_scale = float(layout.get('note_width_scaling', 0.75) or 0.75)
            note_height_scale = float(layout.get('notehead_height_scaling', 1.0) or 1.0)
            note_tilt = float(layout.get('notehead_tilt', 0.0) or 0.0)
            stem_len = float(layout.get('note_stem_length_semitone', 3) or 3) * semitone_mm
            stem_w = float(layout.get('note_stem_thickness_mm', 0.5) or 0.5) * self.scale
            black_rule = str(layout.get('black_note_rule', 'below_stem') or 'below_stem')
            line_start = float(line.get('time_start', 0.0) or 0.0)
            line_end = float(line.get('time_end', line_start) or line_start)

            def _is_line_continuation(note_dict: dict) -> bool:
                start_t = float(note_dict.get('time', 0.0) or 0.0)
                end_t = float(note_dict.get('end', 0.0) or 0.0)
                return op.gt(float(line_start), start_t) and op.gt(end_t, float(line_start))

            # Build per-hand stem groups where each group is either a single
            # note or a chord (multiple notes with equal start time by Operator).
            stem_groups_by_hand: dict[str, list[dict]] = {'l': [], 'r': []}
            for candidate in sorted(notes, key=lambda n: float(n.get('time', 0.0) or 0.0)):
                if _is_line_continuation(candidate):
                    continue
                hand_norm = 'l' if str(candidate.get('hand', 'l') or 'l') == 'l' else 'r'
                t = float(candidate.get('time', 0.0) or 0.0)
                groups = stem_groups_by_hand[hand_norm]
                match = None
                for grp in groups:
                    if op.eq(float(grp.get('time', 0.0) or 0.0), t):
                        match = grp
                        break
                if match is None:
                    match = {'time': t, 'notes': []}
                    groups.append(match)
                match['notes'].append(candidate)

            for note in notes:
                pitch = int(note.get('pitch', 0) or 0)
                time = float(note.get('time', 0.0) or 0.0)
                n_end = float(note.get('end', time) or time)
                hand_key = str(note.get('hand', 'l') or 'l')
                if pitch < 1 or pitch > PIANO_KEY_AMOUNT:
                    continue
                x = key_to_x(pitch)
                y = time_to_y(line, time)
                y_end = time_to_y(line, n_end)
                # Use legacy notehead spec logic
                default_black_above = bool(
                    pitch in BLACK_KEYS and black_note_above_stem(note, black_rule, notes)
                )
                spec = resolve_notehead_spec(note.get('raw', {}) or {}, default_black_above=default_black_above)
                outline = sheared_notehead_outline_points(
                    hand=str(note.get('hand', 'l') or 'l'),
                    is_up=bool(getattr(spec, 'is_up', False)),
                    semitone_space_mm=semitone_mm,
                    width_scale=max(0.05, note_width_scale),
                    height_scale=max(0.1, note_height_scale),
                    base_tilt=max(0.0, min(1.0, note_tilt)),
                    sample_count=128,
                )
                # Center outline at (x, y)
                points = [(x + dx, y + dy) for (dx, dy) in outline]
                # Use notation color for now (MIDI color can be added later)
                self.du.add_polygon(
                    points,
                    fill_color=self.notation_color if note.get('pitch') in BLACK_KEYS else self.paper_color,
                    stroke_color=self.notation_color,
                    stroke_width_mm=max(0.05, stem_w),
                    id=0,
                    tags=["notehead"],
                )

                continues_from_prev_line = _is_line_continuation(note)
                continues_to_next_line = op.lt(time, float(line_end)) and op.gt(n_end, float(line_end))

                # create list of continuation dot positions based on other notes in the same hand and barline positions
                dot_times: list[float] = []
                for other in notes:
                    if int(other.get('idx', -1) or -1) == int(note.get('idx', -2) or -2):
                        continue
                    if str(other.get('hand', 'l') or 'l') != hand_key:
                        continue
                    s = float(other.get('time', 0.0) or 0.0)
                    e = float(other.get('end', 0.0) or 0.0)
                    if op.gt(s, time) and op.lt(s, n_end):
                        dot_times.append(s)
                    if op.gt(e, time) and op.lt(e, n_end):
                        dot_times.append(e)
                for bt in barline_positions:
                    bt = float(bt)
                    if op.eq(bt, float(line_start)) or op.eq(bt, float(line_end)):
                        continue
                    if op.gt(bt, time) and op.lt(bt, n_end):
                        dot_times.append(bt)

                if continues_from_prev_line:
                    dot_times.append(float(line_start))
                if continues_to_next_line:
                    dot_times.append(float(line_end))

                '''CONTINUATION DOT DRAWING'''
                if dot_times and bool(layout.get('note_continuation_dot_visible', True)):
                    dot_d = float(layout.get('note_continuation_dot_size_mm', 0.0) or 0.0)
                    if dot_d > 0.0:
                        dot_d *= self.scale
                    else:
                        dot_d = semitone_mm * 0.8
                    dot_x = float(x)
                    min_collision_gap = max(0.0, float(semitone_mm) * 2.0 - 1e-6)
                    for t in sorted(set(dot_times)):
                        y_center = time_to_y(line, float(t)) + float(semitone_mm)
                        if any(op.eq(float(t), dbt) for dbt in double_bar_ticks):
                            y_center += float(semitone_mm)

                        has_adjacent_start = False
                        for other in notes:
                            if int(other.get('idx', -1) or -1) == int(note.get('idx', -2) or -2):
                                continue
                            if _is_line_continuation(other):
                                continue
                            if not op.eq(float(other.get('time', 0.0) or 0.0), float(t)):
                                continue
                            other_pitch = int(other.get('pitch', 0) or 0)
                            if abs(other_pitch - int(pitch)) == 1:
                                other_black_above = (
                                    other_pitch in BLACK_KEYS
                                    and black_note_above_stem(other, black_rule, notes)
                                )
                                if other_black_above:
                                    continue
                                other_x = float(key_to_x(int(other_pitch)))
                                if abs(other_x - dot_x) >= min_collision_gap:
                                    continue
                                has_adjacent_start = True
                                break
                        if has_adjacent_start:
                            y_center += float(semitone_mm) * 2.0

                        self.du.add_oval(
                            dot_x - dot_d / 2.0,
                            y_center - dot_d / 2.0,
                            dot_x + dot_d / 2.0,
                            y_center + dot_d / 2.0,
                            fill_color=self.notation_color,
                            stroke_color=None,
                            id=0,
                            tags=['continuation_dot'],
                        )

                '''STOP SYMBOL DRAWING'''
                if (
                    bool(layout.get('note_stop_visible', True))
                    and not continues_to_next_line
                    and _has_followed_rest(note)
                ):
                    w_stop = float(semitone_mm) * 2.0
                    points = [
                        (x - w_stop / 2.0, y_end - w_stop),
                        (x, y_end),
                        (x + w_stop / 2.0, y_end - w_stop),
                    ]
                    self.du.add_polyline(
                        points,
                        stroke_color=self.notation_color,
                        stroke_width_mm=float(layout.get('note_stopsign_thickness_mm', 0.4) or 0.4) * self.scale,
                        id=0,
                        tags=['stop_sign'],
                    )

            '''STEM DRAWING'''
            if bool(layout.get('note_stem_visible', True)):
                for hand_norm in ('l', 'r'):
                    for grp in stem_groups_by_hand.get(hand_norm, []):
                        chord_notes = list(grp.get('notes', []) or [])
                        if not chord_notes:
                            continue
                        highest = max(chord_notes, key=lambda n: int(n.get('pitch', 0) or 0))
                        lowest = min(chord_notes, key=lambda n: int(n.get('pitch', 0) or 0))
                        y = time_to_y(line, float(grp.get('time', 0.0) or 0.0))

                        if hand_norm == 'l':
                            x_start = float(key_to_x(int(highest.get('pitch', 0) or 0)))
                            x_end = float(key_to_x(int(lowest.get('pitch', 0) or 0))) - float(stem_len)
                        else:
                            x_start = float(key_to_x(int(lowest.get('pitch', 0) or 0)))
                            x_end = float(key_to_x(int(highest.get('pitch', 0) or 0))) + float(stem_len)

                        self.du.add_line(
                            float(x_start),
                            float(y),
                            float(x_end),
                            float(y),
                            color=self.notation_color,
                            width_mm=float(stem_w),
                            id=0,
                            tags=['stem'],
                        )

            '''BEAM DRAWING'''
            if bool(layout.get('beam_visible', True)):
                beam_w = float(layout.get('beam_thickness_mm', 1.0) or 1.0) * self.scale
                beam_half = max(0.1, float(beam_w) * 0.5)
                stem_half = max(0.05, float(stem_w) * 0.5)
                beam_corner_r = max(0.0, float(layout.get('beam_corner_radius_mm', 0.2) or 0.2) * self.scale)

                notes_by_hand_line: dict[str, list[dict]] = {'l': [], 'r': []}
                for item in notes:
                    hk = str(item.get('hand', 'l') or 'l')
                    hand_norm = 'l' if hk == 'l' else 'r'
                    notes_by_hand_line[hand_norm].append(item)

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
                    self.du.add_polygon(
                        poly,
                        stroke_color=None,
                        fill_color=self.notation_color,
                        id=0,
                        tags=['beam'],
                    )

                for hand_norm in ('r', 'l'):
                    notes_for_hand = notes_by_hand_line.get(hand_norm, [])
                    markers_for_hand = beam_by_hand.get(hand_norm, [])
                    groups, windows = group_by_beam_markers(
                        notes_for_hand,
                        markers_for_hand,
                        line_start,
                        line_end,
                        base_grid,
                        all_barlines,
                    )
                    for idx, grp in enumerate(groups):
                        if not grp:
                            continue
                        t0, t1 = windows[idx] if idx < len(windows) else (line_start, line_end)
                        starts_in = [
                            float(n.get('time', 0.0) or 0.0)
                            for n in grp
                            if op.ge(float(n.get('time', 0.0) or 0.0), float(t0))
                            and op.lt(float(n.get('time', 0.0) or 0.0), float(t1))
                        ]
                        if not starts_in:
                            continue
                        s_min, s_max = min(starts_in), max(starts_in)
                        if op.eq(float(s_min), float(s_max)):
                            continue

                        t_first = min(starts_in)
                        t_last = max(starts_in)
                        if hand_norm == 'r':
                            highest = max(grp, key=lambda n: int(n.get('pitch', 0) or 0))
                            x1 = float(key_to_x(int(highest.get('pitch', 0) or 0))) + float(stem_len)
                            x2 = x1 + float(semitone_mm)
                        else:
                            lowest = min(grp, key=lambda n: int(n.get('pitch', 0) or 0))
                            x1 = float(key_to_x(int(lowest.get('pitch', 0) or 0))) - float(stem_len)
                            x2 = x1 - float(semitone_mm)

                        yb1 = float(time_to_y(line, float(t_first)))
                        yb2 = float(time_to_y(line, float(t_last)))
                        _draw_beam(x1, yb1, x2, yb2)

                        # Connect beam to outer stem tip for each unique start-time chord group.
                        time_anchors: list[float] = []
                        for n in grp:
                            mt = float(n.get('time', t_first) or t_first)
                            if not (op.ge(mt, float(t0)) and op.lt(mt, float(t1))):
                                continue
                            if any(op.eq(mt, existing) for existing in time_anchors):
                                continue
                            time_anchors.append(mt)

                        for mt in sorted(time_anchors):
                            stem_group = None
                            for candidate in stem_groups_by_hand.get(hand_norm, []):
                                if op.eq(float(candidate.get('time', 0.0) or 0.0), float(mt)):
                                    stem_group = candidate
                                    break
                            if stem_group is None:
                                continue
                            stem_notes = list(stem_group.get('notes', []) or [])
                            if not stem_notes:
                                continue

                            if hand_norm == 'r':
                                tip_note = max(stem_notes, key=lambda n: int(n.get('pitch', 0) or 0))
                                x_tip = float(key_to_x(int(tip_note.get('pitch', 0) or 0))) + float(stem_len)
                            else:
                                tip_note = min(stem_notes, key=lambda n: int(n.get('pitch', 0) or 0))
                                x_tip = float(key_to_x(int(tip_note.get('pitch', 0) or 0))) - float(stem_len)

                            y_note = float(time_to_y(line, float(mt)))
                            if abs(yb2 - yb1) > 1e-6:
                                t_ratio = (y_note - yb1) / (yb2 - yb1)
                                x_on_beam = x1 + t_ratio * (x2 - x1)
                            else:
                                x_on_beam = x1

                            self.du.add_line(
                                float(x_tip),
                                float(y_note),
                                float(x_on_beam),
                                float(y_note),
                                color=self.notation_color,
                                width_mm=max(0.15, stem_w),
                                id=0,
                                tags=['beam_stem'],
                            )
