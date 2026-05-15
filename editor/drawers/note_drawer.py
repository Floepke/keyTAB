from __future__ import annotations
from typing import TYPE_CHECKING, cast, Iterable
import bisect
from file_model.SCORE import SCORE
from utils.CONSTANT import BLACK_KEYS, QUARTER_NOTE_UNIT, BE_KEYS, SHORTEST_DURATION
from ui.widgets.draw_util import DrawUtil
from utils.tiny_tool import key_class_filter
from utils.operator import Operator
from typing import Tuple
from ui.style import Style
from symbol_design.noteheads import Notehead, normalize_notehead_literal, resolve_notehead_spec
from editor.editor_defaults import NOTE_WIDTH_SCALING

if TYPE_CHECKING:
    from editor.editor import Editor


class NoteDrawerMixin:
    '''
        Note drawing pipeline adapted from legacy project:
        - Entry `_draw_notes()` computes x/y once and dispatches components
        - `_draw_single_note()` draws all parts (rectangle, head, stem, etc.)
        - Skips centered dashed chord guide for now; beams come later
    '''

    # Local key-class sets (approximate groups used for small positional tweaks)
    _CF_KEYS: set[int] = set(key_class_filter('CF'))
    _ADG_KEYS: set[int] = set(key_class_filter('ADG'))
    # Thresholded time comparator: 7 ticks (smallest app unit is 8)
    _time_op: Operator = Operator(7)
    # Cached sorted notes and indices for current draw pass
    _cached_notes_sorted: list | None = None
    _cached_notes_starts: list[float] | None = None
    _cached_window_lo: int | None = None
    _cached_window_hi: int | None = None
    _cached_notes_view: list | None = None
    _cached_barline_positions: list[float] | None = None
    _note_lookup_source_id: int | None = None
    _notes_by_hand_lookup: dict[str, list] | None = None
    _start_times_by_hand: dict[str, list[float]] | None = None
    _end_times_by_hand: dict[str, list[float]] | None = None
    _end_notes_by_hand: dict[str, list] | None = None
    _STEM_WIDTH_FACTOR: float = 1.0

    def _editor_line_width_mm(self) -> float:
        try:
            return max(0.01, float(getattr(self, 'editor_line_width_global', 0.1) or 0.1))
        except Exception:
            return 0.1

    def _layout_stem_length_mm(self) -> float:
        self = cast("Editor", self)
        score = self.current_score()
        layout = score.layout if score else None
        stem_len_units = float(getattr(layout, 'note_stem_length_semitone', 3) or 3) if layout is not None else 3.0
        return stem_len_units * float(self.semitone_dist or 0.5)

    def _rebuild_note_lookup(self, notes_source: list | None = None) -> None:
        self = cast("Editor", self)
        source = notes_source if notes_source is not None else (self._cached_notes_view or [])
        source_id = id(source)
        if self._note_lookup_source_id == source_id and self._notes_by_hand_lookup is not None:
            return

        by_hand: dict[str, list] = {'l': [], 'r': []}
        for n in source:
            hk = 'l' if str(getattr(n, 'hand', 'l') or 'l') == 'l' else 'r'
            by_hand[hk].append(n)

        start_times_by_hand: dict[str, list[float]] = {'l': [], 'r': []}
        end_times_by_hand: dict[str, list[float]] = {'l': [], 'r': []}
        end_notes_by_hand: dict[str, list] = {'l': [], 'r': []}

        for hk in ('l', 'r'):
            notes_sorted = sorted(by_hand[hk], key=lambda n: float(getattr(n, 'time', 0.0) or 0.0))
            by_hand[hk] = notes_sorted
            start_times_by_hand[hk] = [float(getattr(n, 'time', 0.0) or 0.0) for n in notes_sorted]

            end_pairs = sorted(
                ((float(getattr(n, 'time', 0.0) or 0.0) + float(getattr(n, 'duration', 0.0) or 0.0), n) for n in notes_sorted),
                key=lambda p: p[0],
            )
            end_times_by_hand[hk] = [float(p[0]) for p in end_pairs]
            end_notes_by_hand[hk] = [p[1] for p in end_pairs]

        self._note_lookup_source_id = source_id
        self._notes_by_hand_lookup = by_hand
        self._start_times_by_hand = start_times_by_hand
        self._end_times_by_hand = end_times_by_hand
        self._end_notes_by_hand = end_notes_by_hand

    def _notes_starting_at_time(self, t: float, hand: str | None = None) -> list:
        self = cast("Editor", self)
        self._rebuild_note_lookup()
        if self._notes_by_hand_lookup is None or self._start_times_by_hand is None:
            return []

        thr = float(self._time_op.threshold)
        hands = [hand] if hand in ('l', 'r') else ['l', 'r']
        out: list = []
        for hk in hands:
            times = self._start_times_by_hand.get(hk, [])
            notes = self._notes_by_hand_lookup.get(hk, [])
            lo = bisect.bisect_left(times, float(t) - thr)
            hi = bisect.bisect_right(times, float(t) + thr)
            for i in range(lo, hi):
                n = notes[i]
                if self._time_op.eq(float(getattr(n, 'time', 0.0) or 0.0), float(t)):
                    out.append(n)
        return out

    def _notes_in_open_time_interval(self, hand: str, t0: float, t1: float, by_end: bool = False) -> list:
        self = cast("Editor", self)
        self._rebuild_note_lookup()
        if self._notes_by_hand_lookup is None or self._start_times_by_hand is None or self._end_times_by_hand is None or self._end_notes_by_hand is None:
            return []

        hk = 'l' if str(hand or 'l') == 'l' else 'r'
        if by_end:
            times = self._end_times_by_hand.get(hk, [])
            notes = self._end_notes_by_hand.get(hk, [])
        else:
            times = self._start_times_by_hand.get(hk, [])
            notes = self._notes_by_hand_lookup.get(hk, [])

        lo = bisect.bisect_right(times, float(t0))
        hi = bisect.bisect_left(times, float(t1))
        return notes[lo:hi] if hi > lo else []

    def draw_note(self, du: DrawUtil) -> None:
        """Editor drawer entry point as used by draw_all()."""
        self._draw_notes(du, draw_mode='note')

    def _draw_notes(self, du: DrawUtil, draw_mode: str = 'note') -> None:
        self = cast("Editor", self)
        score: SCORE = self.current_score()
        if score is None:
            return

        # Layout metrics
        margin = float(self.margin or 0.0)
        zpq = float(getattr(score.app_state, 'zoom_mm_per_quarter', 25.0) or 25.0)

        def time_to_mm(ticks: float) -> float:
            return margin + (float(ticks) / float(QUARTER_NOTE_UNIT)) * zpq

        # Viewport culling: compute visible time range with small bleed
        top_mm = float(getattr(self, '_view_y_mm_offset', 0.0) or 0.0)
        vp_h_mm = float(getattr(self, '_viewport_h_mm', 0.0) or 0.0)
        bottom_mm = top_mm + vp_h_mm
        bleed_mm = max(2.0, zpq * 0.25)  # ~quarter-note/4 or 2mm minimum
        time_begin = float(self.mm_to_time(top_mm - bleed_mm))
        time_end = float(self.mm_to_time(bottom_mm + bleed_mm))

        # Use shared render cache from Editor if available
        cache = cast("Editor", self)._draw_cache or None
        if cache is not None:
            notes_sorted = cache.get('notes_sorted') or []
            candidate_indices = cache.get('candidate_indices') or []
            self._cached_notes_view = cache.get('notes_view') or []
            self._cached_notes_sorted = cache.get('notes_sorted') or []
            self._cached_notes_starts = cache.get('starts') or []
            self._cached_barline_positions = cache.get('barline_positions') or []
        else:
            # Fallback: minimal local candidate selection (start-only)
            notes_sorted = sorted(score.events.note or [], key=lambda n: (n.time, n.pitch))
            starts = [float(n.time) for n in notes_sorted]
            lo = bisect.bisect_left(starts, time_begin)
            hi = bisect.bisect_right(starts, time_end)
            candidate_indices = list(range(max(0, lo - 1), hi))
            self._cached_notes_view = [notes_sorted[i] for i in candidate_indices]
            self._cached_barline_positions = self._get_barline_positions()

        self._rebuild_note_lookup(self._cached_notes_view)

        # Per-frame arpeggio Y overrides (set by _build_render_cache)
        arp_y_overrides: dict[int, float] = getattr(self, '_arpeggio_y_overrides', {}) or {}

        # Iterate candidate set only
        for idx in candidate_indices:
            if idx < 0 or idx >= len(notes_sorted):
                continue
            n = notes_sorted[idx]
            # Final interval intersection test in time domain
            n_start = float(n.time)
            n_end = float(n.time + n.duration)
            if self._time_op.lt(n_end, time_begin) or self._time_op.gt(n_start, time_end):
                continue
            # Compute positions once and draw parts
            x = self.pitch_to_x(n.pitch)
            nid = int(getattr(n, '_id', 0) or 0)
            y1 = arp_y_overrides[nid] if nid in arp_y_overrides else time_to_mm(n_start)
            y2 = time_to_mm(n_end)
            self._draw_single_note(du, n, x, y1, y2, draw_mode=draw_mode)

        # Draw stems in one batched pass to avoid per-note duplicates.
        self._draw_stem(du, draw_mode=draw_mode)

        # Do not clear caches here; when using shared cache, Editor manages lifecycle

    def _draw_single_note(self, du: DrawUtil, n, x: float, y1: float, y2: float, draw_mode: str = 'note') -> None:
        self = cast("Editor", self)
        # In tiny mode, render only noteheads and a simple hit rect
        if getattr(self, 'is_tiny_mode', None) and self.is_tiny_mode():
            self._draw_notehead(du, n, x, y1, draw_mode)
            self._draw_midinote(du, n, x, y1, y2, draw_mode)
            self._draw_note_accidental(du, n, x, y1)
            try:
                w = float(self.semitone_dist or 0.5)
                layout = self.current_score().layout
                spec = resolve_notehead_spec(n, default_black_above=self._black_note_above_stem(n, layout))
                y_top = float(y1)
                if bool(getattr(spec, 'is_up', False)):
                    y_top -= w * 2.0
                rect_id = int(getattr(n, '_id', 0) or 0)
                self.register_hit_rect('note', rect_id, float(x - w), float(y_top), float(x + w), float(y1 + (w * 2.0)))
            except Exception:
                pass
            return

        # Draw all parts of the note
        self._draw_midinote(du, n, x, y1, y2, draw_mode)
        self._draw_notehead(du, n, x, y1, draw_mode)
        self._draw_notestop(du, n, x, y2, draw_mode)
        self._draw_note_accidental(du, n, x, y1)
        self._draw_note_continuation_dot(du, n, x, y1, y2, draw_mode)

    def _midinote_color(self, n, draw_mode: str) -> tuple[float, float, float, float]:
        if draw_mode in ('cursor', 'edit', 'selected'):
            return self.accent_color
        hand = getattr(n, 'hand', 'l')
        key = 'midi_left' if hand == 'l' else 'midi_right'
        r, g, b = Style.get_named_rgb(key, (153, 179, 204))
        return (float(r) / 255.0, float(g) / 255.0, float(b) / 255.0, 1.0)

    def _draw_midinote(self, du: DrawUtil, n, x: float, y1: float, y2: float, draw_mode: str) -> None:
        '''Draw the MIDI note rectangle for visualizing note durations'''
        self = cast("Editor", self)
        fill = self._midinote_color(n, draw_mode)
        w = float(self.semitone_dist or 0.5)
        # Use hardcoded editor default for note width scaling (not from file layout)
        head_half_w = w * max(0.05, NOTE_WIDTH_SCALING)
        du.add_polygon(
            [
                (x, y1),
                (x - w, y1 + self.semitone_dist),
                (x - w, y2),
                (x + w, y2),
                (x + w, y1 + self.semitone_dist),
            ],
            stroke_color=None,
            fill_color=fill,
            id=n._id,
            tags=["midi_note"],
        )
        # Register a clickable rectangle covering the full note (notehead top to note end).
        # Sub-zone detection (notehead vs body) is handled in note_tool by comparing
        # cursor Y against note_start_mm + 2 * semitone_dist.
        x_left = x - max(w, head_half_w)
        x_right = x + max(w, head_half_w)
        y_top = y1           # top of notehead
        score = self.current_score()
        layout = score.layout if score else None
        spec = resolve_notehead_spec(n, default_black_above=self._black_note_above_stem(n, layout))
        if bool(getattr(spec, 'is_up', False)):
            y_top -= w * 2.0
        y_bottom = y2        # actual end of note polygon
        rect_id = int(getattr(n, '_id', 0) or 0)
        self.register_hit_rect('note', rect_id, float(x_left), float(y_top), float(x_right), float(y_bottom))

    def _draw_notehead(self, du: DrawUtil, n, x: float, y1: float, draw_mode: str) -> None:
        self = cast("Editor", self)
        layout = self.current_score().layout
        is_narrow = self._should_tune_under_stem_black_width(n, layout)
        scale = layout.scale if layout else 1.0
        outline_w = layout.note_stem_thickness_mm * scale if layout else 0.8
        paper_r, paper_g, paper_b = Style.get_named_rgb('paper', (255, 255, 255))
        bg_fill = (paper_r / 255.0, paper_g / 255.0, paper_b / 255.0, 1.0)
        notehead = Notehead.from_note(
            x_mm=float(x),
            y_mm=float(y1),
            note=n,
            layout=layout,
            semitone_space_mm=float(self.semitone_dist or 0.5),
            notation_color=self.notation_color,
            paper_color=bg_fill,
            default_black_above=self._black_note_above_stem(n, layout),
            outline_width_mm_override=outline_w,
            black_note_narrow=is_narrow,
        )
        tag = "notehead_black" if bool(getattr(notehead, 'filled', False)) else "notehead_white"
        notehead.draw_notehead(du, item_id=int(getattr(n, '_id', 0) or 0), tags=[tag], use_custom_color=True)

    def _draw_notestop(self, du: DrawUtil, n, x: float, y2: float, draw_mode: str) -> None:
        self = cast("Editor", self)
        if getattr(self, 'is_tiny_mode', None) and self.is_tiny_mode():
            return
        layout = self.current_score().layout
        # Show stop triangle if followed by a rest in same hand
        if not self._is_followed_by_rest(n):
            return
        
        # Draw triangle pointing down at end of note
        w = float(self.semitone_dist or 0.5) * 1.8
        points = [
            (x - w / 2, y2 - w),
            (x, y2),
            (x + w / 2, y2 - w),
        ]
        fill = self.notation_color
        # For cursor/edit/selected, emphasize
        if draw_mode in ('cursor', 'edit', 'selected'):
            fill = self.accent_color
        stroke_w = .5
        du.add_polyline(
            points,
            stroke_color=fill,
            stroke_width_mm=stroke_w,
            id=0,
            tags=["stop_sign"],
        )

    def _draw_stem(self, du: DrawUtil, draw_mode: str = 'note') -> None:
        self = cast("Editor", self)
        score = self.current_score()
        layout = score.layout if score else None
        if layout is None:
            return

        notes_view = list(self._cached_notes_view or [])
        if not notes_view:
            return

        # Reuse hand-grouped notes from shared render cache when available.
        cache = getattr(self, '_draw_cache', None) or {}
        cached_by_hand = cache.get('notes_by_hand') or {}

        stem_len = self._layout_stem_length_mm()
        scale = layout.scale
        stem_w = layout.note_stem_thickness_mm * scale
        arp_y_overrides: dict[int, float] = getattr(self, '_arpeggio_y_overrides', {}) or {}

        for hand_key in ('l', 'r'):
            hand_notes = list(cached_by_hand.get(hand_key, []) or [])
            if not hand_notes:
                # Fallback when cache is unavailable/incomplete.
                hand_notes = [
                    n for n in notes_view
                    if ('l' if str(getattr(n, 'hand', 'l') or 'l') == 'l' else 'r') == hand_key
                ]
            if not hand_notes:
                continue
            hand_notes_sorted = sorted(hand_notes, key=lambda n: (float(getattr(n, 'time', 0.0) or 0.0), int(getattr(n, 'pitch', 0) or 0)))

            i = 0
            while i < len(hand_notes_sorted):
                t0 = float(getattr(hand_notes_sorted[i], 'time', 0.0) or 0.0)
                cluster: list = [hand_notes_sorted[i]]
                j = i + 1
                while j < len(hand_notes_sorted):
                    tj = float(getattr(hand_notes_sorted[j], 'time', 0.0) or 0.0)
                    if not self._time_op.eq(tj, t0):
                        break
                    cluster.append(hand_notes_sorted[j])
                    j += 1

                # Arpeggio chords own their diagonal stem; skip overridden notes.
                cluster_no_arp: list = []
                for n in cluster:
                    nid = int(getattr(n, '_id', 0) or 0)
                    if nid in arp_y_overrides:
                        continue
                    cluster_no_arp.append(n)

                if cluster_no_arp:
                    x_values = [float(self.pitch_to_x(int(getattr(n, 'pitch', 0) or 0))) for n in cluster_no_arp]
                    y_values = [float(self.time_to_mm(float(getattr(n, 'time', 0.0) or 0.0))) for n in cluster_no_arp]
                    y_line = float(y_values[0])

                    if len(cluster_no_arp) == 1:
                        x_center = float(x_values[0])
                        x_tip = float(x_center - stem_len) if hand_key == 'l' else float(x_center + stem_len)
                        x1, x2 = (x_center, x_tip)
                    else:
                        x_low = float(min(x_values))
                        x_high = float(max(x_values))
                        if hand_key == 'l':
                            x1, x2 = (float(x_low - stem_len), x_high)
                        else:
                            x1, x2 = (x_low, float(x_high + stem_len))

                    du.add_line(
                        x1,
                        y_line,
                        x2,
                        y_line,
                        color=self.notation_color,
                        width_mm=stem_w,
                        id=0,
                        tags=["stem"],
                    )

                i = j

    def _draw_note_continuation_dot(self, du: DrawUtil, n, x: float, y1: float, y2: float, draw_mode: str) -> None:
        self = cast("Editor", self)
        if getattr(self, 'is_tiny_mode', None) and self.is_tiny_mode():
            return
        layout = self.current_score().layout
        # Draw dots where other notes in same hand start or end within this note duration
        hand = getattr(n, 'hand', 'l')
        start = float(n.time)
        end = float(n.time + n.duration)
        w = float(self.semitone_dist or 0.5)

        # Collect dot times from indexed hand-specific starts/ends in (start, end).
        dot_times: list[float] = []
        for m in self._notes_in_open_time_interval(hand, start, end, by_end=False):
            if int(getattr(m, '_id', -1) or -1) != int(getattr(n, '_id', -2) or -2):
                dot_times.append(float(getattr(m, 'time', 0.0) or 0.0))
        for m in self._notes_in_open_time_interval(hand, start, end, by_end=True):
            if int(getattr(m, '_id', -1) or -1) != int(getattr(n, '_id', -2) or -2):
                dot_times.append(float(getattr(m, 'time', 0.0) or 0.0) + float(getattr(m, 'duration', 0.0) or 0.0))

        # Add a continuation dot at any crossed barline.
        barlines = self._cached_barline_positions or self._get_barline_positions()
        for bt in barlines:
            bt = float(bt)
            if self._time_op.gt(bt, start) and self._time_op.lt(bt, end):
                dot_times.append(bt)
        if not dot_times:
            return

        # Draw dots using notehead center for consistent positioning
        dot_d = w * 0.8
        dot_pitch = int(getattr(n, 'pitch', 0) or 0)
        layout = self.current_score().layout
        # Build a set of double-barline tick positions to avoid overlap.
        _double_bar_ticks: set[float] = {
            float(getattr(ev, 'time', 0.0) or 0.0)
            for ev in (getattr(self.current_score().events, 'double_bar', []) or [])
        }
        dot_x = float(self.pitch_to_x(dot_pitch))
        min_collision_gap = max(0.0, float(self.semitone_dist or 0.5) * 2.0 - 1e-6)
        for t in sorted(set(dot_times)):
            y_center = float(self.time_to_mm(t)) + w
            # Shift dot down one semitone when it lands on a double barline
            # so the two vertical lines don't overlap the dot.
            if any(self._time_op.eq(float(t), dbt) for dbt in _double_bar_ticks):
                y_center += float(self.semitone_dist or 0.5)

            # If another note starts at this exact time on adjacent pitch,
            # push continuation dot forward two semitone distances to avoid overlap.
            has_adjacent_start = False
            for m in self._notes_starting_at_time(float(t), hand=None):
                if int(getattr(m, '_id', -1) or -1) == int(getattr(n, '_id', -2) or -2):
                    continue
                mp = int(getattr(m, 'pitch', 0) or 0)
                if abs(mp - dot_pitch) == 1:
                    # Black adjacent notes drawn above stem do not collide with the
                    # default continuation-dot position.
                    if mp in BLACK_KEYS and self._black_note_above_stem(m, layout):
                        continue
                    other_x = float(self.pitch_to_x(mp))
                    if abs(other_x - dot_x) >= min_collision_gap:
                        continue
                    has_adjacent_start = True
                    break
            if has_adjacent_start:
                y_center += float(self.semitone_dist or 0.5) * 2.0

            du.add_oval(
                x - dot_d / 2.0,
                y_center - dot_d / 2.0,
                x + dot_d / 2.0,
                y_center + dot_d / 2.0,
                fill_color=self.notation_color,
                stroke_color=None,
                id=0,
                tags=["left_dot"],
            )

    def _editor_background_rgba(self) -> Tuple[float, float, float, float]:
        """Return the editor background as RGBA floats (0..1), alpha=1.0.

        Reads from Style.get_paper_color() without instantiating Style
        to avoid side effects.
        """
        from ui.style import Style
        rgb = Style.get_paper_color()
        r, g, b = tuple(int(c) for c in rgb)
        return (r / 255.0, g / 255.0, b / 255.0, 1.0)

    def _black_note_above_stem(self, n, layout) -> bool:
        rule = str(getattr(layout, 'black_note_rule', 'below_stem') or 'below_stem')
        if rule == 'above_stem':
            return True
        t0 = float(getattr(n, 'time', 0.0) or 0.0)
        p0 = int(getattr(n, 'pitch', 0) or 0)
        if rule in ('above_stem_if_collision', 'only_above_stem_if_collision'):
            for note in self._notes_starting_at_time(float(t0), hand=None):
                if getattr(note, '_id', None) == getattr(n, '_id', None):
                    continue
                if abs(int(getattr(note, 'pitch', 0) or 0) - p0) == 1:
                    return True
            return False
        if rule == 'above_stem_if_chord_and_white_note':
            for note in self._notes_starting_at_time(float(t0), hand=None):
                if getattr(note, '_id', None) == getattr(n, '_id', None):
                    continue
                mp = int(getattr(note, 'pitch', 0) or 0)
                if mp not in BLACK_KEYS and mp != p0:
                    return True
            return False
        if rule != 'above_stem_if_chord_and_white_note_same_hand':
            return False
        hand0 = str(getattr(n, 'hand', 'l') or 'l')
        for note in self._notes_starting_at_time(float(t0), hand=hand0):
            if getattr(note, '_id', None) == getattr(n, '_id', None):
                continue
            mp = int(getattr(note, 'pitch', 0) or 0)
            if mp not in BLACK_KEYS and mp != p0:
                return True
        return False

    def _adjacent_white_same_hand(self, n, layout) -> bool:
        t0 = float(getattr(n, 'time', 0.0) or 0.0)
        p0 = int(getattr(n, 'pitch', 0) or 0)
        h0 = str(getattr(n, 'hand', 'l') or 'l')
        for m in self._notes_starting_at_time(float(t0), hand=h0):
            if getattr(m, '_id', None) == getattr(n, '_id', None):
                continue
            mp = int(getattr(m, 'pitch', 0) or 0)
            if mp not in BLACK_KEYS and abs(mp - p0) == 1:
                return True
        return False

    def _should_tune_under_stem_black_width(self, n, layout) -> bool:
        """Narrow black noteheads for under-stem small-second collisions."""
        rule = str(getattr(layout, 'black_note_rule', 'below_stem') or 'below_stem').strip().lower()
        if rule not in ('under_stem', 'below_stem'):
            return False

        # Custom noteheads that explicitly point above the stem should not be
        # narrowed by the under-stem collision rule.
        custom_notehead = normalize_notehead_literal(getattr(n, 'notehead', 'auto'))
        if custom_notehead != 'auto':
            custom_spec = resolve_notehead_spec(n, default_black_above=False)
            if bool(getattr(custom_spec, 'is_up', False)):
                return False

        p0 = int(getattr(n, 'pitch', 0) or 0)
        if p0 not in BLACK_KEYS:
            return False
        t0 = float(getattr(n, 'time', 0.0) or 0.0)
        for m in self._notes_starting_at_time(float(t0), hand=None):
            if getattr(m, '_id', None) == getattr(n, '_id', None):
                continue
            if abs(int(getattr(m, 'pitch', 0) or 0) - p0) == 1:
                return True
        return False

    # ---- Helpers ----
    def _get_barline_positions(self) -> list[float]:
        score: SCORE = cast("Editor", self).current_score()
        pos: list[float] = []
        cur = 0.0
        for bg in score.base_grid:
            measure_len = float(bg.numerator) * (4.0 / float(bg.denominator)) * float(QUARTER_NOTE_UNIT)
            for _ in range(int(bg.measure_amount)):
                pos.append(cur)
                cur += measure_len
        return pos

    def _is_followed_by_rest(self, n) -> bool:
        # True if there is a gap after this note before next note in same hand
        self = cast("Editor", self)
        hand = getattr(n, 'hand', 'l')
        end = float(n.time + n.duration)
        cache = getattr(self, '_draw_cache', None) or {}
        op: Operator = cache.get('op') or self._time_op
        thr = float(op.threshold)

        # Prefer hand-specific lists from cache for accuracy and speed
        notes_by_hand = cache.get('notes_by_hand') or {}
        hand_list = notes_by_hand.get(hand)
        if hand_list:
            starts_hand = [float(m.time) for m in hand_list]
            idx = bisect.bisect_left(starts_hand, float(end - thr))
            min_delta = None
            for j in range(idx, len(hand_list)):
                m = hand_list[j]
                if m._id == n._id:
                    continue
                delta = float(m.time) - end
                if delta >= -thr:
                    min_delta = delta
                    break
            if min_delta is None:
                return True
            return op.gt(float(min_delta), 0.0)

        # Fallback: scan globally if cache lacks hand grouping
        starts = cache.get('starts') or (self._cached_notes_starts or [])
        notes_sorted = cache.get('notes_sorted') or (self._cached_notes_sorted or [])
        if not starts or not notes_sorted:
            score: SCORE = self.current_score()
            notes_sorted = sorted(getattr(score.events, 'note', []) or [], key=lambda nn: (float(nn.time), int(nn.pitch)))
            starts = [float(nn.time) for nn in notes_sorted]
        idx = bisect.bisect_left(starts, float(end - thr)) if starts else 0
        min_delta = None
        for j in range(idx, len(notes_sorted)):
            m = notes_sorted[j]
            if m._id == n._id or getattr(m, 'hand', 'l') != hand:
                continue
            delta = float(m.time) - end
            if delta >= -thr:
                min_delta = delta
                break
        if min_delta is None:
            return True
        return op.gt(float(min_delta), 0.0)
