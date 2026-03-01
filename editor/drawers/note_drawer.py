from __future__ import annotations
from typing import TYPE_CHECKING, cast, Iterable
import bisect
from file_model.SCORE import SCORE
from utils.CONSTANT import BLACK_KEYS, QUARTER_NOTE_UNIT, BE_KEYS, SHORTEST_DURATION
from ui.widgets.draw_util import DrawUtil
from utils.tiny_tool import key_class_filter
from utils.operator import Operator
from typing import Tuple

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
        zpq = float(score.editor.zoom_mm_per_quarter)

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
            y1 = time_to_mm(n_start)
            y2 = time_to_mm(n_end)
            self._draw_single_note(du, n, x, y1, y2, draw_mode=draw_mode)

        # Do not clear caches here; when using shared cache, Editor manages lifecycle

    def _draw_single_note(self, du: DrawUtil, n, x: float, y1: float, y2: float, draw_mode: str = 'note') -> None:
        
        # Draw all parts of the note
        self._draw_midinote(du, n, x, y1, y2, draw_mode)
        self._draw_hand_split_indicator(du, n, x, y1)
        self._draw_notehead(du, n, x, y1, draw_mode)
        self._draw_notestop(du, n, x, y2, draw_mode)
        self._draw_stem(du, n, x, y1, draw_mode)
        self._draw_note_continuation_dot(du, n, x, y1, y2, draw_mode)
        self._draw_connect_stem(du, n, x, y1, draw_mode)
        self._draw_left_dot(du, n, x, y1, draw_mode)

    def _midinote_color(self, n, draw_mode: str) -> tuple[float, float, float, float]:
        if draw_mode in ('cursor', 'edit', 'selected'):
            return self.accent_color
        return (0.6, 0.7, 0.8, 1.0) if (getattr(n, 'hand', '<') in ('l', '<')) else (0.8, 0.7, 0.6, 1.0)

    def _draw_midinote(self, du: DrawUtil, n, x: float, y1: float, y2: float, draw_mode: str) -> None:
        fill = self._midinote_color(n, draw_mode)
        w = float(self.semitone_dist or 0.5)
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
        # Register a clickable rectangle covering the main midinote body
        x_left = x - w
        x_right = x + w
        # Use computed w to avoid relying on raw semitone_dist
        y_top = y1 + w
        y_bottom = y2
        if x_left > x_right:
            x_left, x_right = x_right, x_left
        if y_top > y_bottom:
            y_top, y_bottom = y_bottom, y_top
        rect_id = int(getattr(n, '_id', 0) or 0)
        self.register_note_hit_rect(rect_id, float(x_left), float(y_top), float(x_right), float(y_bottom))

    def _draw_hand_split_indicator(self, du: DrawUtil, n, x: float, y1: float) -> None:
        barlines = self._cached_barline_positions or []
        if not barlines:
            return
        t = float(getattr(n, 'time', 0.0) or 0.0)
        on_barline = False
        for bt in barlines:
            if self._time_op.eq(float(bt), t):
                on_barline = True
                break
        if not on_barline:
            return
        layout = cast("Editor", self).current_score().layout
        w = float(self.semitone_dist or 0.5)
        stem_len = float(layout.note_stem_length_semitone or 3) * w
        thickness = float(layout.grid_barline_thickness_mm or 0.25)
        hand = getattr(n, 'hand', '<')
        if hand in ('l', '<'):
            x1 = x
            x2 = x + (w * 2.0)
            x3 = x - stem_len
        else:
            x1 = x
            x2 = x - (w * 2.0)
            x3 = x + stem_len
        du.add_line(
            x1,
            y1,
            x2,
            y1,
            color=self._editor_background_rgba(),
            width_mm=thickness,
            line_cap="butt",
            id=0,
            tags=["stem_hand_split"],
        )
        du.add_line(
            x3,
            y1,
            x2,
            y1,
            color=self._editor_background_rgba(),
            width_mm=thickness,
            line_cap="butt",
            id=0,
            tags=["stem_hand_split"],
        )

    def _draw_notehead(self, du: DrawUtil, n, x: float, y1: float, draw_mode: str) -> None:
        w = float(self.semitone_dist or 0.5)
        layout = cast("Editor", self).current_score().layout
        outline_w = 0.5
        # Adjust vertical for black-note rule
        if n.pitch in BLACK_KEYS and self._black_note_above_stem(n, layout):
            y1 = y1 - (w * 2.0)
        if n.pitch in BLACK_KEYS:
            du.add_oval(
                x - (w*.8),
                y1,
                x + (w*.8),
                y1 + w * 2.0,
                stroke_color=self.notation_color,
                stroke_width_mm=0.3,
                fill_color=self.notation_color,
                id=n._id,
                tags=["notehead_black"],
            )
        else:
            # Use explicit white for notehead fill to avoid theme bleed
            bg_fill = (1.0, 1.0, 1.0, 1.0)
            du.add_oval(
                x - w,
                y1,
                x + w,
                y1 + w * 2.0,
                stroke_color=self.notation_color,
                stroke_width_mm=outline_w,
                fill_color=bg_fill,
                id=n._id,
                tags=["notehead_white"],
            )

    def _draw_notestop(self, du: DrawUtil, n, x: float, y2: float, draw_mode: str) -> None:
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

    def _draw_stem(self, du: DrawUtil, n, x: float, y1: float, draw_mode: str) -> None:
        layout = cast("Editor", self).current_score().layout
        stem_len = float(layout.note_stem_length_semitone or 3) * float(self.semitone_dist or 0.5)
        stem_w = 0.75
        # Stem direction based on hand
        if getattr(n, 'hand', '<') in ('l', '<'):
            x2 = x - stem_len
        else:
            x2 = x + stem_len
        du.add_line(
            x,
            y1,
            x2,
            y1,
            color=self.notation_color,
            width_mm=stem_w,
            id=0,
            tags=["stem"],
        )

    def _draw_note_continuation_dot(self, du: DrawUtil, n, x: float, y1: float, y2: float, draw_mode: str) -> None:
        # Draw dots where other notes in same hand start or end within this note duration
        hand = getattr(n, 'hand', '<')
        start = float(n.time)
        end = float(n.time + n.duration)
        w = float(self.semitone_dist or 0.5)

        # Collect dot times
        dot_times: list[float] = []
        # Prefer shared cached viewport notes
        cache = cast("Editor", self)._draw_cache or {}
        notes_view = cache.get('notes_view') or (self._cached_notes_view or [])
        for m in notes_view:
            if m._id == n._id or getattr(m, 'hand', '<') != hand:
                continue
            s = float(m.time)
            e = float(m.time + m.duration)
            if self._time_op.gt(s, start) and self._time_op.lt(s, end):
                dot_times.append(s)
            if self._time_op.gt(e, start) and self._time_op.lt(e, end):
                dot_times.append(e)

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
        for t in sorted(set(dot_times)):
            y_center = float(self.time_to_mm(t)) + w
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

    def _draw_connect_stem(self, du: DrawUtil, n, x: float, y1: float, draw_mode: str) -> None:
        # Connect notes in a chord (same start time, same hand)
        stem_w = 0.75
        hand = getattr(n, 'hand', '<')
        t = float(n.time)
        cache = cast("Editor", self)._draw_cache or {}
        notes_view = cache.get('notes_view') or (self._cached_notes_view or [])
        same_time = [m for m in notes_view if getattr(m, 'hand', '<') == hand and self._time_op.eq(float(m.time), t)]
        if len(same_time) < 2:
            return
        lowest = min(same_time, key=lambda m: m.pitch)
        highest = max(same_time, key=lambda m: m.pitch)
        x1 = self.pitch_to_x(lowest.pitch)
        x2 = self.pitch_to_x(highest.pitch)
        du.add_line(
            x1,
            y1,
            x2,
            y1,
            color=self.notation_color,
            width_mm=stem_w,
            id=0,
            tags=["chord_connect"],
        )

    def _draw_left_dot(self, du: DrawUtil, n, x: float, y1: float, draw_mode: str) -> None:
        # Simple left-hand indicator dot in notehead (optional)
        if getattr(n, 'hand', '<') not in ('l', '<'):
            return
        layout = cast("Editor", self).current_score().layout
        if n.pitch in BLACK_KEYS and self._black_note_above_stem(n, layout):
            y1 = y1 - (float(self.semitone_dist or 0.5) * 2.0)
        w = float(self.semitone_dist or 0.5) * 2.0
        dot_d = w * 0.35
        cy = y1 + (w / 2.0)
        fill = (1.0, 1.0, 1.0, 1.0) if (n.pitch in BLACK_KEYS) else self.notation_color
        du.add_oval(
            x - dot_d / 3.0,
            cy - dot_d / 3.0,
            x + dot_d / 3.0,
            cy + dot_d / 3.0,
            stroke_color=None,
            fill_color=fill,
            id=0,
            tags=["left_dot"],
        )

    def _editor_background_rgba(self) -> Tuple[float, float, float, float]:
        """Return the editor background as RGBA floats (0..1), alpha=1.0.

        Reads from Style.get_editor_background_color() without instantiating Style
        to avoid side effects.
        """
        from ui.style import Style
        rgb = Style.get_editor_background_color()
        r, g, b = tuple(int(c) for c in rgb)
        return (r / 255.0, g / 255.0, b / 255.0, 1.0)

    def _black_note_above_stem(self, n, layout) -> bool:
        rule = str(getattr(layout, 'black_note_rule', 'below_stem') or 'below_stem')
        if rule == 'above_stem':
            return True
        try:
            cache = cast("Editor", self)._draw_cache or {}
            notes_view = cache.get('notes_view') or (self._cached_notes_view or [])
        except Exception:
            notes_view = self._cached_notes_view or []
        t0 = float(getattr(n, 'time', 0.0) or 0.0)
        p0 = int(getattr(n, 'pitch', 0) or 0)
        if rule in ('above_stem_if_collision', 'only_above_stem_if_collision'):
            for m in notes_view:
                if getattr(m, '_id', None) == getattr(n, '_id', None):
                    continue
                if not self._time_op.eq(float(getattr(m, 'time', 0.0) or 0.0), t0):
                    continue
                if abs(int(getattr(m, 'pitch', 0) or 0) - p0) == 1:
                    return True
            return False
        if rule == 'above_stem_if_chord_and_white_note':
            for m in notes_view:
                if getattr(m, '_id', None) == getattr(n, '_id', None):
                    continue
                if not self._time_op.eq(float(getattr(m, 'time', 0.0) or 0.0), t0):
                    continue
                mp = int(getattr(m, 'pitch', 0) or 0)
                if mp not in BLACK_KEYS and mp != p0:
                    return True
            return False
        if rule != 'above_stem_if_chord_and_white_note_same_hand':
            return False
        hand0 = str(getattr(n, 'hand', '<') or '<')
        for m in notes_view:
            if getattr(m, '_id', None) == getattr(n, '_id', None):
                continue
            if not self._time_op.eq(float(getattr(m, 'time', 0.0) or 0.0), t0):
                continue
            if str(getattr(m, 'hand', '<') or '<') != hand0:
                continue
            mp = int(getattr(m, 'pitch', 0) or 0)
            if mp not in BLACK_KEYS and mp != p0:
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
        hand = getattr(n, 'hand', '<')
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
            if m._id == n._id or getattr(m, 'hand', '<') != hand:
                continue
            delta = float(m.time) - end
            if delta >= -thr:
                min_delta = delta
                break
        if min_delta is None:
            return True
        return op.gt(float(min_delta), 0.0)
