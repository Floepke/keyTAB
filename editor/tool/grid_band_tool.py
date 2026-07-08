from typing import Optional, Tuple
import math
from editor.tool.base_tool import BaseTool
from file_model.SCORE import SCORE
from file_model.events.grid_band import GridBand
from utils.CONSTANT import QUARTER_NOTE_UNIT, SHORTEST_DURATION
from utils.operator import Operator


class GridBandTool(BaseTool):
    TOOL_NAME = 'grid_band'

    def __init__(self):
        super().__init__()
        self._op = Operator()
        self._dur_op = Operator(float(SHORTEST_DURATION))
        self._drag_marker = None
        self._drag_start_time: float = 0.0
        self._drag_press_time: float = 0.0
        self._drag_initial_duration: float = 0.0
        self._hand: str = 'l'
        self._press_hit = None
        self._min_duration_val: float = 8.0
        self._drag_markers: dict[str, GridBand] = {}
        self._drag_initial_durations: dict[str, float] = {}
        self._drag_hands: list[str] = []

    def toolbar_spec(self) -> list[dict]:
        return [
            {
                'name': 'clear_all_grid_bands',
                'text': 'X',
                'icon': '',
                'tooltip': 'Clear all grid band markers.',
            }
        ]

    def on_toolbar_button(self, name: str) -> None:
        if name != 'clear_all_grid_bands':
            return
        score = self._score()
        if score is None:
            return
        layout = getattr(score, 'layout', None)
        if layout is None:
            return

        had_any = bool(list(getattr(layout, 'grid_band_track', []) or []))
        if not had_any:
            had_any = bool(list(getattr(layout, 'grid_band_left_track', []) or []))
        if not had_any:
            had_any = bool(list(getattr(layout, 'grid_band_right_track', []) or []))

        setattr(layout, 'grid_band_track', [])
        try:
            setattr(layout, 'grid_band_left_track', [])
            setattr(layout, 'grid_band_right_track', [])
        except Exception:
            pass

        self._drag_marker = None
        self._drag_markers = {}
        self._drag_initial_durations = {}
        self._drag_hands = []
        self._press_hit = None

        if self._editor is not None and had_any:
            self._editor._snapshot_if_changed(coalesce=True, label='grid_band_clear_all')
        elif self._editor is not None:
            if hasattr(self._editor, 'force_redraw_from_model'):
                self._editor.force_redraw_from_model()
            else:
                self._editor.draw_frame()

    # ---- Helpers ----
    def _score(self) -> Optional[SCORE]:
        try:
            return self._editor.current_score()
        except Exception:
            return None

    def _current_hand(self) -> str:
        # Single-track mode: hand is irrelevant; keep a fixed marker.
        return 'l'

    def _hand_for_x(self, _x: float) -> str:
        # Single-track mode: always use one shared track.
        return 'l'

    def _hands_for_x(self, _x: float) -> list[str]:
        # Single-track mode: operate on a single track.
        return ['*']

    def _get_hand_tracks(self, _hand: str) -> Tuple[str, list]:
        """Get the single grid band track (merging legacy tracks when present)."""
        score = self._score()
        if score is None:
            return (None, [])
        layout = getattr(score, 'layout', None)
        if layout is None:
            return (None, [])

        raw_markers = list(getattr(layout, 'grid_band_track', []) or [])
        if not raw_markers:
            legacy_left = list(getattr(layout, 'grid_band_left_track', []) or [])
            legacy_right = list(getattr(layout, 'grid_band_right_track', []) or [])
            raw_markers = legacy_left + legacy_right
        markers: list[GridBand] = []
        changed = False
        for mk in raw_markers:
            if isinstance(mk, GridBand):
                markers.append(mk)
                continue
            if isinstance(mk, dict):
                try:
                    markers.append(
                        GridBand(
                            time=float(mk.get('time', 0.0) or 0.0),
                            duration=float(mk.get('duration', 0.0) or 0.0),
                            _id=int(mk.get('_id', mk.get('id', 0)) or 0),
                        )
                    )
                    changed = True
                    continue
                except Exception:
                    changed = True
                    continue
            changed = True

        if changed:
            setattr(layout, 'grid_band_track', markers)
            try:
                setattr(layout, 'grid_band_left_track', [])
                setattr(layout, 'grid_band_right_track', [])
            except Exception:
                pass
        return ('grid_band_track', markers)

    def _marker_field(self, marker, name: str, default):
        if isinstance(marker, dict):
            return marker.get(name, default)
        return getattr(marker, name, default)

    def _set_hand_track(self, _hand: str, markers: list[GridBand]) -> None:
        score = self._score()
        if score is None:
            return
        layout = getattr(score, 'layout', None)
        if layout is None:
            return
        setattr(layout, 'grid_band_track', markers)
        try:
            setattr(layout, 'grid_band_left_track', [])
            setattr(layout, 'grid_band_right_track', [])
        except Exception:
            pass

    def _normalize_markers(self, markers: list[GridBand], active: GridBand | None = None) -> list[GridBand]:
        """Return a valid marker list: sorted, de-duplicated, and overlap-pruned."""
        valid: list[GridBand] = []
        for mk in markers:
            try:
                mt = float(self._marker_field(mk, 'time', 0.0) or 0.0)
                dur = float(self._marker_field(mk, 'duration', 0.0) or 0.0)
            except Exception:
                continue
            if float(dur) < 0.0:
                continue
            if float(mt) < 0.0:
                continue
            valid.append(mk)

        valid.sort(
            key=lambda m: (
                float(self._marker_field(m, 'time', 0.0) or 0.0),
                int(self._marker_field(m, '_id', 0) or 0),
            )
        )

        # Prevent double markers at the same time; keep active marker when applicable.
        deduped: list[GridBand] = []
        for mk in valid:
            mt = float(self._marker_field(mk, 'time', 0.0) or 0.0)
            if not deduped:
                deduped.append(mk)
                continue
            last = deduped[-1]
            last_t = float(self._marker_field(last, 'time', 0.0) or 0.0)
            if self._op.eq(mt, last_t):
                if active is mk and active is not last:
                    deduped[-1] = mk
                continue
            deduped.append(mk)

        # If a marker is expanded, remove any future markers it overlaps.
        if active is not None and active in deduped:
            a_start = float(self._marker_field(active, 'time', 0.0) or 0.0)
            a_dur = float(self._marker_field(active, 'duration', 0.0) or 0.0)
            a_end = a_start + max(0.0, a_dur)
            pruned: list[GridBand] = []
            for mk in deduped:
                if mk is active:
                    pruned.append(mk)
                    continue
                mt = float(self._marker_field(mk, 'time', 0.0) or 0.0)
                if self._op.gt(mt, a_start) and self._op.lt(mt, a_end):
                    continue
                pruned.append(mk)
            deduped = pruned

        deduped.sort(
            key=lambda m: (
                float(self._marker_field(m, 'time', 0.0) or 0.0),
                int(self._marker_field(m, '_id', 0) or 0),
            )
        )

        # Remove unnecessary future markers when duration does not change.
        # Keep the first marker in a same-duration run; drop later ones.
        # During active drag, preserve the active marker, but still prune others.
        compressed: list[GridBand] = []
        for mk in deduped:
            if not compressed:
                compressed.append(mk)
                continue
            prev = compressed[-1]
            prev_d = float(self._marker_field(prev, 'duration', 0.0) or 0.0)
            cur_d = float(self._marker_field(mk, 'duration', 0.0) or 0.0)
            if self._dur_op.eq(cur_d, prev_d):
                if active is not None and mk is active:
                    compressed.append(mk)
                continue
            compressed.append(mk)

        deduped = compressed
        return deduped

    def _find_controller_marker(self, t_raw: float, markers: list[GridBand]) -> Optional[GridBand]:
        """Find marker controlling time by marker order range [time, next_marker_time)."""
        track = [m for m in markers if not self._op.lt(float(self._marker_field(m, 'duration', 0.0) or 0.0), 0.0)]
        track.sort(
            key=lambda m: (
                float(self._marker_field(m, 'time', 0.0) or 0.0),
                int(self._marker_field(m, '_id', 0) or 0),
            )
        )
        for i, mk in enumerate(track):
            mt = float(self._marker_field(mk, 'time', 0.0) or 0.0)
            next_t = float(self._marker_field(track[i + 1], 'time', 0.0) or 0.0) if (i + 1) < len(track) else float('inf')
            dur = float(self._marker_field(mk, 'duration', 0.0) or 0.0)
            if self._op.le(dur, 0.0):
                continue
            if self._op.ge(t_raw, mt) and self._op.lt(t_raw, next_t):
                return mk
        return None

    def _band_start_for_time(self, t_snap: float, controller: GridBand) -> float:
        """Return start of alternating band (dark or light) containing t_snap."""
        c_start = float(self._marker_field(controller, 'time', 0.0) or 0.0)
        c_step = float(self._marker_field(controller, 'duration', 0.0) or 0.0)
        if self._op.le(c_step, 0.0):
            return float(t_snap)

        # Robust against floating-point boundary drift (e.g. 85.3333 * n).
        # Without tolerance, a click exactly on a boundary can be interpreted
        # as the previous band due to tiny rounding errors.
        tol = max(float(self._op.threshold), 1e-9)
        ratio = (float(t_snap) - c_start) / c_step
        n = int(math.floor(ratio + (tol / max(c_step, 1e-9))))
        if n < 0:
            n = 0

        # Clamp/adjust around boundaries using Operator tolerance.
        while self._op.gt(c_start + (float(n) * c_step), float(t_snap)) and n > 0:
            n -= 1
        while self._op.le(c_start + (float(n + 1) * c_step), float(t_snap)):
            n += 1

        return float(c_start + (float(n) * c_step))

    def _normalize_and_store(self, hand: str, active: GridBand | None = None) -> tuple[str, list[GridBand]]:
        track_name, markers = self._get_hand_tracks(hand)
        if track_name is None:
            return (None, [])
        normalized = self._normalize_markers(markers, active=active)
        self._set_hand_track(hand, normalized)
        return (track_name, normalized)

    def _find_marker_at_time(self, markers: list, t: float) -> Optional[GridBand]:
        for mk in markers:
            mt = float(self._marker_field(mk, 'time', 0.0) or 0.0)
            if self._op.eq(mt, float(t)):
                return mk
        return None

    def _find_hit(self, hand: str, t_raw: float, markers: list) -> Optional[GridBand]:
        track = [m for m in markers if self._op.gt(float(self._marker_field(m, 'duration', 0.0) or 0.0), 0.0)]
        track.sort(
            key=lambda m: (
                float(self._marker_field(m, 'time', 0.0) or 0.0),
                int(self._marker_field(m, '_id', 0) or 0),
            )
        )
        for mk in track:
            mt = float(self._marker_field(mk, 'time', 0.0) or 0.0)
            dur = float(self._marker_field(mk, 'duration', 0.0) or 0.0)
            end = mt + max(0.0, dur)
            # Marker editable only inside its own band [time, time + duration)
            if self._op.ge(t_raw, mt) and self._op.lt(t_raw, end):
                return mk
        return None

    def _default_duration(self) -> float:
        units = float(getattr(self._editor, 'snap_size_units', 0.0) or 0.0)
        base = units if self._op.gt(units, 0.0) else float(QUARTER_NOTE_UNIT)
        return max(self._min_duration(), base)

    def _min_duration(self) -> float:
        snap_units = float(getattr(self._editor, 'snap_size_units', 0.0) or 0.0)
        return snap_units if self._op.gt(snap_units, 0.0) else float(self._min_duration_val)

    def _next_id(self, markers: list) -> int:
        """Get the next available ID for a new marker."""
        if not markers:
            return 1
        max_id = max(int(self._marker_field(m, '_id', 0) or 0) for m in markers)
        return max_id + 1

    def _all_barlines(self) -> list[float]:
        score = self._score()
        if score is None:
            return [0.0]
        bars: list[float] = []
        cur = 0.0
        for bg in getattr(score, 'base_grid', []) or []:
            numer = int(getattr(bg, 'numerator', 4) or 4)
            denom = int(getattr(bg, 'denominator', 4) or 4)
            measure_len = float(numer) * (4.0 / float(max(1, denom))) * float(QUARTER_NOTE_UNIT)
            mcount = int(getattr(bg, 'measure_amount', 1) or 1)
            for _ in range(max(1, mcount)):
                bars.append(float(cur))
                cur += measure_len
        bars.append(float(cur))
        out = sorted(list(dict.fromkeys(round(float(v), 6) for v in bars)))
        return out if out else [0.0]

    def _next_barline_after(self, t: float) -> float:
        op = self._op
        bars = self._all_barlines()
        tf = float(t)
        for b in bars:
            bf = float(b)
            if op.gt(bf, tf):
                return bf
        return float(bars[-1]) if bars else tf

    # ---- Events ----
    def on_left_press(self, x: float, y: float) -> None:
        super().on_left_press(x, y)
        score = self._score()
        if score is None:
            return
        hands = self._hands_for_x(x)
        self._drag_hands = list(hands)
        self._drag_markers = {}
        self._drag_initial_durations = {}
        self._hand = hands[0] if hands else self._hand_for_x(x)
        t_raw = float(self._editor.widget_px_to_time(x, y))
        t_snap = float(self._editor.snap_time(t_raw))

        first_marker = None
        for hand in hands:
            track_name, markers = self._get_hand_tracks(hand)
            if track_name is None:
                continue

            controller = self._find_controller_marker(t_snap, markers)
            marker_time = float(t_snap)
            base_duration = self._default_duration()

            if controller is not None:
                marker_time = float(self._editor.snap_time(self._band_start_for_time(t_snap, controller)))
                base_duration = max(
                    self._min_duration(),
                    float(self._marker_field(controller, 'duration', base_duration) or base_duration),
                )

            at_start = self._find_marker_at_time(markers, marker_time)
            if at_start is not None and self._op.gt(float(self._marker_field(at_start, 'duration', 0.0) or 0.0), 0.0):
                mk = at_start
                init_dur = max(
                    self._min_duration(),
                    float(self._marker_field(at_start, 'duration', base_duration) or base_duration),
                )
            else:
                mk = GridBand(
                    time=marker_time,
                    duration=base_duration,
                    _id=self._next_id(markers)
                )
                markers.append(mk)
                init_dur = base_duration
                self._set_hand_track(hand, markers)

            self._drag_markers[hand] = mk
            self._drag_initial_durations[hand] = float(init_dur)
            if first_marker is None:
                first_marker = mk

            self._normalize_and_store(hand, active=mk)

        self._drag_marker = first_marker
        self._press_hit = first_marker
        self._drag_start_time = float(self._marker_field(first_marker, 'time', t_snap) or t_snap) if first_marker is not None else t_snap
        self._drag_press_time = t_snap
        self._drag_initial_duration = float(self._drag_initial_durations.get(hands[0], self._default_duration())) if hands else self._default_duration()

    def on_left_drag(self, x: float, y: float, dx: float, dy: float) -> None:
        super().on_left_drag(x, y, dx, dy)
        score = self._score()
        if score is None:
            return
        if not self._drag_markers:
            return
        t_raw = float(self._editor.widget_px_to_time(x, y))
        t_snap = float(self._editor.snap_time(t_raw))
        delta = float(t_snap) - float(self._drag_press_time)
        for hand, mk in list(self._drag_markers.items()):
            base_duration = float(self._drag_initial_durations.get(hand, self._drag_initial_duration))
            proposed = max(self._min_duration(), base_duration + delta)
            mk_start = float(self._marker_field(mk, 'time', 0.0) or 0.0)
            next_bar = float(self._next_barline_after(mk_start))
            max_duration = max(0.0, next_bar - mk_start)
            mk.duration = min(proposed, max_duration)

            # Drag edits may overlap future markers; normalize resolves this.
            self._normalize_and_store(hand, active=mk)

        if hasattr(self._editor, 'force_redraw_from_model'):
            self._editor.force_redraw_from_model()
        else:
            self._editor.draw_frame()

    def on_left_unpress(self, x: float, y: float) -> None:
        super().on_left_unpress(x, y)
        score = self._score()
        if score is None:
            return
        if self._drag_markers:
            for hand in self._drag_hands or list(self._drag_markers.keys()):
                self._normalize_and_store(hand)
            self._editor._snapshot_if_changed(coalesce=True, label='grid_band_edit')
        self._drag_marker = None
        self._drag_markers = {}
        self._drag_initial_durations = {}
        self._drag_hands = []
        self._press_hit = None

    def on_left_drag_end(self, x: float, y: float) -> None:
        super().on_left_drag_end(x, y)
        score = self._score()
        if score is None:
            return
        if self._drag_markers:
            for hand in self._drag_hands or list(self._drag_markers.keys()):
                self._normalize_and_store(hand)
            self._editor._snapshot_if_changed(coalesce=True, label='grid_band_edit')
        self._drag_marker = None
        self._drag_markers = {}
        self._drag_initial_durations = {}
        self._drag_hands = []
        self._press_hit = None

    def on_left_click(self, x: float, y: float) -> None:
        super().on_left_click(x, y)
        # Creation happens on press; no extra click behavior

    def on_right_click(self, x: float, y: float) -> None:
        super().on_right_click(x, y)
        score = self._score()
        if score is None:
            return
        hands = self._hands_for_x(x)
        
        # Use click position time and snap it.
        t_raw = float(self._editor.widget_px_to_time(x, y))
        t_snap = float(self._editor.snap_time(t_raw))

        changed_any = False
        deleted_any = False
        for hand in hands:
            track_name, markers = self._get_hand_tracks(hand)
            if track_name is None:
                continue

            # Delete stop marker when clicking its indicator time.
            zero_target = None
            for mk in markers:
                mt = float(self._marker_field(mk, 'time', 0.0) or 0.0)
                md = float(self._marker_field(mk, 'duration', 0.0) or 0.0)
                if self._op.eq(md, 0.0) and self._op.eq(mt, t_snap):
                    zero_target = mk
                    break
            if zero_target is not None:
                try:
                    markers.remove(zero_target)
                except ValueError:
                    pass
                self._set_hand_track(hand, markers)
                self._normalize_and_store(hand)
                changed_any = True
                deleted_any = True
                continue

            # Insert a stop marker at the START of the clicked band.
            controller = self._find_controller_marker(t_raw, markers)
            changed = False
            if controller is not None:
                c_start = float(self._marker_field(controller, 'time', 0.0) or 0.0)
                c_step = float(self._marker_field(controller, 'duration', 0.0) or 0.0)
                if self._op.gt(c_step, 0.0):
                    band_start = float(self._editor.snap_time(self._band_start_for_time(t_raw, controller)))
                    same_time = None
                    for mk in markers:
                        mt = float(self._marker_field(mk, 'time', 0.0) or 0.0)
                        if self._op.eq(mt, band_start):
                            same_time = mk
                            break
                    if same_time is not None:
                        try:
                            if self._op.not_equal(float(self._marker_field(same_time, 'duration', 0.0) or 0.0), 0.0):
                                same_time.duration = 0.0
                                changed = True
                        except Exception:
                            pass
                    else:
                        markers.append(
                            GridBand(
                                time=band_start,
                                duration=0.0,
                                _id=self._next_id(markers),
                            )
                        )
                        changed = True
            if changed:
                self._set_hand_track(hand, markers)
                self._normalize_and_store(hand)
                changed_any = True

        if changed_any:
            self._editor._snapshot_if_changed(
                coalesce=True,
                label='grid_band_stop_delete' if deleted_any else 'grid_band_stop_insert'
            )
        if hasattr(self._editor, 'force_redraw_from_model'):
            self._editor.force_redraw_from_model()
        else:
            self._editor.draw_frame()
