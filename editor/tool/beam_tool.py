from typing import Optional, Tuple
from editor.tool.base_tool import BaseTool
from file_model.SCORE import SCORE
from utils.CONSTANT import QUARTER_NOTE_UNIT, SHORTEST_DURATION
from utils.operator import Operator


class BeamTool(BaseTool):
    TOOL_NAME = 'beam'

    def __init__(self):
        super().__init__()
        self._drag_marker = None
        self._drag_start_time: float = 0.0
        self._drag_press_time: float = 0.0
        self._drag_initial_duration: float = 0.0
        self._hand: str = 'l'
        self._press_hit = None
        self._min_duration_val: float = 8.0

    # ---- Helpers ----
    def _score(self) -> Optional[SCORE]:
        try:
            return self._editor.current_score()
        except Exception:
            return None

    def _hand_for_xy(self, x: float, y: float) -> str:
        try:
            pitch = float(self._editor.widget_px_to_pitch(x, y))
        except Exception:
            pitch = getattr(self._editor, 'pitch_cursor', None)
        if pitch is None:
            return self._hand
        return 'l' if float(pitch) <= 40.0 else 'r'

    def _current_hand(self, x: float | None = None) -> str:
        if x is not None:
            return self._hand_for_x(x)
        pitch = getattr(self._editor, 'pitch_cursor', None)
        if pitch is None:
            return self._hand
        return 'l' if float(pitch) <= 40.0 else 'r'

    def _barlines(self) -> list[float]:
        score = self._score()
        if score is None:
            return []
        bars: list[float] = []
        cur = 0.0
        for bg in getattr(score, 'base_grid', []) or []:
            numer = int(getattr(bg, 'numerator', 4) or 4)
            denom = int(getattr(bg, 'denominator', 4) or 4)
            measure_len = float(numer) * (4.0 / float(denom)) * float(QUARTER_NOTE_UNIT)
            for _ in range(int(getattr(bg, 'measure_amount', 1) or 1)):
                bars.append(float(cur))
                cur += measure_len
        bars.append(float(cur))
        return bars

    def _override_window(self, t: float, dur: float, barlines: list[float]) -> Tuple[float, float]:
        if not barlines:
            return (t, t + max(0.0, dur))
        op = Operator(float(SHORTEST_DURATION))
        start_bar = barlines[0]
        for b in barlines:
            if op.ge(t, b):
                start_bar = b
            else:
                break
        end_target = t + max(0.0, dur)
        end_bar = barlines[-1]
        for b in barlines:
            if op.gt(b, end_target):
                end_bar = b
                break
        if op.le(end_bar, start_bar) and len(barlines) > 1:
            for b in barlines:
                if op.gt(b, start_bar):
                    end_bar = b
                    break
        return (float(start_bar), float(end_bar))

    def _find_hit(self, hand: str, t_raw: float, markers: list):
        op = Operator(float(SHORTEST_DURATION))
        for mk in markers:
            if str(getattr(mk, 'hand', 'l')) != hand:
                continue
            mt = float(getattr(mk, 'time', 0.0) or 0.0)
            dur = float(getattr(mk, 'duration', 0.0) or 0.0)
            end = mt + max(0.0, dur)
            # Inside when raw time is within [mt, end) using thresholded comparator.
            if op.ge(t_raw, mt) and op.lt(t_raw, end):
                return mk
        return None

    def _default_duration(self) -> float:
        units = float(getattr(self._editor, 'snap_size_units', 0.0) or 0.0)
        base = units if units > 0 else float(QUARTER_NOTE_UNIT)
        return max(self._min_duration(), base)

    def _min_duration(self) -> float:
        return float(self._min_duration_val)

    # ---- Events ----
    def on_left_press(self, x: float, y: float) -> None:
        super().on_left_press(x, y)
        score = self._score()
        if score is None:
            return
        events = self._editor.current_events(score)
        if events is None:
            return
        self._hand = self._hand_for_xy(x, y)
        t_raw = float(self._editor.widget_px_to_time(x, y))
        t_snap = float(self._editor.snap_time(t_raw))
        markers = list(getattr(events, 'beam', []) or [])
        hit = self._find_hit(self._hand, t_raw, markers)
        self._press_hit = hit
        if hit is not None:
            self._drag_marker = hit
            mt = float(getattr(hit, 'time', t_snap) or t_snap)
            self._drag_start_time = float(self._editor.snap_time(mt))
            self._drag_press_time = t_snap
            self._drag_initial_duration = max(self._min_duration(), float(getattr(hit, 'duration', 0.0) or 0.0))
        else:
            dur = self._default_duration()
            mk = score.new_beam(time=t_snap, duration=dur, hand=self._hand)
            self._drag_marker = mk
            self._drag_start_time = t_snap
            self._drag_press_time = t_snap
            self._drag_initial_duration = dur

    def on_left_drag(self, x: float, y: float, dx: float, dy: float) -> None:
        super().on_left_drag(x, y, dx, dy)
        score = self._score()
        if score is None:
            return
        mk = self._drag_marker
        if mk is None:
            return
        t_raw = float(self._editor.widget_px_to_time(x, y))
        t_snap = float(self._editor.snap_time(t_raw))
        base_duration = float(self._drag_initial_duration)
        mk.duration = max(self._min_duration(), base_duration + (t_snap - float(self._drag_press_time)))
        if hasattr(self._editor, 'force_redraw_from_model'):
            self._editor.force_redraw_from_model()
        else:
            self._editor.draw_frame()

    def on_left_unpress(self, x: float, y: float) -> None:
        super().on_left_unpress(x, y)
        score = self._score()
        if score is None:
            return
        if self._drag_marker is not None:
            try:
                self._editor._snapshot_if_changed(coalesce=True, label='beam_edit')
            except Exception:
                pass
        self._drag_marker = None
        self._press_hit = None

    def on_left_drag_end(self, x: float, y: float) -> None:
        super().on_left_drag_end(x, y)
        score = self._score()
        if score is None:
            return
        if self._drag_marker is not None:
            try:
                self._editor._snapshot_if_changed(coalesce=True, label='beam_edit')
            except Exception:
                pass
        self._drag_marker = None
        self._press_hit = None

    def on_left_click(self, x: float, y: float) -> None:
        super().on_left_click(x, y)
        # Creation happens on press; no extra click behavior

    def on_right_click(self, x: float, y: float) -> None:
        super().on_right_click(x, y)
        score = self._score()
        if score is None:
            return
        events = self._editor.current_events(score)
        if events is None:
            return
        hand = self._hand_for_xy(x, y)
        markers = list(getattr(events, 'beam', []) or [])
        t_raw = float(self._editor.widget_px_to_time(x, y))
        target = self._find_hit(hand, t_raw, markers)
        if target is not None:
            try:
                events.beam.remove(target)
            except ValueError:
                tid = int(getattr(target, '_id', -1) or -1)
                events.beam = [m for m in markers if int(getattr(m, '_id', -2) or -2) != tid]
            try:
                self._editor._snapshot_if_changed(coalesce=True, label='beam_delete')
            except Exception:
                pass
        else:
            # create new marker at snapped cursor time (prefer time_cursor)
            t_snap = float(getattr(self._editor, 'time_cursor', None)) if getattr(self._editor, 'time_cursor', None) is not None else float(self._editor.snap_time(t_raw))
            dur = self._default_duration()
            score.new_beam(time=t_snap, duration=dur, hand=hand)
            try:
                self._editor._snapshot_if_changed(coalesce=True, label='beam_create')
            except Exception:
                pass
        if hasattr(self._editor, 'force_redraw_from_model'):
            self._editor.force_redraw_from_model()
        else:
            self._editor.draw_frame()
