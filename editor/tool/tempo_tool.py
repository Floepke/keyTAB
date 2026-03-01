from __future__ import annotations
from PySide6 import QtCore, QtWidgets
from typing import Optional
from editor.tool.base_tool import BaseTool
from utils.CONSTANT import QUARTER_NOTE_UNIT
from utils.operator import Operator


class TempoTool(BaseTool):
    TOOL_NAME = 'tempo'

    def __init__(self):
        super().__init__()
        self._active_tempo_id: Optional[int] = None
        self._active_time: Optional[float] = None
        self._min_duration: float = 0.0
        self._drag_anchor_y_px: Optional[float] = None
        self._drag_anchor_time: Optional[float] = None
        self._drag_initial_duration: float = 0.0

    def toolbar_spec(self) -> list[dict]:
        return []

    def _find_active_ts_at_time(self, t: float) -> tuple[int, int]:
        """Return (numer, denom) for base grid segment active at time t."""
        if self._editor is None:
            return (4, 4)
        score = self._editor.current_score()
        cur_t = 0.0
        for bg in getattr(score, 'base_grid', []) or []:
            numer = int(getattr(bg, 'numerator', 4) or 4)
            denom = int(getattr(bg, 'denominator', 4) or 4)
            measure_len = float(numer) * (4.0 / float(denom)) * float(QUARTER_NOTE_UNIT)
            count = int(getattr(bg, 'measure_amount', 1) or 1)
            seg_len = float(count) * measure_len
            if t < cur_t + seg_len:
                return (numer, denom)
            cur_t += seg_len
        return (4, 4)

    def _beat_length_ticks(self, numer: int, denom: int) -> float:
        measure_len = float(numer) * (4.0 / float(denom)) * float(QUARTER_NOTE_UNIT)
        return measure_len / max(1, int(numer))

    def _find_tempo_at_time(self, t: float) -> Optional[object]:
        if self._editor is None:
            return None
        score = self._editor.current_score()
        op = Operator(threshold=1e-6)
        for ev in list(getattr(score.events, 'tempo', []) or []):
            if op.equal(float(getattr(ev, 'time', 0.0) or 0.0), float(t)):
                return ev
        return None

    def _find_tempo_by_id(self, tempo_id: int) -> Optional[object]:
        if self._editor is None:
            return None
        score = self._editor.current_score()
        for ev in list(getattr(score.events, 'tempo', []) or []):
            if int(getattr(ev, '_id', -1) or -1) == int(tempo_id):
                return ev
        return None

    def on_left_click(self, x: float, y: float) -> None:
        if self._editor is None:
            return
        score = self._editor.current_score()
        tempo_id = self._editor.hit_test_tempo(x, y) if hasattr(self._editor, 'hit_test_tempo') else None
        existing = self._find_tempo_by_id(tempo_id) if tempo_id is not None else None
        t = self._editor.snap_time(self._editor.y_to_time(y))
        # If clicked on existing, edit tempo value
        if existing is not None:
            cur_tempo = int(getattr(existing, 'tempo', 60) or 60)
            parent_w = None
            parent_w = QtWidgets.QApplication.activeWindow()
            dlg = QtWidgets.QDialog(parent_w)
            dlg.setWindowTitle("Edit Tempo")
            dlg.setModal(True)
            dlg.setWindowModality(QtCore.Qt.NonModal)
            lay = QtWidgets.QFormLayout(dlg)
            tempo = QtWidgets.QSpinBox(dlg)
            tempo.setRange(1, 1000)
            tempo.setValue(cur_tempo)
            lay.addRow("This many of these units in one minute:", tempo)
            btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, parent=dlg)
            lay.addRow(btns)
            btns.accepted.connect(dlg.accept)
            btns.rejected.connect(dlg.reject)
            def _apply():
                existing.tempo = int(tempo.value())
                self._editor._snapshot_if_changed(coalesce=True, label='tempo_edit')
                if hasattr(self._editor, 'force_redraw_from_model'):
                    self._editor.force_redraw_from_model()
                else:
                    self._editor.draw_frame()
            dlg.accepted.connect(_apply)
            dlg.raise_()
            dlg.activateWindow()
            self._tempo_dialog = dlg
            dlg.show()
            return
        # Create new tempo with minimum duration = one beat of active time signature
        numer, denom = self._find_active_ts_at_time(t)
        min_dur = self._beat_length_ticks(numer, denom)
        tp = score.new_tempo(time=float(t), duration=float(min_dur), tempo=60)
        self._active_tempo_id = int(getattr(tp, '_id', 0) or 0)
        self._active_time = float(t)
        self._min_duration = float(min_dur)
        self._drag_initial_duration = float(getattr(tp, 'duration', min_dur) or min_dur)
        self._editor._snapshot_if_changed(coalesce=True, label='tempo_create')
        if hasattr(self._editor, 'force_redraw_from_model'):
            self._editor.force_redraw_from_model()
        else:
            self._editor.draw_frame()

    def on_left_drag_start(self, x: float, y: float) -> None:
        # Capture the mouse anchor so duration edits track delta movement.
        if self._editor is None:
            return
        self._drag_anchor_y_px = float(y)
        try:
            self._drag_anchor_time = float(self._editor.y_to_time(y))
        except Exception:
            self._drag_anchor_time = None

    def on_left_drag(self, x: float, y: float, dx: float, dy: float) -> None:
        if self._editor is None or self._active_tempo_id is None or self._active_time is None:
            return
        score = self._editor.current_score()
        anchor_time = self._drag_anchor_time
        if anchor_time is None:
            anchor_time = float(self._editor.y_to_time(y))
        try:
            cur_time = float(self._editor.y_to_time(y))
        except Exception:
            cur_time = anchor_time
        delta_time = float(cur_time - anchor_time)
        unsnapped_end = float(self._active_time) + float(self._drag_initial_duration) + delta_time
        snap_units = max(1e-6, float(getattr(self._editor, 'snap_size_units', QUARTER_NOTE_UNIT)))
        snapped_end = self._editor.snap_time(float(unsnapped_end + 0.5 * snap_units))
        new_du = max(self._min_duration, float(snapped_end - float(self._active_time)))
        if new_du <= 0.0:
            new_du = self._min_duration
        for ev in list(getattr(score.events, 'tempo', []) or []):
            if int(getattr(ev, '_id', -1) or -1) == int(self._active_tempo_id):
                try:
                    ev.duration = float(new_du)
                except Exception:
                    pass
                break
        if hasattr(self._editor, 'force_redraw_from_model'):
            self._editor.force_redraw_from_model()
        else:
            self._editor.draw_frame()

    def on_left_drag_end(self, x: float, y: float) -> None:
        if self._editor is None:
            return
        self._editor._snapshot_if_changed(coalesce=True, label='tempo_resize')
        self._active_tempo_id = None
        self._active_time = None
        self._drag_anchor_y_px = None
        self._drag_anchor_time = None
        self._drag_initial_duration = 0.0

    def on_left_press(self, x: float, y: float) -> None:
        # If pressing on an existing tempo marker, prepare for duration drag
        if self._editor is None:
            return
        score = self._editor.current_score()
        tempo_id = self._editor.hit_test_tempo(x, y) if hasattr(self._editor, 'hit_test_tempo') else None
        existing = self._find_tempo_by_id(tempo_id) if tempo_id is not None else None
        if existing is None:
            t = self._editor.snap_time(self._editor.y_to_time(y))
            existing = self._find_tempo_at_time(t)
        if existing is None:
            return
        numer, denom = self._find_active_ts_at_time(float(getattr(existing, 'time', 0.0) or 0.0))
        min_du = self._beat_length_ticks(numer, denom)
        self._active_tempo_id = int(getattr(existing, '_id', 0) or 0)
        self._active_time = float(getattr(existing, 'time', 0.0) or 0.0)
        self._min_duration = float(min_du)
        self._drag_initial_duration = float(getattr(existing, 'duration', min_du) or min_du)

    def on_right_click(self, x: float, y: float) -> None:
        if self._editor is None:
            return
        score = self._editor.current_score()
        tempo_id = self._editor.hit_test_tempo(x, y) if hasattr(self._editor, 'hit_test_tempo') else None
        t = self._editor.snap_time(self._editor.y_to_time(y))
        op = Operator(threshold=1e-6)
        lst = list(getattr(score.events, 'tempo', []) or [])
        if not lst:
            return
        # Do not delete the first tempo marker (earliest time)
        earliest_time = min(float(getattr(ev, 'time', 0.0) or 0.0) for ev in lst)
        for i, ev in enumerate(lst):
            ev_time = float(getattr(ev, 'time', 0.0) or 0.0)
            ev_id = int(getattr(ev, '_id', -1) or -1)
            hit = tempo_id is not None and ev_id == int(tempo_id)
            same_time = op.equal(ev_time, float(t))
            if hit or same_time:
                # If this event is at the earliest time, skip deletion
                if op.equal(ev_time, earliest_time):
                    return
                try:
                    del lst[i]
                    score.events.tempo = lst
                except Exception:
                    pass
                self._editor._snapshot_if_changed(coalesce=True, label='tempo_delete')
                if hasattr(self._editor, 'force_redraw_from_model'):
                    self._editor.force_redraw_from_model()
                else:
                    self._editor.draw_frame()
                return

    def on_mouse_move(self, x: float, y: float) -> None:
        super().on_mouse_move(x, y)
