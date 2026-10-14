from __future__ import annotations
from PySide6 import QtWidgets
from typing import Optional
from editor.tool.base_tool import BaseTool
from utils.CONSTANT import QUARTER_NOTE_UNIT, SHORTEST_DURATION
from utils.operator import Operator


class VoltaTool(BaseTool):
    TOOL_NAME = 'volta'

    def __init__(self):
        super().__init__()
        self._active_volta_id: Optional[int] = None
        self._active_time: Optional[float] = None
        self._min_duration: float = 0.0
        self._drag_anchor_y_px: Optional[float] = None
        self._drag_anchor_time: Optional[float] = None
        self._drag_initial_duration: float = 0.0
        self._volta_dialog: Optional[QtWidgets.QDialog] = None

    def toolbar_spec(self) -> list[dict]:
        return []

    def _snap_duration(self) -> float:
        if self._editor is None:
            return float(QUARTER_NOTE_UNIT)
        return max(float(SHORTEST_DURATION), float(getattr(self._editor, 'snap_size_units', QUARTER_NOTE_UNIT)))

    def _find_volta_at_time(self, t: float) -> Optional[object]:
        if self._editor is None:
            return None
        score = self._editor.current_score()
        op = Operator(float(SHORTEST_DURATION))
        for ev in list(getattr(score, 'volta', []) or []):
            ev_time = float(getattr(ev, 'time', 0.0) or 0.0)
            ev_end = ev_time + max(0.0, float(getattr(ev, 'duration', 0.0) or 0.0))
            if op.ge(float(t), ev_time) and op.le(float(t), ev_end):
                return ev
        return None

    def _find_volta_by_id(self, volta_id: int) -> Optional[object]:
        if self._editor is None:
            return None
        score = self._editor.current_score()
        for ev in list(getattr(score, 'volta', []) or []):
            if int(getattr(ev, '_id', -1) or -1) == int(volta_id):
                return ev
        return None

    def _redraw(self) -> None:
        if self._editor is None:
            return
        if hasattr(self._editor, 'force_redraw_from_model'):
            self._editor.force_redraw_from_model()
        else:
            self._editor.draw_frame()

    def _hit_volta(self, x: float, y: float) -> Optional[object]:
        if self._editor is None:
            return None
        volta_id = self._editor.hit_test_volta(x, y) if hasattr(self._editor, 'hit_test_volta') else None
        return self._find_volta_by_id(volta_id) if volta_id is not None else None

    def on_left_click(self, x: float, y: float) -> None:
        if self._editor is None:
            return
        score = self._editor.current_score()
        if score is None:
            return
        existing = self._hit_volta(x, y)
        t = self._editor.snap_time(self._editor.widget_px_to_time(x, y))
        # Clicking an existing bracket opens the properties dialog.
        if existing is not None:
            from ui.widgets.volta_dialog import VoltaDialog
            parent_w = QtWidgets.QApplication.activeWindow()
            dlg = VoltaDialog(
                parent_w,
                initial_number=int(getattr(existing, 'number', 1) or 1),
                initial_x_offset=float(getattr(existing, 'x_offset', 0.0) or 0.0),
            )

            def _apply():
                number, x_offset = dlg.get_values()
                existing.number = int(number)
                existing.x_offset = float(x_offset)
                self._editor._snapshot_if_changed(coalesce=True, label='volta_edit')
                self._redraw()

            dlg.accepted.connect(_apply)
            dlg.raise_()
            dlg.activateWindow()
            self._volta_dialog = dlg
            dlg.show()
            return
        # New brackets start at the current snap duration.
        min_dur = self._snap_duration()
        existing_numbers = [int(getattr(ev, 'number', 1) or 1) for ev in (getattr(score, 'volta', []) or [])]
        number = (max(existing_numbers) + 1) if existing_numbers else 1
        vt = score.new_volta(time=float(t), duration=float(min_dur), number=int(number))
        self._active_volta_id = int(getattr(vt, '_id', 0) or 0)
        self._active_time = float(t)
        self._min_duration = float(min_dur)
        self._drag_initial_duration = float(getattr(vt, 'duration', min_dur) or min_dur)
        self._editor._snapshot_if_changed(coalesce=True, label='volta_create')
        self._redraw()

    def on_left_drag_start(self, x: float, y: float) -> None:
        # Capture the mouse anchor so duration edits track delta movement.
        if self._editor is None:
            return
        self._drag_anchor_y_px = float(y)
        try:
            self._drag_anchor_time = float(self._editor.widget_px_to_time(x, y))
        except Exception:
            self._drag_anchor_time = None

    def on_left_drag(self, x: float, y: float, dx: float, dy: float) -> None:
        if self._editor is None or self._active_volta_id is None or self._active_time is None:
            return
        score = self._editor.current_score()
        anchor_time = self._drag_anchor_time
        if anchor_time is None:
            anchor_time = float(self._editor.widget_px_to_time(x, y))
        try:
            cur_time = float(self._editor.widget_px_to_time(x, y))
        except Exception:
            cur_time = anchor_time
        delta_time = float(cur_time - anchor_time)
        unsnapped_end = float(self._active_time) + float(self._drag_initial_duration) + delta_time
        snap_units = self._snap_duration()
        snapped_end = self._editor.snap_time(float(unsnapped_end + 0.5 * snap_units))
        new_du = max(self._min_duration, float(snapped_end - float(self._active_time)))
        if new_du <= 0.0:
            new_du = self._min_duration
        for ev in list(getattr(score, 'volta', []) or []):
            if int(getattr(ev, '_id', -1) or -1) == int(self._active_volta_id):
                try:
                    ev.duration = float(new_du)
                except Exception:
                    pass
                break
        self._redraw()

    def on_left_drag_end(self, x: float, y: float) -> None:
        if self._editor is None:
            return
        self._editor._snapshot_if_changed(coalesce=True, label='volta_resize')
        self._active_volta_id = None
        self._active_time = None
        self._drag_anchor_y_px = None
        self._drag_anchor_time = None
        self._drag_initial_duration = 0.0

    def on_left_press(self, x: float, y: float) -> None:
        # If pressing on an existing bracket, prepare for a duration drag.
        if self._editor is None:
            return
        existing = self._hit_volta(x, y)
        if existing is None:
            t = self._editor.snap_time(self._editor.widget_px_to_time(x, y))
            existing = self._find_volta_at_time(t)
        if existing is None:
            return
        min_du = self._snap_duration()
        self._active_volta_id = int(getattr(existing, '_id', 0) or 0)
        self._active_time = float(getattr(existing, 'time', 0.0) or 0.0)
        self._min_duration = float(min_du)
        self._drag_initial_duration = float(getattr(existing, 'duration', min_du) or min_du)

    def on_right_unpress(self, x: float, y: float) -> None:
        if self._editor is None:
            return
        score = self._editor.current_score()
        if score is None:
            return
        lst = list(getattr(score, 'volta', []) or [])
        if not lst:
            return
        target = self._hit_volta(x, y)
        kept: list[object] = []
        deleted_any = False
        if target is not None:
            for ev in lst:
                if ev is target:
                    deleted_any = True
                    continue
                kept.append(ev)
        else:
            t = self._editor.snap_time(self._editor.widget_px_to_time(x, y))
            op = Operator(float(SHORTEST_DURATION))
            for ev in lst:
                ev_time = float(getattr(ev, 'time', 0.0) or 0.0)
                ev_end = ev_time + max(0.0, float(getattr(ev, 'duration', 0.0) or 0.0))
                if op.ge(float(t), ev_time) and op.le(float(t), ev_end):
                    deleted_any = True
                    continue
                kept.append(ev)
        if not deleted_any:
            return
        try:
            score.volta = kept
        except Exception:
            return
        self._editor._snapshot_if_changed(coalesce=True, label='volta_delete')
        self._redraw()

    def on_mouse_move(self, x: float, y: float) -> None:
        super().on_mouse_move(x, y)
