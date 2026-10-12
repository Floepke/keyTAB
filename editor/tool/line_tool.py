from __future__ import annotations

from typing import Tuple

from PySide6 import QtWidgets

from editor.tool.base_tool import BaseTool
from file_model.SCORE import SCORE


class LineTool(BaseTool):
    TOOL_NAME = "line"

    def __init__(self) -> None:
        super().__init__()
        self._active_line = None
        self._active_part: str | None = None
        self._drag_anchor: tuple[float, float, float, int, int, float] | None = None

    def _cursor_mm(self, x_px: float, y_px: float) -> Tuple[float, float]:
        return self._editor.widget_px_to_page_mm(float(x_px), float(y_px))

    def _cursor_position(self, x: float, y: float) -> tuple[int, float]:
        x_mm, _ = self._cursor_mm(x, y)
        return self.x_mm_to_rpitch_clamped(x_mm), float(self._editor.snap_time(self._editor.widget_px_to_time(x, y)))

    def _redraw(self) -> None:
        if hasattr(self._editor, "force_redraw_from_model"):
            self._editor.force_redraw_from_model()
        else:
            self._editor.draw_frame()

    @staticmethod
    def _sync_time(line) -> None:
        line.time = min(float(line.time1), float(line.time2))

    def _capture_drag_anchor(self, x: float, y: float) -> None:
        if self._active_line is None:
            return
        x_mm, _ = self._cursor_mm(x, y)
        cursor_time = float(self._editor.widget_px_to_time(x, y))
        self._drag_anchor = (
            x_mm,
            cursor_time,
            float(self._active_line.time1),
            int(self._active_line.rpitch1),
            int(self._active_line.rpitch2),
            float(self._active_line.time2),
        )

    def on_left_press(self, x: float, y: float) -> None:
        super().on_left_press(x, y)
        if self._editor is None:
            return
        score: SCORE = self._editor.current_score()
        events = self._editor.current_events(score) if score is not None else None
        if events is None:
            return
        self._editor.draw_frame()
        x_mm, y_mm = self._cursor_mm(x, y)
        self._active_line, self._active_part = self._editor.hit_test_line_mm(x_mm, y_mm)
        self._drag_anchor = None
        if self._active_line is not None:
            if self._active_part == "line":
                self._capture_drag_anchor(x, y)
            return

        rpitch, time = self._cursor_position(x, y)
        self._active_line = score.new_line(time1=time, time2=time, rpitch1=rpitch, rpitch2=rpitch)
        self._active_part = "end"
        self._redraw()

    def on_left_drag(self, x: float, y: float, dx: float, dy: float) -> None:
        super().on_left_drag(x, y, dx, dy)
        if self._editor is None or self._active_line is None:
            return
        rpitch, time = self._cursor_position(x, y)
        if self._active_part == "start":
            self._active_line.rpitch1 = rpitch
            self._active_line.time1 = time
        elif self._active_part == "end":
            self._active_line.rpitch2 = rpitch
            self._active_line.time2 = time
        elif self._active_part == "line" and self._drag_anchor is not None:
            anchor_x, anchor_time, time1, rpitch1, rpitch2, time2 = self._drag_anchor
            semitone_mm = float(getattr(self._editor, "semitone_dist", 0.0) or 0.0)
            pitch_delta = int(round((self._cursor_mm(x, y)[0] - anchor_x) / semitone_mm)) if semitone_mm > 1e-6 else 0
            min_rpitch, max_rpitch = self.rpitch_bounds()
            pitch_delta = max(min_rpitch - min(rpitch1, rpitch2), min(max_rpitch - max(rpitch1, rpitch2), pitch_delta))
            time_delta = float(self._editor.widget_px_to_time(x, y)) - anchor_time
            self._active_line.rpitch1 = rpitch1 + pitch_delta
            self._active_line.rpitch2 = rpitch2 + pitch_delta
            self._active_line.time1 = float(self._editor.snap_time(time1 + time_delta))
            self._active_line.time2 = float(self._editor.snap_time(time2 + time_delta))
        self._sync_time(self._active_line)
        self._redraw()

    def on_left_unpress(self, x: float, y: float) -> None:
        super().on_left_unpress(x, y)
        self._active_line = None
        self._active_part = None
        self._drag_anchor = None

    def on_left_double_click(self, x: float, y: float) -> None:
        super().on_left_double_click(x, y)
        if self._editor is None:
            return
        self._editor.draw_frame()
        x_mm, y_mm = self._cursor_mm(x, y)
        line, _ = self._editor.hit_test_line_mm(x_mm, y_mm)
        if line is None:
            return

        from ui.dialogs.line_dialog import LineDialog

        dialog = LineDialog(line, parent=getattr(self._editor, "widget", None))
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        for name, value in dialog.values().items():
            setattr(line, name, value)
        self._redraw()
        try:
            self._editor._snapshot_if_changed(coalesce=False, label="line_properties")
        except Exception:
            pass

    def on_right_click(self, x: float, y: float) -> None:
        super().on_right_click(x, y)
        if self._editor is None:
            return
        score: SCORE = self._editor.current_score()
        events = self._editor.current_events(score) if score is not None else None
        if events is None:
            return
        self._editor.draw_frame()
        x_mm, y_mm = self._cursor_mm(x, y)
        line, _ = self._editor.hit_test_line_mm(x_mm, y_mm)
        if line is None:
            return
        try:
            events.line.remove(line)
        except ValueError:
            return
        self._redraw()
