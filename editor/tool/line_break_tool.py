from __future__ import annotations
from typing import Optional, Tuple

from PySide6 import QtWidgets

from editor.tool.base_tool import BaseTool
from file_model.events.line_break import LineBreak
from utils.CONSTANT import QUARTER_NOTE_UNIT
from utils.operator import Operator


class LineBreakTool(BaseTool):
    TOOL_NAME = 'line_break'

    def __init__(self) -> None:
        super().__init__()
        self._drag_target: Optional[LineBreak] = None
        self._dialog_open: bool = False

    def toolbar_spec(self) -> list[dict]:
        return [
            
        ]

    def on_toolbar_button(self, name: str) -> None:
        if name == 'quick_line_breaks':
            self._open_line_break_dialog(None)
            return

    def _time_tol_ticks(self, tol_mm: float = 5.0) -> float:
        if self._editor is None:
            return 0.0
        score = self._editor.current_score()
        zpq = float(getattr(score.editor, 'zoom_mm_per_quarter', 25.0) or 25.0)
        return (float(tol_mm) / max(1e-6, zpq)) * float(QUARTER_NOTE_UNIT)

    def _marker_positions_mm(self) -> Tuple[float, float]:
        if self._editor is None:
            return (0.0, 0.0)
        try:
            du = self._editor.draw_util()
            page_w_mm, _ = du.current_page_size_mm()
        except Exception:
            page_w_mm = 210.0
        margin = float(getattr(self._editor, 'margin', 0.0) or 0.0)
        semitone_dx = float(getattr(self._editor, 'semitone_dist', 2.5) or 2.5)
        stave_right = float(page_w_mm) - margin
        marker_offset = max(2.0, semitone_dx * 0.8)
        marker_x = min(stave_right + marker_offset, float(page_w_mm) - (margin * 0.4))
        if marker_x <= stave_right:
            marker_x = stave_right + marker_offset
        return (stave_right, marker_x)

    def _build_measure_starts(self) -> list[float]:
        if self._editor is None:
            return []
        score = self._editor.current_score()
        starts: list[float] = [0.0]
        cursor = 0.0
        for bg in list(getattr(score, 'base_grid', []) or []):
            try:
                numer = int(getattr(bg, 'numerator', 4) or 4)
                denom = int(getattr(bg, 'denominator', 4) or 4)
                measures = int(getattr(bg, 'measure_amount', 1) or 1)
            except Exception:
                continue
            if measures <= 0:
                continue
            measure_len = float(numer) * (4.0 / float(max(1, denom))) * float(QUARTER_NOTE_UNIT)
            for _ in range(measures):
                cursor += float(measure_len)
                starts.append(float(cursor))
        return starts

    def _hit_test_line_break(self, x_px: float, y_px: float) -> Optional[LineBreak]:
        if self._editor is None:
            return None
        score = self._editor.current_score()
        events = list(getattr(score.events, 'line_break', []) or [])
        if not events:
            return None
        tol_ticks = self._time_tol_ticks()
        click_time = float(self._editor.y_to_time(y_px))
        cursor_time = getattr(self._editor, 'time_cursor', None)
        if cursor_time is not None:
            try:
                click_time = float(cursor_time)
            except Exception:
                pass
        op = Operator(threshold=float(tol_ticks))
        w_px_per_mm = float(getattr(self._editor, '_widget_px_per_mm', 1.0) or 1.0)
        x_mm = float(x_px) / max(1e-6, w_px_per_mm)
        y_mm = float(y_px) / max(1e-6, w_px_per_mm) + float(getattr(self._editor, '_view_y_mm_offset', 0.0) or 0.0)
        stave_right, marker_x = self._marker_positions_mm()
        try:
            du = self._editor.draw_util()
        except Exception:
            du = None
        if du is not None:
            try:
                page_w_mm, _ = du.current_page_size_mm()
            except Exception:
                page_w_mm = float(stave_right + float(getattr(self._editor, 'margin', 0.0) or 0.0))
        else:
            page_w_mm = float(stave_right + float(getattr(self._editor, 'margin', 0.0) or 0.0))
        try:
            from fonts import register_font_from_bytes
            font_family = register_font_from_bytes('C059') or 'C059'
        except Exception:
            font_family = 'C059'
        # Prefer hits on the marker region using snapped time; fallback to nearest by time
        best = None
        best_dt = None
        for lb in events:
            try:
                t0 = float(getattr(lb, 'time', 0.0) or 0.0)
                is_page = bool(getattr(lb, 'page_break', False))
            except Exception:
                continue
            if not op.eq(t0, click_time):
                continue
            label = 'P' if is_page else 'L'
            try:
                if du is not None:
                    _xb, _yb, w_mm, h_mm = du._get_text_extents_mm(label, font_family, 18.0, False, True)
                else:
                    w_mm, h_mm = (6.0, 6.0)
            except Exception:
                w_mm, h_mm = (6.0, 6.0)
            rect_w = max(6.0, float(w_mm) + 4.0)
            rect_x1 = 0.0
            rect_x2 = rect_x1 + rect_w
            if rect_x1 <= x_mm <= rect_x2:
                dt = abs(click_time - t0)
                if best is None or dt < best_dt:
                    best = lb
                    best_dt = dt
        if best is not None:
            return best
        # Fallback: nearest by time within tolerance
        nearest = None
        nearest_dt = None
        for lb in events:
            try:
                t0 = float(getattr(lb, 'time', 0.0) or 0.0)
            except Exception:
                continue
            dt = abs(click_time - t0)
            if dt <= tol_ticks and (nearest_dt is None or dt < nearest_dt):
                nearest = lb
                nearest_dt = dt
        return nearest

    def _sort_line_breaks(self) -> None:
        if self._editor is None:
            return
        score = self._editor.current_score()
        try:
            score.events.line_break.sort(key=lambda lb: float(getattr(lb, 'time', 0.0) or 0.0))
        except Exception:
            pass

    def _is_time_zero(self, t: float) -> bool:
        return abs(float(t)) <= self._time_tol_ticks()

    def _cursor_time(self, y: float) -> float:
        if self._editor is None:
            return float(y)
        tc = getattr(self._editor, 'time_cursor', None)
        if tc is not None:
            try:
                return float(tc)
            except Exception:
                pass
        return float(self._editor.y_to_time(y))

    def on_left_click(self, x: float, y: float) -> None:
        super().on_left_click(x, y)
        if self._editor is None:
            return
        score = self._editor.current_score()
        click_time = self._cursor_time(y)
        hit = self._hit_test_line_break(x, y)
        if hit is not None:
            try:
                hit.page_break = not bool(getattr(hit, 'page_break', False))
                if hasattr(self._editor, '_snapshot_if_changed'):
                    self._editor._snapshot_if_changed(coalesce=False, label='line_break_toggle')
            except Exception:
                pass
            try:
                if hasattr(self._editor, 'force_redraw_from_model'):
                    self._editor.force_redraw_from_model()
                else:
                    self._editor.draw_frame()
            except Exception:
                pass
            try:
                self._editor.score_changed.emit()
            except Exception:
                pass
            return
        if self._is_time_zero(click_time):
            # Never insert at time 0
            return

        # Always insert with dataclass defaults (auto range enabled on new line breaks).
        defaults = LineBreak()
        margin_mm = list(defaults.margin_mm)
        stave_range = defaults.stave_range

        try:
            score.new_line_break(time=click_time, margin_mm=margin_mm, stave_range=stave_range, page_break=False)
            self._sort_line_breaks()
            self._editor._snapshot_if_changed(coalesce=False, label='line_break_insert')
            if hasattr(self._editor, 'force_redraw_from_model'):
                self._editor.force_redraw_from_model()
            else:
                self._editor.draw_frame()
        except Exception:
            pass

    def _open_line_break_dialog(self, lb: LineBreak | None) -> None:
        if self._editor is None:
            return
        if self._dialog_open:
            return
        self._dialog_open = True
        from ui.widgets.line_break_dialog import LineBreakDialog
        parent_w = QtWidgets.QApplication.activeWindow() if hasattr(QtWidgets, 'QApplication') else None
        score = self._editor.current_score()

        def _apply_dialog_values() -> None:
            if hasattr(self._editor, 'force_redraw_from_model'):
                self._editor.force_redraw_from_model()
            else:
                self._editor.draw_frame()
            try:
                self._editor.score_changed.emit()
            except Exception:
                pass

        dlg = LineBreakDialog(
            parent=parent_w,
            score=score,
            selected_line_break=lb,
            measure_resolver=(lambda t: self._editor.get_measure_index_for_time(t)) if hasattr(self._editor, 'get_measure_index_for_time') else None,
            on_change=_apply_dialog_values,
        )

        def _finalize_dialog(result: int) -> None:
            try:
                if result == QtWidgets.QDialog.Accepted:
                    self._editor._snapshot_if_changed(coalesce=False, label='line_break_edit')
            finally:
                self._dialog_open = False

        dlg.finished.connect(_finalize_dialog)
        dlg.show()

    def on_left_drag_start(self, x: float, y: float) -> None:
        super().on_left_drag_start(x, y)
        hit = self._hit_test_line_break(x, y)
        if hit is None:
            self._drag_target = None
            return
        if self._is_time_zero(float(getattr(hit, 'time', 0.0) or 0.0)):
            self._drag_target = None
            return
        self._drag_target = hit

    def on_left_drag(self, x: float, y: float, dx: float, dy: float) -> None:
        super().on_left_drag(x, y, dx, dy)
        if self._editor is None or self._drag_target is None:
            return
        score = self._editor.current_score()
        new_time = self._cursor_time(y)
        if self._is_time_zero(new_time):
            return
        tol_ticks = self._time_tol_ticks()
        try:
            for lb in list(getattr(score.events, 'line_break', []) or []):
                if lb is self._drag_target:
                    continue
                t0 = float(getattr(lb, 'time', 0.0) or 0.0)
                if abs(new_time - t0) <= tol_ticks:
                    return
        except Exception:
            pass
        try:
            self._drag_target.time = float(new_time)
            self._sort_line_breaks()
            if hasattr(self._editor, 'force_redraw_from_model'):
                self._editor.force_redraw_from_model()
            else:
                self._editor.draw_frame()
        except Exception:
            pass

    def on_left_drag_end(self, x: float, y: float) -> None:
        super().on_left_drag_end(x, y)
        if self._drag_target is not None and self._editor is not None:
            try:
                self._editor._snapshot_if_changed(coalesce=False, label='line_break_move')
            except Exception:
                pass
        self._drag_target = None

    def on_right_click(self, x: float, y: float) -> None:
        super().on_right_click(x, y)
        if self._editor is None:
            return
        score = self._editor.current_score()
        hit = self._hit_test_line_break(x, y)
        if hit is None:
            return
        if self._is_time_zero(float(getattr(hit, 'time', 0.0) or 0.0)):
            return
        try:
            score.events.line_break = [lb for lb in list(getattr(score.events, 'line_break', []) or []) if lb is not hit]
            self._editor._snapshot_if_changed(coalesce=False, label='line_break_delete')
            if hasattr(self._editor, 'force_redraw_from_model'):
                self._editor.force_redraw_from_model()
            else:
                self._editor.draw_frame()
        except Exception:
            pass
