from typing import Tuple

from editor.tool.base_tool import BaseTool
from file_model.SCORE import SCORE



class CountLineTool(BaseTool):
    TOOL_NAME = 'count_line'

    def __init__(self):
        super().__init__()
        self._active_line = None
        self._active_handle: str | None = None  # 'start', 'end', or 'line'
        self._line_drag_anchor_cursor_time: float | None = None
        self._line_drag_anchor_cursor_x_mm: float | None = None
        self._line_drag_anchor_time: float | None = None
        self._line_drag_anchor_rp1: int | None = None
        self._line_drag_anchor_rp2: int | None = None

    # ---- Helpers ----
    def _request_light_repaint(self) -> None:
        if self._editor is None:
            return
        w = getattr(self._editor, 'widget', None)
        if w is not None and hasattr(w, 'update'):
            w.update()
            return
        self._editor.draw_frame()

    def _x_mm_to_rpitch(self, x_mm: float) -> int:
        return self.x_mm_to_rpitch_clamped(float(x_mm))

    def _cursor_mm(self, x_px: float, y_px: float) -> Tuple[float, float]:
        if self._editor is None:
            return (0.0, 0.0)
        return self._editor.widget_px_to_page_mm(float(x_px), float(y_px))

    def _clear_line_drag_anchor(self) -> None:
        self._line_drag_anchor_cursor_time = None
        self._line_drag_anchor_cursor_x_mm = None
        self._line_drag_anchor_time = None
        self._line_drag_anchor_rp1 = None
        self._line_drag_anchor_rp2 = None

    def _capture_line_drag_anchor(self, x: float, y: float) -> None:
        if self._editor is None or self._active_line is None:
            return
        self._line_drag_anchor_cursor_time = float(self._editor.widget_px_to_time(x, y))
        x_mm, _y_mm = self._cursor_mm(x, y)
        self._line_drag_anchor_cursor_x_mm = float(x_mm)
        self._line_drag_anchor_time = float(getattr(self._active_line, 'time', 0.0) or 0.0)
        self._line_drag_anchor_rp1 = int(getattr(self._active_line, 'rpitch1', 0) or 0)
        rp2_raw = getattr(self._active_line, 'rpitch2', 4)
        self._line_drag_anchor_rp2 = int(4 if rp2_raw is None else rp2_raw)

    def toolbar_spec(self) -> list[dict]:
        return []

    def on_left_press(self, x: float, y: float) -> None:
        super().on_left_press(x, y)
        if self._editor is None:
            return
        score: SCORE = self._editor.current_score()
        if score is None:
            return
        events = self._editor.current_events(score)
        if events is None:
            return

        self._editor.draw_frame()

        # Convert mouse to page-space mm for hit testing.
        x_mm, y_mm = self._cursor_mm(x, y)

        hit, hit_handle = self._editor.hit_test_count_line_mm(float(x_mm), float(y_mm))

        if hit is not None:
            self._active_line = hit
            self._active_handle = str(hit_handle or 'line')
            if hit_handle == 'line':
                self._capture_line_drag_anchor(x, y)
            return

        # Create a new count line at the snapped time
        x_mm, _y_mm = self._cursor_mm(x, y)
        t_press_raw = float(self._editor.widget_px_to_time(x, y))
        t_press_snap = float(self._editor.snap_time(t_press_raw))
        rp_press = self._x_mm_to_rpitch(x_mm)
        rp2 = int(rp_press)
        self._active_line = score.new_count_line(time=t_press_snap, rpitch1=rp_press, rpitch2=rp2)
        self._active_handle = 'end'
        self._clear_line_drag_anchor()

        # Request a light repaint to show the new line immediately.
        if hasattr(self._editor, 'force_redraw_from_model'):
            self._editor.force_redraw_from_model()
        else:
            self._editor.draw_frame()
    def on_left_unpress(self, x: float, y: float) -> None:
        super().on_left_unpress(x, y)
        self._active_line = None
        self._active_handle = None
        self._clear_line_drag_anchor()
    def on_left_click(self, x: float, y: float) -> None:
        super().on_left_click(x, y)
        return
    def on_left_double_click(self, x: float, y: float) -> None:
        super().on_left_double_click(x, y)
        print('CountLineTool: on_left_double_click()')
    def on_left_drag_start(self, x: float, y: float) -> None:
        super().on_left_drag_start(x, y)
        return
    def on_left_drag(self, x: float, y: float, dx: float, dy: float) -> None:
        super().on_left_drag(x, y, dx, dy)
        if self._editor is None or self._active_line is None:
            return
        # Update time from y
        x_mm, _y_mm = self._cursor_mm(x, y)
        t_raw = float(self._editor.widget_px_to_time(x, y))
        t_snap = float(self._editor.snap_time(t_raw))
        self._active_line.time = float(t_snap)

        # Update pitch for active handle
        rpitch = self._x_mm_to_rpitch(x_mm)
        min_gap = 2  # Keep handles at least two semitone steps apart.
        if self._active_handle == 'start':
            other = int(getattr(self._active_line, 'rpitch2', 0) or 0)
            current = int(getattr(self._active_line, 'rpitch1', 0) or 0)
            target = int(self.clamp_rpitch(rpitch))
            if current <= other:
                target = min(target, int(other - min_gap))
            else:
                target = max(target, int(other + min_gap))
            self._active_line.rpitch1 = int(self.clamp_rpitch(target))
        elif self._active_handle == 'end':
            other = int(getattr(self._active_line, 'rpitch1', 0) or 0)
            current = int(getattr(self._active_line, 'rpitch2', 0) or 0)
            target = int(self.clamp_rpitch(rpitch))
            if current >= other:
                target = max(target, int(other + min_gap))
            else:
                target = min(target, int(other - min_gap))
            if current == other:
                # Initial collapsed state: choose side from cursor direction.
                target = int(other + min_gap) if target >= other else int(other - min_gap)
            self._active_line.rpitch2 = int(self.clamp_rpitch(target))
        elif self._active_handle == 'line':
            if (
                self._line_drag_anchor_cursor_time is None
                or self._line_drag_anchor_cursor_x_mm is None
                or self._line_drag_anchor_time is None
                or self._line_drag_anchor_rp1 is None
                or self._line_drag_anchor_rp2 is None
            ):
                self._capture_line_drag_anchor(x, y)

            anchor_cursor_time = float(self._line_drag_anchor_cursor_time)
            base_time = float(self._line_drag_anchor_time)
            target_time = base_time + (float(t_raw) - anchor_cursor_time)
            self._active_line.time = float(self._editor.snap_time(target_time))

            semitone_mm = float(getattr(self._editor, 'semitone_dist', 0.0) or 0.0)
            if semitone_mm > 1e-6:
                rp_delta = int(round((x_mm - float(self._line_drag_anchor_cursor_x_mm)) / semitone_mm))
            else:
                rp_delta = 0

            base_rp1 = int(self._line_drag_anchor_rp1)
            base_rp2 = int(self._line_drag_anchor_rp2)
            min_rp, max_rp = self.rpitch_bounds()
            lower_allowed_delta = int(min_rp - min(base_rp1, base_rp2))
            upper_allowed_delta = int(max_rp - max(base_rp1, base_rp2))
            if rp_delta < lower_allowed_delta:
                rp_delta = lower_allowed_delta
            if rp_delta > upper_allowed_delta:
                rp_delta = upper_allowed_delta

            self._active_line.rpitch1 = int(base_rp1 + rp_delta)
            self._active_line.rpitch2 = int(base_rp2 + rp_delta)
        self._request_light_repaint()
    def on_left_drag_end(self, x: float, y: float) -> None:
        super().on_left_drag_end(x, y)
        self._active_line = None
        self._active_handle = None
        self._clear_line_drag_anchor()
    def on_right_press(self, x: float, y: float) -> None:
        super().on_right_press(x, y)
        print('CountLineTool: on_right_press()')
    def on_right_unpress(self, x: float, y: float) -> None:
        super().on_right_unpress(x, y)
        print('CountLineTool: on_right_unpress()')
    def on_right_click(self, x: float, y: float) -> None:
        super().on_right_click(x, y)
        if self._editor is None:
            return
        score: SCORE = self._editor.current_score()
        if score is None:
            return
        events = self._editor.current_events(score)
        if events is None:
            return

        try:
            self._editor.draw_frame()
        except Exception:
            pass

        # Delete when clicking either handle or the line body.
        x_mm, y_mm = self._cursor_mm(x, y)
        hit, _hit_handle = self._editor.hit_test_count_line_mm(float(x_mm), float(y_mm))
        if hit is None:
            return

        hit_id = int(getattr(hit, '_id', -1) or -1)
        events.count_line = [
            ev for ev in list(getattr(events, 'count_line', []) or [])
            if int(getattr(ev, '_id', -2) or -2) != hit_id
        ]
        if hasattr(self._editor, 'force_redraw_from_model'):
            self._editor.force_redraw_from_model()
        else:
            self._editor.draw_frame()
    def on_right_double_click(self, x: float, y: float) -> None:
        super().on_right_double_click(x, y)
        print('CountLineTool: on_right_double_click()')
    def on_right_drag_start(self, x: float, y: float) -> None:
        super().on_right_drag_start(x, y)
        print('CountLineTool: on_right_drag_start()')
    def on_right_drag(self, x: float, y: float, dx: float, dy: float) -> None:
        super().on_right_drag(x, y, dx, dy)
        print('CountLineTool: on_right_drag()')
    def on_right_drag_end(self, x: float, y: float) -> None:
        super().on_right_drag_end(x, y)
        print('CountLineTool: on_right_drag_end()')
    def on_mouse_move(self, x: float, y: float) -> None:
        super().on_mouse_move(x, y)
        # print('CountLineTool: on_mouse_move()')

    def on_toolbar_button(self, name: str) -> None:
        print(f"CountLineTool: on_toolbar_button(name='{name}')")
