from __future__ import annotations
import math
from typing import Optional, Tuple

from PySide6 import QtCore
from editor.tool.base_tool import BaseTool
from file_model.SCORE import SCORE


class SlurTool(BaseTool):
    TOOL_NAME = 'slur'

    def __init__(self) -> None:
        super().__init__()
        self._active_slur = None
        self._active_handle: Optional[int] = None  # 1..4
        self._hit_threshold_mm: float = 4.0
        self._created_on_press: bool = False
        self._hand: str = 'l'
        self._pressed_existing: bool = False

    def toolbar_spec(self) -> list[dict]:
        hand = str(getattr(self._editor, 'hand_cursor', self._hand) or self._hand) if self._editor is not None else self._hand
        return [
            {'name': 'slur_hand_left', 'icon': 'mirror:slur', 'active': hand == 'l', 'tooltip': QtCore.QCoreApplication.translate('SlurTool', 'Slur direction: left hand')},
            {'name': 'slur_hand_right', 'icon': 'slur', 'active': hand == 'r', 'tooltip': QtCore.QCoreApplication.translate('SlurTool', 'Slur direction: right hand')},
        ]

    # ---- Helpers ----
    def _score(self) -> Optional[SCORE]:
        try:
            return self._editor.current_score()
        except Exception:
            return None

    def _view_width_mm(self) -> float:
        try:
            du = self.draw_util()
            w, _ = du.current_page_size_mm()
            return float(w or 0.0)
        except Exception:
            return 0.0

    def relative_x_to_x_mm(self, rpitch: int) -> float:
        """Map semitone offset from key 40 to absolute X in mm."""
        return float(self._editor.relative_c4pitch_to_x(int(rpitch)))

    def x_mm_to_relative_x(self, x_mm: float) -> int:
        """Inverse of relative_x_to_x_mm: X mm to semitone offset from key 40."""
        return self.x_mm_to_rpitch_clamped(float(x_mm))

    def _cursor_mm(self, x_px: float, y_px: float) -> Tuple[float, float]:
        return self._editor.widget_px_to_page_mm(float(x_px), float(y_px))

    def _find_nearest_handle(self, x_mm: float, y_mm: float):
        score = self._score()
        if score is None:
            return (None, None)
        best = None
        best_dist = float('inf')
        for sl in getattr(score.events, 'slur', []) or []:
            pts = self._slur_points_mm(sl)
            for idx, (px, py) in enumerate(pts, start=1):
                d = math.hypot(px - x_mm, py - y_mm)
                if d < best_dist and d <= self._hit_threshold_mm:
                    best_dist = d
                    best = (sl, idx)
        if best is None:
            return (None, None)
        return best

    def _slur_points_mm(self, sl) -> list[Tuple[float, float]]:
        return [
            (self.relative_x_to_x_mm(getattr(sl, 'x1_rpitch', 0)), self._editor.time_to_mm(getattr(sl, 'y1_time', 0.0))),
            (self.relative_x_to_x_mm(getattr(sl, 'x2_rpitch', 0)), self._editor.time_to_mm(getattr(sl, 'y2_time', 0.0))),
            (self.relative_x_to_x_mm(getattr(sl, 'x3_rpitch', 0)), self._editor.time_to_mm(getattr(sl, 'y3_time', 0.0))),
            (self.relative_x_to_x_mm(getattr(sl, 'x4_rpitch', 0)), self._editor.time_to_mm(getattr(sl, 'y4_time', 0.0))),
        ]

    def _apply_drag(self, sl, handle: int, rpitch: int, time_val: float) -> None:
        offset = 6 if str(self._hand) == 'r' else -6
        if handle == 1:
            sl.x1_rpitch = self.clamp_rpitch(rpitch)
            sl.y1_time = time_val
            sl.x2_rpitch = self.clamp_rpitch(rpitch + offset)
            sl.y2_time = time_val
        elif handle == 2:
            sl.x2_rpitch = self.clamp_rpitch(rpitch)
            sl.y2_time = time_val
        elif handle == 3:
            sl.x3_rpitch = self.clamp_rpitch(rpitch)
            sl.y3_time = time_val
        elif handle == 4:
            sl.x4_rpitch = self.clamp_rpitch(rpitch)
            sl.y4_time = time_val
            sl.x3_rpitch = self.clamp_rpitch(rpitch + offset)
            sl.y3_time = time_val

    def _redraw(self) -> None:
        if hasattr(self._editor, 'force_redraw_from_model'):
            self._editor.force_redraw_from_model()
        else:
            self._editor.draw_frame()

    def _create_slur_at(self, x: float, y: float):
        score = self._score()
        if score is None:
            return None, None, None
        try:
            t_raw = float(self._editor.widget_px_to_time(x, y))
            t_snap = float(self._editor.snap_time(t_raw))
        except Exception:
            t_snap = 0.0
        try:
            x_mm, _ = self._cursor_mm(x, y)
            rpitch = self.x_mm_to_relative_x(x_mm)
        except Exception:
            rpitch = 0
        # Respect current hand selection for slur direction
        self._hand = str(getattr(self._editor, 'hand_cursor', self._hand) or self._hand)
        sl = score.new_slur(
            x1_rpitch=rpitch, y1_time=t_snap,
            x2_rpitch=rpitch, y2_time=t_snap,
            x3_rpitch=rpitch, y3_time=t_snap,
            x4_rpitch=rpitch, y4_time=t_snap,
        )
        # Initialize as if we dragged handle 1, then handle 4.
        self._apply_drag(sl, 1, rpitch, t_snap)
        self._apply_drag(sl, 4, rpitch, t_snap)
        self._active_slur = sl
        self._active_handle = 4
        self._redraw()
        return sl, rpitch, t_snap

    # ---- Events ----
    def on_left_press(self, x: float, y: float) -> None:
        super().on_left_press(x, y)
        self._hand = str(getattr(self._editor, 'hand_cursor', self._hand) or self._hand)
        x_mm, y_mm = self._cursor_mm(x, y)
        sl, handle = self._find_nearest_handle(x_mm, y_mm)
        self._active_slur = sl
        self._active_handle = handle
        self._created_on_press = False
        self._pressed_existing = sl is not None
        if sl is None:
            sl, rp, ts = self._create_slur_at(x, y)
            if sl is not None:
                self._created_on_press = True
            self._pressed_existing = False

    def on_left_drag_start(self, x: float, y: float) -> None:
        super().on_left_drag_start(x, y)
        if self._active_slur is None:
            x_mm, y_mm = self._cursor_mm(x, y)
            sl, handle = self._find_nearest_handle(x_mm, y_mm)
            self._active_slur = sl
            self._active_handle = handle

    def on_left_drag(self, x: float, y: float, dx: float, dy: float) -> None:
        super().on_left_drag(x, y, dx, dy)
        if self._active_slur is None or self._active_handle is None:
            return
        try:
            t_raw = float(self._editor.widget_px_to_time(x, y))
            t_snap = float(self._editor.snap_time(t_raw))
        except Exception:
            t_snap = 0.0
        try:
            x_mm, _ = self._cursor_mm(x, y)
            rpitch = self.x_mm_to_relative_x(x_mm)
        except Exception:
            rpitch = 0
        self._apply_drag(self._active_slur, int(self._active_handle), rpitch, t_snap)
        self._redraw()

    def on_left_drag_end(self, x: float, y: float) -> None:
        super().on_left_drag_end(x, y)
        self._active_handle = None
        self._active_slur = None
        self._created_on_press = False
        self._pressed_existing = False

    def on_left_unpress(self, x: float, y: float) -> None:
        super().on_left_unpress(x, y)
        self._active_handle = None
        self._active_slur = None
        self._created_on_press = False
        self._pressed_existing = False

    def on_left_click(self, x: float, y: float) -> None:
        super().on_left_click(x, y)
        if (self._created_on_press and self._active_slur is not None) or self._pressed_existing:
            return
        self._create_slur_at(x, y)

    def on_right_click(self, x: float, y: float) -> None:
        super().on_right_click(x, y)
        x_mm, y_mm = self._cursor_mm(x, y)
        sl, handle = self._find_nearest_handle(x_mm, y_mm)
        if sl is None:
            return
        score = self._score()
        if score is None:
            return
        try:
            score.events.slur.remove(sl)
        except ValueError:
            try:
                sl_id = int(getattr(sl, '_id', -1) or -1)
                score.events.slur = [s for s in score.events.slur if int(getattr(s, '_id', -2) or -2) != sl_id]
            except Exception:
                pass
        self._active_slur = None
        self._active_handle = None
        self._redraw()
        try:
            self._editor._snapshot_if_changed(coalesce=True, label='slur_delete')
        except Exception:
            pass

    def on_mouse_move(self, x: float, y: float) -> None:
        super().on_mouse_move(x, y)

    def on_toolbar_button(self, name: str) -> None:
        if self._editor is None:
            return
        if name == 'slur_hand_left':
            self._hand = 'l'
            self._editor.hand_cursor = 'l'
        elif name == 'slur_hand_right':
            self._hand = 'r'
            self._editor.hand_cursor = 'r'
        # Refresh overlay so visual aids reflect the chosen direction
        if hasattr(self._editor, 'widget') and getattr(self._editor, 'widget', None) is not None:
            w = getattr(self._editor, 'widget')
            if hasattr(w, 'request_overlay_refresh'):
                w.request_overlay_refresh()
