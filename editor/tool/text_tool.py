from __future__ import annotations
import math
from copy import deepcopy
from typing import Optional, Tuple

from PySide6 import QtWidgets, QtCore

from ui.dialogs.text_dialog import TextDialog

from editor.tool.base_tool import BaseTool
from file_model.SCORE import SCORE
from file_model.font import Font


class TextTool(BaseTool):
    TOOL_NAME = 'text'

    def __init__(self) -> None:
        super().__init__()
        self._active_text = None
        self._active_mode: Optional[str] = None  # 'move' or 'rotate'
        self._created_on_press: bool = False
        self._pending_new_text = None
        self._hit_threshold_mm: float = 6.0
        self._cached_center: Optional[Tuple[float, float]] = None
        self._rotation_steps: int = 16  # snap rotation to N steps per full turn
        self._preview_timer: QtCore.QTimer | None = None
        self._move_anchor_cursor_time: Optional[float] = None
        self._move_anchor_cursor_x_mm: Optional[float] = None
        self._move_anchor_text_time: Optional[float] = None
        self._move_anchor_text_rpitch: Optional[int] = None

    def on_activate(self) -> None:
        super().on_activate()

    def on_deactivate(self) -> None:
        super().on_deactivate()

    def toolbar_spec(self) -> list[dict]:
        return []

    def _ensure_preview_timer(self) -> QtCore.QTimer:
        if self._preview_timer is None:
            self._preview_timer = QtCore.QTimer()
            self._preview_timer.setSingleShot(True)
            self._preview_timer.setInterval(150)
            self._preview_timer.timeout.connect(self._emit_preview)
        return self._preview_timer

    def _emit_preview(self) -> None:
        if self._editor is None:
            return
        try:
            self._editor.force_redraw_from_model()
        except Exception:
            pass
        try:
            self._editor.score_changed.emit()
        except Exception:
            pass

    def _schedule_preview(self) -> None:
        try:
            timer = self._ensure_preview_timer()
            timer.stop()
            timer.start()
        except Exception:
            pass

    # ---- Helpers ----
    def _score(self) -> Optional[SCORE]:
        try:
            return self._editor.current_score()
        except Exception:
            return None

    def relative_x_to_x_mm(self, rpitch: int) -> float:
        return float(self._editor.relative_c4pitch_to_x(int(rpitch)))

    def x_mm_to_relative_x(self, x_mm: float) -> int:
        base_x = float(self._editor.pitch_to_x(40))
        dist = float(self._editor.semitone_dist or 0.0)
        if dist <= 1e-6:
            return 0
        rp = (float(x_mm) - base_x) / dist
        # Clamp horizontal drag in rpitch space so it stays inside the stave.
        min_rp = -68.0
        max_rp = 73.0
        rp = max(min_rp, min(rp, max_rp))
        return int(round(rp))

    def _cursor_mm(self, x_px: float, y_px: float) -> Tuple[float, float]:
        return self._editor.widget_px_to_page_mm(float(x_px), float(y_px))

    def _text_geom(self, ev) -> Optional[dict]:
        if ev is None:
            return None
        try:
            txt = str(getattr(ev, 'text', ''))
            display_txt = txt if txt.strip() else "(no text set)"
            score = self._score()
            layout = getattr(score, 'layout', None) if score is not None else None
            use_custom = bool(getattr(ev, 'use_custom_font', False))
            font = self._coerce_font(getattr(ev, 'font', None), getattr(layout, 'font_text', None))
            if (not use_custom) or font is None:
                font = self._coerce_font(getattr(layout, 'font_text', None), getattr(layout, 'font_text', None))
            family = font.resolve_family() if font and hasattr(font, 'resolve_family') else getattr(font, 'family', 'Courier New')
            size_pt = float(getattr(font, 'size_pt', 12.0) or 12.0)
            italic = bool(getattr(font, 'italic', False))
            bold = bool(getattr(font, 'bold', False))
            pad_mm = float(getattr(layout, 'text_background_padding_mm', 0.0) or 0.0)
            width_off_mm = float(getattr(ev, 'text_background_width_offset_mm', 0.0) or 0.0)
            x_off = float(getattr(ev, 'x_offset_mm', 0.0) or 0.0)
            y_off = float(getattr(ev, 'y_offset_mm', 0.0) or 0.0)
            angle = float(getattr(ev, 'rotation', 0.0) or 0.0)
            x_mm = float(self.relative_x_to_x_mm(int(getattr(ev, 'x_rpitch', 0) or 0))) + x_off
            y_mm = float(self._editor.time_to_mm(float(getattr(ev, 'time', 0.0) or 0.0))) + y_off
            du = self.draw_util()
            _xb, _yb, ink_w_mm, ink_h_mm = du._get_text_extents_mm(display_txt, family, size_pt, italic, bold)
            base_w_mm = max(0.0, float(ink_w_mm) + (pad_mm * 2.0))
            h_mm = max(0.0, float(ink_h_mm) + (pad_mm * 2.0))
            base_hw = base_w_mm * 0.5
            x0 = -base_hw
            x1 = base_hw + width_off_mm
            if x1 < x0:
                x1 = x0
            w_mm = max(0.0, x1 - x0)
            hh = h_mm * 0.5
            r = min(max(0.0, pad_mm), w_mm * 0.5, hh)
            ang = math.radians(angle)
            sin_a = math.sin(ang)
            cos_a = math.cos(ang)

            def _rounded_rect_points(x0_val: float, x1_val: float, hh_val: float, radius: float) -> list[tuple[float, float]]:
                if radius <= 1e-6:
                    return [(x0_val, -hh_val), (x1_val, -hh_val), (x1_val, hh_val), (x0_val, hh_val)]
                pts: list[tuple[float, float]] = []
                corner_defs = [
                    (x0_val + radius, -hh_val + radius, 180.0, 270.0),
                    (x1_val - radius, -hh_val + radius, 270.0, 360.0),
                    (x1_val - radius, hh_val - radius, 0.0, 90.0),
                    (x0_val + radius, hh_val - radius, 90.0, 180.0),
                ]
                step = 15.0
                for cx, cy, start_deg, end_deg in corner_defs:
                    deg = start_deg
                    while deg < end_deg + 0.01:
                        rad_ang = math.radians(deg)
                        pts.append((cx + radius * math.cos(rad_ang), cy + radius * math.sin(rad_ang)))
                        deg += step
                return pts

            base_poly = _rounded_rect_points(-base_hw, base_hw, hh, min(max(0.0, pad_mm), base_hw, hh))
            draw_poly = _rounded_rect_points(x0, x1, hh, r)
            rot: list[tuple[float, float]] = []
            min_y = float('inf')
            for dx, dy in base_poly:
                rx = dx * cos_a - dy * sin_a
                ry = dx * sin_a + dy * cos_a
                if ry < min_y:
                    min_y = ry
            for dx, dy in draw_poly:
                rx = dx * cos_a - dy * sin_a
                ry = dx * sin_a + dy * cos_a
                rot.append((rx, ry))
            offset_down = max(0.0, -min_y)
            cy = y_mm + offset_down
            poly = [(x_mm + dx, cy + dy) for (dx, dy) in rot]
            min_x = min(p[0] for p in poly)
            max_x = max(p[0] for p in poly)
            min_y_abs = min(p[1] for p in poly)
            max_y_abs = max(p[1] for p in poly)
            bbox = (min_x, max_x, min_y_abs, max_y_abs)
            gap = max(1.5, (self._editor.semitone_dist or 2.5) * 0.3)
            rad = (w_mm * 0.5) + gap
            hx = x_mm + rad * cos_a
            hy = cy + rad * sin_a
            handle_size = max(2.0, (self._editor.semitone_dist or 2.5) * 0.6)
            handle_hit = max(handle_size, self._hit_threshold_mm * 1.25)
            hx1 = hx - handle_hit * 0.5
            hx2 = hx + handle_hit * 0.5
            hy1 = hy - handle_hit * 0.5
            hy2 = hy + handle_hit * 0.5
            return {
                'family': family,
                'size_pt': size_pt,
                'italic': italic,
                'bold': bold,
                'angle': angle,
                'x': x_mm,
                'y': y_mm,
                'cy': cy,
                'poly': poly,
                'bbox': bbox,
                'handle': (hx, hy),
                'handle_rect': (hx1, hy1, hx2, hy2),
            }
        except Exception:
            return None

    def _point_in_poly(self, x: float, y: float, poly: list[tuple[float, float]]) -> bool:
        inside = False
        n = len(poly)
        if n < 3:
            return False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)):
                x_int = (xj - xi) * (y - yi) / max(1e-9, (yj - yi)) + xi
                if x < x_int:
                    inside = not inside
            j = i
        return inside

    def _hit_test_legacy(self, x_mm: float, y_mm: float):
        score = self._score()
        if score is None:
            return (None, None, None)
        best_ev = None
        best_mode = None
        best_geom = None
        best_dist = float('inf')
        for ev in list(getattr(score.events, 'text', []) or []):
            geom = self._text_geom(ev)
            if geom is None:
                continue
            hx, hy = geom['handle']
            hr = geom.get('handle_rect')
            if hr is not None:
                hx1, hy1, hx2, hy2 = hr
                if hx1 <= x_mm <= hx2 and hy1 <= y_mm <= hy2:
                    best_ev = ev
                    best_mode = 'rotate'
                    best_geom = geom
                    best_dist = 0.0
                    continue
            d = math.hypot(x_mm - hx, y_mm - hy)
            if d <= self._hit_threshold_mm and d < best_dist:
                best_ev = ev
                best_mode = 'rotate'
                best_geom = geom
                best_dist = d
                continue
            if self._point_in_poly(x_mm, y_mm, geom['poly']) and best_mode is None:
                best_ev = ev
                best_mode = 'move'
                best_geom = geom
                best_dist = 0.0
                continue
            bx1, bx2, by1, by2 = geom.get('bbox', (None, None, None, None))
            if bx1 is not None:
                pad = max(self._hit_threshold_mm * 3.0, 12.0)
                if (bx1 - pad) <= x_mm <= (bx2 + pad) and (by1 - pad) <= y_mm <= (by2 + pad):
                    best_ev = ev
                    best_mode = 'move'
                    best_geom = geom
                    best_dist = 0.0
                    continue
            cx = geom.get('x', 0.0)
            cy = geom.get('cy', 0.0)
            d_center = math.hypot(x_mm - cx, y_mm - cy)
            if d_center <= max(self._hit_threshold_mm * 3.5, 16.0) and d_center < best_dist:
                best_ev = ev
                best_mode = 'move'
                best_geom = geom
                best_dist = d_center
        return (best_ev, best_mode, best_geom)

    def _hit_test(self, x_mm: float, y_mm: float):
        score = self._score()
        if score is None:
            return (None, None, None)
        try:
            if hasattr(self._editor, 'hit_test_text_mm'):
                text_id, is_handle, _rect = self._editor.hit_test_text_mm(x_mm, y_mm)
                if text_id is not None:
                    ev = self._find_text_by_id(text_id)
                    if ev is not None:
                        return (ev, 'rotate' if is_handle else 'move', self._text_geom(ev))
        except Exception:
            pass
        return self._hit_test_legacy(x_mm, y_mm)

    def _compute_center_mm(self, ev) -> Optional[Tuple[float, float]]:
        if ev is None or self._editor is None:
            return None
        geom = self._text_geom(ev)
        if geom and 'x' in geom and 'cy' in geom:
            return (geom['x'], geom['cy'])
        try:
            x_mm = float(self.relative_x_to_x_mm(int(getattr(ev, 'x_rpitch', 0) or 0)))
        except Exception:
            x_mm = None
        try:
            y_mm = float(self._editor.time_to_mm(float(getattr(ev, 'time', 0.0) or 0.0)))
        except Exception:
            y_mm = None
        if x_mm is None or y_mm is None:
            return None
        return (x_mm, y_mm)

    def _find_text_by_id(self, text_id: int):
        score = self._score()
        if score is None:
            return None
        try:
            for ev in list(getattr(score.events, 'text', []) or []):
                if int(getattr(ev, '_id', -1) or -1) == int(text_id):
                    return ev
        except Exception:
            return None
        return None

    def _capture_move_anchor(self, x_px: float, y_px: float, x_mm: float) -> None:
        if self._editor is None or self._active_text is None or self._active_mode != 'move':
            return
        self._move_anchor_cursor_time = float(self._editor.widget_px_to_time(x_px, y_px))
        self._move_anchor_cursor_x_mm = float(x_mm)
        self._move_anchor_text_time = float(getattr(self._active_text, 'time', 0.0) or 0.0)
        self._move_anchor_text_rpitch = int(getattr(self._active_text, 'x_rpitch', 0) or 0)

    def _clear_move_anchor(self) -> None:
        self._move_anchor_cursor_time = None
        self._move_anchor_cursor_x_mm = None
        self._move_anchor_text_time = None
        self._move_anchor_text_rpitch = None

    # ---- Dialog ----
    def _coerce_font(self, value, default_font: Font | None) -> Font:
        if isinstance(value, Font):
            return deepcopy(value)
        if isinstance(value, dict):
            return Font(
                family=value.get('family', getattr(default_font, 'family', 'Courier New')),
                size_pt=float(value.get('size_pt', getattr(default_font, 'size_pt', 12.0) or 12.0)),
                bold=bool(value.get('bold', getattr(default_font, 'bold', False))),
                italic=bool(value.get('italic', getattr(default_font, 'italic', False))),
                x_offset=float(value.get('x_offset', getattr(default_font, 'x_offset', 0.0) or 0.0)),
                y_offset=float(value.get('y_offset', getattr(default_font, 'y_offset', 0.0) or 0.0)),
            )
        return deepcopy(default_font or Font())

    def _open_text_dialog(self, ev) -> None:
        if self._editor is None or ev is None:
            return
        score = self._score()
        default_font = getattr(score.layout, 'font_text', None) if score is not None else None
        dlg = TextDialog(ev, default_font, parent=QtWidgets.QApplication.activeWindow())
        original_state = TextDialog.snapshot_from_event(ev)

        def _apply_live(commit_snapshot: bool = False) -> None:
            dlg.apply_to_event(ev)
            if commit_snapshot:
                self._editor._snapshot_if_changed(coalesce=False, label='text_edit')
            self._schedule_preview()

        def _apply():
            _apply_live(commit_snapshot=True)

        def _revert_state() -> None:
            TextDialog.restore_event(ev, original_state)
            self._schedule_preview()

        dlg.valueChanged.connect(lambda: _apply_live(False))

        dlg.accepted.connect(_apply)
        dlg.raise_()
        dlg.activateWindow()
        dlg.show()

        dlg.rejected.connect(_revert_state)

    # ---- Events ----
    def on_left_press(self, x: float, y: float) -> None:
        super().on_left_press(x, y)
        if self._editor is None:
            return
        self._clear_move_anchor()
        x_mm, y_mm = self._cursor_mm(x, y)
        hit, mode, geom = self._hit_test(x_mm, y_mm)
        self._active_text = hit
        self._active_mode = mode
        self._created_on_press = False
        self._cached_center = (geom['x'], geom['cy']) if geom else self._compute_center_mm(hit)
        
        # If we hit a rotate handle, prefer rotation immediately
        if hit is not None and mode == 'rotate':
            if self._cached_center is None and hit is not None:
                self._cached_center = self._compute_center_mm(hit)
            return

        if hit is None:
            score = self._score()
            if score is None:
                return
            t_raw = float(self._editor.widget_px_to_time(x, y))
            t_snap = float(self._editor.snap_time(t_raw))
            rp = self.x_mm_to_relative_x(x_mm)
            df = deepcopy(getattr(score.layout, 'font_text', Font()))
            tx = score.new_text(time=t_snap, x_rpitch=rp, rotation=0.0, text='', font=df)
            self._active_text = tx
            self._active_mode = 'move'
            self._created_on_press = True
            self._pending_new_text = tx
            self._cached_center = None
            self._editor._snapshot_if_changed(coalesce=True, label='text_create')
            if hasattr(self._editor, 'force_redraw_from_model'):
                self._editor.force_redraw_from_model()
            else:
                self._editor.draw_frame()

        self._capture_move_anchor(x, y, x_mm)

        super().on_left_drag_start(x, y)

    def on_left_drag(self, x: float, y: float, dx: float, dy: float) -> None:
        super().on_left_drag(x, y, dx, dy)
        if self._editor is None or self._active_text is None:
            return
        x_mm, y_mm = self._cursor_mm(x, y)
        if self._active_mode == 'rotate':
            if self._cached_center is None:
                self._cached_center = self._compute_center_mm(self._active_text)
            cx, cy = self._cached_center if self._cached_center else (x_mm, y_mm)
            ang = (math.degrees(math.atan2(y_mm - cy, x_mm - cx)) + 360.0) % 360.0
            ctrl_down = bool(getattr(self._editor, '_ctrl_down', False)) if self._editor else False
            shift_down = bool(QtWidgets.QApplication.keyboardModifiers() & QtCore.Qt.ShiftModifier)
            if self._rotation_steps and self._rotation_steps > 0 and not ctrl_down and not shift_down:
                step = 360.0 / float(self._rotation_steps)
                ang = round(ang / step) * step
            self._active_text.rotation = float(ang)
        elif self._active_mode == 'move':
            if (
                self._move_anchor_cursor_time is None
                or self._move_anchor_cursor_x_mm is None
                or self._move_anchor_text_time is None
                or self._move_anchor_text_rpitch is None
            ):
                self._capture_move_anchor(x, y, x_mm)

            t_raw = float(self._editor.widget_px_to_time(x, y))
            anchor_time = float(self._move_anchor_cursor_time or t_raw)
            base_text_time = float(self._move_anchor_text_time or 0.0)
            t_snap = max(0.0, float(self._editor.snap_time(base_text_time + (t_raw - anchor_time))))

            semitone_mm = float(getattr(self._editor, 'semitone_dist', 0.0) or 0.0)
            if semitone_mm > 1e-6:
                rp_delta = int(round((x_mm - float(self._move_anchor_cursor_x_mm or x_mm)) / semitone_mm))
            else:
                rp_delta = 0
            base_rp = int(self._move_anchor_text_rpitch or 0)
            rp = max(-68, min(73, base_rp + rp_delta))
            self._active_text.time = float(t_snap)
            self._active_text.x_rpitch = int(rp)
        if hasattr(self._editor, 'force_redraw_from_model'):
            self._editor.force_redraw_from_model()
        else:
            self._editor.draw_frame()

    def on_left_drag_end(self, x: float, y: float) -> None:
        super().on_left_drag_end(x, y)
        if self._editor is None:
            return
        if self._active_text is not None:
            label = 'text_rotate' if self._active_mode == 'rotate' else 'text_move'
            self._editor._snapshot_if_changed(coalesce=True, label=label)
        self._active_mode = None
        self._cached_center = None
        self._clear_move_anchor()

    def on_left_unpress(self, x: float, y: float) -> None:
        super().on_left_unpress(x, y)
        if self._editor is None:
            return
        # Do not open dialog here; defer to click handler to avoid double-open
        if not self._created_on_press:
            self._pending_new_text = None
        self._active_text = None
        self._active_mode = None
        self._cached_center = None
        self._clear_move_anchor()

    def on_left_click(self, x: float, y: float) -> None:
        super().on_left_click(x, y)
        if self._editor is None:
            return
        if self._pending_new_text is not None:
            self._open_text_dialog(self._pending_new_text)
            self._pending_new_text = None
            self._created_on_press = False
            return
        x_mm, y_mm = self._cursor_mm(x, y)
        hit, mode, _ = self._hit_test(x_mm, y_mm)
        if hit is not None and mode != 'rotate':
            self._open_text_dialog(hit)

    def on_right_click(self, x: float, y: float) -> None:
        super().on_right_click(x, y)
        if self._editor is None:
            return
        score = self._score()
        if score is None:
            return
        x_mm, y_mm = self._cursor_mm(x, y)
        hit, _mode, _ = self._hit_test(x_mm, y_mm)
        if hit is None:
            return
        lst = list(getattr(score.events, 'text', []) or [])
        lst = [t for t in lst if int(getattr(t, '_id', -1) or -1) != int(getattr(hit, '_id', -2) or -2)]
        score.events.text = lst
        self._editor._snapshot_if_changed(coalesce=True, label='text_delete')
        if hasattr(self._editor, 'force_redraw_from_model'):
            self._editor.force_redraw_from_model()
        else:
            self._editor.draw_frame()

    def on_mouse_move(self, x: float, y: float) -> None:
        super().on_mouse_move(x, y)

    def on_toolbar_button(self, name: str) -> None:
        return
