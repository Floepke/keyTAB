from __future__ import annotations
import math
from copy import deepcopy
from typing import Optional, Tuple

from PySide6 import QtWidgets, QtCore

from ui.widgets.style_dialog import FontPicker, FloatSliderEdit

from editor.tool.base_tool import BaseTool
from file_model.SCORE import SCORE
from file_model.layout import LayoutFont


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
        px_per_mm = float(getattr(self._editor, '_widget_px_per_mm', 1.0) or 1.0)
        view_offset = float(getattr(self._editor, '_view_y_mm_offset', 0.0) or 0.0)
        x_mm = float(x_px) / max(1e-6, px_per_mm)
        y_mm_local = float(y_px) / max(1e-6, px_per_mm)
        y_mm = y_mm_local + view_offset
        return x_mm, y_mm

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
            x_off = float(getattr(ev, 'x_offset_mm', 0.0) or 0.0)
            y_off = float(getattr(ev, 'y_offset_mm', 0.0) or 0.0)
            angle = float(getattr(ev, 'rotation', 0.0) or 0.0)
            x_mm = float(self.relative_x_to_x_mm(int(getattr(ev, 'x_rpitch', 0) or 0))) + x_off
            y_mm = float(self._editor.time_to_mm(float(getattr(ev, 'time', 0.0) or 0.0))) + y_off
            du = self.draw_util()
            _xb, _yb, w_mm, h_mm = du._get_text_extents_mm(display_txt, family, size_pt, italic, bold)
            w_mm += pad_mm * 2.0
            h_mm += pad_mm * 2.0
            hw = w_mm * 0.5
            hh = h_mm * 0.5
            r = min(max(0.0, pad_mm), hw, hh)
            ang = math.radians(angle)
            sin_a = math.sin(ang)
            cos_a = math.cos(ang)

            def _rounded_rect_points(hw_val: float, hh_val: float, radius: float) -> list[tuple[float, float]]:
                if radius <= 1e-6:
                    return [(-hw_val, -hh_val), (hw_val, -hh_val), (hw_val, hh_val), (-hw_val, hh_val)]
                pts: list[tuple[float, float]] = []
                corner_defs = [
                    (-hw_val + radius, -hh_val + radius, 180.0, 270.0),
                    (hw_val - radius, -hh_val + radius, 270.0, 360.0),
                    (hw_val - radius, hh_val - radius, 0.0, 90.0),
                    (-hw_val + radius, hh_val - radius, 90.0, 180.0),
                ]
                step = 15.0
                for cx, cy, start_deg, end_deg in corner_defs:
                    deg = start_deg
                    while deg < end_deg + 0.01:
                        rad_ang = math.radians(deg)
                        pts.append((cx + radius * math.cos(rad_ang), cy + radius * math.sin(rad_ang)))
                        deg += step
                return pts

            base_poly = _rounded_rect_points(hw, hh, r)
            rot: list[tuple[float, float]] = []
            min_y = float('inf')
            for dx, dy in base_poly:
                rx = dx * cos_a - dy * sin_a
                ry = dx * sin_a + dy * cos_a
                rot.append((rx, ry))
                if ry < min_y:
                    min_y = ry
            offset_down = max(0.0, -min_y)
            cy = y_mm + offset_down
            poly = [(x_mm + dx, cy + dy) for (dx, dy) in rot]
            min_x = min(p[0] for p in poly)
            max_x = max(p[0] for p in poly)
            min_y_abs = min(p[1] for p in poly)
            max_y_abs = max(p[1] for p in poly)
            bbox = (min_x, max_x, min_y_abs, max_y_abs)
            gap = max(1.5, (self._editor.semitone_dist or 2.5) * 0.3)
            rad = hw + gap
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

    # ---- Dialog ----
    def _coerce_font(self, value, default_font: LayoutFont | None) -> LayoutFont:
        if isinstance(value, LayoutFont):
            return deepcopy(value)
        if isinstance(value, dict):
            return LayoutFont(
                family=value.get('family', getattr(default_font, 'family', 'Courier New')),
                size_pt=float(value.get('size_pt', getattr(default_font, 'size_pt', 12.0) or 12.0)),
                bold=bool(value.get('bold', getattr(default_font, 'bold', False))),
                italic=bool(value.get('italic', getattr(default_font, 'italic', False))),
                x_offset=float(value.get('x_offset', getattr(default_font, 'x_offset', 0.0) or 0.0)),
                y_offset=float(value.get('y_offset', getattr(default_font, 'y_offset', 0.0) or 0.0)),
            )
        return deepcopy(default_font or LayoutFont())

    def _open_text_dialog(self, ev) -> None:
        if self._editor is None or ev is None:
            return
        score = self._score()
        default_font = getattr(score.layout, 'font_text', None) if score is not None else None
        cur_font_raw = getattr(ev, 'font', None)
        cur_font = self._coerce_font(cur_font_raw, default_font)
        # If using default, show default in picker; if custom, use event font
        if not bool(getattr(ev, 'use_custom_font', False)):
            cur_font = deepcopy(self._coerce_font(default_font, default_font))

        dlg = QtWidgets.QDialog(QtWidgets.QApplication.activeWindow())
        dlg.setWindowTitle("Edit Text")
        layout = QtWidgets.QFormLayout(dlg)

        txt_edit = QtWidgets.QLineEdit(dlg)
        txt_edit.setText(str(getattr(ev, 'text', '')))
        layout.addRow("Text", txt_edit)

        x_off_edit = FloatSliderEdit(float(getattr(ev, 'x_offset_mm', 0.0) or 0.0), -25.0, 25.0, 0.1, dlg)
        y_off_edit = FloatSliderEdit(float(getattr(ev, 'y_offset_mm', 0.0) or 0.0), -25.0, 25.0, 0.1, dlg)
        rot_edit = FloatSliderEdit(float(getattr(ev, 'rotation', 0.0) or 0.0), 0.0, 360.0, 0.1, dlg)
        layout.addRow("X offset (mm)", x_off_edit)
        layout.addRow("Y offset (mm)", y_off_edit)
        layout.addRow("Rotation (degrees)", rot_edit)

        use_custom_chk = QtWidgets.QCheckBox("Use custom font", dlg)
        use_custom_chk.setChecked(bool(getattr(ev, 'use_custom_font', False)))
        layout.addRow(use_custom_chk)

        font_picker = FontPicker(cur_font, parent=dlg)
        layout.addRow(font_picker)
        font_picker.setVisible(use_custom_chk.isChecked())

        def toggle_custom(state: bool):
            font_picker.setVisible(state)
            if not state:
                font_picker.set_value(default_font or LayoutFont())

        use_custom_chk.toggled.connect(toggle_custom)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, parent=dlg)
        layout.addRow(btns)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)

        original_state = {
            'text': getattr(ev, 'text', ''),
            'use_custom_font': bool(getattr(ev, 'use_custom_font', False)),
            'font': deepcopy(getattr(ev, 'font', None)),
            'x_offset_mm': float(getattr(ev, 'x_offset_mm', 0.0) or 0.0),
            'y_offset_mm': float(getattr(ev, 'y_offset_mm', 0.0) or 0.0),
            'rotation': float(getattr(ev, 'rotation', 0.0) or 0.0),
        }

        def _apply_live(commit_snapshot: bool = False) -> None:
            if bool(use_custom_chk.isChecked()):
                ev.use_custom_font = True
                ev.font = deepcopy(font_picker.value())
            else:
                ev.use_custom_font = False
                ev.font = deepcopy(default_font or LayoutFont())
            ev.text = txt_edit.text()
            ev.x_offset_mm = float(x_off_edit.value())
            ev.y_offset_mm = float(y_off_edit.value())
            try:
                ev.rotation = float(rot_edit.value())
            except Exception:
                pass
            if commit_snapshot:
                try:
                    self._editor._snapshot_if_changed(coalesce=False, label='text_edit')
                except Exception:
                    pass
            self._schedule_preview()

        def _apply():
            _apply_live(commit_snapshot=True)

        def _revert_state() -> None:
            try:
                ev.text = original_state['text']
                ev.use_custom_font = original_state['use_custom_font']
                ev.font = deepcopy(original_state['font'])
                ev.x_offset_mm = float(original_state['x_offset_mm'])
                ev.y_offset_mm = float(original_state['y_offset_mm'])
                ev.rotation = float(original_state['rotation'])
            except Exception:
                pass
            self._schedule_preview()

        txt_edit.textChanged.connect(lambda _t: _apply_live(False))
        x_off_edit.valueChanged.connect(lambda _v: _apply_live(False))
        y_off_edit.valueChanged.connect(lambda _v: _apply_live(False))
        rot_edit.valueChanged.connect(lambda _v: _apply_live(False))
        use_custom_chk.toggled.connect(lambda _v: _apply_live(False))
        font_picker.valueChanged.connect(lambda: _apply_live(False))

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
            t_raw = float(self._editor.y_to_time(y))
            t_snap = float(self._editor.snap_time(t_raw))
            rp = self.x_mm_to_relative_x(x_mm)
            df = deepcopy(getattr(score.layout, 'font_text', LayoutFont()))
            tx = score.new_text(time=t_snap, x_rpitch=rp, rotation=0.0, text='', font=df)
            self._active_text = tx
            self._active_mode = 'move'
            self._created_on_press = True
            self._pending_new_text = tx
            self._cached_center = None
            try:
                self._editor._snapshot_if_changed(coalesce=True, label='text_create')
            except Exception:
                pass
            if hasattr(self._editor, 'force_redraw_from_model'):
                self._editor.force_redraw_from_model()
            else:
                self._editor.draw_frame()

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
            if self._rotation_steps and self._rotation_steps > 0 and not ctrl_down:
                step = 360.0 / float(self._rotation_steps)
                ang = round(ang / step) * step
            try:
                self._active_text.rotation = float(ang)
            except Exception:
                pass
        elif self._active_mode == 'move':
            try:
                t_raw = float(self._editor.y_to_time(y))
                t_snap = float(self._editor.snap_time(t_raw))
            except Exception:
                t_snap = 0.0
            try:
                rp = self.x_mm_to_relative_x(x_mm)
            except Exception:
                rp = 0
            try:
                self._active_text.time = float(t_snap)
                self._active_text.x_rpitch = int(rp)
            except Exception:
                pass
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
            try:
                self._editor._snapshot_if_changed(coalesce=True, label=label)
            except Exception:
                pass
        self._active_mode = None
        self._cached_center = None

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
        try:
            lst = list(getattr(score.events, 'text', []) or [])
            lst = [t for t in lst if int(getattr(t, '_id', -1) or -1) != int(getattr(hit, '_id', -2) or -2)]
            score.events.text = lst
            self._editor._snapshot_if_changed(coalesce=True, label='text_delete')
        except Exception:
            pass
        if hasattr(self._editor, 'force_redraw_from_model'):
            self._editor.force_redraw_from_model()
        else:
            self._editor.draw_frame()

    def on_mouse_move(self, x: float, y: float) -> None:
        super().on_mouse_move(x, y)

    def on_toolbar_button(self, name: str) -> None:
        return
