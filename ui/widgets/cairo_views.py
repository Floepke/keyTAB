from __future__ import annotations
from PySide6 import QtCore, QtGui, QtWidgets
import os
import sys
import cairo
import math
from typing import Optional
from editor.editor import Editor
from ui.widgets.draw_util import DrawUtil, make_image_surface, finalize_image_surface
from ui.style import Style
from settings_manager import get_preferences
# Stripped renderer, tile cache, and spatial index for static viewport simplicity


def _draw_editor_background(ctx: cairo.Context, w: int, h: int, color=(0.12, 0.12, 0.12)):
    # Neutral background; no demo drawings.
    ctx.set_source_rgb(*color)
    ctx.paint()


class CairoEditorWidget(QtWidgets.QWidget):
    # Signal: inform container to adjust external scrollbar
    viewportMetricsChanged = QtCore.Signal(int, int, float, float)
    # Signal: emit logical pixel scroll value when wheel scrolling changes it
    scrollLogicalPxChanged = QtCore.Signal(int)
    # Signal: emitted when the mouse wheel scrolls inside the editor view
    scrollWheelUsed = QtCore.Signal()
    def __init__(self, parent=None):
        super().__init__(parent)
        # Allow splitter to fully collapse this view
        self.setMinimumWidth(0)
        self.setMouseTracking(True)
        # Ensure this widget can receive keyboard focus for shortcuts
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        # Apply the dedicated DrawUtil background color to the editor widget, too
        try:
            color = Style.get_named_qcolor('editor')
            pal = self.palette()
            pal.setColor(QtGui.QPalette.Window, color)
            self.setPalette(pal)
            self.setAutoFillBackground(True)
        except Exception:
            pass
        self._current_tool: str | None = None
        self._editor: Optional[Editor] = None
        self._last_pos: QtCore.QPointF | None = None
        # Throttled mouse move state (30 Hz cap)
        self._pending_move: QtCore.QPointF | None = None
        self._last_sent_pos: QtCore.QPointF | None = None
        self._move_timer: QtCore.QTimer | None = QtCore.QTimer(self)
        self._fps_interval_ms: int = 33
        try:
            self._move_timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
            self._move_timer.setInterval(self._fps_interval_ms)  # default ~30 FPS
            self._move_timer.timeout.connect(self._dispatch_throttled_move)
        except Exception:
            pass
        # Load FPS limit from preferences
        try:
            self._configure_move_timer_from_prefs()
        except Exception:
            pass
        self._du: DrawUtil | None = None
        self._last_px_per_mm: float = 1.0
        self._last_dpr: float = 1.0
        self._content_h_px: int = 0
        # External scroll (logical px), controlled by an external QScrollBar
        self._scroll_logical_px: int = 0
        # Track mouse button state to decide when overlay-only redraw is safe
        self._left_down: bool = False
        self._right_down: bool = False
        # Quick overlay repaint hint for paintEvent
        self._overlay_only_repaint: bool = False
        # Cached static layers for current viewport
        self._content_cache_image: QtGui.QImage | None = None
        self._content_cache_key: tuple | None = None  # (px_per_mm, dpr, vis_w_px, vis_h_px, clip_x_mm, clip_y_mm, clip_w_mm, clip_h_mm)
        # Debug logging toggle (env: PIANOSCRIPT_DEBUG_SCROLL=1)
        self._debug_scroll: bool = os.getenv('PIANOSCRIPT_DEBUG_SCROLL', '0') in ('1', 'true', 'True')
        self._last_debug_key: tuple | None = None
        # Static viewport: no tiling/cache/renderer state
        self._last_cache_params: tuple[float, float, float] | None = None
        # Last hovered note id to avoid redundant status updates
        self._last_hover_note_id: int | None = None

    def set_editor(self, editor: Editor) -> None:
        self._editor = editor

    def request_overlay_refresh(self) -> None:
        """Trigger an overlay-only repaint for guide updates (e.g., cursor changes).

        Sets a hint to reuse cached content and redraw only guides on next paint.
        """
        try:
            self._overlay_only_repaint = True
        except Exception:
            pass
        self.update()

    def force_full_redraw(self) -> None:
        """Invalidate cached content and request a full repaint from the model."""
        try:
            self._content_cache_image = None
            self._content_cache_key = None
            self._overlay_only_repaint = False
        except Exception:
            pass
        self.update()

    def set_tool(self, tool_name: str | None) -> None:
        self._current_tool = tool_name
        self.update()

    def set_scroll_logical_px(self, value: int) -> None:
        """Set external logical pixel scroll offset and repaint."""
        self._scroll_logical_px = max(0, int(value))
        self.update()

    def paintEvent(self, ev: QtGui.QPaintEvent) -> None:
        # Use widget size as static viewport; do not rely on QScrollArea.
        vp = self
        dpr = float(self.devicePixelRatioF())
        vp_w = self.size().width()
        vp_h = self.size().height()
        # Device pixel dimensions (rounded) for the backing image
        vis_w_px = int(max(1, round(float(vp_w) * dpr)))
        vis_h_px = int(max(1, round(float(vp_h) * dpr)))
        # Prepare DrawUtil with page dimensions from SCORE/layout and Editor layout
        page_w_mm = 210.0
        page_h_mm = 297.0
        if self._editor is not None:
            sc = self._editor.current_score()
            if sc is not None:
                lay = getattr(sc, 'layout', None)
                if lay is not None:
                    page_w_mm = float(getattr(lay, 'page_width_mm', page_w_mm))
            # Calculate editor layout metrics (margin, stave_width, editor_height)
            self._editor._calculate_layout(page_w_mm)
            try:
                page_h_mm = float(getattr(self._editor, 'editor_height', page_h_mm) or page_h_mm)
            except Exception:
                page_h_mm = page_h_mm
        # Invalidate cache when scale or page dimensions change
        # Derive px_per_mm from the actual backing image width to avoid anisotropy
        px_per_mm = (float(vis_w_px)) / page_w_mm
        h_px_content = int(page_h_mm * px_per_mm)
        cache_params = (round(px_per_mm, 6), round(page_w_mm, 3), round(page_h_mm, 3))
        if self._last_cache_params is None or self._last_cache_params != cache_params:
            self._last_cache_params = cache_params

        # Always use fresh DrawUtils to avoid item accumulation
        self._last_px_per_mm = px_per_mm
        self._last_dpr = dpr
        self._content_h_px = h_px_content
        # Keep widget height independent from content to maintain a static viewport

        # Visible region equals viewport; use external scroll for offset (logical px)
        # No overscan/bleed: viewport is strictly the visible area.
        bleed_px = 0
        vis_h_px_bleed = vis_h_px
        scroll_val_px = int(self._scroll_logical_px)  # logical px
        # Compute clip rectangle in mm using device px per mm (honors zoom)
        clip_x_mm = 0.0
        clip_y_mm = float(scroll_val_px) * dpr / max(1e-6, px_per_mm)
        clip_w_mm = page_w_mm
        clip_h_mm = float(vis_h_px) / max(1e-6, px_per_mm)
        # No bleed: clip is exactly the viewport size in mm
        clip_y_mm_bleed = clip_y_mm
        clip_h_mm_bleed = float(vis_h_px_bleed) / max(1e-6, px_per_mm)

        # Debug logging (only when values change)
        if self._debug_scroll:
            dbg_key = (scroll_val_px, round(dpr, 3), round(px_per_mm, 6), int(vp_w), int(vp_h),
                       int(vis_w_px), int(vis_h_px), round(clip_y_mm, 3), round(clip_h_mm, 3))
            if dbg_key != self._last_debug_key:
                self._last_debug_key = dbg_key
                print(f"[ScrollDbg] scroll_px={scroll_val_px} dpr={dpr:.3f} px_per_mm={px_per_mm:.6f} "
                      f"vp=({vp_w}x{vp_h}) vis=({vis_w_px}x{vis_h_px}) clip_y_mm={clip_y_mm:.3f} clip_h_mm={clip_h_mm:.3f}")
        # Static viewport: tiling disabled and not used

        # Emit metrics so a container can configure an external scrollbar
        try:
            self.viewportMetricsChanged.emit(h_px_content, vis_h_px, px_per_mm, dpr)
        except Exception:
            print('CairoEditorWidget.paintEvent: Warning: failed to emit viewportMetricsChanged')
        
        # Provide view metrics to the editor for fast px↔mm/time conversions
        try:
            if self._editor is not None:
                widget_px_per_mm = float(vis_w_px) / max(1e-6, page_w_mm) / max(1e-6, dpr)
                self._editor.set_view_metrics(px_per_mm, widget_px_per_mm, dpr)
                # Provide current clip origin offset and viewport height in mm so drawers can cull
                self._editor.set_view_offset_mm(clip_y_mm)
                try:
                    self._editor.set_viewport_height_mm(clip_h_mm)
                except Exception:
                    pass
                # Recompute cursor from last mouse position so guides follow scroll
                if self._last_pos is not None:
                    try:
                        t = self._editor.y_to_time(self._last_pos.y())
                        t = self._editor.snap_time(t)
                        self._editor.time_cursor = t
                        abs_mm = self._editor.time_to_mm(float(t))
                        self._editor.mm_cursor = abs_mm - float(getattr(self._editor, '_view_y_mm_offset', 0.0) or 0.0)
                    except Exception:
                        pass
        except Exception:
            pass

        # Compute a stable cache key for the current content viewport
        cache_key = (round(px_per_mm, 6), round(dpr, 3), vis_w_px, vis_h_px,
                     round(clip_x_mm, 3), round(clip_y_mm, 3), round(clip_w_mm, 3), round(clip_h_mm, 3))

        painter = QtGui.QPainter(self)
        try:
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
            painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing, True)
            painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)
        except Exception:
            print('CairoEditorWidget.paintEvent: Warning: failed to set QPainter render hints')
        try:
            # Use a pure viewport clip rect (no bleed). Overscan applied only inside DrawUtil if needed.
            clip_mm = (clip_x_mm, clip_y_mm, clip_w_mm, clip_h_mm)

            # Fast path: if only overlays changed (mouse move, no buttons), reuse cached content
            fast_overlay = False
            if self._overlay_only_repaint and self._content_cache_image is not None and self._content_cache_key == cache_key:
                fast_overlay = True
                # Draw cached content first
                content_img = self._content_cache_image
                content_img.setDevicePixelRatio(dpr)
                painter.drawImage(QtCore.QRectF(0.0, 0.0, float(vp_w), float(vp_h)), content_img)

                # Rebuild only guides layer and composite on top
                du_guides = DrawUtil()
                du_guides.set_current_page_size_mm(page_w_mm, page_h_mm)
                if self._editor is not None:
                    try:
                        self._editor.draw_guides(du_guides)
                    except Exception:
                        pass
                # Offscreen buffer for overlays
                ov_img, ov_surf, _ov_buf = make_image_surface(vis_w_px, vis_h_px)
                ov_ctx = cairo.Context(ov_surf)
                try:
                    ov_ctx.set_antialias(cairo.ANTIALIAS_BEST)
                except Exception:
                    pass
                du_guides.render_to_cairo(ov_ctx, du_guides.current_page_index(), px_per_mm, clip_mm, overscan_mm=0.0)
                ov_img_detached = finalize_image_surface(ov_img, device_pixel_ratio=dpr)
                painter.drawImage(QtCore.QRectF(0.0, 0.0, float(vp_w), float(vp_h)), ov_img_detached)
            else:
                # Full path: rebuild content (without guides), cache it, then draw guides on top
                du_content = DrawUtil()
                du_content.set_current_page_size_mm(page_w_mm, page_h_mm)
                if self._editor is not None:
                    self._editor.draw_all(du_content)
                # Rasterize content to cache image
                c_img, c_surf, _c_buf = make_image_surface(vis_w_px, vis_h_px)
                c_ctx = cairo.Context(c_surf)
                try:
                    c_ctx.set_antialias(cairo.ANTIALIAS_BEST)
                except Exception:
                    print('CairoEditorWidget.paintEvent: Warning: failed to set antialiasing mode')
                du_content.render_to_cairo(c_ctx, du_content.current_page_index(), px_per_mm, clip_mm, overscan_mm=0.0)
                c_img_detached = finalize_image_surface(c_img, device_pixel_ratio=dpr)
                # Cache the content layer for overlay-only repaints
                self._content_cache_image = c_img_detached
                self._content_cache_key = cache_key
                painter.drawImage(QtCore.QRectF(0.0, 0.0, float(vp_w), float(vp_h)), c_img_detached)

                # Now render guides and composite
                du_guides = DrawUtil()
                du_guides.set_current_page_size_mm(page_w_mm, page_h_mm)
                if self._editor is not None:
                    try:
                        self._editor.draw_guides(du_guides)
                    except Exception:
                        pass
                g_img, g_surf, _g_buf = make_image_surface(vis_w_px, vis_h_px)
                g_ctx = cairo.Context(g_surf)
                try:
                    g_ctx.set_antialias(cairo.ANTIALIAS_BEST)
                except Exception:
                    pass
                du_guides.render_to_cairo(g_ctx, du_guides.current_page_index(), px_per_mm, clip_mm, overscan_mm=0.0)
                g_img_detached = finalize_image_surface(g_img, device_pixel_ratio=dpr)
                painter.drawImage(QtCore.QRectF(0.0, 0.0, float(vp_w), float(vp_h)), g_img_detached)

            # Optional viewport debug overlay: draw a red border around viewport
            if os.getenv('PIANOSCRIPT_DEBUG_VIEWPORT', '0') in ('1', 'true', 'True'):
                pen = QtGui.QPen(QtGui.QColor(220, 40, 40))
                pen.setWidth(1)
                painter.setPen(pen)
                painter.setBrush(QtGui.QBrush())
                painter.drawRect(QtCore.QRectF(0.5, 0.5, float(vp_w) - 1.0, float(vp_h) - 1.0))
        finally:
            painter.end()
        # Reset the overlay-only hint after a paint pass
        self._overlay_only_repaint = False

    def apply_zoom_steps(self, steps: int) -> None:
        """Adjust zoom multiplicatively and preserve time-cursor anchoring."""
        if steps == 0 or self._editor is None:
            return
        sc = self._editor.current_score()
        if sc is None:
            return
        ed = getattr(sc, 'editor', None)
        if ed is None:
            return
        anchor_units = getattr(self._editor, 'time_cursor', None)
        anchor_y_logical_px = None
        if anchor_units is not None:
            try:
                abs_mm_before = self._editor.time_to_mm(float(anchor_units))
                clip_y_mm = float(self._scroll_logical_px) * float(self._last_dpr) / max(1e-6, float(self._last_px_per_mm))
                anchor_y_logical_px = (abs_mm_before - clip_y_mm) * (float(self._last_px_per_mm) / max(1e-6, float(self._last_dpr)))
            except Exception:
                anchor_y_logical_px = None
        current = float(getattr(ed, 'zoom_mm_per_quarter', 5.0) or 5.0)
        factor = (1.10 ** steps)
        new_zoom = max(10.0, min(200.0, current * factor))
        try:
            ed.zoom_mm_per_quarter = float(new_zoom)
        except Exception:
            pass
        if anchor_y_logical_px is not None:
            try:
                abs_mm_after = self._editor.time_to_mm(float(anchor_units))
                new_clip_y_mm = abs_mm_after - (float(anchor_y_logical_px) * float(self._last_dpr) / max(1e-6, float(self._last_px_per_mm)))
                new_scroll = int(round(new_clip_y_mm * float(self._last_px_per_mm) / max(1e-6, float(self._last_dpr))))
                new_scroll = max(0, new_scroll)
                vp_h_px = int(max(1, self.size().height() * float(self._last_dpr)))
                max_scroll = max(0, int(round((int(self._content_h_px) - vp_h_px) / max(1.0, float(self._last_dpr)))))
                if new_scroll > max_scroll:
                    new_scroll = max_scroll
                if new_scroll != self._scroll_logical_px:
                    self._scroll_logical_px = new_scroll
                    self.scrollLogicalPxChanged.emit(new_scroll)
            except Exception:
                pass
        try:
            self.scrollWheelUsed.emit()
        except Exception:
            pass
        # Repaint; metrics will be recomputed and emitted in paintEvent
        self.update()

    def wheelEvent(self, ev: QtGui.QWheelEvent) -> None:
        # Ctrl+Wheel: adjust vertical zoom via SCORE.editor.zoom_mm_per_quarter
        angle = ev.angleDelta().y()
        if angle == 0:
            ev.accept()
            return
        mods = ev.modifiers()
        try:
            ctrl_down = bool(mods & QtCore.Qt.KeyboardModifier.ControlModifier)
        except Exception:
            ctrl_down = False

        if ctrl_down and self._editor is not None:
            steps = int(round(angle / 120.0))
            if steps != 0:
                self.apply_zoom_steps(steps)
            ev.accept()
            return

        # Default: bounded wheel scrolling using external scrollbar semantics
        scale = max(1.0, float(self._last_dpr))
        step_logical_px = int(max(1, round(40.0 * (float(self._last_px_per_mm) / scale))))
        steps = int(round(angle / 120.0))
        delta = -steps * step_logical_px  # negative angle scrolls down visually
        new_val = max(0, int(self._scroll_logical_px + delta))
        vp_h_px = int(max(1, self.size().height() * scale))
        max_scroll = max(0, int(round((int(self._content_h_px) - vp_h_px) / scale)))
        if new_val > max_scroll:
            new_val = max_scroll
        if new_val != self._scroll_logical_px:
            self._scroll_logical_px = new_val
            self.scrollLogicalPxChanged.emit(new_val)
            self.scrollWheelUsed.emit()
            self.update()
        ev.accept()

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        # Take focus on any mouse press so shortcuts are active
        self.setFocus()
        self._last_pos = ev.position()
        self._last_sent_pos = ev.position()
        # Update modifier state on editor (Shift/Ctrl tracking)
        try:
            mods = ev.modifiers()
            shift_down = bool(mods & QtCore.Qt.KeyboardModifier.ShiftModifier)
            ctrl_down = bool(mods & QtCore.Qt.KeyboardModifier.ControlModifier)
            if self._editor is not None:
                self._editor.set_shift_down(shift_down)
                self._editor.set_ctrl_down(ctrl_down)
        except Exception:
            pass
        # On content changes, invalidate cached content
        self._content_cache_image = None
        self._content_cache_key = None
        if self._editor:
            if ev.button() == QtCore.Qt.MouseButton.LeftButton:
                self._left_down = True
                self._editor.mouse_press(1, ev.position().x(), ev.position().y())
                # Immediately request repaint for direct press feedback
                self.update()
            elif ev.button() == QtCore.Qt.MouseButton.RightButton:
                self._right_down = True
                self._editor.mouse_press(2, ev.position().x(), ev.position().y())
                # Immediate repaint for right press as well
                self.update()
        # Ensure we receive the matching release even if the pointer leaves the widget
        if self._left_down or self._right_down:
            self.grabMouse()
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent) -> None:
        # Always record last raw position
        self._last_pos = ev.position()
        # Update modifier state continuously during drag/move
        try:
            mods = ev.modifiers()
            shift_down = bool(mods & QtCore.Qt.KeyboardModifier.ShiftModifier)
            ctrl_down = bool(mods & QtCore.Qt.KeyboardModifier.ControlModifier)
            if self._editor is not None:
                self._editor.set_shift_down(shift_down)
                self._editor.set_ctrl_down(ctrl_down)
        except Exception:
            pass
        # Coalesce moves and dispatch at most 30 times/sec
        self._pending_move = ev.position()
        try:
            if self._move_timer and self._fps_interval_ms > 0 and not self._move_timer.isActive():
                # Fire one immediately for responsiveness, then continue at 30 Hz
                self._dispatch_throttled_move()
                self._move_timer.start()
        except Exception:
            # Fallback: if timer unavailable, deliver immediately (no throttle)
            if self._editor:
                lp = self._last_sent_pos or self._last_pos or ev.position()
                dx = ev.position().x() - lp.x()
                dy = ev.position().y() - lp.y()
                self._editor.mouse_move(ev.position().x(), ev.position().y(), dx, dy)
                self._last_sent_pos = ev.position()
                # Request repaint so shared guides render immediately
                # Use overlay-only repaint if no buttons are pressed
                if not (self._left_down or self._right_down):
                    self._overlay_only_repaint = True
                self.update()
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent) -> None:
        # Flush any pending move before release to deliver final position
        try:
            if self._pending_move is not None:
                self._dispatch_throttled_move()
            if self._move_timer and self._move_timer.isActive():
                self._move_timer.stop()
        except Exception:
            pass
        # Content may have changed during drag; drop cache
        self._content_cache_image = None
        self._content_cache_key = None
        if self._editor:
            if ev.button() == QtCore.Qt.MouseButton.LeftButton:
                self._left_down = False
                self._editor.mouse_release(1, ev.position().x(), ev.position().y())
                # Ensure a repaint after left click/release actions
                self.update()
            elif ev.button() == QtCore.Qt.MouseButton.RightButton:
                self._right_down = False
                self._editor.mouse_release(2, ev.position().x(), ev.position().y())
                # Ensure a repaint after right click/release actions
                self.update()
        # Release mouse capture when no buttons remain pressed
        if not (self._left_down or self._right_down):
            self.releaseMouse()
        super().mouseReleaseEvent(ev)

    def mouseDoubleClickEvent(self, ev: QtGui.QMouseEvent) -> None:
        if self._editor:
            if ev.button() == QtCore.Qt.MouseButton.LeftButton:
                self._editor.mouse_double_click(1, ev.position().x(), ev.position().y())
            elif ev.button() == QtCore.Qt.MouseButton.RightButton:
                self._editor.mouse_double_click(2, ev.position().x(), ev.position().y())
        super().mouseDoubleClickEvent(ev)

    def keyPressEvent(self, ev: QtGui.QKeyEvent) -> None:
        key = ev.key()
        mods = ev.modifiers()
        if self._editor is not None:
            try:
                if ev.matches(QtGui.QKeySequence.StandardKey.SelectAll):
                    self._editor.select_all()
                    if hasattr(self, 'request_overlay_refresh'):
                        self.request_overlay_refresh()
                    else:
                        self.update()
                    ev.accept()
                    return
            except Exception:
                pass
            if key in (QtCore.Qt.Key_BracketLeft, QtCore.Qt.Key_BracketRight):
                try:
                    hand = '<' if key == QtCore.Qt.Key_BracketLeft else '>'
                    if self._editor.set_selected_notes_hand(hand):
                        # Force full redraw (not overlay-only) so note styling updates immediately
                        self._content_cache_image = None
                        self._content_cache_key = None
                        self.update()
                        ev.accept()
                        return
                except Exception:
                    pass
            if key in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Right):
                try:
                    delta = -1 if key == QtCore.Qt.Key_Left else 1
                    if self._editor.transpose_selected_notes(delta):
                        self.update()
                        ev.accept()
                        return
                except Exception:
                    pass
            if key in (QtCore.Qt.Key_Up, QtCore.Qt.Key_Down):
                try:
                    units = float(getattr(self._editor, 'snap_size_units', 0.0) or 0.0)
                    if units <= 0.0:
                        units = 1.0
                    delta = -units if key == QtCore.Qt.Key_Up else units
                    if self._editor.shift_selected_notes_time(delta):
                        self.update()
                        ev.accept()
                        return
                except Exception:
                    pass
            # Delete selection on Backspace/Delete
            if key in (QtCore.Qt.Key_Backspace, QtCore.Qt.Key_Delete):
                try:
                    if self._editor.delete_selection():
                        self.update()
                        ev.accept()
                        return
                except Exception:
                    pass
            if key == QtCore.Qt.Key_Comma:
                self._editor.hand_cursor = '<'
                # Overlay-only guide refresh is enough
                if hasattr(self, 'request_overlay_refresh'):
                    self.request_overlay_refresh()
                ev.accept()
                return
            if key == QtCore.Qt.Key_Period:
                self._editor.hand_cursor = '>'
                if hasattr(self, 'request_overlay_refresh'):
                    self.request_overlay_refresh()
                ev.accept()
                return
            if key == QtCore.Qt.Key_Escape:
                # Trigger window close; MainWindow.closeEvent will run save prompt
                try:
                    w = self.window()
                    if isinstance(w, QtWidgets.QWidget):
                        w.close()
                    ev.accept()
                    return
                except Exception:
                    pass
            # Explicit per-platform shortcuts
            try:
                is_mac = (sys.platform == "darwin")
                ctrl = bool(mods & QtCore.Qt.KeyboardModifier.ControlModifier)
                meta = bool(mods & QtCore.Qt.KeyboardModifier.MetaModifier)
                shift = bool(mods & QtCore.Qt.KeyboardModifier.ShiftModifier)
                # Undo
                if (not is_mac and ctrl and (key == QtCore.Qt.Key_Z) and not shift) or \
                   (is_mac and meta and (key == QtCore.Qt.Key_Z) and not shift):
                    self._editor.undo()
                    self.update()
                    ev.accept()
                    return
                # Redo (Shift+Z)
                if (not is_mac and ctrl and shift and (key == QtCore.Qt.Key_Z)) or \
                   (is_mac and meta and shift and (key == QtCore.Qt.Key_Z)):
                    self._editor.redo()
                    self.update()
                    ev.accept()
                    return
                # Fallback to platform-aware key sequences
                if ev.matches(QtGui.QKeySequence.StandardKey.Undo):
                    self._editor.undo()
                    self.update()
                    ev.accept()
                    return
                if ev.matches(QtGui.QKeySequence.StandardKey.Redo):
                    self._editor.redo()
                    self.update()
                    ev.accept()
                    return
            except Exception:
                pass
        super().keyPressEvent(ev)

    def enterEvent(self, ev: QtCore.QEvent) -> None:
        # Show guides when the mouse enters the editor
        if self._editor is not None:
            self._editor.guides_active = True
            self.request_overlay_refresh()
        super().enterEvent(ev)

    def leaveEvent(self, ev: QtCore.QEvent) -> None:
        # Hide guides when the mouse leaves the editor
        if self._editor is not None:
            self._editor.guides_active = False
            self.request_overlay_refresh()
        super().leaveEvent(ev)

    def focusOutEvent(self, ev: QtGui.QFocusEvent) -> None:
        # Do not steal focus back when a modal dialog is active (e.g., time signature dialog)
        try:
            active_modal = None
            try:
                active_modal = QtWidgets.QApplication.activeModalWidget()
            except Exception:
                active_modal = None
            w = self.window()
            if isinstance(w, QtWidgets.QWidget) and w.isActiveWindow() and active_modal is None:
                QtCore.QTimer.singleShot(0, self.setFocus)
        except Exception:
            pass
        super().focusOutEvent(ev)

    def _dispatch_throttled_move(self) -> None:
        """Deliver at most one coalesced move event per timer tick (~30 Hz)."""
        if not self._editor:
            self._pending_move = None
            return
        pos = self._pending_move
        self._pending_move = None
        if pos is None:
            # No new data; stop timer to save CPU
            try:
                if self._move_timer and self._move_timer.isActive():
                    self._move_timer.stop()
            except Exception:
                pass
            return
        lp = self._last_sent_pos or pos
        dx = pos.x() - lp.x()
        dy = pos.y() - lp.y()
        self._editor.mouse_move(pos.x(), pos.y(), dx, dy)
        self._last_sent_pos = pos
        # Request repaint so shared guides render at the new position
        # Use overlay-only repaint when just moving the mouse (no buttons)
        if not (self._left_down or self._right_down):
            self._overlay_only_repaint = True
        self.update()
        # Update status bar with note attributes if hovering a note rect
        try:
            self._update_hover_note_status(pos.x(), pos.y())
        except Exception:
            pass

    def _configure_move_timer_from_prefs(self) -> None:
        prefs = get_preferences()
        raw = prefs.get('editor_fps_limit', 30)
        try:
            fps = int(raw)
        except Exception:
            fps = 30
        # 0 disables throttling (deliver immediately); otherwise clamp 1..1000
        if fps <= 0:
            self._fps_interval_ms = 0
            try:
                if self._move_timer:
                    self._move_timer.stop()
            except Exception:
                pass
            return
        fps = max(1, min(1000, fps))
        self._fps_interval_ms = max(1, int(round(1000.0 / float(fps))))
        if self._move_timer:
            try:
                self._move_timer.setInterval(self._fps_interval_ms)
            except Exception:
                pass

    def _update_hover_note_status(self, x_px: float, y_px: float) -> None:
        """If hovering a note rectangle, show its attributes in the status bar."""
        if self._editor is None:
            return
        try:
            note_id = self._editor.hit_test_note_id(float(x_px), float(y_px))
        except Exception:
            note_id = None
        if note_id is None:
            # Clear memo and clear status bar message when leaving any note
            self._last_hover_note_id = None
            try:
                from ui.main_window import MainWindow  # local import to avoid cycle
            except Exception:
                MainWindow = None  # type: ignore
            w = self.window()
            if MainWindow is not None and isinstance(w, MainWindow):
                try:
                    w._status("", 0)
                except Exception:
                    pass
            return
        if self._last_hover_note_id == int(note_id):
            return
        self._last_hover_note_id = int(note_id)
        # Fetch note details and emit to MainWindow status bar
        try:
            n = self._editor.get_note_by_id(int(note_id))
            if n is None:
                return
            time_val = float(getattr(n, 'time', 0.0) or 0.0)
            pitch_val = int(getattr(n, 'pitch', 0) or 0)
            dur_val = float(getattr(n, 'duration', 0.0) or 0.0)
            vel_val = int(getattr(n, 'velocity', 64) or 64)
            hand_val = str(getattr(n, 'hand', '<') or '<')
            # Compute measure index and whether a rest follows this note
            try:
                measure_idx = int(self._editor.get_measure_index_for_time(time_val))
            except Exception:
                measure_idx = 0
            msg = (
                f"Note: time: {time_val:.0f}, pitch: {pitch_val}, duration: {dur_val:.0f}, "
                f"velocity: {vel_val}, hand: {hand_val}, id: {int(note_id)}, "
                f"measure: {measure_idx}"
            )
            w = self.window()
            # Call MainWindow._status if available
            try:
                from ui.main_window import MainWindow  # local import to avoid cycle at module load
            except Exception:
                MainWindow = None  # type: ignore
            if MainWindow is not None and isinstance(w, MainWindow):
                try:
                    w._status(msg, 0)
                except Exception:
                    pass
        except Exception:
            pass
