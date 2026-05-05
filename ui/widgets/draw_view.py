from __future__ import annotations
from PySide6 import QtCore, QtGui, QtWidgets
import cairo
from ui.widgets.draw_util import DrawUtil, make_image_surface, finalize_image_surface
from ui.style import Style
from utils.CONSTANT import ENGRAVER_LAYERING
from engraver.engraver import do_engrave



class RenderEmitter(QtCore.QObject):
    rendered = QtCore.Signal(QtGui.QImage, int, float)


class RenderTask(QtCore.QRunnable):
    def __init__(self, draw_util: DrawUtil, w_px: int, h_px: int, px_per_mm: float, dpr: float, page_index: int, emitter: RenderEmitter, score: dict | None = None, perform_engrave: bool = False, render_zoom_factor: float = 1.0):
        super().__init__()
        self.setAutoDelete(True)
        self._du = draw_util
        self._w_px = w_px
        self._h_px = h_px
        self._px_per_mm = px_per_mm
        self._dpr = dpr
        self._page_index = page_index
        self._emitter = emitter
        self._score = score
        self._perform_engrave = perform_engrave
        self._render_zoom_factor = float(render_zoom_factor)

    def run(self) -> None:
        # Optionally run engraving to update DrawUtil from score before rendering.
        if self._perform_engrave and self._score is not None:
            try:
                do_engrave(self._score, self._du)
            except Exception as e:
                # Fail engraving silently for now; could emit an error signal if desired.
                print(f"Engrave error: {e}")
        image, surface, _buf = make_image_surface(self._w_px, self._h_px)
        ctx = cairo.Context(surface)
        self._du.render_to_cairo(ctx, self._page_index, self._px_per_mm, layering=ENGRAVER_LAYERING)
        # Detach the image from the temporary buffer so Python memory can be reclaimed
        final = finalize_image_surface(image, device_pixel_ratio=self._dpr)
        # Emit back to the UI thread, but skip if the emitter is gone (e.g., view closed)
        try:
            if self._emitter is not None:
                self._emitter.rendered.emit(final, self._page_index, self._render_zoom_factor)
        except RuntimeError:
            # Emitter already deleted; ignore
            pass


class DrawUtilView(QtWidgets.QWidget):
    def __init__(self, draw_util: DrawUtil, parent=None):
        super().__init__(parent)
        self._du = draw_util
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self._image: QtGui.QImage | None = None
        self._prev_image: QtGui.QImage | None = None
        self._fade_progress: float = 1.0
        self._fade_elapsed_ms: int = 0
        self._fade_duration_ms: int = 500
        self._fade_timer = QtCore.QTimer(self)
        self._fade_timer.setInterval(16)
        self._fade_timer.timeout.connect(self._on_fade_tick)
        self._page_index = max(0, self._du.current_page_index())
        self._pool = QtCore.QThreadPool.globalInstance()
        self._emitter = RenderEmitter()
        self._emitter.rendered.connect(self._on_rendered)
        # Allow splitter to fully collapse this view
        self.setMinimumWidth(0)
        self._last_px_per_mm: float = 1.0  # device px per mm
        self._last_widget_px_per_mm: float = 1.0  # widget px per mm
        self._last_dpr: float = 1.0
        self._last_w_px: int = 0
        self._last_h_px: int = 0
        self._scroll_x_px: float = 0.0
        self._scroll_y_px: float = 0.0
        self._zoom_factor: float = 1.0
        self._image_render_zoom_factor: float = 1.0
        self._prev_image_render_zoom_factor: float = 1.0
        self._zoom_step_factor: float = 1.1
        self._max_zoom_factor: float = 8.0
        self._score: dict | None = None
        self._page_prev_cb = None
        # Playhead overlay: set by set_playhead_overlay(), drawn in paintEvent
        self._playhead_y_mm: float | None = None
        self._playhead_x1_mm: float | None = None
        self._playhead_x2_mm: float | None = None
        self._page_next_cb = None
        # Resize throttling: scale existing image during drag, re-render after settle
        self._resizing: bool = False
        self._resize_timer = QtCore.QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(180)
        self._resize_timer.timeout.connect(self._on_resize_settle)
        self._zoom_rerender_timer = QtCore.QTimer(self)
        self._zoom_rerender_timer.setSingleShot(True)
        self._zoom_rerender_timer.setInterval(150)
        self._zoom_rerender_timer.timeout.connect(self._on_zoom_settle)
        self._suppress_fade_once: bool = False
        # Apply a dedicated background color for DrawUtil views
        try:
            accent = Style.get_named_qcolor('alternate_background_color')
            pal = self.palette()
            pal.setColor(QtGui.QPalette.Window, accent)
            self.setPalette(pal)
            self.setAutoFillBackground(True)
            self.setAttribute(QtCore.Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        except Exception:
            pass

    def set_page(self, index: int, request_render: bool = True):
        self._page_index = index
        self._scroll_x_px = 0.0
        self._scroll_y_px = 0.0
        if request_render:
            self.request_render()

    def set_page_turn_callbacks(self, prev_cb, next_cb) -> None:
        self._page_prev_cb = prev_cb
        self._page_next_cb = next_cb

    def set_playhead_overlay(self, y_mm: float, x1_mm: float, x2_mm: float) -> None:
        """Set the print-view playhead position and trigger a repaint."""
        self._playhead_y_mm = float(y_mm)
        self._playhead_x1_mm = float(x1_mm)
        self._playhead_x2_mm = float(x2_mm)
        self._update_playhead_scroll()
        self.update()

    def clear_playhead_overlay(self) -> None:
        """Remove the playhead overlay and trigger a repaint."""
        self._playhead_y_mm = None
        self._playhead_x1_mm = None
        self._playhead_x2_mm = None
        self.update()

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(600, 800)

    def _is_horizontal_read_direction(self) -> bool:
        layout = (self._score or {}).get('layout', {}) or {}
        return str(layout.get('read_direction', 'vertical') or 'vertical').strip().lower() == 'horizontal'

    def _fit_scale_for_page(self, page_w_mm: float, page_h_mm: float, dpr: float) -> float:
        return (max(1, self.width()) * float(dpr)) / max(1e-6, float(page_w_mm))

    def _zoom_modifiers_active(self, mods: QtCore.Qt.KeyboardModifiers) -> bool:
        try:
            return bool(
                (mods & QtCore.Qt.KeyboardModifier.ControlModifier)
                or (mods & QtCore.Qt.KeyboardModifier.MetaModifier)
            )
        except Exception:
            return False

    def _apply_zoom_steps(self, steps: int, anchor_pos: QtCore.QPointF | None = None) -> None:
        if steps == 0:
            return
        old_zoom = float(self._zoom_factor)
        new_zoom = float(old_zoom) * (float(self._zoom_step_factor) ** float(steps))
        new_zoom = max(1.0, min(float(self._max_zoom_factor), float(new_zoom)))
        if abs(new_zoom - old_zoom) <= 1e-6:
            return

        if anchor_pos is None:
            anchor_x = float(self.width()) * 0.5
            anchor_y = float(self.height()) * 0.5
        else:
            anchor_x = float(anchor_pos.x())
            anchor_y = float(anchor_pos.y())

        zoom_ratio = float(new_zoom) / max(1e-6, float(old_zoom))
        self._scroll_x_px = max(0.0, (float(self._scroll_x_px) + anchor_x) * zoom_ratio - anchor_x)
        self._scroll_y_px = max(0.0, (float(self._scroll_y_px) + anchor_y) * zoom_ratio - anchor_y)
        self._zoom_factor = float(new_zoom)
        self.update()
        self._zoom_rerender_timer.start()

    def reset_view_state(self) -> None:
        self._zoom_factor = 1.0
        self._scroll_x_px = 0.0
        self._scroll_y_px = 0.0
        self.update()
        self._zoom_rerender_timer.start()

    def _scaled_image_metrics(self, img: QtGui.QImage, render_zoom_factor: float | None = None) -> tuple[int, int, int, int, float, float, float]:
        img_w = float(img.width()) / max(1e-6, float(img.devicePixelRatio()))
        img_h = float(img.height()) / max(1e-6, float(img.devicePixelRatio()))
        image_zoom = float(render_zoom_factor) if render_zoom_factor is not None else float(self._image_render_zoom_factor)
        scale = max(0.05, float(self._zoom_factor) / max(1e-6, image_zoom))
        if self._resizing and img_w > 0.0 and img_h > 0.0:
            # During live resize, preserve zoom while approximating new fit-to-width.
            scale *= float(max(1, self.width())) / float(img_w)
        tgt_w = int(round(img_w * scale))
        tgt_h = int(round(img_h * scale))
        max_scroll_x = max(0.0, float(tgt_w - self.width()))
        max_scroll_y = max(0.0, float(tgt_h - self.height()))
        self._scroll_x_px = max(0.0, min(float(max_scroll_x), float(self._scroll_x_px)))
        self._scroll_y_px = max(0.0, min(float(max_scroll_y), float(self._scroll_y_px)))
        if tgt_w <= self.width():
            img_x = (self.width() - tgt_w) // 2
        else:
            img_x = -int(round(self._scroll_x_px))
        if tgt_h <= self.height():
            img_y = (self.height() - tgt_h) // 2
        else:
            img_y = -int(round(self._scroll_y_px))
        return (tgt_w, tgt_h, img_x, img_y, scale, max_scroll_x, max_scroll_y)

    def _playhead_midpoint_output_px(self, ppm: float) -> tuple[float, float] | None:
        if (
            self._playhead_y_mm is None
            or self._playhead_x1_mm is None
            or self._playhead_x2_mm is None
        ):
            return None
        x1_mm_out, y1_mm_out = self._du.map_page_point_to_output_mm(
            float(self._playhead_x1_mm),
            float(self._playhead_y_mm),
            self._page_index,
        )
        x2_mm_out, y2_mm_out = self._du.map_page_point_to_output_mm(
            float(self._playhead_x2_mm),
            float(self._playhead_y_mm),
            self._page_index,
        )
        return (
            ((float(x1_mm_out) + float(x2_mm_out)) * 0.5) * float(ppm),
            ((float(y1_mm_out) + float(y2_mm_out)) * 0.5) * float(ppm),
        )

    def _update_playhead_scroll(self) -> None:
        if self._image is None:
            return

        # Compute the playhead midpoint in output pixels, then adjust scroll to center it if possible.
        tgt_w, tgt_h, _img_x, _img_y, _scale, max_scroll_x, max_scroll_y = self._scaled_image_metrics(self._image)
        page_w_mm, page_h_mm = self._du.current_page_size_mm()
        ppm_x = float(tgt_w) / max(1e-6, float(page_w_mm))
        ppm_y = float(tgt_h) / max(1e-6, float(page_h_mm))
        ppm = min(ppm_x, ppm_y) if ppm_x > 0.0 and ppm_y > 0.0 else max(ppm_x, ppm_y)
        midpoint_px = self._playhead_midpoint_output_px(ppm)
        if midpoint_px is None:
            return
        mid_x_px, mid_y_px = midpoint_px
        changed = False
        if self._is_horizontal_read_direction():
            if max_scroll_x > 0.0:
                target_scroll_x = max(0.0, min(float(max_scroll_x), float(mid_x_px - (self.width() * 0.5))))
                if abs(float(self._scroll_x_px) - float(target_scroll_x)) > 0.5:
                    self._scroll_x_px = float(target_scroll_x)
                    changed = True
        else:
            if max_scroll_y > 0.0:
                target_scroll_y = max(0.0, min(float(max_scroll_y), float(mid_y_px - (self.height() * 0.5))))
                if abs(float(self._scroll_y_px) - float(target_scroll_y)) > 0.5:
                    self._scroll_y_px = float(target_scroll_y)
                    changed = True
        if changed:
            self.update()

    @QtCore.Slot()
    def request_render(self):
        w = max(1, self.width())
        h = max(1, self.height())
        dpr = float(self.devicePixelRatioF())
        page_count = self._du.page_count()
        if page_count <= 0:
            return
        if self._page_index >= page_count:
            self._page_index = max(0, page_count - 1)
        page_w_mm, page_h_mm = self._du.current_page_size_mm()
        if page_w_mm <= 0 or page_h_mm <= 0:
            return
        render_zoom = max(1.0, float(self._zoom_factor))
        px_per_mm = self._fit_scale_for_page(page_w_mm, page_h_mm, dpr) * render_zoom
        h_px = int(page_h_mm * px_per_mm)
        w_px = int(page_w_mm * px_per_mm)
        # Store metrics for hit-testing
        self._last_px_per_mm = px_per_mm
        self._last_widget_px_per_mm = float(px_per_mm) / max(1e-6, float(dpr))
        self._last_dpr = dpr
        self._last_w_px = w_px
        self._last_h_px = h_px
        task = RenderTask(self._du, w_px, h_px, px_per_mm, dpr, self._page_index, self._emitter, self._score, False, render_zoom)
        self._pool.start(task)

    def _on_zoom_settle(self) -> None:
        self._suppress_fade_once = True
        self.request_render()

    def _on_resize_settle(self) -> None:
        self._resizing = False
        self.request_render()

    @QtCore.Slot()
    def request_engrave_and_render(self):
        """Deprecated: engraving is managed by Engraver. Kept for compatibility."""
        self.request_render()

    def resizeEvent(self, ev: QtGui.QResizeEvent) -> None:
        super().resizeEvent(ev)
        if self._image is None:
            self.request_render()
            return
        # Scale the current image while dragging; re-render once the resize settles
        self._resizing = True
        self._suppress_fade_once = True
        self._resize_timer.start()
        self.update()

    def paintEvent(self, ev: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        try:
            painter.fillRect(self.rect(), self.palette().window())
        except Exception:
            painter.fillRect(self.rect(), QtCore.Qt.GlobalColor.white)
        if self._image is not None:
            try:
                paper_qcolor = Style.get_paper_qcolor()
            except Exception:
                paper_qcolor = QtCore.Qt.GlobalColor.white

            tgt_w, tgt_h, x, y, _scale, _max_scroll_x, _max_scroll_y = self._scaled_image_metrics(self._image)

            # Keep a permanent paper layer behind transition frames so cross-fades remain white.
            painter.fillRect(QtCore.QRect(x, y, tgt_w, tgt_h), paper_qcolor)

            def _draw_image(img: QtGui.QImage, opacity: float) -> None:
                painter.save()
                painter.setOpacity(opacity)
                if img is self._prev_image:
                    img_zoom = float(self._prev_image_render_zoom_factor)
                else:
                    img_zoom = float(self._image_render_zoom_factor)
                tgt_w, tgt_h, x, y, _scale, _max_scroll_x, _max_scroll_y = self._scaled_image_metrics(img, img_zoom)
                painter.drawImage(QtCore.QRect(x, y, tgt_w, tgt_h), img)
                painter.restore()

            if self._prev_image is not None and self._fade_progress < 1.0:
                # Keep the old frame fully visible and fade only the new frame on top.
                _draw_image(self._prev_image, 1.0)
                _draw_image(self._image, self._fade_progress)
            else:
                _draw_image(self._image, 1.0)

        # --- Playhead overlay (drawn on top of the page image) ---
        if (
            self._playhead_y_mm is not None
            and self._playhead_x1_mm is not None
            and self._playhead_x2_mm is not None
            and self._image is not None
        ):
            try:
                from ui.style import Style as _Style
                _accent_rgb = _Style.get_named_rgb('accent', fallback=(51, 153, 255))
                _ph_color = QtGui.QColor(
                    int(_accent_rgb[0]),
                    int(_accent_rgb[1]),
                    int(_accent_rgb[2]),
                    int(0.8 * 255),
                )
                # Convert mm → widget px using the same transform as the image
                _tgt_w, _tgt_h, _img_x, _img_y, _scale, _max_scroll_x, _max_scroll_y = self._scaled_image_metrics(self._image)
                # widget px per mm: tgt_w / page_w_mm
                try:
                    _page_w_mm, _page_h_mm = self._du.current_page_size_mm()
                    _ppm_x = float(_tgt_w) / max(1e-6, float(_page_w_mm))
                    _ppm_y = float(_tgt_h) / max(1e-6, float(_page_h_mm))
                    _ppm = min(_ppm_x, _ppm_y) if _ppm_x > 0.0 and _ppm_y > 0.0 else max(_ppm_x, _ppm_y)
                except Exception:
                    _ppm = float(_tgt_w) / 210.0
                # Map from unrotated drawing-space mm to rendered output-space mm
                # so the overlay follows the same page rotation as DrawUtil.
                _x1_mm_out, _y1_mm_out = self._du.map_page_point_to_output_mm(
                    float(self._playhead_x1_mm),
                    float(self._playhead_y_mm),
                    self._page_index,
                )
                _x2_mm_out, _y2_mm_out = self._du.map_page_point_to_output_mm(
                    float(self._playhead_x2_mm),
                    float(self._playhead_y_mm),
                    self._page_index,
                )
                _x1_px = _img_x + int(round(_x1_mm_out * _ppm))
                _y1_px = _img_y + int(round(_y1_mm_out * _ppm))
                _x2_px = _img_x + int(round(_x2_mm_out * _ppm))
                _y2_px = _img_y + int(round(_y2_mm_out * _ppm))
                _pen = QtGui.QPen(_ph_color)
                _pen.setWidthF(max(1.5, _ppm * 1.0))
                _pen.setCapStyle(QtCore.Qt.PenCapStyle.FlatCap)
                painter.save()
                painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
                painter.setPen(_pen)
                painter.drawLine(_x1_px, _y1_px, _x2_px, _y2_px)
                painter.restore()
            except Exception:
                pass

        painter.end()

    def wheelEvent(self, ev: QtGui.QWheelEvent) -> None:
        if self._zoom_modifiers_active(ev.modifiers()):
            angle_delta = ev.angleDelta()
            raw = angle_delta.y() if angle_delta.y() != 0 else angle_delta.x()
            steps = int(round(float(raw) / 120.0))
            if steps != 0:
                self._apply_zoom_steps(steps, ev.position())
            ev.accept()
            return

        if self._image is None:
            ev.ignore()
            return

        _tgt_w, _tgt_h, _img_x, _img_y, _scale, max_scroll_x, max_scroll_y = self._scaled_image_metrics(self._image)
        shift_down = bool(ev.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier)
        use_horizontal = bool(shift_down and max_scroll_x > 0.0)
        if not use_horizontal and max_scroll_y <= 0.0 and max_scroll_x > 0.0:
            use_horizontal = True
        if not use_horizontal and max_scroll_y <= 0.0:
            ev.ignore()
            return

        pixel_delta = ev.pixelDelta()
        angle_delta = ev.angleDelta()
        delta = pixel_delta.x() if use_horizontal else pixel_delta.y()
        if delta == 0:
            delta = angle_delta.x() if use_horizontal else angle_delta.y()
        if delta == 0:
            delta = angle_delta.y() if use_horizontal else angle_delta.x()
        if delta != 0:
            delta = delta / 2
        if delta == 0:
            ev.ignore()
            return
        if use_horizontal:
            self._scroll_x_px = max(0.0, min(float(max_scroll_x), float(self._scroll_x_px - delta)))
        else:
            self._scroll_y_px = max(0.0, min(float(max_scroll_y), float(self._scroll_y_px - delta)))
        self.update()
        ev.accept()

    def keyPressEvent(self, ev: QtGui.QKeyEvent) -> None:
        if self._zoom_modifiers_active(ev.modifiers()) and ev.key() == QtCore.Qt.Key.Key_Down:
            self._zoom_factor = 1.0
            self.update()
            self._zoom_rerender_timer.start()
            ev.accept()
            return
        super().keyPressEvent(ev)

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        if ev.button() == QtCore.Qt.MouseButton.LeftButton and callable(self._page_prev_cb):
            self._page_prev_cb()
            return
        if ev.button() == QtCore.Qt.MouseButton.RightButton and callable(self._page_next_cb):
            self._page_next_cb()
            return
        if self._image is None:
            return
        # Convert from widget px to page mm
        _tgt_w, _tgt_h, x_offset_px, y_offset_px, view_scale, _max_scroll_x, _max_scroll_y = self._scaled_image_metrics(self._image)
        x_px = ev.position().x() - x_offset_px
        y_px = ev.position().y() - y_offset_px
        if x_px < 0 or y_px < 0 or x_px > float(_tgt_w) or y_px > float(_tgt_h):
            return
        # Use widget px per mm for conversion (since event positions are in widget px)
        px_per_mm_widget = self._last_widget_px_per_mm * max(1e-6, float(view_scale))
        x_mm = float(x_px) / px_per_mm_widget
        y_mm = float(y_px) / px_per_mm_widget
        hit = self._du.hit_test_point_mm(x_mm, y_mm, self._page_index)
        if hit is not None:
            # Simple console feedback for now
            hit_id = getattr(hit, "id", 0)
            hit_tags = getattr(hit, "tags", [])
            hit_rect = getattr(hit, "hit_rect_mm", None)
            print(f"Hit: type={type(hit).__name__} id={hit_id} tags={hit_tags} rect_mm={hit_rect}")
        else:
            print("Hit: none")

    def document_changed(self) -> None:
        # Convenience for callers after mutating the DrawUtil
        self.request_render()

    def set_score(self, score: dict | None) -> None:
        self._score = score
        # Reflect paper size from file model layout into DrawUtil
        try:
            layout = (score or {}).get('layout', {}) or {}
            w_mm = float(layout.get('page_width_mm', 0.0) or 0.0)
            h_mm = float(layout.get('page_height_mm', 0.0) or 0.0)
            if w_mm > 0 and h_mm > 0:
                self._du.set_current_page_size_mm(w_mm, h_mm)
                # Trigger rerender with new dimensions
                self.request_render()
        except Exception:
            pass

    @QtCore.Slot(QtGui.QImage, int, float)
    def _on_rendered(self, image: QtGui.QImage, page_index: int, render_zoom_factor: float):
        if page_index != self._page_index:
            return
        if self._suppress_fade_once:
            self._prev_image = None
            self._fade_progress = 1.0
            self._fade_elapsed_ms = 0
            self._fade_timer.stop()
            self._suppress_fade_once = False
        elif self._image is not None:
            self._prev_image = self._image
            self._prev_image_render_zoom_factor = float(self._image_render_zoom_factor)
            self._fade_progress = 0.0
            self._fade_elapsed_ms = 0
            self._fade_timer.start()
        else:
            self._prev_image = None
            self._fade_progress = 1.0
        self._image = image
        self._image_render_zoom_factor = max(1.0, float(render_zoom_factor))
        self._update_playhead_scroll()
        self.update()

    def _on_fade_tick(self) -> None:
        self._fade_elapsed_ms += int(self._fade_timer.interval())
        if self._fade_duration_ms <= 0:
            self._fade_progress = 1.0
        else:
            self._fade_progress = min(1.0, float(self._fade_elapsed_ms) / float(self._fade_duration_ms))
        if self._fade_progress >= 1.0:
            self._fade_timer.stop()
            self._prev_image = None
        self.update()

    def closeEvent(self, ev: QtGui.QCloseEvent) -> None:
        # No persistent threads; nothing special to stop.
        super().closeEvent(ev)

    def shutdown(self) -> None:
        # Using QThreadPool tasks that finish automatically; nothing to do.
        pass
