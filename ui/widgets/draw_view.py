from __future__ import annotations
from PySide6 import QtCore, QtGui, QtWidgets
import cairo
from ui.widgets.draw_util import DrawUtil, make_image_surface, finalize_image_surface
from ui.style import Style
from utils.CONSTANT import ENGRAVER_LAYERING
from engraver.engraver import do_engrave



class RenderEmitter(QtCore.QObject):
    rendered = QtCore.Signal(QtGui.QImage, int)


class RenderTask(QtCore.QRunnable):
    def __init__(self, draw_util: DrawUtil, w_px: int, h_px: int, px_per_mm: float, dpr: float, page_index: int, emitter: RenderEmitter, score: dict | None = None, perform_engrave: bool = False):
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
                self._emitter.rendered.emit(final, self._page_index)
        except RuntimeError:
            # Emitter already deleted; ignore
            pass


class DrawUtilView(QtWidgets.QWidget):
    def __init__(self, draw_util: DrawUtil, parent=None):
        super().__init__(parent)
        self._du = draw_util
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
        self._last_h_px: int = 0
        self._scroll_px: float = 0.0
        self._score: dict | None = None
        self._page_prev_cb = None
        self._page_next_cb = None
        # Resize throttling: scale existing image during drag, re-render after settle
        self._resizing: bool = False
        self._resize_timer = QtCore.QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(180)
        self._resize_timer.timeout.connect(self._on_resize_settle)
        self._suppress_fade_once: bool = False
        # Apply a dedicated background color for DrawUtil views
        try:
            accent = Style.get_named_qcolor('accent')
            pal = self.palette()
            pal.setColor(QtGui.QPalette.Window, accent)
            self.setPalette(pal)
            self.setAutoFillBackground(True)
            self.setAttribute(QtCore.Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        except Exception:
            pass

    def set_page(self, index: int, request_render: bool = True):
        self._page_index = index
        self._scroll_px = 0.0
        if request_render:
            self.request_render()

    def set_page_turn_callbacks(self, prev_cb, next_cb) -> None:
        self._page_prev_cb = prev_cb
        self._page_next_cb = next_cb

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(600, 800)

    @QtCore.Slot()
    def request_render(self):
        w = max(1, self.width())
        dpr = float(self.devicePixelRatioF())
        page_count = self._du.page_count()
        if page_count <= 0:
            return
        if self._page_index >= page_count:
            self._page_index = max(0, page_count - 1)
        page_w_mm, page_h_mm = self._du.current_page_size_mm()
        if page_w_mm <= 0 or page_h_mm <= 0:
            return
        px_per_mm = (w * dpr) / page_w_mm
        h_px = int(page_h_mm * px_per_mm)
        w_px = int(w * dpr)
        # Store metrics for hit-testing
        self._last_px_per_mm = px_per_mm
        self._last_widget_px_per_mm = (w) / page_w_mm
        self._last_dpr = dpr
        self._last_h_px = h_px
        task = RenderTask(self._du, w_px, h_px, px_per_mm, dpr, self._page_index, self._emitter, self._score, False)
        self._pool.start(task)

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

            def _draw_image(img: QtGui.QImage, opacity: float) -> None:
                painter.save()
                painter.setOpacity(opacity)
                img_w = img.width() / img.devicePixelRatio()
                img_h = img.height() / img.devicePixelRatio()
                # During resize, scale current image to widget width to avoid re-engraving per frame
                if self._resizing and img_w > 0:
                    scale = float(self.width()) / float(img_w)
                else:
                    scale = 1.0
                tgt_w = int(round(img_w * scale))
                tgt_h = int(round(img_h * scale))
                x = 0
                if tgt_h <= self.height():
                    y = (self.height() - tgt_h) // 2
                else:
                    max_scroll = max(0, tgt_h - self.height())
                    self._scroll_px = max(0.0, min(float(max_scroll), float(self._scroll_px)))
                    y = -int(round(self._scroll_px))
                # Keep the page area light during fades to avoid a dark flash between frames.
                painter.fillRect(QtCore.QRect(x, y, tgt_w, tgt_h), paper_qcolor)
                painter.drawImage(QtCore.QRect(x, y, tgt_w, tgt_h), img)
                painter.restore()

            if self._prev_image is not None and self._fade_progress < 1.0:
                _draw_image(self._prev_image, 1.0 - self._fade_progress)
                _draw_image(self._image, self._fade_progress)
            else:
                _draw_image(self._image, 1.0)
        painter.end()

    def wheelEvent(self, ev: QtGui.QWheelEvent) -> None:
        if self._image is None:
            return
        img_h = int(self._image.height() / self._image.devicePixelRatio())
        if img_h <= self.height():
            return
        delta = ev.pixelDelta().y()
        if delta == 0:
            delta = ev.angleDelta().y() / 2
        if delta == 0:
            return
        max_scroll = max(0, img_h - self.height())
        self._scroll_px = max(0.0, min(float(max_scroll), float(self._scroll_px - delta)))
        self.update()
        ev.accept()

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
        img_h = int(self._image.height() / self._last_dpr)
        if img_h <= self.height():
            y_offset_px = (self.height() - img_h) // 2
        else:
            y_offset_px = -int(round(self._scroll_px))
        x_px = ev.position().x()
        y_px = ev.position().y() - y_offset_px
        if y_px < 0 or y_px > (self._last_h_px / self._last_dpr) or x_px < 0:
            return
        # Use widget px per mm for conversion (since event positions are in widget px)
        x_mm = float(x_px) / self._last_widget_px_per_mm
        y_mm = float(y_px) / self._last_widget_px_per_mm
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

    @QtCore.Slot(QtGui.QImage, int)
    def _on_rendered(self, image: QtGui.QImage, page_index: int):
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
            self._fade_progress = 0.0
            self._fade_elapsed_ms = 0
            self._fade_timer.start()
        else:
            self._prev_image = None
            self._fade_progress = 1.0
        self._image = image
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
