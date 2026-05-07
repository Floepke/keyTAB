from __future__ import annotations

from typing import Callable, Optional

from PySide6 import QtCore, QtGui, QtWidgets
from ui.style import Style


class _JumpToClickScrollBarStyle(QtWidgets.QProxyStyle):
    """Scrollbar style override to jump on groove click."""

    def styleHint(self, hint, option=None, widget=None, returnData=None):
        if hint == QtWidgets.QStyle.StyleHint.SH_ScrollBar_LeftClickAbsolutePosition:
            return 1
        if hint == QtWidgets.QStyle.StyleHint.SH_ToolTip_WakeUpDelay:
            return 0
        if hint == QtWidgets.QStyle.StyleHint.SH_ToolTip_FallAsleepDelay:
            return 0
        return super().styleHint(hint, option, widget, returnData)


class EditorScrollBar(QtWidgets.QScrollBar):
    """Editor scrollbar with jump-click behavior and instant predictive tooltip."""

    def __init__(self, orientation: QtCore.Qt.Orientation, parent=None) -> None:
        super().__init__(orientation, parent)
        self.setMouseTracking(True)
        self._tooltip_provider: Optional[Callable[[int], str]] = None
        self._measure_index_provider: Optional[Callable[[int], int]] = None
        self._jump_target_provider: Optional[Callable[[int], int]] = None
        self._measure_tooltip_font_point_size: Optional[float] = 32.0
        self._measure_tooltip_font_family: str = "Edwin"
        self._measure_popup = QtWidgets.QLabel(self.window() if isinstance(self.window(), QtWidgets.QWidget) else self)
        self._measure_popup.setWindowFlags(
            QtCore.Qt.WindowType.ToolTip
            | QtCore.Qt.WindowType.FramelessWindowHint
        )
        self._measure_popup.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self._measure_popup.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._measure_popup.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._measure_popup.hide()
        self._jump_style = _JumpToClickScrollBarStyle(self.style())
        self.setStyle(self._jump_style)

    def _hide_measure_popup(self) -> None:
        self._measure_popup.hide()

    def _show_measure_popup(self, text: str, global_pos: QtCore.QPoint) -> None:
        self._ensure_measure_popup_parent()

        tooltip_font = QtGui.QFont(QtWidgets.QToolTip.font())
        tooltip_font.setFamily(self._measure_tooltip_font_family)
        custom_pt = self._measure_tooltip_font_point_size
        if custom_pt is not None:
            tooltip_font.setPointSizeF(custom_pt)
        self._measure_popup.setFont(tooltip_font)

        bg_rgb = Style.get_named_rgb('bg', (240, 240, 240))
        fg_rgb = Style.get_named_rgb('text', (0, 0, 0))
        bg = QtGui.QColor(int(bg_rgb[0]), int(bg_rgb[1]), int(bg_rgb[2]))
        fg = QtGui.QColor(int(fg_rgb[0]), int(fg_rgb[1]), int(fg_rgb[2]))
        border = fg
        self._measure_popup.setStyleSheet(
            f"QLabel {{ background: {bg.name()}; color: {fg.name()}; "
            f"border: 1px solid {border.name()}; border-radius: 3px; padding: 2px 4px; }}"
        )
        self._measure_popup.setText(str(text))
        self._measure_popup.adjustSize()

        tip_w = int(self._measure_popup.width())
        tip_h = int(self._measure_popup.height())
        if self.orientation() == QtCore.Qt.Orientation.Vertical:
            scrollbar_left_global_x = self.mapToGlobal(QtCore.QPoint(0, 0)).x()
            tip_x = int(scrollbar_left_global_x - tip_w - 2)
            tip_y = int(global_pos.y() - (tip_h // 2))
        else:
            scrollbar_top_global_y = self.mapToGlobal(QtCore.QPoint(0, 0)).y()
            tip_x = int(global_pos.x() - (tip_w // 2))
            tip_y = int(scrollbar_top_global_y - tip_h - 2)
        self._measure_popup.move(tip_x, tip_y)
        self._measure_popup.show()

    def _ensure_measure_popup_parent(self) -> None:
        host = self.window() if isinstance(self.window(), QtWidgets.QWidget) else None
        if host is not None and self._measure_popup.parentWidget() is not host:
            self._measure_popup.setParent(host)
            self._measure_popup.setWindowFlags(
                QtCore.Qt.WindowType.ToolTip
                | QtCore.Qt.WindowType.FramelessWindowHint
            )
            self._measure_popup.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
            self._measure_popup.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        popup_handle = self._measure_popup.windowHandle()
        host_handle = host.windowHandle() if host is not None else None
        if popup_handle is not None and host_handle is not None:
            popup_handle.setTransientParent(host_handle)

    def set_tooltip_provider(self, provider: Optional[Callable[[int], str]]) -> None:
        self._tooltip_provider = provider

    def set_measure_index_provider(self, provider: Optional[Callable[[int], int]]) -> None:
        self._measure_index_provider = provider

    def set_jump_target_provider(self, provider: Optional[Callable[[int], int]]) -> None:
        self._jump_target_provider = provider

    def _style_option(self) -> QtWidgets.QStyleOptionSlider:
        opt = QtWidgets.QStyleOptionSlider()
        self.initStyleOption(opt)
        return opt

    def _slider_rect(self) -> QtCore.QRect:
        opt = self._style_option()
        return self.style().subControlRect(
            QtWidgets.QStyle.ComplexControl.CC_ScrollBar,
            opt,
            QtWidgets.QStyle.SubControl.SC_ScrollBarSlider,
            self,
        )

    def _groove_rect(self) -> QtCore.QRect:
        opt = self._style_option()
        return self.style().subControlRect(
            QtWidgets.QStyle.ComplexControl.CC_ScrollBar,
            opt,
            QtWidgets.QStyle.SubControl.SC_ScrollBarGroove,
            self,
        )

    def _predicted_value_for_pos(self, pos: QtCore.QPointF) -> int:
        groove = self._groove_rect()
        slider = self._slider_rect()
        if not groove.isValid() or groove.width() <= 0 or groove.height() <= 0:
            return int(self.value())

        opt = self._style_option()
        if self.orientation() == QtCore.Qt.Orientation.Vertical:
            span = max(1, groove.height() - slider.height())
            rel_pos = int(round(float(pos.y()) - float(groove.top()) - (float(slider.height()) * 0.5)))
        else:
            span = max(1, groove.width() - slider.width())
            rel_pos = int(round(float(pos.x()) - float(groove.left()) - (float(slider.width()) * 0.5)))

        rel_pos = max(0, min(span, rel_pos))
        v = QtWidgets.QStyle.sliderValueFromPosition(
            int(self.minimum()),
            int(self.maximum()),
            int(rel_pos),
            int(span),
            bool(getattr(opt, 'upsideDown', False)),
        )
        return int(max(self.minimum(), min(self.maximum(), v)))

    def _update_tooltip(self, pos: QtCore.QPointF, global_pos: QtCore.QPoint) -> None:
        provider = self._tooltip_provider
        measure_index_provider = self._measure_index_provider
        if provider is None and measure_index_provider is None:
            self._hide_measure_popup()
            return

        if self._slider_rect().contains(pos.toPoint()):
            self._hide_measure_popup()
            return

        predicted_value = self._predicted_value_for_pos(pos)
        text = ""
        if measure_index_provider is not None:
            measure_index = int(measure_index_provider(predicted_value))
            text = str(max(1, measure_index + 1))
        elif provider is not None:
            text = str(provider(predicted_value) or "").strip()

        if not text:
            self._hide_measure_popup()
            return
        self._show_measure_popup(text, global_pos)

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        if ev.button() == QtCore.Qt.MouseButton.LeftButton and not self._slider_rect().contains(ev.position().toPoint()):
            jump_target_provider = self._jump_target_provider
            if jump_target_provider is not None:
                predicted_value = self._predicted_value_for_pos(ev.position())
                target_value = int(jump_target_provider(predicted_value))
                clamped_target = int(max(self.minimum(), min(self.maximum(), target_value)))
                if int(self.value()) != clamped_target:
                    self.setValue(clamped_target)
                ev.accept()
                return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent) -> None:
        self._update_tooltip(ev.position(), ev.globalPosition().toPoint())
        super().mouseMoveEvent(ev)

    def enterEvent(self, ev: QtCore.QEvent) -> None:
        local = self.mapFromGlobal(QtGui.QCursor.pos())
        self._update_tooltip(QtCore.QPointF(local), QtGui.QCursor.pos())
        super().enterEvent(ev)

    def leaveEvent(self, ev: QtCore.QEvent) -> None:
        self._hide_measure_popup()
        super().leaveEvent(ev)

    def hideEvent(self, ev: QtGui.QHideEvent) -> None:
        self._hide_measure_popup()
        super().hideEvent(ev)

    def refresh_hover_tooltip(self) -> None:
        """Re-evaluate hover tooltip for the current cursor position."""
        try:
            if not self.isVisible():
                self._hide_measure_popup()
                return
            global_pos = QtGui.QCursor.pos()
            local_pos = self.mapFromGlobal(global_pos)
            if self.rect().contains(local_pos):
                self._update_tooltip(QtCore.QPointF(local_pos), global_pos)
            else:
                self._hide_measure_popup()
        except Exception:
            self._hide_measure_popup()

    def event(self, ev: QtCore.QEvent) -> bool:
        # Keep predictive measure tooltip responsive even when Qt only emits
        # tooltip events (for example after modal dialogs or when cursor rests).
        if ev.type() == QtCore.QEvent.Type.ToolTip and isinstance(ev, QtGui.QHelpEvent):
            self._update_tooltip(QtCore.QPointF(ev.pos()), ev.globalPos())
            ev.accept()
            return True
        return super().event(ev)
