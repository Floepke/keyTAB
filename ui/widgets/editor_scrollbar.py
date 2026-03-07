from __future__ import annotations
import html

from typing import Callable, Optional

from PySide6 import QtCore, QtGui, QtWidgets


class _JumpToClickScrollBarStyle(QtWidgets.QProxyStyle):
    """Scrollbar style override to jump on groove click."""

    def styleHint(self, hint, option=None, widget=None, returnData=None):
        if hint == QtWidgets.QStyle.StyleHint.SH_ScrollBar_LeftClickAbsolutePosition:
            return 1
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
        self._jump_style = _JumpToClickScrollBarStyle(self.style())
        self.setStyle(self._jump_style)

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
            QtWidgets.QToolTip.hideText()
            return

        if self._slider_rect().contains(pos.toPoint()):
            QtWidgets.QToolTip.hideText()
            return

        predicted_value = self._predicted_value_for_pos(pos)
        text = ""
        if measure_index_provider is not None:
            measure_index = int(measure_index_provider(predicted_value))
            text = str(max(1, measure_index + 1))
        elif provider is not None:
            text = str(provider(predicted_value) or "").strip()

        if not text:
            QtWidgets.QToolTip.hideText()
            return

        tooltip_font = QtGui.QFont(QtWidgets.QToolTip.font())
        tooltip_font.setFamily(self._measure_tooltip_font_family)
        custom_pt = self._measure_tooltip_font_point_size
        if custom_pt is not None:
            tooltip_font.setPointSizeF(custom_pt)

        fm = QtGui.QFontMetrics(tooltip_font)
        text_w = int(fm.horizontalAdvance(text))
        text_h = int(fm.height())
        tip_w = text_w + 6
        tip_h = text_h - (custom_pt)

        scrollbar_left_global_x = self.mapToGlobal(QtCore.QPoint(0, 0)).x()
        tip_x = int(scrollbar_left_global_x - tip_w)
        tip_y = int(global_pos.y() - (tip_h * 1.7))

        safe_text = html.escape(text)
        style_parts = [f"font-family:{self._measure_tooltip_font_family}"]
        if custom_pt is not None:
            style_parts.append(f"font-size:{custom_pt}pt")
        rich_text = f"<span style='{' ; '.join(style_parts)}'>{safe_text}</span>"
        QtWidgets.QToolTip.showText(QtCore.QPoint(tip_x, tip_y), rich_text, self)

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
        QtWidgets.QToolTip.hideText()
        super().leaveEvent(ev)
