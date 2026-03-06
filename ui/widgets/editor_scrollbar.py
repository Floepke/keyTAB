from __future__ import annotations

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
        self._jump_style = _JumpToClickScrollBarStyle(self.style())
        self.setStyle(self._jump_style)

    def set_tooltip_provider(self, provider: Optional[Callable[[int], str]]) -> None:
        self._tooltip_provider = provider

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
        if provider is None:
            QtWidgets.QToolTip.hideText()
            return

        if self._slider_rect().contains(pos.toPoint()):
            QtWidgets.QToolTip.hideText()
            return

        predicted_value = self._predicted_value_for_pos(pos)
        text = str(provider(predicted_value) or "").strip()

        if not text:
            QtWidgets.QToolTip.hideText()
            return

        fm = QtGui.QFontMetrics(QtWidgets.QToolTip.font())
        text_w = int(fm.horizontalAdvance(text))
        text_h = int(fm.height())
        tip_w = text_w + 40
        tip_h = text_h + 10

        scrollbar_left_global_x = self.mapToGlobal(QtCore.QPoint(0, 0)).x()
        tip_x = int(scrollbar_left_global_x - tip_w)
        tip_y = int(global_pos.y() - (tip_h * 1.7))

        QtWidgets.QToolTip.showText(QtCore.QPoint(tip_x, tip_y), text, self)

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
