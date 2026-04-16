from __future__ import annotations
import json
from pathlib import Path
from dataclasses import asdict, fields
from typing import Any, get_args, get_origin, get_type_hints, Literal, TYPE_CHECKING

from PySide6 import QtCore, QtGui, QtWidgets

from file_model.layout import LAYOUT_FLOAT_CONFIG
from file_model.font import Font
from file_model.layout import Layout
from file_model.SCORE import SCORE


FONT_OFFSET_FIELDS = {
    'font_title',
    'font_composer',
    'font_copyright',
    'font_arranger',
    'font_lyricist',
}

class ClickSlider(QtWidgets.QSlider):
    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            if self.orientation() == QtCore.Qt.Orientation.Horizontal:
                pos = ev.position().x()
                span = max(1.0, float(self.width()))
                val = self.minimum() + (self.maximum() - self.minimum()) * (pos / span)
            else:
                pos = ev.position().y()
                span = max(1.0, float(self.height()))
                val = self.maximum() - (self.maximum() - self.minimum()) * (pos / span)
            self.setSliderPosition(int(round(val)))
            self.setSliderDown(True)
            self.sliderMoved.emit(self.sliderPosition())
            ev.accept()
            return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent) -> None:
        if self.isSliderDown():
            if self.orientation() == QtCore.Qt.Orientation.Horizontal:
                pos = ev.position().x()
                span = max(1.0, float(self.width()))
                val = self.minimum() + (self.maximum() - self.minimum()) * (pos / span)
            else:
                pos = ev.position().y()
                span = max(1.0, float(self.height()))
                val = self.maximum() - (self.maximum() - self.minimum()) * (pos / span)
            self.setSliderPosition(int(round(val)))
            self.sliderMoved.emit(self.sliderPosition())
            ev.accept()
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent) -> None:
        if self.isSliderDown():
            self.setSliderDown(False)
        super().mouseReleaseEvent(ev)


class FloatSliderEdit(QtWidgets.QWidget):
    valueChanged = QtCore.Signal(float)

    def __init__(self, value: float, min_value: float, max_value: float, step: float, parent=None) -> None:
        super().__init__(parent)
        self._min = float(min_value)
        self._max = float(max_value)
        self._step = float(step)
        self._decimals = self._step_decimals(self._step)
        self._slider = ClickSlider(QtCore.Qt.Orientation.Horizontal, self)
        self._edit = QtWidgets.QLineEdit(self)
        self._edit.setMinimumWidth(70)
        self._edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self._edit.setValidator(QtGui.QRegularExpressionValidator(QtCore.QRegularExpression(r"[0-9.,]+"), self))
        self._dec_btn = QtWidgets.QToolButton(self)
        self._dec_btn.setText("-")
        self._inc_btn = QtWidgets.QToolButton(self)
        self._inc_btn.setText("+")
        for btn in (self._dec_btn, self._inc_btn):
            btn.setAutoRepeat(True)
            btn.setAutoRepeatDelay(300)
            btn.setAutoRepeatInterval(75)
            btn.setFixedWidth(28)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._slider, 1)
        layout.addWidget(self._edit, 0)
        layout.addWidget(self._dec_btn, 0)
        layout.addWidget(self._inc_btn, 0)
        self._apply_range()
        self.set_value(value)
        self._slider.valueChanged.connect(self._on_slider_changed)
        self._edit.editingFinished.connect(self._on_edit_finished)
        self._dec_btn.clicked.connect(lambda: self._nudge(-1))
        self._inc_btn.clicked.connect(lambda: self._nudge(1))
        self._slider.installEventFilter(self)

    @staticmethod
    def _step_decimals(step: float) -> int:
        s = f"{float(step):.10f}".rstrip("0")
        if "." not in s:
            return 0
        return max(0, min(6, len(s.split(".", 1)[1])))

    def _format_value(self, value: float) -> str:
        return f"{float(value):.{self._decimals}f}"

    def _apply_range(self) -> None:
        steps = max(1, int(round((self._max - self._min) / max(1e-6, self._step))))
        self._slider.setRange(0, steps)
        self._slider.setSingleStep(1)
        self._slider.setPageStep(max(1, int(round(steps / 10.0))))

    def _clamp(self, val: float) -> float:
        return max(self._min, min(self._max, val))

    def _snap(self, val: float) -> float:
        if self._step <= 0:
            return val
        snapped = round(val / self._step) * self._step
        return round(snapped, self._decimals)

    def _slider_to_value(self, sv: int) -> float:
        return self._min + float(sv) * self._step

    def _value_to_slider(self, val: float) -> int:
        return int(round((val - self._min) / self._step))

    def set_value(self, value: float) -> None:
        val = self._snap(self._clamp(float(value)))
        self._slider.blockSignals(True)
        self._slider.setValue(self._value_to_slider(val))
        self._slider.blockSignals(False)
        self._edit.setText(self._format_value(val))

    def value(self) -> float:
        val = self._slider_to_value(self._slider.value())
        return self._snap(self._clamp(val))

    def _on_slider_changed(self, _v: int) -> None:
        val = self.value()
        self._edit.setText(self._format_value(val))
        self.valueChanged.emit(val)

    def _on_edit_finished(self) -> None:
        text = self._edit.text().strip()
        try:
            val = float(text.replace(',', '.'))
        except Exception:
            val = self.value()
        val = self._snap(self._clamp(val))
        self.set_value(val)
        self.valueChanged.emit(val)

    def eventFilter(self, obj: QtCore.QObject, ev: QtCore.QEvent) -> bool:
        if obj is self._slider and ev.type() == QtCore.QEvent.Type.Wheel:
            delta = ev.angleDelta().y() or ev.angleDelta().x()
            if delta:
                steps = int(delta / 120)
                if steps != 0:
                    self.set_value(self.value() + steps * self._step)
                    self.valueChanged.emit(self.value())
                    ev.accept()
                    return True
        return super().eventFilter(obj, ev)

    def wheelEvent(self, ev: QtGui.QWheelEvent) -> None:
        delta = ev.angleDelta().y()
        if delta == 0:
            return
        step = self._step if self._step > 0 else 1.0
        cur = self.value()
        direction = 1.0 if delta > 0 else -1.0
        self.set_value(cur + (direction * step))
        self.valueChanged.emit(self.value())
        ev.accept()

    def _nudge(self, direction: int) -> None:
        base_step = self._step if self._step > 0 else max((self._max - self._min) / 200.0, 0.01)
        new_val = self.value() + float(direction) * base_step
        self.set_value(new_val)
        self.valueChanged.emit(self.value())


class ColorPickerEdit(QtWidgets.QWidget):
    valueChanged = QtCore.Signal(str)

    PRESET_COLORS = (
        '#777', '#888', '#999', '#aaa', '#bbb', '#ccc', '#ddd', '#eee',
        '#d88ba0', '#d78490', '#d98f83', '#d49a78', '#cfaa72', '#c6b86d',
        '#b6c46f', '#9ec173', '#86bf82', '#77bc96', '#6db8aa', '#6eaec1',
        '#769fca', '#818fd1', '#907fd0', '#a079cc', '#b274c6', '#c070bf',
        '#ca74b0', '#cf7f9f',
    )

    class _SwatchDelegate(QtWidgets.QStyledItemDelegate):
        def paint(self, painter: QtGui.QPainter, option: QtWidgets.QStyleOptionViewItem, index: QtCore.QModelIndex) -> None:
            color_code = str(index.data(QtCore.Qt.ItemDataRole.UserRole) or '').strip()
            color = QtGui.QColor(color_code)
            rect = option.rect.adjusted(4, 3, -4, -3)
            painter.save()
            if option.state & QtWidgets.QStyle.StateFlag.State_Selected:
                painter.fillRect(option.rect, option.palette.highlight())
            fill = color if color.isValid() else QtGui.QColor('#000000')
            border = QtGui.QColor('#666666')
            painter.setPen(QtGui.QPen(border, 1.0))
            painter.setBrush(QtGui.QBrush(fill))
            painter.drawRect(rect)
            painter.restore()

        def sizeHint(self, option: QtWidgets.QStyleOptionViewItem, index: QtCore.QModelIndex) -> QtCore.QSize:
            return QtCore.QSize(max(120, option.rect.width()), 26)

    class _ClosedSwatchCombo(QtWidgets.QComboBox):
        def __init__(self, parent=None) -> None:
            super().__init__(parent)
            self._current_color_code: str = '#000000'

        def set_current_color(self, code: str) -> None:
            txt = str(code or '').strip()
            if txt and not txt.startswith('#'):
                txt = f"#{txt}"
            self._current_color_code = txt if txt else '#000000'
            self.update()

        def paintEvent(self, event: QtGui.QPaintEvent) -> None:
            painter = QtWidgets.QStylePainter(self)
            opt = QtWidgets.QStyleOptionComboBox()
            self.initStyleOption(opt)

            # Draw normal frame and arrow button first.
            painter.drawComplexControl(QtWidgets.QStyle.ComplexControl.CC_ComboBox, opt)

            # Fill the visible combo field (excluding arrow button) with selected color.
            edit_rect = self.style().subControlRect(
                QtWidgets.QStyle.ComplexControl.CC_ComboBox,
                opt,
                QtWidgets.QStyle.SubControl.SC_ComboBoxEditField,
                self,
            )
            fill = QtGui.QColor(self._current_color_code)
            if not fill.isValid():
                fill = QtGui.QColor('#000000')
            painter.fillRect(edit_rect.adjusted(1, 1, -1, -1), fill)
            painter.setPen(QtGui.QPen(QtGui.QColor('#666666'), 1.0))
            painter.drawRect(edit_rect.adjusted(1, 1, -1, -1))

    @staticmethod
    def _make_swatch_icon(code: str, size: int = 18) -> QtGui.QIcon:
        pix = QtGui.QPixmap(size, size)
        pix.fill(QtCore.Qt.GlobalColor.transparent)
        p = QtGui.QPainter(pix)
        try:
            rect = pix.rect().adjusted(1, 1, -1, -1)
            color = QtGui.QColor(str(code or '').strip())
            if not color.isValid():
                color = QtGui.QColor('#000000')
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
            p.setPen(QtGui.QPen(QtGui.QColor('#666666'), 1.0))
            p.setBrush(QtGui.QBrush(color))
            p.drawRect(rect)
        finally:
            p.end()
        return QtGui.QIcon(pix)

    def __init__(self, value: str, parent=None) -> None:
        super().__init__(parent)
        self._combo = self._ClosedSwatchCombo(self)
        self._combo.setEditable(False)
        self._populate_presets()
        self._combo.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        self._combo.setItemDelegate(self._SwatchDelegate(self._combo))
        self._combo.setIconSize(QtCore.QSize(16, 16))
        view = self._combo.view()
        if view is not None:
            view.setStyleSheet("QListView::item { min-height: 24px; padding: 2px 4px; }")
        self._hex_edit = QtWidgets.QLineEdit(self)
        self._hex_edit.setMinimumWidth(92)
        self._hex_edit.setMaximumWidth(110)
        self._hex_edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self._hex_edit.setValidator(
            QtGui.QRegularExpressionValidator(QtCore.QRegularExpression(r"#?[0-9a-fA-F]{0,8}"), self)
        )
        self._button = QtWidgets.QPushButton(self.tr("Pick"), self)
        self._button.setFixedWidth(48)
        self._combo.setMinimumWidth(170)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._combo, 1)
        layout.addWidget(self._hex_edit, 0)
        layout.addWidget(self._button, 0)
        self.set_value(str(value))
        self._button.clicked.connect(self._open_dialog)
        self._combo.activated.connect(self._on_combo_activated)
        self._hex_edit.editingFinished.connect(self._on_edit_finished)

    def _populate_presets(self) -> None:
        self._combo.clear()
        # Dynamic first row reflects the current selected value when it is not in presets.
        self._combo.addItem('')
        self._combo.setItemData(0, '#000000', QtCore.Qt.ItemDataRole.UserRole)
        self._combo.setItemIcon(0, self._make_swatch_icon('#000000'))
        for idx, code in enumerate(self.PRESET_COLORS):
            self._combo.addItem('')
            row = idx + 1
            self._combo.setItemData(row, code, QtCore.Qt.ItemDataRole.UserRole)
            self._combo.setItemIcon(row, self._make_swatch_icon(code))

    def _set_current_swatch(self, txt: str) -> None:
        self._combo.setItemData(0, txt, QtCore.Qt.ItemDataRole.UserRole)
        self._combo.setItemIcon(0, self._make_swatch_icon(txt))
        if isinstance(self._combo, self._ClosedSwatchCombo):
            self._combo.set_current_color(txt)

    def set_value(self, value: str) -> None:
        txt = str(value or '').strip()
        if txt and not txt.startswith('#'):
            txt = f"#{txt}"
        self._set_current_swatch(txt if txt else '#000000')
        self._hex_edit.setText(txt)
        idx = self._combo.findData(txt, QtCore.Qt.ItemDataRole.UserRole)
        self._combo.blockSignals(True)
        self._combo.setCurrentIndex(idx if idx >= 1 else 0)
        self._combo.blockSignals(False)

    def value(self) -> str:
        txt = self._hex_edit.text().strip()
        if txt and not txt.startswith('#'):
            txt = f"#{txt}"
        return txt

    def _on_combo_activated(self, index: int) -> None:
        txt = str(self._combo.itemData(index, QtCore.Qt.ItemDataRole.UserRole) or '').strip()
        if not txt:
            txt = self.value()
        self.set_value(txt)
        self.valueChanged.emit(txt)

    def _open_dialog(self) -> None:
        col = QtGui.QColor(self.value())
        if not col.isValid():
            col = QtGui.QColor(0, 0, 0)
        picked = QtWidgets.QColorDialog.getColor(col, self)
        if not picked.isValid():
            return
        self.set_value(picked.name())
        self.valueChanged.emit(self.value())

    def _on_edit_finished(self) -> None:
        txt = self.value()
        self.set_value(txt)
        self.valueChanged.emit(txt)


class FlexibleDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        try:
            self.setLocale(QtCore.QLocale.c())
        except Exception:
            pass

    def _normalize_text(self, text: str) -> str:
        return text.replace(',', '.')

    def validate(self, text: str, pos: int) -> QtGui.QValidator.State:
        normalized = self._normalize_text(text)
        return super().validate(normalized, pos)

    def valueFromText(self, text: str) -> float:
        normalized = self._normalize_text(text)
        return super().valueFromText(normalized)

    def fixup(self, text: str) -> str:
        return self._normalize_text(text)

    def keyPressEvent(self, ev: QtGui.QKeyEvent) -> None:
        if ev.text() == ',':
            ev = QtGui.QKeyEvent(ev.type(), ev.key(), ev.modifiers(), '.')
        super().keyPressEvent(ev)


class FontPicker(QtWidgets.QWidget):
    valueChanged = QtCore.Signal()

    def __init__(self, value: Font | dict[str, Any], parent=None, show_offsets: bool = False) -> None:
        super().__init__(parent)
        coerced = self._coerce_font_value(value)
        self._font_cls = type(coerced) if isinstance(coerced, Font) else Font
        self._show_offsets = bool(show_offsets)
        self._combo = QtWidgets.QFontComboBox(self)
        self._size = QtWidgets.QSpinBox(self)
        self._size.setRange(1, 200)
        try:
            # Emit changes while typing in the spinbox
            self._size.setKeyboardTracking(True)
        except Exception:
            pass
        self._bold = QtWidgets.QCheckBox(self.tr("Bold"), self)
        self._italic = QtWidgets.QCheckBox(self.tr("Italic"), self)
        self._underline = QtWidgets.QCheckBox(self.tr("Underline"), self)
        self._x_offset: FlexibleDoubleSpinBox | None = None
        self._y_offset: FlexibleDoubleSpinBox | None = None
        if self._show_offsets:
            self._x_offset = FlexibleDoubleSpinBox(self)
            self._y_offset = FlexibleDoubleSpinBox(self)
            for spin, axis in ((self._x_offset, 'X'), (self._y_offset, 'Y')):
                spin.setRange(-500.0, 500.0)
                spin.setDecimals(2)
                spin.setSingleStep(0.25)
                spin.setMinimumWidth(70)
                spin.setKeyboardTracking(True)
                spin.setToolTip(self.tr("{axis}-offset (mm).").format(axis=axis))

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._combo, 1)
        layout.addWidget(self._size, 0)
        layout.addWidget(self._bold, 0)
        layout.addWidget(self._italic, 0)
        layout.addWidget(self._underline, 0)
        if self._show_offsets and self._x_offset and self._y_offset:
            layout.addWidget(self._x_offset, 0)
            layout.addWidget(self._y_offset, 0)

        self.set_value(coerced)
        self._combo.currentFontChanged.connect(lambda _f: self.valueChanged.emit())
        self._size.valueChanged.connect(lambda _v: self.valueChanged.emit())
        try:
            self._size.editingFinished.connect(lambda: self.valueChanged.emit())
        except Exception:
            pass
        self._bold.stateChanged.connect(lambda _v: self.valueChanged.emit())
        self._italic.stateChanged.connect(lambda _v: self.valueChanged.emit())
        self._underline.stateChanged.connect(lambda _v: self.valueChanged.emit())
        if self._show_offsets and self._x_offset and self._y_offset:
            self._x_offset.valueChanged.connect(lambda _v: self.valueChanged.emit())
            self._y_offset.valueChanged.connect(lambda _v: self.valueChanged.emit())

    def _coerce_font_value(self, value: Font | dict[str, Any]) -> Font:
        if isinstance(value, Font):
            return value
        if isinstance(value, dict):
            try:
                return Font(**value)
            except Exception:
                pass
        return Font()

    def set_value(self, value: Font | dict[str, Any]) -> None:
        value = self._coerce_font_value(value)
        try:
            self._combo.setCurrentFont(QtGui.QFont(str(value.family)))
        except Exception:
            pass
        try:
            self._size.setValue(int(value.size_pt))
        except Exception:
            self._size.setValue(10)
        self._bold.setChecked(bool(value.bold))
        self._italic.setChecked(bool(value.italic))
        self._underline.setChecked(bool(getattr(value, 'underline', False)))
        if self._show_offsets and self._x_offset and self._y_offset:
            try:
                self._x_offset.setValue(float(getattr(value, 'x_offset', 0.0) or 0.0))
                self._y_offset.setValue(float(getattr(value, 'y_offset', 0.0) or 0.0))
            except Exception:
                self._x_offset.setValue(0.0)
                self._y_offset.setValue(0.0)

    def set_family(self, family: str) -> None:
        self._combo.setCurrentFont(QtGui.QFont(str(family)))

    def value(self) -> Font:
        font_cls = self._font_cls if isinstance(self._font_cls, type) and issubclass(self._font_cls, Font) else Font
        x_off = float(self._x_offset.value()) if self._x_offset is not None else 0.0
        y_off = float(self._y_offset.value()) if self._y_offset is not None else 0.0
        return font_cls(
            family=str(self._combo.currentFont().family()),
            size_pt=float(self._size.value()),
            bold=bool(self._bold.isChecked()),
            italic=bool(self._italic.isChecked()),
            underline=bool(self._underline.isChecked()),
            x_offset=x_off,
            y_offset=y_off,
        )


class RadioGroupWidget(QtWidgets.QWidget):
    valueChanged = QtCore.Signal(str)

    def __init__(self, options: list[tuple[str, str]], value: str, parent=None) -> None:
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self._buttons: list[tuple[QtWidgets.QRadioButton, str]] = []
        for label, val in options:
            rb = QtWidgets.QRadioButton(label, self)
            lay.addWidget(rb)
            self._buttons.append((rb, val))
            rb.toggled.connect(lambda checked, v=val: self.valueChanged.emit(v) if checked else None)
        self.set_value(value)

    def set_value(self, value: str) -> None:
        val_str = str(value) if value is not None else ''
        matched = False
        for rb, val in self._buttons:
            rb.blockSignals(True)
            rb.setChecked(val == val_str)
            if val == val_str:
                matched = True
            rb.blockSignals(False)
        if not matched and self._buttons:
            self._buttons[0][0].blockSignals(True)
            self._buttons[0][0].setChecked(True)
            self._buttons[0][0].blockSignals(False)

    def value(self) -> str:
        for rb, val in self._buttons:
            if rb.isChecked():
                return val
        return self._buttons[0][1] if self._buttons else ''


class StyleDialog(QtWidgets.QDialog):
    values_changed = QtCore.Signal()
    tab_changed = QtCore.Signal(int)

    def _style_field_labels(self) -> dict[str, str]:
        return {
            'scale': self.tr('Scale'),
            'page_width_mm': self.tr('Page width (mm)'),
            'page_height_mm': self.tr('Page height (mm)'),
            'page_top_margin_mm': self.tr('Top margin (mm)'),
            'page_bottom_margin_mm': self.tr('Bottom margin (mm)'),
            'page_left_margin_mm': self.tr('Left margin (mm)'),
            'page_right_margin_mm': self.tr('Right margin (mm)'),
            'header_height_mm': self.tr('Header height (mm)'),
            'footer_height_mm': self.tr('Footer height (mm)'),
            'black_note_rule': self.tr('Black note rule'),
            'note_head_visible': self.tr('Note head visible'),
            'note_stem_visible': self.tr('Note stem visible'),
            'accidental_visible': self.tr('Accidental line visible'),
            'chord_connect_visible': self.tr('Chord connector visible'),
            'note_stop_visible': self.tr('Note stop visible'),
            'note_stem_length_semitone': self.tr('Note stem length (semitones)'),
            'note_stem_thickness_mm': self.tr('Note stem thickness (mm) (Applies to both stem and the lines of the notehead symbol)'),
            'note_stopsign_thickness_mm': self.tr('Note stop thickness (mm)'),
            'note_leftdot_visible': self.tr('Left handed centered note symbol dot visible'),
            'note_continuation_dot_visible': self.tr('Continuation dot visible'),
            'note_continuation_dot_size_mm': self.tr('Continuation dot size (mm)'),
            'note_midinote_visible': self.tr('MIDI note blocks visible'),
            'note_midinote_left_color': self.tr('MIDI note color (left hand)'),
            'note_midinote_right_color': self.tr('MIDI note color (right hand)'),
            'note_width_scaling': self.tr('Noot width scaling (1 = Perfectly rounded)'),
            'beam_visible': self.tr('Beam visible'),
            'beam_thickness_mm': self.tr('Beam thickness (mm)'),
            'beam_corner_radius_mm': self.tr('Beam corner radius (mm) (rounded corners)'),
            'grace_note_visible': self.tr('Grace note visible'),
            'grace_note_outline_width_mm': self.tr('Grace note outline thickness (mm)'),
            'grace_note_scale': self.tr('Grace note scale'),
            'pedal_lane_enabled': self.tr('Pedal lane enabled'),
            'pedal_lane_width_mm': self.tr('Pedal lane width (mm)'),
            'text_visible': self.tr('Text visible'),
            'text_background_padding_mm': self.tr('Text background padding (mm)'),
            'slur_visible': self.tr('Slur visible'),
            'slur_width_sides_mm': self.tr('Slur side thickness (mm)'),
            'slur_width_middle_mm': self.tr('Slur middle thickness (mm)'),
            'hairpin_visible': self.tr('Hairpin visible'),
            'hairpin_line_width_mm': self.tr('Hairpin line thickness (mm)'),
            'hairpin_width_mm': self.tr('Hairpin width (mm)'),
            'hairpin_text_gap_mm': self.tr('Hairpin text gap (mm)'),
            'dynamic_symbol_font_size_pt': self.tr('Dynamic symbol font size (pt)'),
            'dynamic_symbol_background_padding_mm': self.tr('Dynamic symbol background padding (mm)'),
            'dynamic_symbol_visible': self.tr('Dynamic symbol visible'),
            'repeat_start_visible': self.tr('Start repeat visible'),
            'repeat_end_visible': self.tr('End repeat visible'),
            'double_barline_visible': self.tr('Double barline visible'),
            'countline_visible': self.tr('Count line visible'),
            'countline_dash_pattern': self.tr('Count line dash pattern'),
            'countline_thickness_mm': self.tr('Count line thickness (mm)'),
            'stave_visible': self.tr('Stave visible'),
            'barline_visible': self.tr('Barline visible'),
            'grid_line_visible': self.tr('Grid line visible'),
            'grid_band_visible': self.tr('Grid band visible'),
            'grid_barline_thickness_mm': self.tr('Grid barline thickness (mm)'),
            'grid_gridline_thickness_mm': self.tr('Grid line thickness (mm)'),
            'grid_gridline_dash_pattern_mm': self.tr('Grid line dash pattern (mm)'),
            'grid_band_color': self.tr('Grid band color'),
            'grid_band_start_phase': self.tr('Grid band start phase'),
            'time_signature_visible': self.tr('Time signature visible'),
            'time_signature_indicator_type': self.tr('Time signature indicator type'),
            'time_signature_indicator_lane_width_mm': self.tr('Time signature lane width (mm)'),
            'time_signature_indicator_guide_thickness_mm': self.tr('Time signature guide thickness (mm)'),
            'time_signature_indicator_divide_guide_thickness_mm': self.tr('Time signature divider thickness (mm)'),
            'time_signature_indicator_classic_font': self.tr('Time signature classic font'),
            'time_signature_indicator_klavarskribo_font': self.tr('Time signature Klavarskribo font'),
            'measure_numbering_guide_thickness_mm': self.tr('Measure numbering guide thickness (mm)'),
            'measure_numbering_placement': self.tr('Measure numbering placement'),
            'measure_numbering_guide_visible': self.tr('Measure numbering guide visible'),
            'measure_numbers_visible': self.tr('Measure numbers visible'),
            'measure_numbering_font': self.tr('Measure numbering font'),
            'font_text': self.tr('Text font'),
            'font_title': self.tr('Title font'),
            'font_composer': self.tr('Composer font'),
            'font_copyright': self.tr('Copyright font'),
            'font_arranger': self.tr('Arranger font'),
            'font_lyricist': self.tr('Lyricist font'),
            'stave_two_line_thickness_mm': self.tr('Stave two-line thickness (mm)'),
            'stave_three_line_thickness_mm': self.tr('Stave three-line thickness (mm)'),
            'stave_clef_line_thickness_mm': self.tr('Stave clef line thickness (mm)'),
            'stave_ledger_line_length_mm': self.tr('Stave ledger line length (mm)'),
            'stave_clef_line_dash_pattern_mm': self.tr('Stave clef line dash pattern (mm)'),
        }

    def _coerce_layout_fonts(self, layout_obj: Layout) -> None:
        """Ensure all LayoutFont-typed fields are dataclass instances, not dict payloads."""
        try:
            hints = getattr(self, '_type_hints', None)
            if not isinstance(hints, dict):
                try:
                    hints = get_type_hints(Layout)
                except Exception:
                    hints = {}
            defaults = Layout()
            for f in fields(Layout):
                name = f.name
                hint = hints.get(name, f.type)
                if hint is not Font:
                    continue
                val = getattr(layout_obj, name, getattr(defaults, name))
                if isinstance(val, dict):
                    try:
                        val = Font(**val)
                    except Exception:
                        val = getattr(defaults, name)
                    try:
                        setattr(layout_obj, name, val)
                    except Exception:
                        pass
        except Exception:
            pass

    def _font_family_from_value(self, value: Any, fallback: str = "Edwin") -> str:
        try:
            if isinstance(value, dict):
                fam = value.get('family', fallback)
            else:
                fam = getattr(value, 'family', fallback)
            fam = str(fam or fallback)
            return fam
        except Exception:
            return str(fallback)

    def __init__(self, parent=None, layout: Layout | None = None, score: SCORE | None = None) -> None:
        super().__init__(parent)
        self.setWindowFlag(QtCore.Qt.WindowType.WindowMinMaxButtonsHint, True)
        self.setSizeGripEnabled(True)
        self.setWindowTitle(self.tr("Style"))
        self.setModal(False)
        self.setWindowModality(QtCore.Qt.NonModal)
        self.setMinimumSize(280, 300)
        self.resize(768, 512)

        self._layout = layout or Layout()
        self._editors: dict[str, QtWidgets.QWidget] = {}
        self._score: SCORE | None = score
        self._tab_scrolls: list[QtWidgets.QScrollArea] = []
        self._tab_contents: list[QtWidgets.QWidget] = []
        self._category_list: QtWidgets.QListWidget | None = None
        self._stack: QtWidgets.QStackedWidget | None = None
        self._tab_titles: list[str] = []
        self._all_fonts_combo: QtWidgets.QFontComboBox | None = None
        self._field_tabs: dict[str, str] = {}

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        pages_row = QtWidgets.QHBoxLayout()
        pages_row.setContentsMargins(0, 0, 0, 0)
        pages_row.setSpacing(8)
        lay.addLayout(pages_row, 1)

        category_list = QtWidgets.QListWidget(self)
        category_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        category_list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        category_list.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        category_list.setMinimumWidth(100)
        category_list.setMaximumWidth(160)
        
        # Make category list respond to wheel events to navigate tabs
        class CategoryListWheelFilter(QtCore.QObject):
            def eventFilter(self, obj: QtCore.QObject, ev: QtCore.QEvent) -> bool:
                if ev.type() == QtCore.QEvent.Type.Wheel:
                    wheel_ev = ev  # type: QtGui.QWheelEvent
                    if wheel_ev.angleDelta().y() > 0:
                        new_row = max(0, category_list.currentRow() - 1)
                    else:
                        new_row = min(category_list.count() - 1, category_list.currentRow() + 1)
                    category_list.setCurrentRow(new_row)
                    ev.accept()
                    return True
                return False
        
        wheel_filter = CategoryListWheelFilter(self)
        category_list.installEventFilter(wheel_filter)
        self._category_list = category_list
        self._category_wheel_filter = wheel_filter
        pages_row.addWidget(category_list, 0)

        stack = QtWidgets.QStackedWidget(self)
        self._stack = stack
        pages_row.addWidget(stack, 1)

        def _on_category_changed(index: int) -> None:
            if self._stack is not None and 0 <= int(index) < self._stack.count():
                self._stack.setCurrentIndex(int(index))
            self.tab_changed.emit(max(0, int(index)))

        category_list.currentRowChanged.connect(_on_category_changed)

        tab_order = [
            self.tr("Page"),
            self.tr("Stave"),
            self.tr("Grid"),
            self.tr("Grid band"),
            self.tr("Time signature"),
            self.tr("Measure Numbering"),
            self.tr("Fonts"),
            self.tr("Note"),
            self.tr("Grace note"),
            self.tr("Beam"),
            self.tr("Dynamic"),
            self.tr("Slur"),
            self.tr("Text"),
            self.tr("Countline"),
            self.tr("Pedal"),
            self.tr("Visibility"),
        ]

        def _make_tab(title: str) -> QtWidgets.QVBoxLayout:
            page = QtWidgets.QWidget(self)
            page_layout = QtWidgets.QVBoxLayout(page)
            page_layout.setContentsMargins(0, 0, 0, 0)
            page_layout.setSpacing(0)
            scroll = QtWidgets.QScrollArea(page)
            scroll.setWidgetResizable(True)
            content = QtWidgets.QWidget(scroll)
            vbox = QtWidgets.QVBoxLayout(content)
            vbox.setContentsMargins(6, 6, 6, 6)
            vbox.setSpacing(4)
            content.setLayout(vbox)
            scroll.setWidget(content)
            page_layout.addWidget(scroll, 1)
            if self._stack is not None:
                self._stack.addWidget(page)
            if self._category_list is not None:
                self._category_list.addItem(title)
            self._tab_titles.append(title)
            self._tab_scrolls.append(scroll)
            self._tab_contents.append(content)
            return vbox

        tab_forms: dict[str, QtWidgets.QVBoxLayout] = {t: _make_tab(t) for t in tab_order}
        if self._category_list is not None and self._category_list.count() > 0:
            self._category_list.setCurrentRow(0)

        field_tabs: dict[str, str] = {
            # Page
            'page_width_mm': 'Page',
            'page_height_mm': 'Page',
            'page_top_margin_mm': 'Page',
            'page_bottom_margin_mm': 'Page',
            'page_left_margin_mm': 'Page',
            'page_right_margin_mm': 'Page',
            'header_height_mm': 'Page',
            'footer_height_mm': 'Page',
            'scale': 'Page',
            # Note
            'black_note_rule': 'Note',
            'note_width_scaling': 'Note',
            'note_stem_length_semitone': 'Note',
            'note_stem_thickness_mm': 'Note',
            'note_stopsign_thickness_mm': 'Note',
            'note_continuation_dot_size_mm': 'Note',
            'note_midinote_left_color': 'Note',
            'note_midinote_right_color': 'Note',
            # Beam
            'beam_thickness_mm': 'Beam',
            'beam_corner_radius_mm': 'Beam',
            # Dynamic
            'hairpin_line_width_mm': 'Dynamic',
            'hairpin_width_mm': 'Dynamic',
            'hairpin_font_size_pt': 'Dynamic',
            'hairpin_text_gap_mm': 'Dynamic',
            'dynamic_symbol_background_padding_mm': 'Dynamic',
            # Pedal
            'pedal_lane_width_mm': 'Pedal',
            # Grace note
            'grace_note_outline_width_mm': 'Grace note',
            'grace_note_scale': 'Grace note',
            # Text
            'text_background_padding_mm': 'Text',
            # Slur
            'slur_width_sides_mm': 'Slur',
            'slur_width_middle_mm': 'Slur',
            # Countline
            'countline_dash_pattern': 'Countline',
            'countline_thickness_mm': 'Countline',
            # Time signature
            'time_signature_indicator_type': 'Time signature',
            'time_signature_indicator_lane_width_mm': 'Time signature',
            'time_signature_indicator_guide_thickness_mm': 'Time signature',
            'time_signature_indicator_divide_guide_thickness_mm': 'Time signature',
            # Grid
            'grid_barline_thickness_mm': 'Grid',
            'grid_gridline_thickness_mm': 'Grid',
            'grid_gridline_dash_pattern_mm': 'Grid',
            # Grid band
            'grid_band_color': 'Grid band',
            'grid_band_start_phase': 'Grid band',
            # Stave
            'stave_two_line_thickness_mm': 'Stave',
            'stave_three_line_thickness_mm': 'Stave',
            'stave_clef_line_thickness_mm': 'Stave',
            'stave_ledger_line_length_mm': 'Stave',
            'stave_clef_line_dash_pattern_mm': 'Stave',
            # Fonts
            'font_text': 'Text',
            'font_title': 'Fonts',
            'font_composer': 'Fonts',
            'font_copyright': 'Fonts',
            'font_arranger': 'Fonts',
            'font_lyricist': 'Fonts',
            'time_signature_indicator_classic_font': 'Time signature',
            'time_signature_indicator_klavarskribo_font': 'Time signature',
            # Measure Numbering
            'measure_numbering_guide_thickness_mm': 'Measure Numbering',
            'measure_numbering_placement': 'Measure Numbering',
            'measure_numbering_font': 'Measure Numbering',
            # Visibility
            'note_head_visible': 'Visibility',
            'note_stem_visible': 'Visibility',
            'note_stop_visible': 'Visibility',
            'note_continuation_dot_visible': 'Visibility',
            'note_leftdot_visible': 'Visibility',
            'note_midinote_visible': 'Visibility',
            'stave_visible': 'Visibility',
            'barline_visible': 'Visibility',
            'grid_line_visible': 'Visibility',
            'grid_band_visible': 'Visibility',
            'beam_visible': 'Visibility',
            'grace_note_visible': 'Visibility',
            'pedal_lane_enabled': 'Visibility',
            'text_visible': 'Visibility',
            'hairpin_visible': 'Visibility',
            'slur_visible': 'Visibility',
            'countline_visible': 'Visibility',
            'repeat_start_visible': 'Visibility',
            'repeat_end_visible': 'Visibility',
            'double_barline_visible': 'Visibility',
            'measure_numbering_guide_visible': 'Visibility',
            'measure_numbers_visible': 'Visibility',
            'time_signature_visible': 'Visibility',
            'dynamic_symbol_visible': 'Visibility',
            'accidental_visible': 'Visibility',
            'chord_connect_visible': 'Visibility',
        }

        type_hints = {}
        try:
            type_hints = get_type_hints(Layout)
        except Exception:
            type_hints = {}
        self._type_hints = type_hints
        self._coerce_layout_fonts(self._layout)
        _hide_fields = {
            "measure_grouping",
        }
        self._field_tabs = field_tabs
        field_labels = self._style_field_labels()

        for f in fields(Layout):
            name = f.name
            if name in _hide_fields:
                continue
            fallback_label = name.replace('_', ' ').capitalize()
            label = field_labels.get(name, self.tr(fallback_label))
            value = getattr(self._layout, name)
            field_type = type_hints.get(name, f.type)
            editor = self._make_editor(field_type, value, name)
            if editor is None:
                continue
            self._editors[name] = editor
            tab_name = self.tr(field_tabs.get(name, 'Page'))
            vbox = tab_forms.get(tab_name, tab_forms[self.tr('Page')])
            box = QtWidgets.QGroupBox(label, self)
            box_layout = QtWidgets.QVBoxLayout(box)
            box_layout.setContentsMargins(6, 2, 6, 6)
            box_layout.setSpacing(0)
            box_layout.addWidget(editor)
            vbox.addWidget(box)
            self._wire_editor_change(editor)

        self._add_all_fonts_control(tab_forms.get(self.tr('Fonts')))  # type: ignore[arg-type]

        # Add stretch to each tab's layout to push groupboxes to the top
        for vbox in tab_forms.values():
            vbox.addStretch(1)

        # Style file actions
        actions_row = QtWidgets.QHBoxLayout()
        actions_row.setContentsMargins(0, 0, 0, 0)
        actions_row.setSpacing(8)
        self.save_style_btn = QtWidgets.QPushButton(self.tr("Save Style"), self)
        self.load_style_btn = QtWidgets.QPushButton(self.tr("Load…"), self)
        self.load_tab_style_btn = QtWidgets.QPushButton(self.tr("Load into current tab"), self)
        actions_row.addWidget(self.save_style_btn, 0)
        actions_row.addWidget(self.load_style_btn, 0)
        actions_row.addWidget(self.load_tab_style_btn, 0)
        actions_row.addStretch(1)
        lay.addLayout(actions_row)

        self.msg_label = QtWidgets.QLabel("", self)
        pal = self.msg_label.palette()
        pal.setColor(self.msg_label.foregroundRole(), QtCore.Qt.GlobalColor.darkGreen)
        self.msg_label.setPalette(pal)
        lay.addWidget(self.msg_label)

        self.btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=self,
        )
        self.btns.accepted.connect(self._on_accept_clicked)
        self.btns.rejected.connect(self.reject)
        lay.addWidget(self.btns)

        self.save_style_btn.clicked.connect(self._save_style_to_disk)
        self.load_style_btn.clicked.connect(lambda _=None: self._show_load_menu(scope="all"))
        self.load_tab_style_btn.clicked.connect(lambda _=None: self._show_load_menu(scope="tab"))
        self.load_style_btn.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.load_style_btn.customContextMenuRequested.connect(lambda pos: self._show_load_menu(scope="all", global_pos=self.load_style_btn.mapToGlobal(pos)))
        self.load_tab_style_btn.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.load_tab_style_btn.customContextMenuRequested.connect(lambda pos: self._show_load_menu(scope="tab", global_pos=self.load_tab_style_btn.mapToGlobal(pos)))

        QtCore.QTimer.singleShot(0, self._fit_to_contents)

    def _fit_to_contents(self) -> None:
        category_list = self._category_list
        if category_list is None or not self._tab_scrolls:
            return
        try:
            list_w = int(max(100, min(160, category_list.sizeHintForColumn(0) + 28 if category_list.count() > 0 else 130)))
            category_list.setMinimumWidth(list_w)
            category_list.setMaximumWidth(list_w)
        except Exception:
            pass

    def set_current_tab(self, index: int) -> None:
        category_list = self._category_list
        if category_list is None:
            return
        safe = max(0, min(int(index), category_list.count() - 1))
        category_list.setCurrentRow(safe)

    def current_tab_index(self) -> int:
        category_list = self._category_list
        if category_list is None:
            return 0
        return int(max(0, category_list.currentRow()))

    def _pstyle_dir(self) -> Path:
        root = Path.home() / ".keyTAB" / "pstyle"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _list_pstyle_paths(self) -> list[Path]:
        root = self._pstyle_dir()
        try:
            return sorted([p for p in root.glob("*.pstyle") if p.is_file()])
        except Exception:
            return []

    def _serialize_layout(self, layout_obj: Layout) -> dict:
        try:
            return asdict(layout_obj)
        except Exception:
            return layout_obj.__dict__

    def _layout_from_dict(self, data: dict) -> Layout:
        if not isinstance(data, dict):
            raise ValueError("Invalid style payload")
        data = dict(data)
        if 'dynamic_symbol_background_padding_mm' not in data:
            if 'dynamic_symbol_background_padding' in data:
                data['dynamic_symbol_background_padding_mm'] = data.get('dynamic_symbol_background_padding')
            elif 'dynamic_background_padding' in data:
                data['dynamic_symbol_background_padding_mm'] = data.get('dynamic_background_padding')
        # Coerce known LayoutFont fields back to dataclasses to keep typing consistent
        fixed: dict[str, Any] = {}
        defaults = Layout()
        for f in fields(Layout):
            name = f.name
            val = data.get(name, getattr(defaults, name))
            hint = self._type_hints.get(name, f.type)
            if hint is Font and isinstance(val, dict):
                try:
                    val = Font(**val)
                except Exception:
                    val = getattr(defaults, name)
            fixed[name] = val
        # Backwards compatibility: migrate legacy text_font_family/size into font_text if missing
        if "font_text" not in data and ("text_font_family" in data or "text_font_size_pt" in data):
            try:
                fam = str(data.get("text_font_family", "Edwin"))
                size = float(data.get("text_font_size_pt", 12.0))
                fixed["font_text"] = Font(family=fam, size_pt=size)
            except Exception:
                pass
        # Legacy migration: merge left/right grid band tracks into the unified track
        if not fixed.get("grid_band_track"):
            legacy_left = data.get("grid_band_left_track", []) or []
            legacy_right = data.get("grid_band_right_track", []) or []
            if legacy_left or legacy_right:
                fixed["grid_band_track"] = list(legacy_left) + list(legacy_right)
        return Layout(**fixed)

    def _load_layout_from_file(self, stem: str | None) -> Layout:
        if stem is None or stem == "keyTAB Default":
            return Layout()
        path = self._pstyle_dir() / f"{stem}.pstyle"
        if not path.is_file():
            raise FileNotFoundError(f"Style not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return self._layout_from_dict(data)

    def _current_tab_name(self) -> str:
        idx = self.current_tab_index()
        if not self._tab_titles:
            return ""
        try:
            if 0 <= int(idx) < len(self._tab_titles):
                return str(self._tab_titles[int(idx)] or "")
            return ""
        except Exception:
            return ""

    def _apply_layout_object(self, layout_obj: Layout) -> None:
        self._layout = layout_obj
        self._apply_layout_to_editors(layout_obj)

    def _apply_layout_to_tab(self, layout_obj: Layout, tab_name: str) -> None:
        if not tab_name:
            return
        # Update only editors mapped to the given tab
        for f in fields(Layout):
            name = f.name
            if self.tr(self._field_tabs.get(name, "Page")) != tab_name:
                continue
            editor = self._editors.get(name)
            if editor is None:
                continue
            field_type = self._type_hints.get(name, f.type)
            value = getattr(layout_obj, name, getattr(self._layout, name, None))
            self._set_editor_value(editor, field_type, value)
            try:
                setattr(self._layout, name, value)
            except Exception:
                pass
        try:
            self.values_changed.emit()
        except Exception:
            pass

    def _make_editor(self, field_type: Any, value: Any, field_name: str) -> QtWidgets.QWidget | None:
        origin = get_origin(field_type)
        args = get_args(field_type)

        if origin is Literal and args:
            literal_labels = {
                'above_stem': self.tr('Above stem (Klavarskribo)'),
                'below_stem': self.tr('Below stem'),
                'above_stem_if_collision': self.tr('Above stem if collision'),
                'above_stem_if_chord_and_white_note': self.tr('Above stem if chord and white note'),
                'above_stem_if_chord_and_white_note_same_hand': self.tr('Above stem if chord and white note same hand'),
                'dark': self.tr('Dark'),
                'light': self.tr('Light'),
                'classical': self.tr('Classical'),
                'klavarskribo': self.tr('Klavarskribo'),
                'classical & klavarskribo': self.tr('Classical & Klavarskribo'),
            }
            options = [(literal_labels.get(str(a), str(a).replace('_', ' ').capitalize()), str(a)) for a in args]
            return RadioGroupWidget(options, str(value) if value is not None else str(args[0]), self)

        if field_type is bool:
            cb = QtWidgets.QCheckBox(self)
            cb.setChecked(bool(value))
            return cb

        if field_name == 'measure_numbering_placement':
            options = [
                (self.tr('Place measure numbering on top of every system'), 'system'),
                (self.tr('Place measure numbering on every barline'), 'barline'),
            ]
            return RadioGroupWidget(options, str(value) if value is not None else 'system', self)

        if field_type is int:
            sb = QtWidgets.QSpinBox(self)
            sb.setRange(-1000000, 1000000)
            sb.setValue(int(value))
            # Ensure immediate updates while typing
            sb.setKeyboardTracking(True)
            return sb

        if field_type is float:
            if field_name in LAYOUT_FLOAT_CONFIG:
                cfg = LAYOUT_FLOAT_CONFIG[field_name]
                return FloatSliderEdit(float(value), cfg['min'], cfg['max'], cfg['step'], self)
            is_mm = field_name.endswith('_mm')
            is_pt = field_name.endswith('_pt')
            if is_mm:
                return FloatSliderEdit(float(value), 0.0, 1000.0, 0.25, self)
            if is_pt:
                return FloatSliderEdit(float(value), 1.0, 200.0, 0.5, self)
            return FloatSliderEdit(float(value), -1000.0, 1000.0, 0.01, self)

        if field_type is str and (field_name.startswith('color_') or field_name.endswith('_color')):
            return ColorPickerEdit(str(value or ''), self)

        if field_type is str:
            le = QtWidgets.QLineEdit(self)
            le.setText(str(value) if value is not None else "")
            return le

        if field_type is Font:
            if isinstance(value, dict):
                try:
                    value = Font(**value)
                except Exception:
                    value = Font()
            show_offsets = field_name in FONT_OFFSET_FIELDS
            return FontPicker(value, self, show_offsets=show_offsets)

        if origin is list and args and args[0] is float:
            le = QtWidgets.QLineEdit(self)
            le.setText(self._format_float_list(value))
            le.setValidator(
                QtGui.QRegularExpressionValidator(QtCore.QRegularExpression(r"[0-9., ]*"), self)
            )
            return le

        return None

    def _wire_editor_change(self, editor: QtWidgets.QWidget) -> None:
        if isinstance(editor, QtWidgets.QCheckBox):
            editor.stateChanged.connect(lambda _v: self.values_changed.emit())
        elif isinstance(editor, QtWidgets.QSpinBox):
            editor.valueChanged.connect(lambda _v: self.values_changed.emit())
            editor.editingFinished.connect(lambda: self.values_changed.emit())
        elif isinstance(editor, FloatSliderEdit):
            editor.valueChanged.connect(lambda _v: self.values_changed.emit())
        elif isinstance(editor, FontPicker):
            editor.valueChanged.connect(lambda: self.values_changed.emit())
        elif isinstance(editor, QtWidgets.QComboBox):
            editor.currentTextChanged.connect(lambda _v: self.values_changed.emit())
        elif isinstance(editor, QtWidgets.QLineEdit):
            editor.textChanged.connect(lambda _v: self.values_changed.emit())
        elif isinstance(editor, ColorPickerEdit):
            editor.valueChanged.connect(lambda _v: self.values_changed.emit())
        elif isinstance(editor, RadioGroupWidget):
            editor.valueChanged.connect(lambda _v: self.values_changed.emit())

    def _add_all_fonts_control(self, vbox: QtWidgets.QVBoxLayout | None) -> None:
        if vbox is None:
            return
        combo = QtWidgets.QFontComboBox(self)
        self._all_fonts_combo = combo
        font_title = getattr(self._layout, 'font_title', Font())
        combo.setCurrentFont(QtGui.QFont(self._font_family_from_value(font_title)))
        combo.currentFontChanged.connect(lambda f: self._set_all_font_families(f.family()))
        box = QtWidgets.QGroupBox(self.tr("Apply family to all fonts"), self)
        box_layout = QtWidgets.QVBoxLayout(box)
        box_layout.setContentsMargins(6, 2, 6, 6)
        box_layout.setSpacing(0)
        box_layout.addWidget(combo)
        vbox.insertWidget(0, box)

    def _set_all_font_families(self, family: str) -> None:
        if not family:
            return
        for editor in self._editors.values():
            if isinstance(editor, FontPicker):
                editor.blockSignals(True)
                editor.set_family(family)
                editor.blockSignals(False)
                editor.valueChanged.emit()
        self.values_changed.emit()

    def _set_editor_value(self, editor: QtWidgets.QWidget, field_type: Any, value: Any) -> None:
        origin = get_origin(field_type)
        args = get_args(field_type)
        if isinstance(editor, QtWidgets.QCheckBox):
            editor.setChecked(bool(value))
        elif isinstance(editor, QtWidgets.QSpinBox):
            editor.setValue(int(value))
        elif isinstance(editor, FloatSliderEdit):
            editor.set_value(float(value))
        elif isinstance(editor, FontPicker):
            # Convert dict payloads to LayoutFont when loading from appdata
            if isinstance(value, dict):
                value = Font(**value)
            editor.set_value(value)
        elif isinstance(editor, ColorPickerEdit):
            editor.set_value(str(value or ''))
        elif origin is list and args and args[0] is float and isinstance(editor, QtWidgets.QLineEdit):
            editor.setText(self._format_float_list(value))
        elif isinstance(editor, QtWidgets.QComboBox):
            if field_type is int:
                int_val = int(value) if value is not None else 0
                matched = False
                for i in range(editor.count()):
                    if editor.itemData(i) == int_val:
                        editor.setCurrentIndex(i)
                        matched = True
                        break
                if not matched:
                    editor.setCurrentText(str(int_val))
            else:
                editor.setCurrentText(str(value))
        elif isinstance(editor, QtWidgets.QLineEdit):
            editor.setText(str(value) if value is not None else "")
        elif isinstance(editor, RadioGroupWidget):
            editor.set_value(str(value) if value is not None else '')

    def _save_style_to_disk(self) -> None:
        name, ok = QtWidgets.QInputDialog.getText(self, self.tr("Save Style"), self.tr("Enter your custom style name here:"))
        if not ok:
            return
        stem = str(name or "").strip()
        if not stem:
            self.msg_label.setText(self.tr("Style name cannot be empty."))
            return
        # Basic character whitelist
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -_.")
        if any(ch not in allowed for ch in stem):
            self.msg_label.setText(self.tr("Name contains invalid characters."))
            return
        stem = " ".join(stem.split())
        if stem.lower().endswith(".pstyle"):
            stem = stem[:-7]
        path = self._pstyle_dir() / f"{stem}.pstyle"
        try:
            payload = self._serialize_layout(self.get_values())
            with path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, ensure_ascii=True)
            self.msg_label.setText(self.tr("Saved style to {name}.").format(name=path.name))
        except Exception:
            self.msg_label.setText(self.tr("Failed to save style."))

    def _show_load_menu(self, scope: str = "all", global_pos: QtCore.QPoint | None = None) -> None:
        menu = QtWidgets.QMenu(self)
        default_action = menu.addAction(self.tr("keyTAB Default"))
        default_action.setData("__default__")
        menu.addSeparator()
        for p in self._list_pstyle_paths():
            menu.addAction(p.stem)
        menu.addSeparator()
        prompt_action = menu.addAction(self.tr("Browse…"))
        prompt_action.setData("__browse__")

        def _handle(action: QtGui.QAction) -> None:
            if action is None:
                return
            marker = str(action.data() or "")
            if marker == "__browse__":
                self._browse_and_load(scope)
                return
            if marker == "__default__":
                self._load_and_apply("keyTAB Default", scope)
                return
            text = action.text()
            self._load_and_apply(text, scope)

        menu.triggered.connect(_handle)
        pos = global_pos or QtGui.QCursor.pos()
        menu.exec(pos)

    def _browse_and_load(self, scope: str) -> None:
        dlg = QtWidgets.QFileDialog(self, self.tr("Load Style"), str(self._pstyle_dir()))
        dlg.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)
        dlg.setNameFilter(self.tr("Style Files (*.pstyle)"))
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        files = dlg.selectedFiles()
        if not files:
            return
        path = Path(files[0])
        stem = path.stem
        try:
            lay = self._layout_from_dict(json.loads(path.read_text(encoding="utf-8")))
            self._apply_loaded_layout(lay, scope)
            self.msg_label.setText(self.tr("Loaded style '{name}'.").format(name=stem))
        except Exception:
            self.msg_label.setText(self.tr("Failed to load style."))

    def _load_and_apply(self, name: str, scope: str) -> None:
        try:
            lay = self._load_layout_from_file(name)
            self._apply_loaded_layout(lay, scope)
            self.msg_label.setText(self.tr("Loaded style '{name}'.").format(name=name))
        except Exception:
            self.msg_label.setText(self.tr("Failed to load style."))

    def _apply_loaded_layout(self, layout_obj: Layout, scope: str) -> None:
        if scope == "tab":
            tab_name = self._current_tab_name()
            self._apply_layout_to_tab(layout_obj, tab_name)
        else:
            self._apply_layout_object(layout_obj)

    def _apply_layout_to_editors(self, layout_obj: Layout) -> None:
        self._coerce_layout_fonts(layout_obj)
        for f in fields(Layout):
            name = f.name
            editor = self._editors.get(name)
            if editor is None:
                continue
            field_type = self._type_hints.get(name, f.type)
            value = getattr(layout_obj, name, None)
            self._set_editor_value(editor, field_type, value)
        if self._all_fonts_combo is not None:
            try:
                self._all_fonts_combo.blockSignals(True)
                font_title = getattr(layout_obj, 'font_title', Font())
                self._all_fonts_combo.setCurrentFont(QtGui.QFont(self._font_family_from_value(font_title)))
            finally:
                self._all_fonts_combo.blockSignals(False)
        self.values_changed.emit()

    def _on_accept_clicked(self) -> None:
        try:
            _ = self.get_values()
        except Exception:
            self.msg_label.setText(self.tr("Invalid layout values."))
            return
        self.msg_label.setText("")
        self.accept()

    def get_values(self) -> Layout:
        # Start from current layout so fields without an editor (e.g. grid band tracks)
        # are preserved when the dialog applies changes.
        data: dict[str, Any] = {}
        for f in fields(Layout):
            data[f.name] = getattr(self._layout, f.name)

        for f in fields(Layout):
            name = f.name
            editor = self._editors.get(name)
            if editor is None:
                continue
            field_type = self._type_hints.get(name, f.type)
            origin = get_origin(field_type)
            args = get_args(field_type)
            if isinstance(editor, QtWidgets.QCheckBox):
                data[name] = bool(editor.isChecked())
            elif isinstance(editor, QtWidgets.QSpinBox):
                data[name] = int(editor.value())
            elif isinstance(editor, FloatSliderEdit):
                data[name] = float(editor.value())
            elif isinstance(editor, FontPicker):
                data[name] = editor.value()
            elif isinstance(editor, ColorPickerEdit):
                data[name] = str(editor.value())
            elif origin is list and args and args[0] is float and isinstance(editor, QtWidgets.QLineEdit):
                data[name] = self._parse_float_list(editor.text())
            elif isinstance(editor, QtWidgets.QComboBox):
                if field_type is int:
                    try:
                        data[name] = int(editor.currentText())
                    except (ValueError, TypeError):
                        data[name] = getattr(self._layout, name, 0)
                else:
                    data[name] = str(editor.currentText())
            elif isinstance(editor, QtWidgets.QLineEdit):
                data[name] = str(editor.text())
            elif isinstance(editor, RadioGroupWidget):
                data[name] = editor.value()
        return Layout(**data)

    def _format_float_list(self, value: Any) -> str:
        if not isinstance(value, list):
            return ""
        parts: list[str] = []
        for v in value:
            try:
                parts.append(f"{float(v):.2f}".rstrip('0').rstrip('.'))
            except Exception:
                continue
        return " ".join(parts)

    def _parse_float_list(self, text: str) -> list[float]:
        if not text:
            return []
        parts = text.replace(',', ' ').split()
        values: list[float] = []
        for part in parts:
            try:
                values.append(float(part))
            except Exception:
                continue
        return values


# Utility functions for managing the default style
def _get_default_style_dir() -> Path:
    """Get the directory where default style is stored."""
    root = Path.home() / ".keyTAB" / "pstyle"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _serialize_layout_dict(layout_obj: Layout) -> dict:
    """Serialize a Layout object to a dictionary."""
    try:
        return asdict(layout_obj)
    except Exception:
        return layout_obj.__dict__


def _layout_from_dict(data: dict) -> Layout:
    """Load a Layout object from a dictionary."""
    if not isinstance(data, dict):
        raise ValueError("Invalid style payload")
    data = dict(data)
    if 'dynamic_symbol_background_padding_mm' not in data:
        if 'dynamic_symbol_background_padding' in data:
            data['dynamic_symbol_background_padding_mm'] = data.get('dynamic_symbol_background_padding')
        elif 'dynamic_background_padding' in data:
            data['dynamic_symbol_background_padding_mm'] = data.get('dynamic_background_padding')
    # Coerce known LayoutFont fields back to dataclasses to keep typing consistent
    fixed: dict[str, Any] = {}
    defaults = Layout()
    type_hints = get_type_hints(Layout)
    for f in fields(Layout):
        name = f.name
        val = data.get(name, getattr(defaults, name))
        hint = type_hints.get(name, f.type)
        if hint is Font and isinstance(val, dict):
            try:
                val = Font(**val)
            except Exception:
                val = getattr(defaults, name)
        fixed[name] = val
    # Backwards compatibility: migrate legacy text_font_family/size into font_text if missing
    if "font_text" not in data and ("text_font_family" in data or "text_font_size_pt" in data):
        try:
            fam = str(data.get("text_font_family", "Edwin"))
            size = float(data.get("text_font_size_pt", 12.0))
            fixed["font_text"] = Font(family=fam, size_pt=size)
        except Exception:
            pass
    # Legacy migration: merge left/right grid band tracks into the unified track
    if not fixed.get("grid_band_track"):
        legacy_left = data.get("grid_band_left_track", []) or []
        legacy_right = data.get("grid_band_right_track", []) or []
        if legacy_left or legacy_right:
            fixed["grid_band_track"] = list(legacy_left) + list(legacy_right)
    return Layout(**fixed)


def save_default_style(layout: Layout) -> None:
    """Save the current layout as the default style."""
    pstyle_dir = _get_default_style_dir()
    path = pstyle_dir / "__default__.pstyle"
    try:
        payload = _serialize_layout_dict(layout)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=True)
    except Exception as e:
        print(f"Failed to save default style: {e}")


def load_default_style() -> Layout | None:
    """Load the default style if it exists, otherwise return None."""
    pstyle_dir = _get_default_style_dir()
    path = pstyle_dir / "__default__.pstyle"
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return _layout_from_dict(data)
    except Exception as e:
        print(f"Failed to load default style: {e}")
        return None


def reset_default_style() -> None:
    """Remove the default style file."""
    pstyle_dir = _get_default_style_dir()
    path = pstyle_dir / "__default__.pstyle"
    try:
        if path.is_file():
            path.unlink()
    except Exception as e:
        print(f"Failed to reset default style: {e}")
