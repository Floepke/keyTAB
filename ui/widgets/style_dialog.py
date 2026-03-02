from __future__ import annotations
import json
from pathlib import Path
from dataclasses import asdict, fields
from typing import Any, get_args, get_origin, get_type_hints, Literal, TYPE_CHECKING

from PySide6 import QtCore, QtGui, QtWidgets

from file_model.layout import LAYOUT_FLOAT_CONFIG, LayoutFont
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
        return round(snapped, 2)

    def _slider_to_value(self, sv: int) -> float:
        return self._min + float(sv) * self._step

    def _value_to_slider(self, val: float) -> int:
        return int(round((val - self._min) / self._step))

    def set_value(self, value: float) -> None:
        val = self._snap(self._clamp(float(value)))
        self._slider.blockSignals(True)
        self._slider.setValue(self._value_to_slider(val))
        self._slider.blockSignals(False)
        self._edit.setText(f"{val:.2f}")

    def value(self) -> float:
        val = self._slider_to_value(self._slider.value())
        return self._snap(self._clamp(val))

    def _on_slider_changed(self, _v: int) -> None:
        val = self.value()
        self._edit.setText(f"{val:.2f}")
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

    def __init__(self, value: str, parent=None) -> None:
        super().__init__(parent)
        self._combo = QtWidgets.QComboBox(self)
        self._combo.setEditable(False)
        self._populate_presets()
        self._combo.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        self._combo.setItemDelegate(self._SwatchDelegate(self._combo))
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
        self._button = QtWidgets.QPushButton("Pick", self)
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
        for idx, code in enumerate(self.PRESET_COLORS):
            self._combo.addItem('')
            self._combo.setItemData(idx, code, QtCore.Qt.ItemDataRole.UserRole)

    def set_value(self, value: str) -> None:
        txt = str(value or '').strip()
        if txt and not txt.startswith('#'):
            txt = f"#{txt}"
        self._hex_edit.setText(txt)
        idx = self._combo.findData(txt, QtCore.Qt.ItemDataRole.UserRole)
        self._combo.blockSignals(True)
        self._combo.setCurrentIndex(idx if idx >= 0 else 0)
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

    def __init__(self, value: LayoutFont, parent=None, show_offsets: bool = False) -> None:
        super().__init__(parent)
        self._font_cls = type(value)
        self._show_offsets = bool(show_offsets)
        self._combo = QtWidgets.QFontComboBox(self)
        self._size = QtWidgets.QSpinBox(self)
        self._size.setRange(1, 200)
        try:
            # Emit changes while typing in the spinbox
            self._size.setKeyboardTracking(True)
        except Exception:
            pass
        self._bold = QtWidgets.QCheckBox("Bold", self)
        self._italic = QtWidgets.QCheckBox("Italic", self)
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
                spin.setToolTip(f"{axis}-offset (mm)")

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._combo, 1)
        layout.addWidget(self._size, 0)
        layout.addWidget(self._bold, 0)
        layout.addWidget(self._italic, 0)
        if self._show_offsets and self._x_offset and self._y_offset:
            layout.addWidget(self._x_offset, 0)
            layout.addWidget(self._y_offset, 0)

        self.set_value(value)
        self._combo.currentFontChanged.connect(lambda _f: self.valueChanged.emit())
        self._size.valueChanged.connect(lambda _v: self.valueChanged.emit())
        try:
            self._size.editingFinished.connect(lambda: self.valueChanged.emit())
        except Exception:
            pass
        self._bold.stateChanged.connect(lambda _v: self.valueChanged.emit())
        self._italic.stateChanged.connect(lambda _v: self.valueChanged.emit())
        if self._show_offsets and self._x_offset and self._y_offset:
            self._x_offset.valueChanged.connect(lambda _v: self.valueChanged.emit())
            self._y_offset.valueChanged.connect(lambda _v: self.valueChanged.emit())

    def set_value(self, value: LayoutFont) -> None:
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
        if self._show_offsets and self._x_offset and self._y_offset:
            try:
                self._x_offset.setValue(float(getattr(value, 'x_offset', 0.0) or 0.0))
                self._y_offset.setValue(float(getattr(value, 'y_offset', 0.0) or 0.0))
            except Exception:
                self._x_offset.setValue(0.0)
                self._y_offset.setValue(0.0)

    def set_family(self, family: str) -> None:
        self._combo.setCurrentFont(QtGui.QFont(str(family)))

    def value(self) -> LayoutFont:
        font_cls = self._font_cls or LayoutFont
        x_off = float(self._x_offset.value()) if self._x_offset is not None else 0.0
        y_off = float(self._y_offset.value()) if self._y_offset is not None else 0.0
        return font_cls(
            family=str(self._combo.currentFont().family()),
            size_pt=float(self._size.value()),
            bold=bool(self._bold.isChecked()),
            italic=bool(self._italic.isChecked()),
            x_offset=x_off,
            y_offset=y_off,
        )


class StyleDialog(QtWidgets.QDialog):
    values_changed = QtCore.Signal()
    tab_changed = QtCore.Signal(int)

    def __init__(self, parent=None, layout: Layout | None = None, score: SCORE | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Style")
        self.setModal(True)
        self.setWindowModality(QtCore.Qt.NonModal)
        try:
            screen = QtWidgets.QApplication.primaryScreen()
            if screen is not None:
                max_h = int(screen.availableGeometry().height() / 3)
        except Exception:
            pass
        try:
            self.setMinimumWidth(600)
            self.resize(750, 400)
        except Exception:
            pass

        self._layout = layout or Layout()
        self._editors: dict[str, QtWidgets.QWidget] = {}
        self._score: SCORE | None = score
        self._tab_scrolls: list[QtWidgets.QScrollArea] = []
        self._tab_contents: list[QtWidgets.QWidget] = []
        self._tabs: QtWidgets.QTabWidget | None = None
        self._all_fonts_combo: QtWidgets.QFontComboBox | None = None
        self._field_tabs: dict[str, str] = {}

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        tabs = QtWidgets.QTabWidget(self)
        self._tabs = tabs
        lay.addWidget(tabs, 1)
        try:
            tabs.currentChanged.connect(self.tab_changed.emit)
        except Exception:
            pass

        tab_order = [
            "Page",
            "Grid",
            "Time signature",
            "Stave",
            "Fonts",
            "Note",
            "Grace note",
            "Beam",
            "Slur",
            "Text",
            "Countline",
            "Pedal",
            "Visibility",
        ]

        def _make_tab(title: str) -> QtWidgets.QFormLayout:
            tab = QtWidgets.QWidget(self)
            tab_layout = QtWidgets.QVBoxLayout(tab)
            tab_layout.setContentsMargins(0, 0, 0, 0)
            tab_layout.setSpacing(6)
            scroll = QtWidgets.QScrollArea(tab)
            scroll.setWidgetResizable(True)
            content = QtWidgets.QWidget(scroll)
            form = QtWidgets.QFormLayout(content)
            form.setContentsMargins(8, 8, 8, 8)
            form.setSpacing(6)
            content.setLayout(form)
            scroll.setWidget(content)
            tab_layout.addWidget(scroll, 1)
            tabs.addTab(tab, title)
            self._tab_scrolls.append(scroll)
            self._tab_contents.append(content)
            return form

        tab_forms: dict[str, QtWidgets.QFormLayout] = {t: _make_tab(t) for t in tab_order}

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
            'black_note_width_scaling': 'Note',
            'note_stem_length_semitone': 'Note',
            'note_stem_thickness_mm': 'Note',
            'note_stopsign_thickness_mm': 'Note',
            'note_continuation_dot_size_mm': 'Note',
            'note_midinote_left_color': 'Note',
            'note_midinote_right_color': 'Note',
            # Beam
            'beam_thickness_mm': 'Beam',
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
            # Grid
            'grid_barline_thickness_mm': 'Grid',
            'grid_gridline_thickness_mm': 'Grid',
            'grid_gridline_dash_pattern_mm': 'Grid',
            # Time signature
            'time_signature_indicator_type': 'Time signature',
            'time_signature_indicator_lane_width_mm': 'Time signature',
            'time_signature_indicator_guide_thickness_mm': 'Time signature',
            'time_signature_indicator_divide_guide_thickness_mm': 'Time signature',
            # Stave
            'stave_two_line_thickness_mm': 'Stave',
            'stave_three_line_thickness_mm': 'Stave',
            'stave_clef_line_dash_pattern_mm': 'Stave',
            # Fonts
            'font_text': 'Fonts',
            'font_title': 'Fonts',
            'font_composer': 'Fonts',
            'font_copyright': 'Fonts',
            'font_arranger': 'Fonts',
            'font_lyricist': 'Fonts',
            'time_signature_indicator_classic_font': 'Fonts',
            'time_signature_indicator_klavarskribo_font': 'Fonts',
            'measure_numbering_font': 'Fonts',
            # Visibility
            'note_head_visible': 'Visibility',
            'note_stem_visible': 'Visibility',
            'note_leftdot_visible': 'Visibility',
            'note_midinote_visible': 'Visibility',
            'beam_visible': 'Visibility',
            'grace_note_visible': 'Visibility',
            'pedal_lane_enabled': 'Visibility',
            'text_visible': 'Visibility',
            'slur_visible': 'Visibility',
            'countline_visible': 'Visibility',
            'repeat_start_visible': 'Visibility',
            'repeat_end_visible': 'Visibility',
        }

        type_hints = {}
        try:
            type_hints = get_type_hints(Layout)
        except Exception:
            type_hints = {}
        self._type_hints = type_hints
        _hide_fields = {
            "measure_grouping",
        }
        self._field_tabs = field_tabs

        for f in fields(Layout):
            name = f.name
            if name in _hide_fields:
                continue
            label = name.replace('_', ' ').capitalize() + ":"
            value = getattr(self._layout, name)
            field_type = type_hints.get(name, f.type)
            editor = self._make_editor(field_type, value, name)
            if editor is None:
                continue
            self._editors[name] = editor
            tab_name = field_tabs.get(name, 'Page')
            form = tab_forms.get(tab_name, tab_forms['Page'])
            form.addRow(QtWidgets.QLabel(label, self), editor)
            self._wire_editor_change(editor)

        self._add_all_fonts_control(tab_forms.get('Fonts'))

        # Style file actions
        actions_row = QtWidgets.QHBoxLayout()
        actions_row.setContentsMargins(0, 0, 0, 0)
        actions_row.setSpacing(8)
        self.save_style_btn = QtWidgets.QPushButton("Save Style", self)
        self.load_style_btn = QtWidgets.QPushButton("Load…", self)
        self.load_tab_style_btn = QtWidgets.QPushButton("Load into current tab", self)
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
        tabs = self._tabs
        if tabs is None or not self._tab_scrolls or not self._tab_contents:
            return
        try:
            screen = QtWidgets.QApplication.primaryScreen()
            max_h = int(screen.availableGeometry().height()) if screen is not None else 800
            max_w = int(screen.availableGeometry().width()) if screen is not None else 1200
        except Exception:
            max_h = 800
            max_w = 1200

        max_content_h = 0
        max_content_w = 0
        for content in self._tab_contents:
            try:
                max_content_h = max(max_content_h, int(content.sizeHint().height()))
                max_content_w = max(max_content_w, int(content.sizeHint().width()))
            except Exception:
                continue

        tab_bar_h = int(tabs.tabBar().sizeHint().height())
        tab_bar_w = int(tabs.tabBar().sizeHint().width())
        action_h = int(getattr(self, 'save_style_btn', QtWidgets.QPushButton()).sizeHint().height())
        action_w = int(getattr(self, 'save_style_btn', QtWidgets.QPushButton()).sizeHint().width()) * 3
        msg_h = int(self.msg_label.sizeHint().height())
        msg_w = int(self.msg_label.sizeHint().width())
        btns_h = int(self.btns.sizeHint().height())
        btns_w = int(self.btns.sizeHint().width())

        lay = self.layout()
        margins = lay.contentsMargins() if lay is not None else QtCore.QMargins()
        spacing = int(lay.spacing()) if lay is not None else 0
        gaps = 3

        non_scroll_h = margins.top() + margins.bottom() + tab_bar_h + action_h + msg_h + btns_h + (spacing * gaps)
        desired_scroll_h = max_content_h
        max_scroll_h = max(1, max_h - non_scroll_h)
        scroll_h = min(desired_scroll_h, max_scroll_h)

        non_scroll_w = margins.left() + margins.right()
        desired_w = max(tab_bar_w, max_content_w, action_w, msg_w, btns_w) + non_scroll_w
        total_w = min(desired_w, max_w)

        for scroll in self._tab_scrolls:
            frame = int(scroll.frameWidth()) * 2
            scroll.setMinimumHeight(scroll_h + frame)
            scroll.setMaximumHeight(scroll_h + frame)
            scroll.setMinimumWidth(total_w - non_scroll_w + frame)
            scroll.setMaximumWidth(total_w - non_scroll_w + frame)

        total_h = non_scroll_h + scroll_h
        if total_h > max_h:
            total_h = max_h
        self.setMinimumHeight(total_h)
        self.setMaximumHeight(total_h)
        self.setMinimumWidth(total_w)
        self.setMaximumWidth(total_w)
        self.resize(total_w, total_h)

    def set_current_tab(self, index: int) -> None:
        tabs = self._tabs
        if tabs is None:
            return
        try:
            safe = max(0, min(int(index), tabs.count() - 1))
            tabs.setCurrentIndex(safe)
        except Exception:
            pass

    def current_tab_index(self) -> int:
        tabs = self._tabs
        if tabs is None:
            return 0
        try:
            return int(tabs.currentIndex())
        except Exception:
            return 0

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
        # Coerce known LayoutFont fields back to dataclasses to keep typing consistent
        fixed: dict[str, Any] = {}
        defaults = Layout()
        for f in fields(Layout):
            name = f.name
            val = data.get(name, getattr(defaults, name))
            hint = self._type_hints.get(name, f.type)
            if hint is LayoutFont and isinstance(val, dict):
                try:
                    val = LayoutFont(**val)
                except Exception:
                    val = getattr(defaults, name)
            fixed[name] = val
        # Backwards compatibility: migrate legacy text_font_family/size into font_text if missing
        if "font_text" not in data and ("text_font_family" in data or "text_font_size_pt" in data):
            try:
                fam = str(data.get("text_font_family", "Edwin"))
                size = float(data.get("text_font_size_pt", 12.0))
                fixed["font_text"] = LayoutFont(family=fam, size_pt=size)
            except Exception:
                pass
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
        tabs = self._tabs
        if tabs is None:
            return ""
        try:
            return str(tabs.tabText(int(tabs.currentIndex())) or "")
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
            if self._field_tabs.get(name, "Page") != tab_name:
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
            combo = QtWidgets.QComboBox(self)
            options = [str(a) for a in args]
            combo.addItems(options)
            try:
                combo.setCurrentText(str(value))
            except Exception:
                pass
            return combo

        if field_type is bool:
            cb = QtWidgets.QCheckBox(self)
            cb.setChecked(bool(value))
            return cb

        if field_type is int:
            sb = QtWidgets.QSpinBox(self)
            sb.setRange(-1000000, 1000000)
            sb.setValue(int(value))
            try:
                # Ensure immediate updates while typing
                sb.setKeyboardTracking(True)
            except Exception:
                pass
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

        if field_type is LayoutFont:
            if isinstance(value, dict):
                try:
                    value = LayoutFont(**value)
                except Exception:
                    value = LayoutFont()
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
        try:
            if isinstance(editor, QtWidgets.QCheckBox):
                editor.stateChanged.connect(lambda _v: self.values_changed.emit())
            elif isinstance(editor, QtWidgets.QSpinBox):
                editor.valueChanged.connect(lambda _v: self.values_changed.emit())
                try:
                    editor.editingFinished.connect(lambda: self.values_changed.emit())
                except Exception:
                    pass
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
        except Exception:
            pass

    def _add_all_fonts_control(self, form: QtWidgets.QFormLayout | None) -> None:
        if form is None:
            return
        label = QtWidgets.QLabel("Apply family to all fonts:", self)
        combo = QtWidgets.QFontComboBox(self)
        self._all_fonts_combo = combo
        try:
            font_title = getattr(self._layout, 'font_title', LayoutFont())
            combo.setCurrentFont(QtGui.QFont(str(font_title.family)))
        except Exception:
            pass
        combo.currentFontChanged.connect(lambda f: self._set_all_font_families(f.family()))
        try:
            form.insertRow(0, label, combo)
        except Exception:
            form.addRow(label, combo)

    def _set_all_font_families(self, family: str) -> None:
        if not family:
            return
        for editor in self._editors.values():
            if isinstance(editor, FontPicker):
                editor.blockSignals(True)
                editor.set_family(family)
                editor.blockSignals(False)
                try:
                    editor.valueChanged.emit()
                except Exception:
                    pass
        self.values_changed.emit()

    def _set_editor_value(self, editor: QtWidgets.QWidget, field_type: Any, value: Any) -> None:
        origin = get_origin(field_type)
        args = get_args(field_type)
        if isinstance(editor, QtWidgets.QCheckBox):
            editor.setChecked(bool(value))
        elif isinstance(editor, QtWidgets.QSpinBox):
            try:
                editor.setValue(int(value))
            except Exception:
                pass
        elif isinstance(editor, FloatSliderEdit):
            try:
                editor.set_value(float(value))
            except Exception:
                pass
        elif isinstance(editor, FontPicker):
            try:
                # Convert dict payloads to LayoutFont when loading from appdata
                if isinstance(value, dict):
                    try:
                        value = LayoutFont(**value)
                    except Exception:
                        pass
                editor.set_value(value)
            except Exception:
                pass
        elif isinstance(editor, ColorPickerEdit):
            editor.set_value(str(value or ''))
        elif origin is list and args and args[0] is float and isinstance(editor, QtWidgets.QLineEdit):
            editor.setText(self._format_float_list(value))
        elif isinstance(editor, QtWidgets.QComboBox):
            try:
                editor.setCurrentText(str(value))
            except Exception:
                pass
        elif isinstance(editor, QtWidgets.QLineEdit):
            editor.setText(str(value) if value is not None else "")

    def _save_style_to_disk(self) -> None:
        try:
            name, ok = QtWidgets.QInputDialog.getText(self, "Save Style", "Style name (no extension):")
        except Exception:
            return
        if not ok:
            return
        stem = str(name or "").strip()
        if not stem:
            self.msg_label.setText("Style name cannot be empty.")
            return
        # Basic character whitelist
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -_.")
        if any(ch not in allowed for ch in stem):
            self.msg_label.setText("Name contains invalid characters.")
            return
        stem = " ".join(stem.split())
        if stem.lower().endswith(".pstyle"):
            stem = stem[:-7]
        path = self._pstyle_dir() / f"{stem}.pstyle"
        try:
            payload = self._serialize_layout(self.get_values())
            with path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, ensure_ascii=False)
            self.msg_label.setText(f"Saved style to {path.name}.")
        except Exception:
            self.msg_label.setText("Failed to save style.")

    def _show_load_menu(self, scope: str = "all", global_pos: QtCore.QPoint | None = None) -> None:
        menu = QtWidgets.QMenu(self)
        default_action = menu.addAction("keyTAB Default")
        menu.addSeparator()
        for p in self._list_pstyle_paths():
            menu.addAction(p.stem)
        menu.addSeparator()
        prompt_action = menu.addAction("Browse…")

        def _handle(action: QtGui.QAction) -> None:
            if action is None:
                return
            text = action.text()
            if text == "Browse…":
                self._browse_and_load(scope)
                return
            self._load_and_apply(text, scope)

        menu.triggered.connect(_handle)
        pos = global_pos or QtGui.QCursor.pos()
        menu.exec(pos)

    def _browse_and_load(self, scope: str) -> None:
        dlg = QtWidgets.QFileDialog(self, "Load Style", str(self._pstyle_dir()))
        dlg.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)
        dlg.setNameFilter("Style Files (*.pstyle)")
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
            self.msg_label.setText(f"Loaded style '{stem}'.")
        except Exception:
            self.msg_label.setText("Failed to load style.")

    def _load_and_apply(self, name: str, scope: str) -> None:
        try:
            lay = self._load_layout_from_file(name)
            self._apply_loaded_layout(lay, scope)
            self.msg_label.setText(f"Loaded style '{name}'.")
        except Exception:
            self.msg_label.setText("Failed to load style.")

    def _apply_loaded_layout(self, layout_obj: Layout, scope: str) -> None:
        if scope == "tab":
            tab_name = self._current_tab_name()
            self._apply_layout_to_tab(layout_obj, tab_name)
        else:
            self._apply_layout_object(layout_obj)

    def _apply_layout_to_editors(self, layout_obj: Layout) -> None:
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
                font_title = getattr(layout_obj, 'font_title', LayoutFont())
                self._all_fonts_combo.setCurrentFont(QtGui.QFont(str(font_title.family)))
            finally:
                self._all_fonts_combo.blockSignals(False)
        self.values_changed.emit()

    def _on_accept_clicked(self) -> None:
        try:
            _ = self.get_values()
        except Exception:
            self.msg_label.setText("Invalid layout values.")
            return
        self.msg_label.setText("")
        self.accept()

    def get_values(self) -> Layout:
        data: dict[str, Any] = {}
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
                data[name] = str(editor.currentText())
            elif isinstance(editor, QtWidgets.QLineEdit):
                data[name] = str(editor.text())
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
