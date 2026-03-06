from __future__ import annotations
import copy
from typing import Callable, Optional
from PySide6 import QtCore, QtGui, QtWidgets

from utils.CONSTANT import BE_KEYS, CF_KEYS, QUARTER_NOTE_UNIT

from file_model.events.line_break import LineBreak


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


class LineBreakDialog(QtWidgets.QDialog):
    valuesChanged = QtCore.Signal()
    def __init__(self,
                 parent=None,
                 score=None,
                 selected_line_break: Optional[LineBreak] = None,
                 measure_resolver: Optional[Callable[[float], int]] = None,
                 on_change: Optional[Callable[[], None]] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Line/Page Break")
        self.setModal(True)
        self.setWindowModality(QtCore.Qt.NonModal)
        try:
            self.resize(900, 600)
        except Exception:
            pass

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        self._score = score
        self._line_breaks: list[LineBreak] = list(getattr(score.events, 'line_break', []) or []) if score is not None else []
        self._selected_line_break: Optional[LineBreak] = selected_line_break if selected_line_break in self._line_breaks else (self._line_breaks[0] if self._line_breaks else None)
        self._measure_resolver = measure_resolver
        self._on_change_cb = on_change
        self._layout = getattr(score, 'layout', None) if score is not None else None
        self._measure_grouping_text = str(getattr(self._layout, 'measure_grouping', "") or "") if self._layout is not None else ""
        self._original_breaks: list[LineBreak] = copy.deepcopy(self._line_breaks)
        self._original_grouping: str = str(self._measure_grouping_text)
        self._measure_starts_mm: list[float] = self._build_measure_starts()
        self._suppress_measure_change: bool = False

        list_label = QtWidgets.QLabel("Line/Page breaks:", self)
        self.break_table = QtWidgets.QTableWidget(self)
        self.break_table.setColumnCount(6)
        self.break_table.setHorizontalHeaderLabels([
            " ",
            " Start Measure ",
            " Type ",
            " Left margin " ,
            " Right margin ",
            " Key range ",
        ])
        self.break_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.break_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.break_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.break_table.verticalHeader().setVisible(False)
        self.break_table.horizontalHeader().setStretchLastSection(True)
        self.break_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.break_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.break_table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.break_table.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.break_table.horizontalHeader().setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.break_table.horizontalHeader().setSectionResizeMode(5, QtWidgets.QHeaderView.ResizeMode.Stretch)
        lay.addWidget(list_label)
        lay.addWidget(self.break_table)

        quick_row = QtWidgets.QHBoxLayout()
        quick_row.setContentsMargins(0, 0, 0, 0)
        quick_row.setSpacing(6)
        self.measure_grouping_label = QtWidgets.QLabel("Measure Grouping:", self)
        self.measure_grouping_edit = QtWidgets.QLineEdit(self)
        self.measure_grouping_edit.setPlaceholderText("e.g. 4 6 4")
        self.measure_grouping_edit.setText(self._measure_grouping_text)
        self.apply_grouping_btn = QtWidgets.QPushButton("Apply Measure Grouping", self)
        self.apply_grouping_btn.clicked.connect(self._on_apply_grouping_clicked)
        quick_row.addWidget(self.measure_grouping_label)
        quick_row.addWidget(self.measure_grouping_edit, 1)
        quick_row.addWidget(self.apply_grouping_btn)
        lay.addLayout(quick_row)

        bulk_row = QtWidgets.QHBoxLayout()
        bulk_row.setContentsMargins(0, 0, 0, 0)
        bulk_row.setSpacing(6)
        self.edit_all_left_btn = QtWidgets.QPushButton("Edit All Left Margins", self)
        self.edit_all_right_btn = QtWidgets.QPushButton("Edit All Right Margins", self)
        self.edit_all_left_btn.clicked.connect(lambda: self._edit_all_margins(side="left"))
        self.edit_all_right_btn.clicked.connect(lambda: self._edit_all_margins(side="right"))
        bulk_row.addWidget(self.edit_all_left_btn)
        bulk_row.addWidget(self.edit_all_right_btn)
        bulk_row.addStretch(1)
        lay.addLayout(bulk_row)

        # Validation message
        self.msg_label = QtWidgets.QLabel("", self)
        pal = self.msg_label.palette()
        pal.setColor(self.msg_label.foregroundRole(), QtCore.Qt.GlobalColor.red)
        self.msg_label.setPalette(pal)
        lay.addWidget(self.msg_label)

        # Buttons
        self.btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Help,
            parent=self,
        )
        self.btns.accepted.connect(self._on_accept_clicked)
        self.btns.rejected.connect(self.reject)
        self.btns.helpRequested.connect(self._on_help_clicked)
        lay.addWidget(self.btns)

        self.valuesChanged.connect(self._validate_form)
        self.valuesChanged.connect(self._on_values_changed)

        # Initialize
        self._populate_break_list()
        if self._selected_line_break is None and self._line_breaks:
            self._selected_line_break = self._line_breaks[0]
        self._select_line_break(self._selected_line_break)
        self._validate_form()

        self.break_table.currentCellChanged.connect(lambda _r, _c, _pr, _pc: self._on_break_selected())

        QtCore.QTimer.singleShot(0, self._focus_first)

    def _focus_first(self) -> None:
        try:
            self.break_table.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)
        except Exception:
            pass

    def _create_type_badge(self, is_page: bool) -> QtWidgets.QToolButton:
        btn = QtWidgets.QToolButton(self)
        btn.setText("P" if is_page else "L")
        btn.setAutoRaise(True)
        btn.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Expanding)
        btn.setMinimumWidth(32)
        btn.setMinimumHeight(28)
        try:
            from fonts import register_font_from_bytes
            marker_family = register_font_from_bytes('Fira Code') or 'Fira Code'
        except Exception:
            marker_family = 'Fira Code'
        marker_font = btn.font()
        marker_font.setFamily(marker_family)
        marker_font.setPointSize(18)
        marker_font.setBold(True)
        btn.setFont(marker_font)
        btn.setStyleSheet(
            "QToolButton {"
            " background: #000000;"
            " color: #ffffff;"
            " border-radius: 4px;"
            " padding: 0 8px;"
            " }"
        )
        btn.setToolTip("Page Break" if is_page else "Line Break")
        return btn

    def _create_margin_spin(self, value: float) -> FlexibleDoubleSpinBox:
        spin = FlexibleDoubleSpinBox(self)
        spin.setRange(0.0, 200.0)
        spin.setDecimals(2)
        spin.setSingleStep(0.5)
        spin.setValue(float(value))
        spin.setKeyboardTracking(True)
        spin.setMinimumWidth(80)
        return spin

    def _create_range_widget(self, lb: LineBreak) -> QtWidgets.QWidget:
        defaults = LineBreak()
        lb_range = getattr(lb, 'stave_range', defaults.stave_range)
        is_auto = bool(lb_range == 'auto' or lb_range is True or lb_range is None)
        fallback = 'auto' if defaults.stave_range == 'auto' else list(defaults.stave_range or [1, 88])
        if is_auto:
            rng = [1, 88]
        else:
            base_range = lb_range if lb_range is not None else ([1, 88] if fallback == 'auto' else fallback)
            rng = list(base_range)

        def _note_name(key_num: int) -> str:
            midi_note = int(key_num) + 20  # Piano key 1 corresponds to MIDI 21 (A0)
            names = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b']
            name = names[midi_note % 12]
            octave = (midi_note // 12) - 1
            return f"{name}{octave}"

        def _closest(keys: list[int], target: int) -> int:
            return min(keys, key=lambda k: abs(int(k) - int(target))) if keys else target

        wrapper = QtWidgets.QWidget(self)
        layout = QtWidgets.QHBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        auto_cb = QtWidgets.QCheckBox(wrapper)
        auto_cb.setChecked(is_auto)
        auto_cb.setText("Automatic key range")

        # Allow starting range at key 1 (A0) for allowing to select full range
        cf_keys = sorted(set(CF_KEYS + [1]))
        be_keys = sorted(BE_KEYS)

        def _build_combo(prefix: str, keys: list[int]) -> QtWidgets.QComboBox:
            combo = QtWidgets.QComboBox(wrapper)
            combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
            for key in keys:
                combo.addItem(f"{prefix} key {key} ({_note_name(key)})", key)
            return combo

        from_combo = _build_combo("from", cf_keys)
        to_combo = _build_combo("to", be_keys)

        def _set_combo_value(combo: QtWidgets.QComboBox, value: int, keys: list[int]) -> None:
            target = _closest(keys, value)
            idx = combo.findData(target)
            combo.setCurrentIndex(idx if idx >= 0 else 0)

        low_val = int(rng[0]) if len(rng) > 0 else 1
        high_val = int(rng[1]) if len(rng) > 1 else 88
        _set_combo_value(from_combo, low_val, cf_keys)
        _set_combo_value(to_combo, high_val, be_keys)

        layout.addWidget(auto_cb)
        layout.addWidget(from_combo)
        layout.addWidget(to_combo)
        layout.addStretch(1)

        def _refresh_combo_style() -> None:
            disabled_style = "QComboBox { color: #7a7a7a; }"
            from_combo.setStyleSheet("" if from_combo.isEnabled() else disabled_style)
            to_combo.setStyleSheet("" if to_combo.isEnabled() else disabled_style)

        def _apply_range_state() -> None:
            is_auto_mode = bool(auto_cb.isChecked())
            from_combo.setEnabled(not is_auto_mode)
            to_combo.setEnabled(not is_auto_mode)
            _refresh_combo_style()
            if is_auto_mode:
                lb.stave_range = 'auto'
            else:
                lb.stave_range = [int(from_combo.currentData()), int(to_combo.currentData())]
            self.valuesChanged.emit()

        def _range_changed(_v: int) -> None:
            if not auto_cb.isChecked():
                lb.stave_range = [int(from_combo.currentData()), int(to_combo.currentData())]
                self.valuesChanged.emit()

        auto_cb.toggled.connect(lambda _v: _apply_range_state())
        from_combo.currentIndexChanged.connect(_range_changed)
        to_combo.currentIndexChanged.connect(_range_changed)

        _apply_range_state()
        _refresh_combo_style()

        return wrapper

    def _build_measure_starts(self) -> list[float]:
        starts: list[float] = [0.0]
        score = self._score
        if score is None:
            return starts
        cursor = 0.0
        try:
            for bg in list(getattr(score, 'base_grid', []) or []):
                try:
                    numer = int(getattr(bg, 'numerator', 4) or 4)
                    denom = int(getattr(bg, 'denominator', 4) or 4)
                    measures = int(getattr(bg, 'measure_amount', 1) or 1)
                except Exception:
                    continue
                if measures <= 0:
                    continue
                measure_len = float(numer) * (4.0 / float(max(1, denom))) * float(QUARTER_NOTE_UNIT)
                for _ in range(measures):
                    cursor += measure_len
                    starts.append(float(cursor))
        except Exception:
            pass
        return starts if starts else [0.0]

    def _measure_index_for_time(self, t: float) -> int:
        starts = self._measure_starts_mm
        if not starts:
            return 0
        for idx, start in enumerate(starts):
            if start > t:
                return max(0, idx - 1)
        return max(0, len(starts) - 1)

    def _measure_time_for_index(self, idx: int) -> float:
        starts = self._measure_starts_mm
        if not starts:
            return 0.0
        idx = max(0, min(int(idx), len(starts) - 1))
        return float(starts[idx])

    def _populate_break_list(self) -> None:
        self.break_table.blockSignals(True)
        self.break_table.setRowCount(0)
        try:
            self._line_breaks.sort(key=lambda b: float(getattr(b, 'time', 0.0) or 0.0))
        except Exception:
            pass
        for lb in self._line_breaks:
            row = self.break_table.rowCount()
            self.break_table.insertRow(row)
            self._set_break_row(row, lb)
        self.break_table.blockSignals(False)

    def _set_break_row(self, row: int, lb: LineBreak) -> None:
        measure_val = self._measure_index_for_time(float(getattr(lb, 'time', 0.0) or 0.0))
        measure_item = QtWidgets.QTableWidgetItem(str(int(measure_val) + 1))
        measure_item.setData(QtCore.Qt.ItemDataRole.UserRole, lb)
        self.break_table.setItem(row, 1, measure_item)

        defaults = LineBreak()
        margin_mm = list(getattr(lb, 'margin_mm', defaults.margin_mm) or defaults.margin_mm)
        left_margin = float(margin_mm[0] if len(margin_mm) > 0 else defaults.margin_mm[0])
        right_margin = float(margin_mm[1] if len(margin_mm) > 1 else defaults.margin_mm[1])

        # Measure spin: allow nudging between neighboring breaks by whole measures
        measure_spin = QtWidgets.QSpinBox(self)
        measure_spin.setMinimumWidth(70)
        measure_spin.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        def _neighbor_measure_bounds() -> tuple[int, int]:
            prev_lb = self._line_breaks[row - 1] if row - 1 >= 0 else None
            next_lb = self._line_breaks[row + 1] if row + 1 < len(self._line_breaks) else None
            prev_idx = self._measure_index_for_time(float(getattr(prev_lb, 'time', 0.0) or 0.0)) if prev_lb else 0
            next_idx = self._measure_index_for_time(float(getattr(next_lb, 'time', 0.0) or 0.0)) if next_lb else max(0, len(self._measure_starts_mm) - 1)
            min_idx = prev_idx + 1 if prev_lb is not None else 0
            max_idx = next_idx - 1 if next_lb is not None else max(0, len(self._measure_starts_mm) - 1)
            return (min_idx, max_idx)

        min_idx, max_idx = _neighbor_measure_bounds()
        display_min = int(min_idx) + 1
        display_max = int(max_idx) + 1
        measure_spin.setRange(display_min, display_max)
        measure_spin.setSingleStep(1)
        clamped_val = max(display_min, min(display_max, int(measure_val) + 1))
        measure_spin.setValue(int(clamped_val))
        if min_idx >= max_idx or row == 0:
            measure_spin.setEnabled(False)

        type_btn = self._create_type_badge(bool(getattr(lb, 'page_break', False)))
        left_spin = self._create_margin_spin(left_margin)
        right_spin = self._create_margin_spin(right_margin)
        range_widget = self._create_range_widget(lb)

        def _toggle_type() -> None:
            lb.page_break = not bool(getattr(lb, 'page_break', False))
            type_btn.setText("P" if lb.page_break else "L")
            type_btn.setToolTip("Page" if lb.page_break else "Line")
            self.valuesChanged.emit()

        def _left_changed(val: float) -> None:
            cur = list(getattr(lb, 'margin_mm', defaults.margin_mm) or defaults.margin_mm)
            if len(cur) < 2:
                cur = [float(cur[0]) if cur else 0.0, 0.0]
            cur[0] = float(val)
            lb.margin_mm = list(cur)
            self.valuesChanged.emit()

        def _right_changed(val: float) -> None:
            cur = list(getattr(lb, 'margin_mm', defaults.margin_mm) or defaults.margin_mm)
            if len(cur) < 2:
                cur = [float(cur[0]) if cur else 0.0, 0.0]
            cur[1] = float(val)
            lb.margin_mm = list(cur)
            self.valuesChanged.emit()

        def _on_measure_changed(val: int) -> None:
            if self._suppress_measure_change:
                return
            self._suppress_measure_change = True
            try:
                new_time = self._measure_time_for_index(int(val) - 1)
                lb.time = float(new_time)
                try:
                    self._line_breaks.sort(key=lambda b: float(getattr(b, 'time', 0.0) or 0.0))
                except Exception:
                    pass
                self._populate_break_list()
                self._select_line_break(lb)
                self.valuesChanged.emit()
            finally:
                self._suppress_measure_change = False

        measure_spin.valueChanged.connect(_on_measure_changed)
        type_btn.clicked.connect(_toggle_type)
        left_spin.valueChanged.connect(_left_changed)
        right_spin.valueChanged.connect(_right_changed)

        # Delete control in column 0
        delete_btn = QtWidgets.QToolButton(self)
        delete_btn.setText("✕")
        delete_btn.setAutoRaise(True)
        delete_btn.setToolTip("Delete")
        delete_btn.setFixedWidth(28)

        def _delete_break() -> None:
            try:
                # Never allow deleting the first line break entry.
                if row == 0:
                    return
                if self._score is not None:
                    try:
                        self._score.events.line_break.remove(lb)
                    except Exception:
                        pass
                try:
                    self._line_breaks.remove(lb)
                except Exception:
                    pass
                self._populate_break_list()
                self.valuesChanged.emit()
            except Exception:
                pass

        delete_btn.clicked.connect(_delete_break)
        if row == 0:
            delete_btn.hide()
            delete_btn.setEnabled(False)

        self.break_table.setCellWidget(row, 0, delete_btn)
        self.break_table.setCellWidget(row, 1, measure_spin)
        self.break_table.setCellWidget(row, 2, type_btn)
        self.break_table.setCellWidget(row, 3, left_spin)
        self.break_table.setCellWidget(row, 4, right_spin)
        self.break_table.setCellWidget(row, 5, range_widget)

    def _select_line_break(self, lb: Optional[LineBreak]) -> None:
        if lb is None:
            self.break_table.clearSelection()
            return
        for row in range(self.break_table.rowCount()):
            item = self.break_table.item(row, 1)
            if item is not None and item.data(QtCore.Qt.ItemDataRole.UserRole) is lb:
                self.break_table.setCurrentCell(row, 1)
                return

    def _current_line_break(self) -> Optional[LineBreak]:
        row = self.break_table.currentRow()
        if row < 0:
            return None
        item = self.break_table.item(row, 1)
        if item is None:
            return None
        return item.data(QtCore.Qt.ItemDataRole.UserRole)

    def _on_break_selected(self) -> None:
        lb = self._current_line_break()
        if lb is None:
            return
        self._selected_line_break = lb

    def _reload_line_breaks(self) -> None:
        if self._score is not None:
            try:
                self._line_breaks = list(getattr(self._score.events, 'line_break', []) or [])
            except Exception:
                self._line_breaks = []
        self._measure_starts_mm = self._build_measure_starts()
        self._populate_break_list()
        if self._line_breaks:
            self._selected_line_break = self._line_breaks[0]
        else:
            self._selected_line_break = None
        self._select_line_break(self._selected_line_break)
        self._validate_form()
        self.valuesChanged.emit()

    def _parse_grouping(self, text: str) -> Optional[list[int]]:
        parts = [p for p in (text or "").strip().split() if p.strip()]
        if not parts:
            return None
        try:
            values = [int(p) for p in parts]
        except Exception:
            return None
        if any(v <= 0 for v in values):
            return None
        return values

    def _on_apply_grouping_clicked(self) -> None:
        if self._score is None:
            return
        txt = self.measure_grouping_edit.text().strip()
        groups = self._parse_grouping(txt)
        if groups is None:
            self.msg_label.setText("Enter one or more positive integers separated by spaces.")
            return
        self.msg_label.setText("")
        if self._layout is not None:
            try:
                self._layout.measure_grouping = str(txt)
            except Exception:
                pass
        ok = False
        try:
            ok = bool(self._score.apply_quick_line_breaks(groups))
        except Exception:
            ok = False
        if ok:
            self._reload_line_breaks()
        else:
            self.msg_label.setText("Could not apply measure grouping.")

    def _on_help_clicked(self) -> None:
        msg = (
            "Measure Grouping lets you generate line breaks by measures.\n"
            "Enter positive integers separated by spaces (e.g. '4 6 4'). Each number\n"
            "is the count of measures on a line; after the list is exhausted, the last\n"
            "number repeats. Existing margins, ranges, and page/line types are reused\n"
            "in order. Click 'Apply Measure Grouping' to generate breaks; OK saves\n"
            "other edits, Cancel reverts to the original state."
        )
        QtWidgets.QMessageBox.information(self, "Line Break Help", msg)

    def _edit_all_margins(self, side: str) -> None:
        if side not in ("left", "right"):
            return
        title = "Edit All Left Margins" if side == "left" else "Edit All Right Margins"
        label = "All left margins (mm):" if side == "left" else "All right margins (mm):"
        val = self._prompt_margin_value(title, label, 5.0)
        if val is None:
            return
        defaults = LineBreak()
        for lb in self._line_breaks:
            margin_mm = list(getattr(lb, 'margin_mm', defaults.margin_mm) or defaults.margin_mm)
            if len(margin_mm) < 2:
                margin_mm = [float(margin_mm[0]) if margin_mm else 0.0, 0.0]
            if side == "left":
                margin_mm[0] = float(val)
            else:
                margin_mm[1] = float(val)
            lb.margin_mm = list(margin_mm)
        self._populate_break_list()
        self.valuesChanged.emit()

    def _prompt_margin_value(self, title: str, label: str, initial_value: float) -> Optional[float]:
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(title)
        dlg.setModal(True)
        layout = QtWidgets.QVBoxLayout(dlg)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        text = QtWidgets.QLabel(label, dlg)
        layout.addWidget(text)

        spin = FlexibleDoubleSpinBox(dlg)
        spin.setRange(0.0, 200.0)
        spin.setDecimals(2)
        spin.setSingleStep(0.5)
        spin.setValue(float(initial_value))
        spin.setKeyboardTracking(True)
        layout.addWidget(spin)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=dlg,
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)

        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return None
        return float(spin.value())

    def _validate_form(self) -> bool:
        msg = ""
        defaults = LineBreak()
        for lb in self._line_breaks:
            lb_range = getattr(lb, 'stave_range', defaults.stave_range)
            if lb_range == 'auto' or lb_range is True:
                continue
            try:
                low, high = int(lb_range[0]), int(lb_range[1])
            except Exception:
                msg = "Key range must contain two numbers."
                break
            if not (1 <= low <= 88 and 1 <= high <= 88):
                msg = "Key range must stay between key 1 and key 88."
                break
            if low >= high:
                msg = "Key range must have 'from key' lower than 'to key'."
                break

        self.msg_label.setText(msg)
        ok_btn = self.btns.button(QtWidgets.QDialogButtonBox.Ok)
        if ok_btn is not None:
            ok_btn.setEnabled(not bool(msg))
        return not bool(msg)

    def _on_values_changed(self) -> None:
        if callable(self._on_change_cb):
            try:
                self._on_change_cb()
            except Exception:
                pass

    def restore_original_state(self) -> None:
        if self._score is None:
            return
        try:
            self._score.events.line_break = copy.deepcopy(self._original_breaks)
        except Exception:
            try:
                self._score.events.line_break = [copy.deepcopy(lb) for lb in (self._original_breaks or [])]
            except Exception:
                self._score.events.line_break = []
        self._line_breaks = list(getattr(self._score.events, 'line_break', []) or [])
        if self._layout is not None:
            try:
                self._layout.measure_grouping = str(self._original_grouping)
            except Exception:
                pass
        self._measure_starts_mm = self._build_measure_starts()
        self._populate_break_list()
        if self._line_breaks:
            self._selected_line_break = self._line_breaks[0]
        else:
            self._selected_line_break = None
        self._select_line_break(self._selected_line_break)
        self._validate_form()
        self.valuesChanged.emit()

    def _persist_measure_grouping(self) -> None:
        if self._layout is None:
            return
        try:
            self._layout.measure_grouping = str(self.measure_grouping_edit.text().strip())
        except Exception:
            pass

    def done(self, result: int) -> None:
        if result == QtWidgets.QDialog.Accepted:
            self._persist_measure_grouping()
            self._validate_form()
            self.valuesChanged.emit()
        else:
            self.restore_original_state()
        super().done(result)

    def _on_accept_clicked(self) -> None:
        if not self._validate_form():
            return
        self.msg_label.setText("")
        self.accept()
