from __future__ import annotations
from typing import Callable, Optional, Tuple
from PySide6 import QtCore, QtGui, QtWidgets
from ui.dialogs import DialogGeometryMixin
from ui.dialogs.style_dialog import FloatSliderEdit

from utils.CONSTANT import BE_KEYS, CF_KEYS, QUARTER_NOTE_UNIT

from file_model.events.line_break import LineBreak


class BulkKeyRangeDialog(QtWidgets.QDialog):
    """Dialog to set key range for all line/page breaks at once."""
    
    def __init__(self, parent=None, default_low: int = 1, default_high: int = 88) -> None:
        super().__init__(parent)
        self.setWindowFlag(QtCore.Qt.WindowType.WindowMinMaxButtonsHint, True)
        self.setSizeGripEnabled(True)
        self.setWindowTitle(self.tr("Set All Key Ranges"))
        self.setModal(False)
        self.setWindowModality(QtCore.Qt.NonModal)
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # Info label
        info = QtWidgets.QLabel(
            self.tr("Apply this key range to all existing line/page start markers:"),
            self
        )
        layout.addWidget(info)
        
        # Combo box row
        combo_layout = QtWidgets.QHBoxLayout()
        combo_layout.setContentsMargins(0, 0, 0, 0)
        combo_layout.setSpacing(8)
        
        # Helper to convert key number to note name
        def _note_name(key_num: int) -> str:
            midi_note = int(key_num) + 20  # Piano key 1 corresponds to MIDI 21 (A0)
            names = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b']
            name = names[midi_note % 12]
            octave = (midi_note // 12) - 1
            return f"{name}{octave}"
        
        # From key combo
        from_label = QtWidgets.QLabel(self.tr("From:"), self)
        self.from_combo = QtWidgets.QComboBox(self)
        self.from_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
        cf_keys = sorted(set(CF_KEYS + [1]))
        for key in cf_keys:
            self.from_combo.addItem(
                self.tr("key {key} ({note})").format(key=key, note=_note_name(key)),
                key
            )
        # Set default
        from_idx = self.from_combo.findData(default_low)
        if from_idx >= 0:
            self.from_combo.setCurrentIndex(from_idx)
        
        combo_layout.addWidget(from_label)
        combo_layout.addWidget(self.from_combo)
        
        # To key combo
        to_label = QtWidgets.QLabel(self.tr("To:"), self)
        self.to_combo = QtWidgets.QComboBox(self)
        self.to_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
        be_keys = sorted(BE_KEYS)
        for key in be_keys:
            self.to_combo.addItem(
                self.tr("key {key} ({note})").format(key=key, note=_note_name(key)),
                key
            )
        # Set default
        to_idx = self.to_combo.findData(default_high)
        if to_idx >= 0:
            self.to_combo.setCurrentIndex(to_idx)
        
        combo_layout.addWidget(to_label)
        combo_layout.addWidget(self.to_combo)
        combo_layout.addStretch(1)
        
        layout.addLayout(combo_layout)
        
        # Buttons
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=self,
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)
    
    def get_range(self) -> Tuple[int, int]:
        """Return (low_key, high_key) selected by user."""
        low = int(self.from_combo.currentData())
        high = int(self.to_combo.currentData())
        return (low, high)


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


class StaveConfigDialog(DialogGeometryMixin, QtWidgets.QDialog):
    DIALOG_KEY = "stave_config"
    valuesChanged = QtCore.Signal()
    def __init__(self,
                 parent=None,
                 score=None,
                 selected_line_break: Optional[LineBreak] = None,
                 measure_resolver: Optional[Callable[[float], int]] = None,
                 on_change: Optional[Callable[[], None]] = None) -> None:
        super().__init__(parent)
        self.setWindowFlag(QtCore.Qt.WindowType.WindowMinMaxButtonsHint, True)
        self.setSizeGripEnabled(True)
        self.setWindowTitle(self.tr("Stave Configuration / Document Layout"))
        self.setModal(False)
        self.setWindowModality(QtCore.Qt.NonModal)
        self.resize(900, 200)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        self._score = score
        self._active_stave_index = 0
        self._line_breaks: list[LineBreak] = []
        self._selected_line_break: Optional[LineBreak] = selected_line_break if selected_line_break in self._line_breaks else (self._line_breaks[0] if self._line_breaks else None)
        self._measure_resolver = measure_resolver
        self._on_change_cb = on_change
        self._layout = getattr(score, 'layout', None) if score is not None else None
        read_direction = str(getattr(self._layout, 'read_direction', 'vertical') or 'vertical').strip().lower()
        self._horizontal_read_direction = read_direction == 'horizontal'
        self._measure_grouping_text = str(getattr(self._layout, 'measure_grouping', "") or "") if self._layout is not None else ""
        self._measure_starts_mm: list[float] = self._build_measure_starts()
        self._suppress_measure_change: bool = False
        self._suppress_stave_meta_change: bool = False

        self._suppress_tab_change: bool = False
        self.stave_tabs = QtWidgets.QTabWidget(self)
        self._tab_pages: list[QtWidgets.QWidget] = []
        lay.addWidget(self.stave_tabs)

        base_color = self.palette().color(QtGui.QPalette.ColorRole.Window).name()
        accent_color = self.palette().color(QtGui.QPalette.ColorRole.Link).name()
        self.stave_tabs.setStyleSheet(
            "QTabWidget::pane {"
            f" background-color: {base_color};"
            " border: 0px solid palette(mid);"
            "}"
            "QTabBar::tab {"
            f" background-color: {base_color};"
            "}"
            "QTabBar::tab:selected {"
            f" background-color: {accent_color};"
            "}"
        )

        self._tab_content_host = QtWidgets.QWidget(self)
        content_lay = QtWidgets.QVBoxLayout(self._tab_content_host)
        content_lay.setContentsMargins(8, 8, 8, 8)
        content_lay.setSpacing(8)
        self._tab_content_host.setAutoFillBackground(True)
        content_palette = self._tab_content_host.palette()
        content_palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(base_color))
        self._tab_content_host.setPalette(content_palette)

        self.staves_group = QtWidgets.QGroupBox(self.tr("Staves"), self._tab_content_host)
        staves_group_lay = QtWidgets.QVBoxLayout(self.staves_group)
        staves_group_lay.setContentsMargins(6, 6, 6, 6)
        staves_group_lay.setSpacing(6)
        stave_row = QtWidgets.QHBoxLayout()
        stave_row.setContentsMargins(0, 0, 0, 0)
        stave_row.setSpacing(6)
        self.stave_enabled_label = QtWidgets.QLabel(self.tr("Enabled:"), self.staves_group)
        self.stave_enabled_cb = QtWidgets.QCheckBox(self.staves_group)
        self.stave_name_label = QtWidgets.QLabel(self.tr("Name:"), self.staves_group)
        self.stave_name_edit = QtWidgets.QLineEdit(self.staves_group)
        self.stave_scale_label = QtWidgets.QLabel(self.tr("Scale:"), self.staves_group)
        self.stave_scale_slider = FloatSliderEdit(1.0, 0.25, 3.0, 0.05, self.staves_group)
        self.stave_scale_slider.setMinimumWidth(240)
        self.stave_name_edit.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.stave_scale_slider.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.stave_enabled_cb.setToolTip(self.tr("Enable or disable this stave in score rendering."))
        self.stave_name_edit.setToolTip(self.tr("Set the display name of this stave."))
        self.stave_scale_slider.setToolTip(self.tr("Set stave scale. 1.00 means original size."))
        stave_row.addWidget(self.stave_enabled_label)
        stave_row.addWidget(self.stave_enabled_cb)
        stave_row.addWidget(self.stave_name_label)
        stave_row.addWidget(self.stave_name_edit, 1)
        stave_row.addWidget(self.stave_scale_label)
        stave_row.addWidget(self.stave_scale_slider, 2)
        staves_group_lay.addLayout(stave_row)
        content_lay.addWidget(self.staves_group)

        self.break_markers_group = QtWidgets.QGroupBox(self.tr("Line/Page break editor"), self._tab_content_host)
        break_group_lay = QtWidgets.QVBoxLayout(self.break_markers_group)
        break_group_lay.setContentsMargins(6, 6, 6, 6)
        break_group_lay.setSpacing(6)
        self.break_table = QtWidgets.QTableWidget(self.break_markers_group)
        self.break_table.setColumnCount(6)
        left_label, right_label = self._margin_side_labels()
        self.break_table.setHorizontalHeaderLabels([
            " ",
            self.tr(" Start Measure "),
            self.tr(" Type "),
            f" {left_label} ",
            f" {right_label} ",
            self.tr(" Key range "),
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
        break_group_lay.addWidget(self.break_table)
        content_lay.addWidget(self.break_markers_group)

        self.layout_group = QtWidgets.QGroupBox(self.tr("Layout"), self)
        layout_group_lay = QtWidgets.QVBoxLayout(self.layout_group)
        layout_group_lay.setContentsMargins(6, 6, 6, 6)
        layout_group_lay.setSpacing(8)

        quick_row = QtWidgets.QHBoxLayout()
        quick_row.setContentsMargins(0, 0, 0, 0)
        quick_row.setSpacing(6)
        self.measure_grouping_label = QtWidgets.QLabel(self.tr("Measure Grouping:"), self.layout_group)
        self.measure_grouping_edit = QtWidgets.QLineEdit(self.layout_group)
        self.measure_grouping_edit.setPlaceholderText(self.tr("e.g. 4 6 4"))
        self.measure_grouping_edit.setText(self._measure_grouping_text)
        self.apply_grouping_btn = QtWidgets.QPushButton(self.tr("Apply Measure Grouping"), self.layout_group)
        self.apply_grouping_btn.clicked.connect(self._on_apply_grouping_clicked)

        grouping_tip = self.tr(
            "Measure Grouping lets you generate line breaks by measures.\n"
            "Enter positive integers separated by spaces (e.g. '4 6 4'). Each number\n"
            "is the count of measures on a line; after the list is exhausted, the last\n"
            "number repeats. Existing margins, ranges, and page/line types are reused\n"
            "in order. Click 'Apply Measure Grouping' to generate breaks; OK saves\n"
            "other edits and Cancel discards the previewed changes."
        )
        self.measure_grouping_label.setToolTip(grouping_tip)
        self.measure_grouping_edit.setToolTip(grouping_tip)
        self.apply_grouping_btn.setToolTip(grouping_tip)

        quick_row.addWidget(self.measure_grouping_label)
        quick_row.addWidget(self.measure_grouping_edit, 1)
        quick_row.addWidget(self.apply_grouping_btn)
        layout_group_lay.addLayout(quick_row)

        bulk_row = QtWidgets.QHBoxLayout()
        bulk_row.setContentsMargins(0, 0, 0, 0)
        bulk_row.setSpacing(8)
        self.edit_all_left_btn = QtWidgets.QPushButton(self._edit_all_margins_button_text("left"), self.layout_group)
        self.edit_all_right_btn = QtWidgets.QPushButton(self._edit_all_margins_button_text("right"), self.layout_group)
        self.set_all_key_ranges_btn = QtWidgets.QPushButton(self.tr("Set All Key Ranges"), self.layout_group)
        self.edit_all_left_btn.clicked.connect(lambda: self._edit_all_margins(side="left"))
        self.edit_all_right_btn.clicked.connect(lambda: self._edit_all_margins(side="right"))
        self.set_all_key_ranges_btn.clicked.connect(self._set_all_key_ranges)
        self.edit_all_left_btn.setToolTip(self._edit_all_margins_tooltip("left"))
        self.edit_all_right_btn.setToolTip(self._edit_all_margins_tooltip("right"))
        self.set_all_key_ranges_btn.setToolTip(self.tr("Set one key range for all current line/page break markers."))
        bulk_row.addWidget(self.edit_all_left_btn)
        bulk_row.addWidget(self.edit_all_right_btn)
        bulk_row.addWidget(self.set_all_key_ranges_btn)
        bulk_row.addStretch(1)
        layout_group_lay.addLayout(bulk_row)

        # Validation message
        self.msg_label = QtWidgets.QLabel("...", self.layout_group)
        pal = self.msg_label.palette()
        pal.setColor(self.msg_label.foregroundRole(), QtCore.Qt.GlobalColor.red)
        self.msg_label.setPalette(pal)
        layout_group_lay.addWidget(self.msg_label)

        self._init_stave_tabs()
        layout_group_row = QtWidgets.QHBoxLayout()
        layout_group_row.setContentsMargins(16, 0, 16, 0)
        layout_group_row.setSpacing(0)
        layout_group_row.addWidget(self.layout_group)
        lay.addLayout(layout_group_row)

        # Buttons
        self.btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=self,
        )
        self.help_btn = QtWidgets.QPushButton(self.tr("Help"), self)
        self.help_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogHelpButton))
        self.help_btn.clicked.connect(self._show_help)
        self.btns.accepted.connect(self._on_accept_clicked)
        self.btns.rejected.connect(self.reject)

        ok_btn = self.btns.button(QtWidgets.QDialogButtonBox.Ok)
        if ok_btn is not None:
            ok_btn.setToolTip(self.tr("Save your line/page break edits and close this dialog."))
        cancel_btn = self.btns.button(QtWidgets.QDialogButtonBox.Cancel)
        if cancel_btn is not None:
            cancel_btn.setToolTip(self.tr("Discard previewed edits and close this dialog."))
        buttons_row = QtWidgets.QHBoxLayout()
        buttons_row.setContentsMargins(8, 8, 8, 8)
        buttons_row.setSpacing(8)
        buttons_row.addWidget(self.help_btn, 0)
        buttons_row.addStretch(1)
        buttons_row.addWidget(self.btns, 0)
        lay.addLayout(buttons_row)

        self.valuesChanged.connect(self._validate_form)
        self.valuesChanged.connect(self._on_values_changed)

        self.stave_enabled_cb.stateChanged.connect(self._on_stave_enabled_changed)
        self.stave_name_edit.textChanged.connect(self._on_stave_name_changed)
        self.stave_scale_slider.valueChanged.connect(self._on_stave_scale_changed)

        # Initialize
        self._reload_line_breaks(keep_row=False)
        self._validate_form()

        self.break_table.currentCellChanged.connect(lambda _r, _c, _pr, _pc: self._on_break_selected())

        QtCore.QTimer.singleShot(0, self._focus_first)

    def _stave_tab_title(self, stave_index: int) -> str:
        default_name = self.tr("Stave {idx}").format(idx=int(stave_index) + 1)
        stave_name = default_name
        staves = list(getattr(self._score, 'staves', []) or []) if self._score is not None else []
        if 0 <= int(stave_index) < len(staves):
            candidate = str(getattr(staves[int(stave_index)], 'name', '') or '').strip()
            if candidate:
                stave_name = candidate
        return self.tr("Stave {idx}: {name}").format(idx=int(stave_index) + 1, name=stave_name)

    def _init_stave_tabs(self) -> None:
        target_count = 4
        self._tab_pages = []
        self.stave_tabs.clear()
        for idx in range(target_count):
            page = QtWidgets.QWidget(self.stave_tabs)
            page_lay = QtWidgets.QVBoxLayout(page)
            page_lay.setContentsMargins(8, 8, 8, 8)
            page_lay.setSpacing(0)
            page.setAutoFillBackground(True)
            page_palette = page.palette()
            page_palette.setColor(QtGui.QPalette.ColorRole.Window, self.palette().color(QtGui.QPalette.ColorRole.Window))
            page.setPalette(page_palette)
            self._tab_pages.append(page)
            self.stave_tabs.addTab(page, self._stave_tab_title(idx))
        self.stave_tabs.currentChanged.connect(self._on_tab_changed)
        current_index = max(0, min(int(self._active_stave_index), max(0, target_count - 1)))
        self._suppress_tab_change = True
        self.stave_tabs.setCurrentIndex(current_index)
        self._suppress_tab_change = False
        self._move_content_into_active_tab()

    def _move_content_into_active_tab(self) -> None:
        if not self._tab_pages:
            return
        idx = max(0, min(int(self._active_stave_index), len(self._tab_pages) - 1))
        old_parent = self._tab_content_host.parentWidget()
        if old_parent is not None and old_parent.layout() is not None:
            old_parent.layout().removeWidget(self._tab_content_host)
        target = self._tab_pages[idx]
        target_lay = target.layout()
        if target_lay is not None:
            target_lay.addWidget(self._tab_content_host)

    def _on_tab_changed(self, tab_index: int) -> None:
        if self._suppress_tab_change:
            return
        self.set_selected_stave_index(int(tab_index))

    def _current_stave_events(self):
        if self._score is None:
            return None
        staves = list(getattr(self._score, 'staves', []) or [])
        if not staves:
            return None
        idx = max(0, min(self._active_stave_index, len(staves) - 1))
        stave = staves[idx]
        return getattr(stave, 'events', None)

    def _current_stave(self):
        if self._score is None:
            return None
        staves = list(getattr(self._score, 'staves', []) or [])
        if not staves:
            return None
        idx = max(0, min(self._active_stave_index, len(staves) - 1))
        return staves[idx]

    def _current_stave_line_breaks(self) -> list[LineBreak]:
        events = self._current_stave_events()
        if events is None:
            return []
        lst = getattr(events, 'line_break', None)
        if isinstance(lst, list):
            return lst
        return []

    def _update_stave_tabs(self) -> None:
        if not self._tab_pages:
            return
        for idx in range(len(self._tab_pages)):
            self.stave_tabs.setTabText(idx, self._stave_tab_title(idx))
        current_index = max(0, min(int(self._active_stave_index), len(self._tab_pages) - 1))
        if self.stave_tabs.currentIndex() != current_index:
            self._suppress_tab_change = True
            self.stave_tabs.setCurrentIndex(current_index)
            self._suppress_tab_change = False
        self._move_content_into_active_tab()
        self._sync_stave_meta_controls()

    def _sync_stave_meta_controls(self) -> None:
        stave = self._current_stave()
        self._suppress_stave_meta_change = True
        try:
            if stave is None:
                self.stave_enabled_cb.setEnabled(False)
                self.stave_name_edit.setEnabled(False)
                self.stave_scale_slider.setEnabled(False)
                self.stave_enabled_cb.setChecked(False)
                self.stave_name_edit.setText("")
                self.stave_scale_slider.set_value(1.0)
                return
            self.stave_enabled_cb.setEnabled(True)
            self.stave_name_edit.setEnabled(True)
            self.stave_scale_slider.setEnabled(True)
            self.stave_enabled_cb.setChecked(bool(getattr(stave, 'enabled', True)))
            self.stave_name_edit.setText(str(getattr(stave, 'name', '') or ''))
            try:
                self.stave_scale_slider.set_value(float(getattr(stave, 'scale', 1.0) or 1.0))
            except Exception:
                self.stave_scale_slider.set_value(1.0)
        finally:
            self._suppress_stave_meta_change = False

    def _on_stave_enabled_changed(self, _state: int) -> None:
        if self._suppress_stave_meta_change:
            return
        stave = self._current_stave()
        if stave is None:
            return
        try:
            stave.enabled = bool(self.stave_enabled_cb.isChecked())
        except Exception:
            return
        self.valuesChanged.emit()

    def _on_stave_name_changed(self, text: str) -> None:
        if self._suppress_stave_meta_change:
            return
        stave = self._current_stave()
        if stave is None:
            return
        try:
            stave.name = str(text or "")
        except Exception:
            return
        self._update_stave_tabs()
        self.valuesChanged.emit()

    def _on_stave_scale_changed(self, value: float) -> None:
        if self._suppress_stave_meta_change:
            return
        stave = self._current_stave()
        if stave is None:
            return
        try:
            stave.scale = float(value)
        except Exception:
            return
        self.valuesChanged.emit()

    def _show_help(self) -> None:
        title = self.tr("Line/Page Break Help")
        text = self.tr(
            "Use this dialog to configure both stave properties and line/page breaks.\n\n"
            "Stave settings (top row in each tab):\n"
            "- Enabled: include/exclude this stave in rendering.\n"
            "- Name: rename the stave.\n"
            "- Scale: change this stave's drawing scale (0.25 to 3.00).\n\n"
            "Line/Page break markers:\n"
            "- Start Measure: choose where a marker starts.\n"
            "- Type: L for line break, P for page break.\n"
            "- Margins: set left/right (or bottom/top in horizontal read direction) per stave.\n"
            "- Key range: set automatic or manual key range for each marker.\n"
            "- Delete: remove a marker (the first marker at measure 1 cannot be deleted).\n\n"
            "Shared fields across staves:\n"
            "- Marker time and marker type are linked across staves. Editing them in one tab updates all tabs.\n"
            "- Margins and key ranges stay per stave.\n\n"
            "Measure Grouping:\n"
            "- Enter positive integers (for example: 4 6 4) to generate repeating line-break distribution by measures.\n"
            "- Existing marker styling is reused in order.\n\n"
            "Bulk tools:\n"
            "- Edit all left/right margins for current stave markers.\n"
            "- Set one key range for all current stave markers.\n\n"
            "OK accepts changes. Cancel discards dialog changes when used with preview restore."
        )
        QtWidgets.QMessageBox.information(self, title, text)

    def set_selected_stave_index(self, stave_index: int) -> None:
        if self._score is None:
            return
        staves = list(getattr(self._score, 'staves', []) or [])
        if not staves:
            return
        normalized = int(int(stave_index) % len(staves))
        if normalized == self._active_stave_index and self._line_breaks:
            self._update_stave_tabs()
            return
        self._active_stave_index = normalized
        self._update_stave_tabs()
        self._reload_line_breaks(keep_row=True)

    def _reload_line_breaks(self, keep_row: bool = True) -> None:
        current_row = self.break_table.currentRow() if keep_row else -1
        self._line_breaks = self._current_stave_line_breaks()
        self._measure_starts_mm = self._build_measure_starts()
        self._update_stave_tabs()
        self._populate_break_list()
        if self._line_breaks:
            if keep_row and current_row >= 0:
                row = max(0, min(current_row, len(self._line_breaks) - 1))
                self._selected_line_break = self._line_breaks[row]
            else:
                self._selected_line_break = self._line_breaks[0]
        else:
            self._selected_line_break = None
        self._select_line_break(self._selected_line_break)
        self._validate_form()

    def _marker_label(self, is_page: bool) -> str:
        # Keep marker glyphs localizable (e.g. Dutch uses R for "Regel").
        return self.tr("P") if is_page else self.tr("L")

    def _margin_side_label(self, side: str) -> str:
        if side == "left":
            return self.tr("Bottom margin") if self._horizontal_read_direction else self.tr("Left margin")
        return self.tr("Top margin") if self._horizontal_read_direction else self.tr("Right margin")

    def _margin_side_labels(self) -> tuple[str, str]:
        return (self._margin_side_label("left"), self._margin_side_label("right"))

    def _edit_all_margins_button_text(self, side: str) -> str:
        side_label = self._margin_side_label(side)
        return self.tr("Edit All {side}").format(side=side_label)

    def _edit_all_margins_tooltip(self, side: str) -> str:
        if side == "left":
            return self.tr("Edit all current line/page break left-side margins in millimeters.")
        return self.tr("Edit all current line/page break right-side margins in millimeters.")

    def _edit_all_margins_prompt(self, side: str) -> tuple[str, str]:
        side_label = self._margin_side_label(side)
        title = self.tr("Edit All {side}").format(side=side_label)
        label = self.tr("All {side} (mm):").format(side=side_label.lower())
        return title, label

    def _focus_first(self) -> None:
        self.break_table.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)

    def _create_type_badge(self, is_page: bool) -> QtWidgets.QToolButton:
        btn = QtWidgets.QToolButton(self)
        btn.setText(self._marker_label(is_page))
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
        btn.setToolTip(self.tr("Page break.") if is_page else self.tr("Line break."))
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
        auto_cb.setText(self.tr("Automatic key range"))

        # Allow starting range at key 1 (A0) for allowing to select full range
        cf_keys = sorted(set(CF_KEYS + [1]))
        be_keys = sorted(BE_KEYS)

        from_text = self.tr("from")
        to_text = self.tr("to")

        def _build_combo(prefix: str, keys: list[int]) -> QtWidgets.QComboBox:
            combo = QtWidgets.QComboBox(wrapper)
            combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
            for key in keys:
                combo.addItem(self.tr("{prefix} key {key} ({note})").format(prefix=prefix, key=key, note=_note_name(key)), key)
            return combo

        from_combo = _build_combo(from_text, cf_keys)
        to_combo = _build_combo(to_text, be_keys)

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
            type_btn.setText(self._marker_label(bool(lb.page_break)))
            type_btn.setToolTip(self.tr("Page break.") if lb.page_break else self.tr("Line break."))
            if hasattr(self._score, 'sync_linked_line_breaks'):
                self._score.sync_linked_line_breaks(self._active_stave_index)
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
                if hasattr(self._score, 'sync_linked_line_breaks'):
                    self._score.sync_linked_line_breaks(self._active_stave_index)
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
        delete_btn.setToolTip(self.tr("Delete this line break."))
        delete_btn.setFixedWidth(28)

        def _delete_break() -> None:
            try:
                # Never allow deleting the first line break entry.
                if row == 0:
                    return
                if self._score is not None:
                    try:
                        current_events = self._current_stave_events()
                        if current_events is not None:
                            current_events.line_break.remove(lb)
                    except Exception:
                        pass
                try:
                    self._line_breaks.remove(lb)
                except Exception:
                    pass
                if hasattr(self._score, 'sync_linked_line_breaks'):
                    self._score.sync_linked_line_breaks(self._active_stave_index)
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
            self.msg_label.setText(self.tr("Enter one or more positive integers separated by spaces."))
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
            if hasattr(self._score, 'sync_linked_line_breaks'):
                self._score.sync_linked_line_breaks(self._active_stave_index)
            self._reload_line_breaks()
            self.valuesChanged.emit()
        else:
            self.msg_label.setText(self.tr("Could not apply measure grouping."))

    def _edit_all_margins(self, side: str) -> None:
        if side not in ("left", "right"):
            return
        title, label = self._edit_all_margins_prompt(side)
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
        if hasattr(self._score, 'sync_linked_line_breaks'):
            self._score.sync_linked_line_breaks(self._active_stave_index)
        self.valuesChanged.emit()

    def _prompt_margin_value(self, title: str, label: str, initial_value: float) -> Optional[float]:
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowFlag(QtCore.Qt.WindowType.WindowMinMaxButtonsHint, True)
        dlg.setSizeGripEnabled(True)
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

    def _set_all_key_ranges(self) -> None:
        """Open dialog to set key range for all line/page breaks at once."""
        if not self._line_breaks:
            return
        
        # Get default range from first line break
        defaults = LineBreak()
        first_lb = self._line_breaks[0]
        first_range = getattr(first_lb, 'stave_range', defaults.stave_range)
        is_auto = bool(first_range == 'auto' or first_range is True or first_range is None)
        if is_auto:
            default_low = 1
            default_high = 88
        else:
            rng = list(first_range) if first_range is not None else [1, 88]
            default_low = int(rng[0]) if len(rng) > 0 else 1
            default_high = int(rng[1]) if len(rng) > 1 else 88
        
        # Show dialog
        dlg = BulkKeyRangeDialog(self, default_low, default_high)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        
        # Apply range to all line breaks
        low_key, high_key = dlg.get_range()
        for lb in self._line_breaks:
            lb.stave_range = [int(low_key), int(high_key)]

        if hasattr(self._score, 'sync_linked_line_breaks'):
            self._score.sync_linked_line_breaks(self._active_stave_index)
        
        self._populate_break_list()
        self.valuesChanged.emit()

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
                msg = self.tr("Key range must contain two numbers.")
                break
            if not (1 <= low <= 88 and 1 <= high <= 88):
                msg = self.tr("Key range must stay between key 1 and key 88.")
                break
            if low >= high:
                msg = self.tr("Key range must have 'from key' lower than 'to key'.")
                break

        self.msg_label.setText(msg)
        ok_btn = self.btns.button(QtWidgets.QDialogButtonBox.Ok)
        if ok_btn is not None:
            ok_btn.setEnabled(not bool(msg))
        return not bool(msg)

    def _on_values_changed(self) -> None:
        callback_ok = False
        if callable(self._on_change_cb):
            try:
                self._on_change_cb()
                callback_ok = True
            except Exception:
                pass
        if callback_ok:
            return

        # Fallback: refresh directly through the parent window when callback
        # wiring is unavailable or fails, so live preview still updates.
        parent = self.parent()

        file_manager = getattr(parent, 'file_manager', None)
        if file_manager is not None and hasattr(file_manager, 'on_model_changed'):
            file_manager.on_model_changed()

        editor_controller = getattr(parent, 'editor_controller', None)
        if editor_controller is not None:
            if hasattr(editor_controller, 'force_redraw_from_model'):
                editor_controller.force_redraw_from_model()
            elif hasattr(editor_controller, 'draw_frame'):
                editor_controller.draw_frame()

    def _persist_measure_grouping(self) -> None:
        if self._layout is None:
            return
        try:
            self._layout.measure_grouping = str(self.measure_grouping_edit.text().strip())
        except Exception:
            pass

    def _on_accept_clicked(self) -> None:
        if not self._validate_form():
            return
        self.msg_label.setText("")
        self._persist_measure_grouping()
        if hasattr(self._score, 'sync_linked_line_breaks'):
            self._score.sync_linked_line_breaks(self._active_stave_index)
        self.valuesChanged.emit()
        self.accept()
