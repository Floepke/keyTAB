from __future__ import annotations

from PySide6 import QtCore, QtWidgets
from ui.dialogs import DialogGeometryMixin


class MidiImportDialog(DialogGeometryMixin, QtWidgets.QDialog):
    DIALOG_KEY = "midi_import"
    """Dialog for assigning MIDI tracks to left/right hand before import.

    Displays a table with one row per non-empty, non-drum track.
    Each row shows track number, name, note count, pitch range, and a
    combo box to assign the track to left hand / right hand / skip.

    The ``assignments_changed`` signal is emitted whenever the user changes
    any assignment; callers can wire this to a live-preview refresh.
    """

    assignments_changed = QtCore.Signal(dict)  # {track_index: 'l' | 'r' | 'skip'}

    def __init__(self, track_infos: list[dict], parent=None) -> None:
        super().__init__(parent)
        self.setWindowFlag(QtCore.Qt.WindowType.WindowMinMaxButtonsHint, True)
        self.setSizeGripEnabled(True)
        self.setWindowTitle(self.tr("MIDI Import – Track Assignment"))
        self.setModal(True)

        self._track_infos = track_infos
        self._combos: dict[int, QtWidgets.QComboBox] = {}

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        # Explanation
        info = QtWidgets.QLabel(
            self.tr(
                "Assign each MIDI track to a hand. "
                "Notes are placed in the assigned hand during import.\n"
                "Choose 'Skip' to exclude a track entirely."
            ),
            self,
        )
        info.setWordWrap(True)
        lay.addWidget(info)

        # Track table
        self.table = QtWidgets.QTableWidget(self)
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels([
            self.tr("Track"),
            self.tr("Name"),
            self.tr("Notes"),
            self.tr("Pitch range"),
            self.tr("Hand"),
        ])
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.setAlternatingRowColors(True)

        self.table.setRowCount(len(self._track_infos))
        for row, ti in enumerate(self._track_infos):
            self._add_row(row, ti)

        lay.addWidget(self.table)

        self._validation_label = QtWidgets.QLabel("", self)
        self._validation_label.setWordWrap(True)
        self._validation_label.setStyleSheet("color: #b00020;")
        lay.addWidget(self._validation_label)

        # Buttons
        self._btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        self._btns.accepted.connect(self.accept)
        self._btns.rejected.connect(self.reject)
        lay.addWidget(self._btns)

        # Size: fit table rows with a sensible cap
        table_rows = max(1, len(self._track_infos))
        height = min(120 + table_rows * 34, 560)
        self.resize(620, height)
        self._update_validation_ui()

    def _add_row(self, row: int, ti: dict) -> None:
        track_idx = int(ti.get('index', row))
        name = str(ti.get('name', f'Track {track_idx + 1}'))
        note_count = int(ti.get('note_count', 0))
        min_p = str(ti.get('min_pitch', '–'))
        max_p = str(ti.get('max_pitch', '–'))
        pitch_range = f"{min_p} – {max_p}" if note_count > 0 else "–"
        default_hand = str(ti.get('default_hand', 'r'))

        # Track number (1-based)
        track_item = QtWidgets.QTableWidgetItem(str(track_idx + 1))
        track_item.setTextAlignment(int(QtCore.Qt.AlignmentFlag.AlignCenter))
        self.table.setItem(row, 0, track_item)

        # Name
        self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(name))

        # Note count
        count_item = QtWidgets.QTableWidgetItem(str(note_count))
        count_item.setTextAlignment(int(QtCore.Qt.AlignmentFlag.AlignCenter))
        self.table.setItem(row, 2, count_item)

        # Pitch range
        range_item = QtWidgets.QTableWidgetItem(pitch_range)
        range_item.setTextAlignment(int(QtCore.Qt.AlignmentFlag.AlignCenter))
        self.table.setItem(row, 3, range_item)

        # Hand combo
        combo = QtWidgets.QComboBox(self)
        combo.addItem(self.tr("Left hand"), "l")
        combo.addItem(self.tr("Right hand"), "r")
        combo.addItem(self.tr("Skip"), "skip")

        target_data = default_hand if default_hand in ("l", "r", "skip") else "r"
        idx = combo.findData(target_data)
        if idx >= 0:
            combo.setCurrentIndex(idx)

        combo.currentIndexChanged.connect(self._on_assignment_changed)
        combo.currentTextChanged.connect(self._on_assignment_changed)
        combo.activated.connect(lambda _idx: self._on_assignment_changed())
        self.table.setCellWidget(row, 4, combo)
        self._combos[track_idx] = combo

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_assignments(self) -> dict[int, str]:
        """Return current {track_index: 'l' | 'r' | 'skip'} mapping."""
        return {idx: str(combo.currentData() or 'r') for idx, combo in self._combos.items()}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _has_any_imported_track(self) -> bool:
        for combo in self._combos.values():
            if str(combo.currentData() or 'r') != 'skip':
                return True
        return False

    def _update_validation_ui(self) -> None:
        ok_btn = self._btns.button(QtWidgets.QDialogButtonBox.StandardButton.Ok) if hasattr(self, '_btns') else None
        has_import = self._has_any_imported_track()
        if has_import:
            self._validation_label.setText("")
            if ok_btn is not None:
                ok_btn.setEnabled(True)
        else:
            self._validation_label.setText(self.tr("At least one track must be imported."))
            if ok_btn is not None:
                ok_btn.setEnabled(False)

    def _on_assignment_changed(self) -> None:
        self._update_validation_ui()
        self.assignments_changed.emit(self.get_assignments())
