from __future__ import annotations

from typing import Optional

from PySide6 import QtCore, QtWidgets

from file_model.SCORE import SCORE, Stave
from ui.dialogs import DialogGeometryMixin


class StaveDialog(DialogGeometryMixin, QtWidgets.QDialog):
    """Edit per-stave metadata (name, scale, enabled) for 4 editor staves."""

    DIALOG_KEY = "stave"

    def __init__(self, parent=None, score: Optional[SCORE] = None, on_change=None) -> None:
        super().__init__(parent)
        self.setWindowFlag(QtCore.Qt.WindowType.WindowMinMaxButtonsHint, True)
        self.setSizeGripEnabled(True)
        self.setWindowTitle(self.tr("Staves"))
        self.setModal(False)
        self.setWindowModality(QtCore.Qt.NonModal)
        self.resize(560, 280)

        self._score = score
        self._on_change = on_change

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        hint = QtWidgets.QLabel(
            self.tr("Configure stave name, scale, and visibility for the 4 editor staves."),
            self,
        )
        lay.addWidget(hint)

        self.table = QtWidgets.QTableWidget(self)
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(
            [
                self.tr("#"),
                self.tr("Name"),
                self.tr("Scale"),
                self.tr("On"),
            ]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setRowCount(4)
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        lay.addWidget(self.table)

        self._name_edits: list[QtWidgets.QLineEdit] = []
        self._scale_spins: list[QtWidgets.QDoubleSpinBox] = []
        self._enabled_checks: list[QtWidgets.QCheckBox] = []

        for i in range(4):
            index_item = QtWidgets.QTableWidgetItem(str(i + 1))
            index_item.setFlags(index_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            index_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(i, 0, index_item)

            name_edit = QtWidgets.QLineEdit(self)
            name_edit.setPlaceholderText(self.tr("Stave {n}").format(n=i + 1))
            self.table.setCellWidget(i, 1, name_edit)
            self._name_edits.append(name_edit)

            scale_spin = QtWidgets.QDoubleSpinBox(self)
            scale_spin.setDecimals(2)
            scale_spin.setRange(0.25, 4.0)
            scale_spin.setSingleStep(0.05)
            scale_spin.setValue(1.0)
            self.table.setCellWidget(i, 2, scale_spin)
            self._scale_spins.append(scale_spin)

            enabled_check = QtWidgets.QCheckBox(self)
            enabled_check.setChecked(True)
            enabled_wrap = QtWidgets.QWidget(self)
            enabled_lay = QtWidgets.QHBoxLayout(enabled_wrap)
            enabled_lay.setContentsMargins(0, 0, 0, 0)
            enabled_lay.setSpacing(0)
            enabled_lay.addWidget(enabled_check, 0, QtCore.Qt.AlignmentFlag.AlignCenter)
            self.table.setCellWidget(i, 3, enabled_wrap)
            self._enabled_checks.append(enabled_check)

            name_edit.textEdited.connect(self._emit_change)
            scale_spin.valueChanged.connect(lambda _v, _idx=i: self._emit_change())
            enabled_check.toggled.connect(lambda _v, _idx=i: self._emit_change())

        self.btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        self.btns.accepted.connect(self._accept_apply)
        self.btns.rejected.connect(self.reject)
        lay.addWidget(self.btns)

        self._load_from_score()

    def _emit_change(self) -> None:
        if callable(self._on_change):
            try:
                self._on_change()
            except Exception:
                pass

    def _ensure_four_staves(self) -> None:
        if self._score is None:
            return
        staves = list(getattr(self._score, "staves", []) or [])
        staves = list(staves[:4])
        for i in range(len(staves), 4):
            staves.append(Stave(name=f"Stave {i + 1}", scale=1.0, enabled=True))
        for i, st in enumerate(staves):
            if not str(getattr(st, "name", "") or "").strip():
                st.name = f"Stave {i + 1}"
            if not hasattr(st, "enabled"):
                st.enabled = True
        self._score.staves = staves

    def _load_from_score(self) -> None:
        self._ensure_four_staves()
        staves = list(getattr(self._score, "staves", []) or []) if self._score is not None else []
        for i in range(4):
            default_name = f"Stave {i + 1}"
            st = staves[i] if i < len(staves) else Stave(name=default_name)
            self._name_edits[i].setText(str(getattr(st, "name", default_name) or default_name))
            self._scale_spins[i].setValue(float(getattr(st, "scale", 1.0) or 1.0))
            self._enabled_checks[i].setChecked(bool(getattr(st, "enabled", True)))

    def apply_to_score(self) -> None:
        if self._score is None:
            return
        self._ensure_four_staves()
        for i in range(4):
            st = self._score.staves[i]
            name = str(self._name_edits[i].text() or "").strip() or f"Stave {i + 1}"
            st.name = name
            st.scale = float(self._scale_spins[i].value())
            st.enabled = bool(self._enabled_checks[i].isChecked())

    def _accept_apply(self) -> None:
        self.apply_to_score()
        self.accept()
