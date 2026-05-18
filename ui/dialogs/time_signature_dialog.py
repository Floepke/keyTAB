from __future__ import annotations

from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets
from utils.CONSTANT import QUARTER_NOTE_UNIT
from ui.dialogs import DialogGeometryMixin

VALID_DENOMS = [1, 2, 4, 8, 16, 32, 64, 128]


class TimeSignatureDialog(DialogGeometryMixin, QtWidgets.QDialog):
    DIALOG_KEY = "time_signature"
    def __init__(
        self,
        parent=None,
        initial_numer: int = 4,
        initial_denom: int = 4,
        initial_grid_positions: Optional[list[float]] = None,
        initial_indicator_enabled: Optional[bool] = True,
        indicator_type: Optional[str] = None,
        editor_widget: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowFlag(QtCore.Qt.WindowType.WindowMinMaxButtonsHint, True)
        self.setSizeGripEnabled(True)
        self.setWindowTitle(self.tr("Set Time Signature"))
        self.setModal(False)
        self.setWindowModality(QtCore.Qt.WindowModality.NonModal)

        self._editor_widget = editor_widget
        self._indicator_type = str(indicator_type or 'classical & klavarskribo')
        self._numer = int(initial_numer or 4)
        self._denom = int(initial_denom or 4)
        if self._denom not in VALID_DENOMS:
            self._denom = 4
        self._grid_positions: list[float] = [float(v) for v in (initial_grid_positions or []) if isinstance(v, (int, float))]
        self._indicator_enabled = bool(initial_indicator_enabled if initial_indicator_enabled is not None else True)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        ts_row = QtWidgets.QHBoxLayout()
        ts_row.setContentsMargins(0, 0, 0, 0)
        ts_row.setSpacing(6)
        ts_row.addWidget(QtWidgets.QLabel(self.tr("Time signature:"), self))
        self.ts_edit = QtWidgets.QLineEdit(self)
        self.ts_edit.setPlaceholderText(self.tr("e.g., 4/4"))
        self.ts_edit.setText(f"{self._numer}/{self._denom}")
        ts_row.addWidget(self.ts_edit, 1)
        lay.addLayout(ts_row)

        self.indicator_enabled_cb = QtWidgets.QCheckBox(self.tr("Time-signature indicator enabled"), self)
        self.indicator_enabled_cb.setChecked(self._indicator_enabled)
        lay.addWidget(self.indicator_enabled_cb)

        self.msg_label = QtWidgets.QLabel("", self)
        pal = self.msg_label.palette()
        pal.setColor(QtGui.QPalette.WindowText, QtGui.QColor(200, 0, 0))
        self.msg_label.setPalette(pal)
        lay.addWidget(self.msg_label)

        self.btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, self)
        self.btns.accepted.connect(self._on_accept)
        self.btns.rejected.connect(self.reject)
        lay.addWidget(self.btns)

        self.ts_edit.textChanged.connect(self._on_any_changed)
        self.indicator_enabled_cb.toggled.connect(self._on_any_changed)

        self._install_validators()
        self._on_any_changed()
        QtCore.QTimer.singleShot(0, self._focus_entry)

    def _install_validators(self) -> None:
        ts_rx = QtCore.QRegularExpression(r"^[0-9/ ]*$")
        self.ts_edit.setValidator(QtGui.QRegularExpressionValidator(ts_rx, self.ts_edit))

    def _focus_entry(self) -> None:
        try:
            self.ts_edit.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)
            self.ts_edit.selectAll()
        except Exception:
            pass

    def _parse_ts(self, text: str) -> tuple[Optional[int], Optional[int], Optional[str]]:
        raw = str(text or "").strip().replace(" ", "")
        if not raw:
            return None, None, self.tr("Enter time signature as N/D.")
        parts = raw.split('/')
        if len(parts) != 2:
            return None, None, self.tr("Format must be N/D (e.g., 4/4).")
        n_s, d_s = parts
        if (not n_s.isdigit()) or (not d_s.isdigit()):
            return None, None, self.tr("Time signature accepts only digits and '/'.")
        n = int(n_s)
        d = int(d_s)
        if n <= 0:
            return None, None, self.tr("Numerator must be > 0.")
        if d not in VALID_DENOMS:
            return None, None, self.tr("Denominator must be one of {values}.").format(values=VALID_DENOMS)
        return n, d, None

    def _default_grid_positions(self, numer: int, denom: int) -> list[float]:
        beat_len = float(4.0 / max(1, int(denom))) * float(QUARTER_NOTE_UNIT)
        return [float(i) * beat_len for i in range(max(1, int(numer)))]

    def _on_any_changed(self) -> None:
        ok_btn = self.btns.button(QtWidgets.QDialogButtonBox.Ok)
        n, d, ts_err = self._parse_ts(self.ts_edit.text())
        if ts_err:
            self.msg_label.setText(ts_err)
            if ok_btn is not None:
                ok_btn.setEnabled(False)
            return

        self.msg_label.setText("")
        if ok_btn is not None:
            ok_btn.setEnabled(True)

        self._numer = int(n)
        self._denom = int(d)
        self._grid_positions = self._default_grid_positions(self._numer, self._denom)
        self._indicator_enabled = bool(self.indicator_enabled_cb.isChecked())

    def _on_accept(self) -> None:
        self._on_any_changed()
        if self.msg_label.text().strip():
            return
        self.accept()

    def get_values(self) -> tuple[int, int, list[float], bool]:
        return int(self._numer), int(self._denom), list(self._grid_positions), bool(self._indicator_enabled)
