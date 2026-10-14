from __future__ import annotations

from typing import Optional

from PySide6 import QtCore, QtWidgets

from ui.dialogs import DialogGeometryMixin


class VoltaDialog(DialogGeometryMixin, QtWidgets.QDialog):
    """Edit the properties of a single volta bracket.

    Exposes two settings:

    * ``number`` — the ending number printed inside the bracket.
    * ``x_offset`` — horizontal displacement in mm. This is only honoured by
      the engraver; the editor always draws the bracket next to the stave.
    """

    DIALOG_KEY = "volta"

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        initial_number: int = 1,
        initial_x_offset: float = 0.0,
    ) -> None:
        super().__init__(parent)
        self.setWindowFlag(QtCore.Qt.WindowType.WindowMinMaxButtonsHint, True)
        self.setSizeGripEnabled(True)
        self.setWindowTitle(self.tr("Edit Volta"))
        self.setModal(False)
        self.setWindowModality(QtCore.Qt.WindowModality.NonModal)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(6)

        self.number_spin = QtWidgets.QSpinBox(self)
        self.number_spin.setRange(1, 99)
        self.number_spin.setValue(max(1, int(initial_number or 1)))
        form.addRow(self.tr("Number:"), self.number_spin)

        self.x_offset_spin = QtWidgets.QDoubleSpinBox(self)
        self.x_offset_spin.setRange(-500.0, 500.0)
        self.x_offset_spin.setDecimals(2)
        self.x_offset_spin.setSingleStep(0.5)
        self.x_offset_spin.setSuffix(self.tr(" mm"))
        self.x_offset_spin.setValue(float(initial_x_offset or 0.0))
        form.addRow(self.tr("X offset:"), self.x_offset_spin)

        lay.addLayout(form)

        info = QtWidgets.QLabel(
            self.tr(
                "INFO:\n"
                "The X offset only affects engraved output; in the editor the volta "
                "bracket is always drawn alongside the stave."
            ),
            self,
        )
        info.setWordWrap(True)
        lay.addWidget(info)

        lay.addStretch(1)

        self.btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        self.btns.accepted.connect(self.accept)
        self.btns.rejected.connect(self.reject)
        lay.addWidget(self.btns)

        self.number_spin.setFocus()
        self.number_spin.selectAll()

    def get_values(self) -> tuple[int, float]:
        return (int(self.number_spin.value()), float(self.x_offset_spin.value()))
