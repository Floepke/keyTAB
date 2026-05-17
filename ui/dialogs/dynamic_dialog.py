from __future__ import annotations

from typing import Callable, Optional

from PySide6 import QtCore, QtWidgets

from ui.dialogs.dynamic_menu import DynamicSymbolGrid
from ui.dialogs.style_dialog import FloatSliderEdit


class DynamicSymbolDialog(QtWidgets.QDialog):
    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        current_value: str = '',
        rotation: float | None = None,
        default_rotation: float = 0.0,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr('Edit Dynamic Symbol'))
        self.setModal(False)

        self._selected_glyph = str(current_value or '')
        self._default_rotation = float(default_rotation or 0.0)
        self._rotation_value = float(rotation if rotation is not None else self._default_rotation)
        self._custom_rotation_enabled = rotation is not None

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        symbol_box = QtWidgets.QGroupBox(self.tr('Symbol'), self)
        symbol_layout = QtWidgets.QVBoxLayout(symbol_box)
        symbol_layout.setContentsMargins(8, 8, 8, 8)
        symbol_layout.setSpacing(6)
        self._grid = DynamicSymbolGrid(symbol_box, current_value=current_value)
        self._grid.symbol_selected.connect(self._on_symbol_selected)
        self._grid.symbol_double_clicked.connect(self._on_symbol_double_clicked)
        symbol_layout.addWidget(self._grid)
        root.addWidget(symbol_box, 1)

        rotation_box = QtWidgets.QGroupBox(self.tr('Rotation'), self)
        rotation_layout = QtWidgets.QVBoxLayout(rotation_box)
        rotation_layout.setContentsMargins(8, 8, 8, 8)
        rotation_layout.setSpacing(6)
        self._rotation_edit = FloatSliderEdit(self._rotation_value, 0.0, 360.0, 1.0, rotation_box)
        self._rotation_edit.setVisible(self._custom_rotation_enabled)
        rotation_layout.addWidget(self._rotation_edit)
        root.addWidget(rotation_box, 0)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons, 0)

        self.resize(560, 480)

    def _on_symbol_selected(self, glyph: str) -> None:
        self._selected_glyph = str(glyph or '')

    def _on_symbol_double_clicked(self, glyph: str) -> None:
        self._selected_glyph = str(glyph or '')
        self.accept()

    def selected_glyph(self) -> str:
        current = self._grid.selected_glyph()
        if current:
            self._selected_glyph = current
        return str(self._selected_glyph or '')

    def rotation_value(self) -> float | None:
        # if not self._custom_rotation_checkbox.isChecked():
        #     return None
        return float(self._rotation_edit.value())

    @classmethod
    def open_dynamic_symbol(
        cls,
        on_accepted: Callable[[str, float | None], None],
        parent: Optional[QtWidgets.QWidget] = None,
        current_value: str = '',
        rotation: float | None = None,
        default_rotation: float = 0.0,
    ) -> 'DynamicSymbolDialog':
        """Open a non-blocking dialog; call on_accepted(glyph, rotation) when Ok is clicked."""
        dialog = cls(
            parent=parent,
            current_value=current_value,
            rotation=rotation,
            default_rotation=default_rotation,
        )

        def _on_accepted() -> None:
            on_accepted(dialog.selected_glyph(), dialog.rotation_value())

        dialog.accepted.connect(_on_accepted)
        dialog.raise_()
        dialog.activateWindow()
        dialog.show()
        return dialog