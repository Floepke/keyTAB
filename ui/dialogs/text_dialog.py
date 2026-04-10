from __future__ import annotations

from copy import deepcopy

from PySide6 import QtCore, QtWidgets

from file_model.font import Font
from ui.dialogs.style_dialog import FontPicker, FloatSliderEdit


class TextDialog(QtWidgets.QDialog):
    valueChanged = QtCore.Signal()

    def _add_labeled_row(self, form: QtWidgets.QFormLayout, label_text: str, field: QtWidgets.QWidget, tooltip: str) -> None:
        label = QtWidgets.QLabel(label_text, self)
        label.setToolTip(tooltip)
        field.setToolTip(tooltip)
        form.addRow(label, field)

    def __init__(self, ev, default_font: Font | None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowFlag(QtCore.Qt.WindowType.WindowMinMaxButtonsHint, True)
        self.setSizeGripEnabled(True)
        self.setWindowTitle(self.tr("Edit Text"))
        self._default_font = deepcopy(default_font or Font())

        cur_font = self._coerce_font(getattr(ev, "font", None), self._default_font)
        if not bool(getattr(ev, "use_custom_font", False)):
            cur_font = deepcopy(self._default_font)

        layout = QtWidgets.QFormLayout(self)

        self.txt_edit = QtWidgets.QLineEdit(self)
        self.txt_edit.setText(str(getattr(ev, "text", "")))
        self._add_labeled_row(layout, self.tr("Text"), self.txt_edit, self.tr("Edit the displayed text content."))

        self.x_off_edit = FloatSliderEdit(float(getattr(ev, "x_offset_mm", 0.0) or 0.0), -25.0, 25.0, 0.1, self)
        self._add_labeled_row(layout, self.tr("X offset (mm)"), self.x_off_edit, self.tr("Shift text horizontally in millimeters."))

        self.y_off_edit = FloatSliderEdit(float(getattr(ev, "y_offset_mm", 0.0) or 0.0), -25.0, 25.0, 0.1, self)
        self._add_labeled_row(layout, self.tr("Y offset (mm)"), self.y_off_edit, self.tr("Shift text vertically in millimeters."))

        self.width_off_edit = FloatSliderEdit(float(getattr(ev, "text_background_width_offset_mm", 0.0) or 0.0), -20.0, 20.0, 0.05, self)
        width_off_tip = self.tr(
            "Some fonts report width differently, so the background can end too early or too late at the last character. "
            "Use this as a manual correction for this text only: positive extends the background to the right, negative makes it narrower. "
            "It changes only the background rectangle, not the text position."
        )
        width_off_row = QtWidgets.QWidget(self)
        width_off_row_layout = QtWidgets.QHBoxLayout(width_off_row)
        width_off_row_layout.setContentsMargins(0, 0, 0, 0)
        width_off_row_layout.addWidget(self.width_off_edit)
        self._add_labeled_row(layout, self.tr("Background width offset (mm)"), width_off_row, width_off_tip)

        self.rot_edit = FloatSliderEdit(float(getattr(ev, "rotation", 0.0) or 0.0), 0.0, 360.0, 0.1, self)
        self._add_labeled_row(layout, self.tr("Rotation (degrees)"), self.rot_edit, self.tr("Rotate text clockwise in degrees."))

        self.use_custom_chk = QtWidgets.QCheckBox(self.tr("Use custom font"), self)
        self.use_custom_chk.setChecked(bool(getattr(ev, "use_custom_font", False)))
        self.use_custom_chk.setToolTip(self.tr("When enabled, this text uses its own font instead of the layout default font."))
        layout.addRow(self.use_custom_chk)

        self.font_picker = FontPicker(cur_font, parent=self)
        self.font_picker.setToolTip(self.tr("Select family, size, weight, and style for this text when custom font is enabled."))
        layout.addRow(self.font_picker)

        self.use_custom_chk.toggled.connect(self._toggle_custom)
        self._toggle_custom(self.use_custom_chk.isChecked())

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, parent=self)
        layout.addRow(btns)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

        self.txt_edit.textChanged.connect(lambda _t: self.valueChanged.emit())
        self.x_off_edit.valueChanged.connect(lambda _v: self.valueChanged.emit())
        self.y_off_edit.valueChanged.connect(lambda _v: self.valueChanged.emit())
        self.width_off_edit.valueChanged.connect(lambda _v: self.valueChanged.emit())
        self.rot_edit.valueChanged.connect(lambda _v: self.valueChanged.emit())
        self.use_custom_chk.toggled.connect(lambda _v: self.valueChanged.emit())
        self.font_picker.valueChanged.connect(self.valueChanged.emit)

    def _coerce_font(self, value, default_font: Font | None) -> Font:
        if isinstance(value, Font):
            return deepcopy(value)
        if isinstance(value, dict):
            return Font(
                family=value.get("family", getattr(default_font, "family", "Courier New")),
                size_pt=float(value.get("size_pt", getattr(default_font, "size_pt", 12.0) or 12.0)),
                bold=bool(value.get("bold", getattr(default_font, "bold", False))),
                italic=bool(value.get("italic", getattr(default_font, "italic", False))),
                x_offset=float(value.get("x_offset", getattr(default_font, "x_offset", 0.0) or 0.0)),
                y_offset=float(value.get("y_offset", getattr(default_font, "y_offset", 0.0) or 0.0)),
            )
        return deepcopy(default_font or Font())

    def _toggle_custom(self, state: bool) -> None:
        self.font_picker.setVisible(bool(state))
        if not state:
            self.font_picker.set_value(deepcopy(self._default_font))

    @staticmethod
    def snapshot_from_event(ev) -> dict:
        return {
            "text": getattr(ev, "text", ""),
            "use_custom_font": bool(getattr(ev, "use_custom_font", False)),
            "font": deepcopy(getattr(ev, "font", None)),
            "text_background_width_offset_mm": float(getattr(ev, "text_background_width_offset_mm", 0.0) or 0.0),
            "x_offset_mm": float(getattr(ev, "x_offset_mm", 0.0) or 0.0),
            "y_offset_mm": float(getattr(ev, "y_offset_mm", 0.0) or 0.0),
            "rotation": float(getattr(ev, "rotation", 0.0) or 0.0),
        }

    def apply_to_event(self, ev) -> None:
        if bool(self.use_custom_chk.isChecked()):
            ev.use_custom_font = True
            ev.font = deepcopy(self.font_picker.value())
        else:
            ev.use_custom_font = False
            ev.font = deepcopy(self._default_font)
        ev.text_background_width_offset_mm = float(self.width_off_edit.value())
        ev.text = self.txt_edit.text()
        ev.x_offset_mm = float(self.x_off_edit.value())
        ev.y_offset_mm = float(self.y_off_edit.value())
        try:
            ev.rotation = float(self.rot_edit.value())
        except Exception:
            pass

    @staticmethod
    def restore_event(ev, snapshot: dict) -> None:
        ev.text = snapshot["text"]
        ev.use_custom_font = snapshot["use_custom_font"]
        ev.font = deepcopy(snapshot["font"])
        ev.text_background_width_offset_mm = float(snapshot["text_background_width_offset_mm"])
        ev.x_offset_mm = float(snapshot["x_offset_mm"])
        ev.y_offset_mm = float(snapshot["y_offset_mm"])
        ev.rotation = float(snapshot["rotation"])
