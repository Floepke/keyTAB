from __future__ import annotations

from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from ui.dialogs import DialogGeometryMixin
from ui.dialogs.style_dialog import ColorPickerEdit, FloatSliderEdit


class LineDialog(DialogGeometryMixin, QtWidgets.QDialog):
    DIALOG_KEY = "line"

    def __init__(self, line: object, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("Line Properties"))
        self.setWindowFlag(QtCore.Qt.WindowType.WindowMinMaxButtonsHint, True)
        self.setSizeGripEnabled(True)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        general_group = QtWidgets.QGroupBox(self.tr("General"), self)
        general_form = QtWidgets.QFormLayout(general_group)
        general_form.setSpacing(8)
        layout.addWidget(general_group)

        self.thickness_edit = FloatSliderEdit(
            float(getattr(line, "thickness_mm", 0.5) or 0.5), 0.05, 10.0, 0.05, self
        )
        self._add_row(
            general_form,
            self.tr("Thickness (mm)"),
            self.thickness_edit,
            self.tr("The line stroke thickness in millimeters."),
        )

        self.dash_pattern_edit = QtWidgets.QLineEdit(self)
        dash_pattern = getattr(line, "dash_pattern_mm", []) or []
        self.dash_pattern_edit.setText(" ".join(self._format_number(value) for value in dash_pattern))
        self.dash_pattern_edit.setValidator(
            QtGui.QRegularExpressionValidator(QtCore.QRegularExpression(r"[0-9. ]*"), self)
        )
        dash_pattern_row = QtWidgets.QWidget(self)
        dash_pattern_layout = QtWidgets.QHBoxLayout(dash_pattern_row)
        dash_pattern_layout.setContentsMargins(0, 0, 0, 0)
        dash_pattern_layout.setSpacing(4)
        dash_pattern_layout.addWidget(self.dash_pattern_edit)
        self.dash_pattern_help_button = QtWidgets.QToolButton(dash_pattern_row)
        self.dash_pattern_help_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogHelpButton)
        )
        self.dash_pattern_help_button.setToolTip(self.tr("Show dash pattern help."))
        self.dash_pattern_help_button.setAccessibleName(self.tr("Dash pattern help"))
        self.dash_pattern_help_button.clicked.connect(self._show_dash_pattern_help)
        dash_pattern_layout.addWidget(self.dash_pattern_help_button)
        self._add_row(
            general_form,
            self.tr("Dash pattern (mm)"),
            dash_pattern_row,
            self.tr("Space-separated dash and gap lengths. Use 0 for a dot; leave empty for a solid line."),
        )

        self.dash_offset_edit = FloatSliderEdit(
            float(getattr(line, "dash_offset_mm", 0.0) or 0.0), -100.0, 100.0, 0.05, self
        )
        self._add_row(
            general_form,
            self.tr("Dash offset (mm)"),
            self.dash_offset_edit,
            self.tr("Offset applied to the dash pattern cycle."),
        )

        color = str(getattr(line, "color", "auto") or "auto")
        self.custom_color_check = QtWidgets.QCheckBox(self.tr("Use Custom Color"), self)
        self.custom_color_check.setChecked(color != "auto")
        general_form.addRow(self.custom_color_check)
        self.color_edit = ColorPickerEdit("" if color == "auto" else color, self)
        self._color_label = self._add_row(
            general_form,
            self.tr("Color"),
            self.color_edit,
            self.tr("Custom line color."),
        )

        zigzag_group = QtWidgets.QGroupBox(self.tr("Zigzag"), self)
        zigzag_form = QtWidgets.QFormLayout(zigzag_group)
        zigzag_form.setSpacing(8)
        layout.addWidget(zigzag_group)

        self.zigzag_type_combo = QtWidgets.QComboBox(self)
        self.zigzag_type_combo.addItem(self.tr("Off"), None)
        self.zigzag_type_combo.addItem(self.tr("Triangle"), "triangle")
        self.zigzag_type_combo.addItem(self.tr("Sine"), "sine")
        zigzag_type = getattr(line, "zigzag_type", None)
        self.zigzag_type_combo.setCurrentIndex(
            max(0, self.zigzag_type_combo.findData(zigzag_type))
        )
        self._add_row(
            zigzag_form,
            self.tr("Zigzag type"),
            self.zigzag_type_combo,
            self.tr("Choose an off, triangle-zigzag, or sine-zigzag line."),
        )

        self.zigzag_amp_edit = FloatSliderEdit(
            float(getattr(line, "zigzag_amp_semitone", 2.0) or 0.0), 0.0, 24.0, 0.25, self
        )
        self._add_row(
            zigzag_form,
            self.tr("Zigzag amplitude (semitones)"),
            self.zigzag_amp_edit,
            self.tr("Maximum zigzag displacement, measured in keyboard semitone spacing."),
        )

        self.zigzag_freq_edit = FloatSliderEdit(
            float(getattr(line, "zigzag_freq", 64.0) or 64.0), 1.0, 4096.0, 1.0, self
        )
        self._add_row(
            zigzag_form,
            self.tr("Zigzag wavelength (time units)"),
            self.zigzag_freq_edit,
            self.tr("The zigzag wavelength in score time units; 256 units equal one quarter note."),
        )

        arrow_group = QtWidgets.QGroupBox(self.tr("Arrow"), self)
        arrow_form_layout = QtWidgets.QFormLayout(arrow_group)
        arrow_form_layout.setSpacing(8)
        layout.addWidget(arrow_group)

        self.arrow_combo = QtWidgets.QComboBox(self)
        self.arrow_combo.addItem(self.tr("Off"), None)
        self.arrow_combo.addItem(self.tr("Left"), "left")
        self.arrow_combo.addItem(self.tr("Right"), "right")
        self.arrow_combo.addItem(self.tr("Both"), "both")
        self.arrow_combo.setCurrentIndex(max(0, self.arrow_combo.findData(getattr(line, "arrow", None))))
        self._add_row(
            arrow_form_layout,
            self.tr("Arrowheads"),
            self.arrow_combo,
            self.tr("Show arrowheads at either endpoint."),
        )

        arrow_form = list(getattr(line, "arrow_form", [0.5, 0.5]) or [])
        arrow_length = float(arrow_form[0]) if arrow_form else 0.5
        arrow_width = float(arrow_form[1]) if len(arrow_form) > 1 else 0.5
        self.arrow_length_edit = FloatSliderEdit(arrow_length, 0.1, 20.0, 0.05, self)
        self._add_row(
            arrow_form_layout,
            self.tr("Arrow length (mm)"),
            self.arrow_length_edit,
            self.tr("Arrowhead length in millimeters."),
        )
        self.arrow_width_edit = FloatSliderEdit(arrow_width, 0.1, 20.0, 0.05, self)
        self._add_row(
            arrow_form_layout,
            self.tr("Arrow width (mm)"),
            self.arrow_width_edit,
            self.tr("Arrowhead width in millimeters."),
        )

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.custom_color_check.toggled.connect(self._update_enabled_controls)
        self.zigzag_type_combo.currentIndexChanged.connect(self._update_enabled_controls)
        self.arrow_combo.currentIndexChanged.connect(self._update_enabled_controls)
        self._update_enabled_controls()

    def _add_row(
        self,
        form: QtWidgets.QFormLayout,
        label_text: str,
        widget: QtWidgets.QWidget,
        tooltip: str,
    ) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(label_text, self)
        label.setToolTip(tooltip)
        widget.setToolTip(tooltip)
        form.addRow(label, widget)
        return label

    @staticmethod
    def _format_number(value: object) -> str:
        try:
            return f"{float(value):g}"
        except (TypeError, ValueError):
            return ""

    @staticmethod
    def _parse_dash_pattern(value: str) -> list[float]:
        result: list[float] = []
        text = str(value or "")
        if "," in text:
            return result
        for part in text.split():
            try:
                length = float(part)
            except ValueError:
                continue
            if length >= 0.0:
                result.append(max(0.01, length))
        return result

    def _show_dash_pattern_help(self) -> None:
        QtWidgets.QMessageBox.information(
            self,
            self.tr("Dash Pattern Help"),
            self.tr(
                "Enter space-separated lengths in millimeters. Values alternate between "
                "drawn dashes and gaps.\n\n"
                'For example, "4 2" draws a 4 mm dash followed by a 2 mm gap. '
                'Use "0 2" for a dot followed by a 2 mm gap. Leave the field empty '
                "for a solid line."
            ),
        )

    def _update_enabled_controls(self, *_args: object) -> None:
        custom_color_enabled = self.custom_color_check.isChecked()
        self._color_label.setVisible(custom_color_enabled)
        self.color_edit.setVisible(custom_color_enabled)
        zigzag_enabled = self.zigzag_type_combo.currentData() is not None
        self.zigzag_amp_edit.setEnabled(zigzag_enabled)
        self.zigzag_freq_edit.setEnabled(zigzag_enabled)
        arrow_enabled = self.arrow_combo.currentData() is not None
        self.arrow_length_edit.setEnabled(arrow_enabled)
        self.arrow_width_edit.setEnabled(arrow_enabled)

    def values(self) -> dict[str, Any]:
        for edit in (
            self.thickness_edit,
            self.dash_offset_edit,
            self.zigzag_amp_edit,
            self.zigzag_freq_edit,
            self.arrow_length_edit,
            self.arrow_width_edit,
        ):
            edit.commit()
        color = self.color_edit.value() if self.custom_color_check.isChecked() else "auto"
        return {
            "thickness_mm": float(self.thickness_edit.value()),
            "dash_pattern_mm": self._parse_dash_pattern(self.dash_pattern_edit.text()),
            "dash_offset_mm": float(self.dash_offset_edit.value()),
            "color": color if color else "auto",
            "zigzag_type": self.zigzag_type_combo.currentData(),
            "zigzag_amp_semitone": float(self.zigzag_amp_edit.value()),
            "zigzag_freq": float(self.zigzag_freq_edit.value()),
            "arrow": self.arrow_combo.currentData(),
            "arrow_form": [
                float(self.arrow_length_edit.value()),
                float(self.arrow_width_edit.value()),
            ],
        }
