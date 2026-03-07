from __future__ import annotations

import sys

from PySide6 import QtCore, QtGui, QtWidgets

from settings_manager import get_preferences_manager


class PreferencesDialog(QtWidgets.QDialog):
    # Tweak this value to adjust the shared width of all first-column descriptions.
    FIRST_COLUMN_WIDTH = 350

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags())
        self.setWindowTitle("Preferences")
        self.setModal(True)
        self.resize(768, 768)

        self._pm = get_preferences_manager()
        self._initial_values = dict(self._pm._values)
        self._fields: dict[str, tuple[str, QtWidgets.QWidget]] = {}

        layout = QtWidgets.QVBoxLayout(self)

        restart_notice = QtWidgets.QLabel(
            "Changes are applied by restarting keyTAB when you press Apply and Restart keyTAB.",
            self,
        )
        restart_notice.setWordWrap(True)
        layout.addWidget(restart_notice)

        prefs_group = QtWidgets.QGroupBox("Preferences", self)
        group_layout = QtWidgets.QVBoxLayout(prefs_group)
        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        group_layout.addWidget(scroll)

        body = QtWidgets.QWidget()
        body.setObjectName("PrefsForm")
        body_layout = QtWidgets.QVBoxLayout(body)
        body_layout.setContentsMargins(12, 8, 12, 8)
        body_layout.setSpacing(8)
        scroll.setWidget(body)

        app_palette = QtWidgets.QApplication.palette(self)
        accent_qcolor = app_palette.color(QtGui.QPalette.ColorRole.Text)
        accent_css = f"rgb({accent_qcolor.red()}, {accent_qcolor.green()}, {accent_qcolor.blue()})"
        first_column_width = int(self.FIRST_COLUMN_WIDTH)
        for key, pref in self._pm.iter_schema():
            widget, kind = self._build_editor(key, pref)
            setting_group = QtWidgets.QGroupBox(self._pretty_label(key), self)
            setting_form = QtWidgets.QFormLayout(setting_group)
            setting_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
            setting_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
            setting_form.setHorizontalSpacing(12)
            setting_form.setVerticalSpacing(0)
            desc = QtWidgets.QLabel(str(getattr(pref, 'description', '') or ''))
            desc.setWordWrap(True)
            desc.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
            desc.setFixedWidth(first_column_width)
            desc.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Preferred)
            desc.setStyleSheet(f"font-size: 11px; color: {accent_css};")
            setting_form.addRow(desc, widget)
            body_layout.addWidget(setting_group)
            self._fields[key] = (kind, widget)

        body_layout.addStretch(1)

        layout.addWidget(prefs_group, stretch=1)

        buttons = QtWidgets.QDialogButtonBox(self)
        apply_button = buttons.addButton("Apply and Restart keyTAB", QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton(QtWidgets.QDialogButtonBox.StandardButton.Close)
        apply_button.clicked.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons, stretch=0)

    def _pretty_label(self, key: str) -> str:
        parts = key.split("_")
        pretty_parts = []
        for p in parts:
            if p in ("ui", "fps"):
                pretty_parts.append(p.upper())
            else:
                pretty_parts.append(p.capitalize())
        return " ".join(pretty_parts)

    def _build_editor(self, key: str, pref) -> tuple[QtWidgets.QWidget, str]:
        value = self._pm.get(key, pref.default)
        if key == "theme":
            combo = QtWidgets.QComboBox()
            combo.addItems(["light", "dark"])
            try:
                idx = combo.findText(str(value))
                combo.setCurrentIndex(idx if idx >= 0 else 0)
            except Exception:
                combo.setCurrentIndex(0)
            return combo, "theme"
        if isinstance(pref.default, bool):
            checkbox = QtWidgets.QCheckBox()
            checkbox.setChecked(bool(value))
            return checkbox, "bool"
        if isinstance(pref.default, int) and not isinstance(pref.default, bool):
            spin = QtWidgets.QSpinBox()
            min_v = -1000000000
            max_v = 1000000000
            try:
                if getattr(pref, 'min', None) is not None:
                    min_v = int(pref.min)
                if getattr(pref, 'max', None) is not None:
                    max_v = int(pref.max)
            except Exception:
                min_v = -1000000000
                max_v = 1000000000
            if max_v < min_v:
                min_v, max_v = max_v, min_v
            spin.setRange(min_v, max_v)
            spin.setSingleStep(1)
            try:
                spin.setValue(int(value))
            except Exception:
                spin.setValue(int(pref.default))
            return spin, "int"
        if isinstance(pref.default, float):
            spin = QtWidgets.QDoubleSpinBox()
            spin.setDecimals(2)
            min_v = -1000000000.0
            max_v = 1000000000.0
            try:
                if getattr(pref, 'min', None) is not None:
                    min_v = float(pref.min)
                if getattr(pref, 'max', None) is not None:
                    max_v = float(pref.max)
            except Exception:
                min_v = -1000000000.0
                max_v = 1000000000.0
            if max_v < min_v:
                min_v, max_v = max_v, min_v
            spin.setRange(min_v, max_v)
            spin.setSingleStep(0.05)
            try:
                spin.setValue(float(value))
            except Exception:
                spin.setValue(float(pref.default))
            return spin, "float"
        line = QtWidgets.QLineEdit()
        line.setText(str(value) if value is not None else "")
        return line, "str"

    def _on_accept(self) -> None:
        self._apply_changes()
        self._restart_application()
        self.accept()

    def _restart_application(self) -> None:
        app = QtWidgets.QApplication.instance()
        if app is None:
            return
        try:
            args = list(sys.argv)
            if args:
                QtCore.QProcess.startDetached(sys.executable, args)
        except Exception:
            pass
        app.quit()

    def _apply_changes(self) -> None:
        for key, (kind, widget) in self._fields.items():
            try:
                if kind == "theme":
                    value = str(widget.currentText())
                elif kind == "bool":
                    value = bool(widget.isChecked())
                elif kind == "int":
                    value = int(widget.value())
                elif kind == "float":
                    value = float(widget.value())
                else:
                    value = str(widget.text())
                self._pm.set(key, value)
            except Exception:
                continue
        try:
            self._pm.save()
        except Exception:
            pass
