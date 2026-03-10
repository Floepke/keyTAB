from __future__ import annotations

from typing import Callable, Dict

from PySide6 import QtCore, QtGui, QtWidgets

from scripting.spec import (
    ArrayField,
    BoolField,
    DialogSpec,
    Field,
    FloatField,
    IntField,
    LabelField,
    StringField,
)


class ScriptDialog(QtWidgets.QDialog):
    def __init__(
        self,
        spec: DialogSpec,
        on_preview: Callable[[Dict[str, object]], None],
        on_apply: Callable[[Dict[str, object]], None],
        on_cancel: Callable[[], None],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._spec = spec
        self._on_preview = on_preview
        self._on_apply = on_apply
        self._on_cancel = on_cancel
        self._widgets: Dict[str, QtWidgets.QWidget] = {}
        self._status = QtWidgets.QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color: gray;")
        self.setWindowTitle(spec.title)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        for field in self._spec.fields:
            widget = self._build_widget_for_field(field)
            if isinstance(field, LabelField):
                form.addRow(widget)
                continue
            self._widgets[field.name] = widget
            form.addRow(field.label, widget)
        layout.addLayout(form)
        layout.addWidget(self._status)

        btn_box = QtWidgets.QDialogButtonBox(self)
        self._preview_btn = btn_box.addButton("Preview", QtWidgets.QDialogButtonBox.ButtonRole.ActionRole)
        self._ok_btn = btn_box.addButton(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        self._cancel_btn = btn_box.addButton(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        self._preview_btn.clicked.connect(self._handle_preview)
        btn_box.accepted.connect(self._handle_apply)
        btn_box.rejected.connect(self._handle_cancel)
        layout.addWidget(btn_box)

        self.resize(420, self.sizeHint().height())

    def _build_widget_for_field(self, field: Field) -> QtWidgets.QWidget:
        if isinstance(field, LabelField):
            w = QtWidgets.QLabel(str(field.text or field.label or ""))
            w.setWordWrap(True)
            return w
        if isinstance(field, BoolField):
            w = QtWidgets.QCheckBox()
            w.setChecked(bool(field.default))
            return w
        if isinstance(field, IntField):
            w = QtWidgets.QSpinBox()
            w.setRange(int(field.minimum), int(field.maximum))
            w.setSingleStep(int(field.step))
            w.setValue(int(field.default))
            return w
        if isinstance(field, FloatField):
            w = QtWidgets.QDoubleSpinBox()
            w.setRange(float(field.minimum), float(field.maximum))
            w.setSingleStep(float(field.step))
            w.setDecimals(int(field.decimals))
            w.setValue(float(field.default))
            return w
        if isinstance(field, StringField):
            w = QtWidgets.QLineEdit()
            w.setText(str(field.default or ""))
            w.setPlaceholderText(str(field.placeholder or ""))
            return w
        if isinstance(field, ArrayField):
            w = QtWidgets.QLineEdit()
            w.setText(str(field.default or ""))
            regex = "^[^\n]*$"
            w.setValidator(QtGui.QRegularExpressionValidator(QtCore.QRegularExpression(regex), w))
            return w
        raise ValueError(f"Unsupported field type: {field}")

    def _collect_values(self) -> Dict[str, object]:
        values: Dict[str, object] = {}
        for field in self._spec.fields:
            if isinstance(field, LabelField):
                continue
            widget = self._widgets[field.name]
            values[field.name] = self._value_for_field(field, widget)
        return values

    def _value_for_field(self, field: Field, widget: QtWidgets.QWidget) -> object:
        if isinstance(field, BoolField):
            return bool(widget.isChecked())  # type: ignore[attr-defined]
        if isinstance(field, IntField):
            return int(widget.value())  # type: ignore[attr-defined]
        if isinstance(field, FloatField):
            return float(widget.value())  # type: ignore[attr-defined]
        if isinstance(field, StringField):
            return str(widget.text())  # type: ignore[attr-defined]
        if isinstance(field, ArrayField):
            raw = str(widget.text())  # type: ignore[attr-defined]
            parts = [p.strip() for p in raw.split(field.separator) if p.strip()]
            if field.element_type == "int":
                return [int(p) for p in parts]
            if field.element_type == "float":
                return [float(p) for p in parts]
            return parts
        raise ValueError(f"Unsupported field type: {field}")

    def _handle_preview(self) -> None:
        try:
            vals = self._collect_values()
            self._on_preview(vals)
            self._set_status("Preview applied", ok=True)
        except Exception as exc:
            self._set_status(str(exc), ok=False)

    def _handle_apply(self) -> None:
        try:
            vals = self._collect_values()
            self._on_apply(vals)
            self.accept()
        except Exception as exc:
            self._set_status(str(exc), ok=False)

    def _handle_cancel(self) -> None:
        try:
            self._on_cancel()
        finally:
            super().reject()

    def reject(self) -> None:  # noqa: D401
        self._handle_cancel()

    def _set_status(self, text: str, ok: bool = True) -> None:
        color = "#007a3d" if ok else "#800020"
        self._status.setStyleSheet(f"color: {color};")
        self._status.setText(text)