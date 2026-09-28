from __future__ import annotations

import json
from typing import Any

from PySide6 import QtCore, QtWidgets
from ui.dialogs import DialogGeometryMixin


class FileInspectorDialog(DialogGeometryMixin, QtWidgets.QDialog):
    """Displays a JSON-compatible score payload as a collapsible tree."""

    DIALOG_KEY = "file_inspector"

    def __init__(self, payload: dict[str, Any], parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("File Inspector"))

        layout = QtWidgets.QVBoxLayout(self)
        self.tree = QtWidgets.QTreeWidget(self)
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels([self.tr("Key"), self.tr("Value")])
        self.tree.setAlternatingRowColors(True)
        self.tree.setUniformRowHeights(True)
        self.tree.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.tree.header().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.tree.header().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.tree)

        self._add_value(None, self.tr("Score"), payload)
        self.tree.expandAll()

        buttons = QtWidgets.QDialogButtonBox(self)
        expand_button = buttons.addButton(self.tr("Expand All"), QtWidgets.QDialogButtonBox.ButtonRole.ActionRole)
        collapse_button = buttons.addButton(self.tr("Collapse All"), QtWidgets.QDialogButtonBox.ButtonRole.ActionRole)
        buttons.addButton(QtWidgets.QDialogButtonBox.StandardButton.Close)
        expand_button.clicked.connect(self.tree.expandAll)
        collapse_button.clicked.connect(self.tree.collapseAll)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _add_value(
        self,
        parent: QtWidgets.QTreeWidgetItem | None,
        key: str,
        value: Any,
    ) -> QtWidgets.QTreeWidgetItem:
        item = QtWidgets.QTreeWidgetItem([key, self._summary(value)])
        if parent is None:
            self.tree.addTopLevelItem(item)
        else:
            parent.addChild(item)

        if isinstance(value, dict):
            for child_key, child_value in value.items():
                self._add_value(item, str(child_key), child_value)
        elif isinstance(value, list):
            for index, child_value in enumerate(value):
                self._add_value(item, f"[{index}]", child_value)
        return item

    @staticmethod
    def _summary(value: Any) -> str:
        if isinstance(value, dict):
            return f"{{{len(value)}}}"
        if isinstance(value, list):
            return f"[{len(value)}]"
        return json.dumps(value, ensure_ascii=False)