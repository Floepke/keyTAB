from __future__ import annotations
from typing import Callable, Optional

from PySide6 import QtCore, QtWidgets, QtGui

from ui.dialogs import DialogGeometryMixin
from ui.dialogs.style_dialog import FloatSliderEdit
from fonts import register_font_from_bytes


DYNAMIC_GLYPH_CHOICES: list[tuple[str, str]] = [
    ('pppppp', '\ue527'),
    ('ppppp', '\ue528'),
    ('pppp', '\ue529'),
    ('ppp', '\ue52a'),
    ('pp', '\ue52b'),
    ('p', '\ue520'),
    ('mp', '\ue52c'),
    ('mf', '\ue52d'),
    ('f', '\ue522'),
    ('pf', '\ue52e'),
    ('ff', '\ue52f'),
    ('fff', '\ue530'),
    ('ffff', '\ue531'),
    ('fffff', '\ue532'),
    ('ffffff', '\ue533'),
    ('fp', '\ue534'),
    ('fz', '\ue535'),
    ('sf', '\ue536'),
    ('sfp', '\ue537'),
    ('sfpp', '\ue538'),
    ('sfz', '\ue539'),
    ('sfzp', '\ue53a'),
    ('sffz', '\ue53b'),
    ('rf', '\ue53c'),
    ('rfz', '\ue53d'),
    ('m', '\ue521'),
    ('r', '\ue523'),
    ('s', '\ue524'),
    ('z', '\ue525'),
    ('n', '\ue526'),
]


class DynamicSymbolGrid(QtWidgets.QListWidget):
    """Grid widget for selecting dynamic symbols."""
    
    symbol_selected = QtCore.Signal(str)       # Emitted when a symbol is single-clicked
    symbol_double_clicked = QtCore.Signal(str)  # Emitted when a symbol is double-clicked
    
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, current_value: str = '') -> None:
        super().__init__(parent)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        self.setFlow(QtWidgets.QListView.Flow.LeftToRight)
        self.setWrapping(True)
        self.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self.setMovement(QtWidgets.QListView.Movement.Static)
        self.setLayoutMode(QtWidgets.QListView.LayoutMode.Batched)
        self.setUniformItemSizes(True)
        self.setSpacing(4)
        self.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # Font setup
        app_font = QtWidgets.QApplication.font()
        leland_family = register_font_from_bytes('LelandText') or 'LelandText'
        symbol_font = QtGui.QFont(app_font)
        symbol_font.setFamily(leland_family)
        base_pt = float(app_font.pointSizeF() if app_font.pointSizeF() > 0 else max(8.0, float(app_font.pointSize())))
        symbol_font.setPointSizeF(max(13.0, base_pt * 2.15))
        
        # Calculate grid size
        metrics = QtGui.QFontMetrics(symbol_font)
        glyph_w = max([metrics.horizontalAdvance(glyph) for _token, glyph in DYNAMIC_GLYPH_CHOICES] or [0])
        glyph_h = max([metrics.boundingRect(glyph).height() for _token, glyph in DYNAMIC_GLYPH_CHOICES] or [metrics.height()])
        cell_w = max(60, int(glyph_w + 20))
        cell_h = max(60, int(glyph_h + 20))
        self.setGridSize(QtCore.QSize(cell_w, cell_h))
        self.setFont(symbol_font)
        
        # Add 'none' option
        no_symbol_item = QtWidgets.QListWidgetItem(self.tr('none'))
        no_symbol_item.setData(QtCore.Qt.ItemDataRole.UserRole, '')
        no_symbol_item.setToolTip(self.tr('Use no dynamic symbol'))
        no_symbol_item.setFont(app_font)
        no_symbol_item.setTextAlignment(int(QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignVCenter))
        no_symbol_item.setSizeHint(QtCore.QSize(cell_w, cell_h))
        self.addItem(no_symbol_item)
        
        # Add glyphs
        for token, glyph in DYNAMIC_GLYPH_CHOICES:
            item = QtWidgets.QListWidgetItem(glyph)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, glyph)
            item.setData(QtCore.Qt.ItemDataRole.UserRole + 1, token)
            item.setToolTip(token)
            item.setFont(symbol_font)
            item.setTextAlignment(int(QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignVCenter))
            item.setSizeHint(QtCore.QSize(cell_w, cell_h))
            self.addItem(item)
        
        # Set current selection
        current = str(current_value or '')
        if current:
            for i in range(self.count()):
                item = self.item(i)
                if item is not None and str(item.data(QtCore.Qt.ItemDataRole.UserRole) or '') == current:
                    self.setCurrentRow(i)
                    break
        else:
            self.setCurrentRow(0)
        
        if self.currentRow() < 0 and self.count() > 0:
            self.setCurrentRow(0)
        
        # Connect single- and double-click
        self.itemClicked.connect(self._on_item_clicked)
        self.itemDoubleClicked.connect(self._on_item_double_clicked)
    
    def _on_item_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        glyph = str(item.data(QtCore.Qt.ItemDataRole.UserRole) or '')
        self.symbol_selected.emit(glyph)

    def _on_item_double_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        glyph = str(item.data(QtCore.Qt.ItemDataRole.UserRole) or '')
        self.symbol_double_clicked.emit(glyph)
    
    def selected_glyph(self) -> str:
        """Get currently selected glyph."""
        item = self.currentItem()
        if item is None:
            return ''
        return str(item.data(QtCore.Qt.ItemDataRole.UserRole) or '')


class DynamicSymbolDialog(DialogGeometryMixin, QtWidgets.QDialog):
    DIALOG_KEY = "dynamic_symbol"

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
