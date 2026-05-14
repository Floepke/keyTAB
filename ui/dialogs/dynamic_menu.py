from __future__ import annotations

from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

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


class DynamicSymbolMenu(QtWidgets.QWidget):
    """Floating menu widget for selecting dynamic symbols."""
    
    # Signal emitted when menu closes with selection
    finished = QtCore.Signal(str, bool)  # (glyph, accepted)
    
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, current_value: str = '') -> None:
        super().__init__(parent)
        self.setWindowFlags(QtCore.Qt.WindowType.Popup | QtCore.Qt.WindowType.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, False)
        
        self._selected_glyph = ''
        self._accept_flag = False
        self._event_loop = None
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(0)
        
        self.grid = DynamicSymbolGrid(self, current_value=current_value)
        self.grid.symbol_selected.connect(self._on_symbol_selected)
        layout.addWidget(self.grid)
        
        # Set size
        self.resize(500, 380)
    
    def _on_symbol_selected(self, glyph: str) -> None:
        """Handle symbol selection."""
        self._selected_glyph = glyph
        self._accept_flag = True
        self.finished.emit(glyph, True)
        self.hide()
        if self._event_loop:
            self._event_loop.quit()
    
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Handle menu close."""
        self.finished.emit(self._selected_glyph, self._accept_flag)
        if self._event_loop:
            self._event_loop.quit()
        super().closeEvent(event)
    
    def hideEvent(self, event: QtGui.QHideEvent) -> None:
        """Handle menu hide."""
        self.finished.emit(self._selected_glyph, self._accept_flag)
        if self._event_loop:
            self._event_loop.quit()
        super().hideEvent(event)
    
    def exec_at(self, pos: QtCore.QPoint) -> tuple[str, bool]:
        """Show menu at position and return selected glyph."""
        self.move(pos)
        self.show()
        self.grid.setFocus()
        self.activateWindow()
        
        # Run event loop until menu closes
        self._event_loop = QtCore.QEventLoop()
        self._event_loop.exec()
        self._event_loop = None
        
        return self._selected_glyph, self._accept_flag
    
    @classmethod
    def get_dynamic_glyph(
        cls,
        parent: Optional[QtWidgets.QWidget] = None,
        current_value: str = '',
        pos: Optional[QtCore.QPoint] = None,
    ) -> tuple[str, bool]:
        """Show menu and return selected glyph."""
        menu = cls(parent=parent, current_value=current_value)
        if pos is None:
            pos = QtGui.QCursor.pos()
        return menu.exec_at(pos)
