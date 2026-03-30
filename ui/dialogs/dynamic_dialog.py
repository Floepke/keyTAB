from __future__ import annotations

from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

from fonts import register_font_from_bytes


class NoAutoActivateListWidget(QtWidgets.QListWidget):
    """Custom list widget that prevents single-click auto-activation."""
    
    # Custom signal to ensure we control when double-click acceptance happens
    itemDoubleClicked = QtCore.Signal(QtWidgets.QListWidgetItem)
    
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
    
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        # Don't call super for single-click to prevent default behavior
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            item = self.itemAt(event.position().toPoint())
            if item is not None:
                self.setCurrentItem(item)
            event.accept()
            return
        super().mousePressEvent(event)
    
    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        # Don't process single-click release
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            event.accept()
            return
        super().mouseReleaseEvent(event)
    
    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        # On double-click, select item and emit signal
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            item = self.itemAt(event.position().toPoint())
            if item is not None:
                self.setCurrentItem(item)
                self.itemDoubleClicked.emit(item)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)
    
    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        # Block Enter key to prevent auto-accepting
        if event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
            event.ignore()
            return
        super().keyPressEvent(event)


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


class DynamicDialog(QtWidgets.QDialog):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, current_value: str = '') -> None:
        super().__init__(parent)
        self.setWindowTitle('Dynamic Symbol Selection')
        self.setModal(True)
        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)

        screen = QtGui.QGuiApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            self.resize(max(760, int(avail.width() * 0.72)), max(520, int(avail.height() * 0.78)))
        else:
            self.resize(980, 700)
        self.setMinimumSize(760, 520)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        info = QtWidgets.QLabel('Select a dynamic symbol:', self)
        info.setWordWrap(True)
        lay.addWidget(info)

        self.grid = NoAutoActivateListWidget(self)
        self.grid.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.grid.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        self.grid.setFlow(QtWidgets.QListView.Flow.LeftToRight)
        self.grid.setWrapping(True)
        self.grid.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self.grid.setMovement(QtWidgets.QListView.Movement.Static)
        self.grid.setLayoutMode(QtWidgets.QListView.LayoutMode.Batched)
        self.grid.setUniformItemSizes(True)
        self.grid.setSpacing(8)
        self.grid.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.grid.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.grid.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)  # Ensure grid can receive focus

        app_font = QtWidgets.QApplication.font()
        leland_family = register_font_from_bytes('LelandText') or 'LelandText'
        symbol_font = QtGui.QFont(app_font)
        symbol_font.setFamily(leland_family)
        base_pt = float(app_font.pointSizeF() if app_font.pointSizeF() > 0 else max(8.0, float(app_font.pointSize())))
        symbol_font.setPointSizeF(max(13.0, base_pt * 2.15))

        metrics = QtGui.QFontMetrics(symbol_font)
        glyph_w = max([metrics.horizontalAdvance(glyph) for _token, glyph in DYNAMIC_GLYPH_CHOICES] or [0])
        glyph_h = max([metrics.boundingRect(glyph).height() for _token, glyph in DYNAMIC_GLYPH_CHOICES] or [metrics.height()])
        cell_w = max(66, int(glyph_w + 28))
        cell_h = max(72, int(glyph_h + 28))
        self.grid.setGridSize(QtCore.QSize(cell_w, cell_h))

        self.grid.setFont(symbol_font)

        no_symbol_item = QtWidgets.QListWidgetItem('none')
        no_symbol_item.setData(QtCore.Qt.ItemDataRole.UserRole, '')
        no_symbol_item.setData(QtCore.Qt.ItemDataRole.UserRole + 1, '')
        no_symbol_item.setToolTip('Use no dynamic symbol')
        no_symbol_item.setFont(app_font)
        no_symbol_item.setTextAlignment(int(QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignVCenter))
        no_symbol_item.setSizeHint(QtCore.QSize(cell_w, cell_h))
        self.grid.addItem(no_symbol_item)

        for token, glyph in DYNAMIC_GLYPH_CHOICES:
            item = QtWidgets.QListWidgetItem(glyph)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, glyph)
            item.setData(QtCore.Qt.ItemDataRole.UserRole + 1, token)
            item.setToolTip(token)
            item.setFont(symbol_font)
            item.setTextAlignment(int(QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignVCenter))
            item.setSizeHint(QtCore.QSize(cell_w, cell_h))
            self.grid.addItem(item)

        current = str(current_value or '')
        if current:
            for i in range(self.grid.count()):
                item = self.grid.item(i)
                if item is not None and str(item.data(QtCore.Qt.ItemDataRole.UserRole) or '') == current:
                    self.grid.setCurrentRow(i)
                    break
        else:
            self.grid.setCurrentRow(0)
        if self.grid.currentRow() < 0 and self.grid.count() > 0:
            self.grid.setCurrentRow(0)

        lay.addWidget(self.grid, 1)

        hint = QtWidgets.QLabel('LelandText symbols are shown in a clickable raster. Double click to choose a symbol or select symbol and hit Ok.', self)
        hint.setWordWrap(True)
        lay.addWidget(hint)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, self)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

        self.grid.itemDoubleClicked.connect(lambda _item: self.accept())

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        """Set focus to grid when dialog becomes visible."""
        super().showEvent(event)
        self.activateWindow()
        self.raise_()
        self.grid.setFocus()

    def hideEvent(self, event: QtGui.QHideEvent) -> None:
        """Clean up when dialog hides."""
        super().hideEvent(event)

    def selected_glyph(self) -> str:
        item = self.grid.currentItem()
        if item is None:
            return ''
        return str(item.data(QtCore.Qt.ItemDataRole.UserRole) or '')

    @classmethod
    def get_dynamic_glyph(
        cls,
        parent: Optional[QtWidgets.QWidget] = None,
        current_value: str = '',
    ) -> tuple[str, bool]:
        dlg = cls(parent=parent, current_value=current_value)
        dlg.show()  # Show dialog first
        dlg.exec()  # Then run modal loop
        if dlg.result() == int(QtWidgets.QDialog.DialogCode.Accepted):
            return dlg.selected_glyph(), True
        return '', False
