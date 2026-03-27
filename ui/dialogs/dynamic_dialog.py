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


class DynamicDialog(QtWidgets.QDialog):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, current_value: str = '') -> None:
        super().__init__(parent)
        self.setWindowTitle('Hairpin Dynamic Direction')
        self.setModal(True)
        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        info = QtWidgets.QLabel('Select a dynamic symbol:', self)
        info.setWordWrap(True)
        lay.addWidget(info)

        self.listbox = QtWidgets.QListWidget(self)
        self.listbox.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)

        app_font = QtWidgets.QApplication.font()
        leland_family = register_font_from_bytes('LelandText') or 'LelandText'
        combo_font = QtGui.QFont(app_font)
        combo_font.setFamily(leland_family)
        base_pt = float(app_font.pointSizeF() if app_font.pointSizeF() > 0 else max(8.0, float(app_font.pointSize())))
        combo_font.setPointSizeF(max(12.0, base_pt * 2.0))

        row_h = max(42, int(QtGui.QFontMetrics(combo_font).height() * 3.0))

        self.listbox.setFont(combo_font)

        no_symbol_item = QtWidgets.QListWidgetItem('(use no symbol)')
        no_symbol_item.setData(QtCore.Qt.ItemDataRole.UserRole, '')
        no_symbol_item.setData(QtCore.Qt.ItemDataRole.UserRole + 1, '')
        no_symbol_item.setFont(app_font)
        no_symbol_item.setTextAlignment(int(QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignVCenter))
        no_symbol_item.setSizeHint(QtCore.QSize(0, row_h))
        self.listbox.addItem(no_symbol_item)

        for token, glyph in DYNAMIC_GLYPH_CHOICES:
            item = QtWidgets.QListWidgetItem(glyph)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, glyph)
            item.setData(QtCore.Qt.ItemDataRole.UserRole + 1, token)
            item.setFont(combo_font)
            item.setTextAlignment(int(QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignVCenter))
            item.setSizeHint(QtCore.QSize(0, row_h))
            self.listbox.addItem(item)

        current = str(current_value or '')
        if current:
            for i in range(self.listbox.count()):
                item = self.listbox.item(i)
                if item is not None and str(item.data(QtCore.Qt.ItemDataRole.UserRole) or '') == current:
                    self.listbox.setCurrentRow(i)
                    break
        else:
            self.listbox.setCurrentRow(0)
        if self.listbox.currentRow() < 0 and self.listbox.count() > 0:
            self.listbox.setCurrentRow(0)

        lay.addWidget(self.listbox)

        hint = QtWidgets.QLabel('We use font LelandText for all dynamic symbols. LelandText is the music font from Musescore 3.', self)
        hint.setWordWrap(True)
        lay.addWidget(hint)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, self)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

        self.listbox.itemDoubleClicked.connect(lambda _item: self.accept())

        QtCore.QTimer.singleShot(0, self.listbox.setFocus)

    def selected_glyph(self) -> str:
        item = self.listbox.currentItem()
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
        if dlg.exec() == int(QtWidgets.QDialog.DialogCode.Accepted):
            return dlg.selected_glyph(), True
        return '', False
