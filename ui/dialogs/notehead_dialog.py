from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

import cairo
from PySide6 import QtCore, QtGui, QtWidgets

from file_model.events.note import Note
from symbol_design.noteheads import Notehead, normalize_notehead_literal, resolve_notehead_spec
from ui.widgets.draw_util import DrawUtil, finalize_image_surface, make_image_surface


_NOTEHEAD_CHOICES: list[tuple[str, str]] = [
    ("auto", "Auto"),
    ("circle_white_up", "Circle White Up"),
    ("circle_white_down", "Circle White Down"),
    ("circle_black_up", "Circle Black Up"),
    ("circle_black_down", "Circle Black Down"),
    ("bullet_white_up", "Bullet White Up"),
    ("bullet_white_down", "Bullet White Down"),
    ("bullet_black_up", "Bullet Black Up"),
    ("bullet_black_down", "Bullet Black Down"),
    ("triangle_white_up", "Triangle White Up"),
    ("triangle_white_down", "Triangle White Down"),
    ("triangle_black_up", "Triangle Black Up"),
    ("triangle_black_down", "Triangle Black Down"),
]


class _NoteheadDelegate(QtWidgets.QStyledItemDelegate):
    def paint(self, painter: QtGui.QPainter, option: QtWidgets.QStyleOptionViewItem, index: QtCore.QModelIndex) -> None:
        painter.save()
        if option.state & QtWidgets.QStyle.StateFlag.State_Selected:
            painter.fillRect(option.rect, option.palette.highlight())
        icon_value = index.data(QtCore.Qt.ItemDataRole.DecorationRole)
        text = str(index.data(QtCore.Qt.ItemDataRole.DisplayRole) or "")
        rect = option.rect.adjusted(8, 4, -8, -4)
        if isinstance(icon_value, QtGui.QIcon):
            pixmap = icon_value.pixmap(88, 52)
            painter.drawPixmap(rect.left(), rect.top(), pixmap)
            text_x = rect.left() + 100
        else:
            text_x = rect.left()
        pen = option.palette.highlightedText().color() if option.state & QtWidgets.QStyle.StateFlag.State_Selected else option.palette.text().color()
        painter.setPen(pen)
        text_rect = QtCore.QRect(text_x, rect.top(), max(10, rect.width() - (text_x - rect.left())), rect.height())
        painter.drawText(text_rect, int(QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft), text)
        painter.restore()

    def sizeHint(self, option: QtWidgets.QStyleOptionViewItem, index: QtCore.QModelIndex) -> QtCore.QSize:
        return QtCore.QSize(max(220, option.rect.width()), 60)


class NoteheadDialog(QtWidgets.QDialog):
    _PREVIEW_LAYERING = ["preview_background", "preview_stem", "notehead_white", "notehead_black", "left_dot"]

    def _translated_choice_label(self, literal: str, fallback: str) -> str:
        labels = {
            "auto": self.tr("Auto"),
            "circle_white_up": self.tr("Circle White Up"),
            "circle_white_down": self.tr("Circle White Down"),
            "circle_black_up": self.tr("Circle Black Up"),
            "circle_black_down": self.tr("Circle Black Down"),
            "bullet_white_up": self.tr("Bullet White Up"),
            "bullet_white_down": self.tr("Bullet White Down"),
            "bullet_black_up": self.tr("Bullet Black Up"),
            "bullet_black_down": self.tr("Bullet Black Down"),
            "triangle_white_up": self.tr("Triangle White Up"),
            "triangle_white_down": self.tr("Triangle White Down"),
            "triangle_black_up": self.tr("Triangle Black Up"),
            "triangle_black_down": self.tr("Triangle Black Down"),
        }
        return labels.get(literal, fallback)

    def __init__(
        self,
        *,
        note: Any,
        layout,
        semitone_space_mm: float,
        notation_color: tuple[float, float, float, float],
        paper_color: tuple[float, float, float, float],
        default_black_above: bool,
        choices: Optional[list[tuple[str, str]]] = None,
        show_stem: bool = True,
        outline_width_mm_override: float | None = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowFlag(QtCore.Qt.WindowType.WindowMinMaxButtonsHint, True)
        self.setSizeGripEnabled(True)
        self.setWindowTitle(self.tr("Notehead Override"))
        self.setModal(False)
        self.setWindowModality(QtCore.Qt.WindowModality.NonModal)
        self.setMinimumWidth(420)

        self._note = note
        self._layout = layout
        self._semitone_space_mm = float(max(0.5, semitone_space_mm))
        self._notation_color = notation_color
        self._paper_color = paper_color
        self._default_black_above = bool(default_black_above)
        self._choices = list(choices or _NOTEHEAD_CHOICES)
        self._show_stem = bool(show_stem)
        self._outline_width_mm_override = None if outline_width_mm_override is None else float(outline_width_mm_override)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        info = QtWidgets.QLabel(self.tr("Choose a manual notehead override. Auto keeps the current layout-driven behavior."), self)
        info.setWordWrap(True)
        lay.addWidget(info)

        self.combo = QtWidgets.QComboBox(self)
        self.combo.setEditable(False)
        self.combo.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        self.combo.setItemDelegate(_NoteheadDelegate(self.combo))
        self.combo.setIconSize(QtCore.QSize(88, 52))
        view = self.combo.view()
        if view is not None:
            view.setStyleSheet("QListView::item { min-height: 56px; padding: 3px 6px; }")
        self._populate_choices()
        lay.addWidget(self.combo)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel, self)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

        current_literal = normalize_notehead_literal(getattr(note, "notehead", "auto"))
        idx = self.combo.findData(current_literal, QtCore.Qt.ItemDataRole.UserRole)
        self.combo.setCurrentIndex(idx if idx >= 0 else 0)

    def _populate_choices(self) -> None:
        self.combo.clear()
        for literal, label in self._choices:
            self.combo.addItem(self._translated_choice_label(literal, label))
            idx = self.combo.count() - 1
            self.combo.setItemData(idx, literal, QtCore.Qt.ItemDataRole.UserRole)
            pix = self._render_preview(literal, size=QtCore.QSize(88, 52), px_per_mm=4.0)
            self.combo.setItemData(idx, QtGui.QIcon(pix), QtCore.Qt.ItemDataRole.DecorationRole)

    def _make_preview_note(self, literal: str):
        return replace(self._note, notehead=literal)

    def _render_preview(self, literal: str, *, size: QtCore.QSize, px_per_mm: float) -> QtGui.QPixmap:
        note = self._make_preview_note(literal)
        width_px = max(1, int(size.width()))
        height_px = max(1, int(size.height()))
        width_mm = float(width_px) / float(px_per_mm)
        height_mm = float(height_px) / float(px_per_mm)
        x_mm = width_mm * 0.5
        y_mm = height_mm * 0.5

        du = DrawUtil()
        du.new_page(width_mm, height_mm)
        du.add_rectangle(
            0.0,
            0.0,
            width_mm,
            height_mm,
            stroke_color=None,
            fill_color=self._paper_color,
            corner_radius=1.25,
            tags=["preview_background"],
        )
        if self._show_stem:
            self._draw_preview_stem(du, note, x_mm=x_mm, y_mm=y_mm)
        notehead = Notehead.from_note(
            x_mm=x_mm,
            y_mm=y_mm,
            note=note,
            layout=self._layout,
            semitone_space_mm=self._semitone_space_mm,
            notation_color=self._notation_color,
            paper_color=self._paper_color,
            default_black_above=self._default_black_above,
            outline_width_mm_override=self._outline_width_mm_override,
        )
        tag = "notehead_black" if bool(getattr(notehead, "filled", False)) else "notehead_white"
        notehead.draw_notehead(du, item_id=0, tags=[tag])

        image, surface, _buf = make_image_surface(width_px, height_px)
        ctx = cairo.Context(surface)
        du.render_to_cairo(ctx, 0, float(px_per_mm), layering=self._PREVIEW_LAYERING)
        surface.flush()
        final = finalize_image_surface(image)
        return QtGui.QPixmap.fromImage(final)

    def _draw_preview_stem(self, du: DrawUtil, note: Note, *, x_mm: float, y_mm: float) -> None:
        stem_len = self._layout_value('note_stem_length_semitone', 6.0) * self._semitone_space_mm
        stem_w = self._layout_value('note_stem_thickness_mm', 1.25) * self._layout_value('scale', 1.0)
        hand = str(getattr(note, 'hand', 'l') or 'l')
        x2 = float(x_mm) - float(stem_len) if hand == 'l' else float(x_mm) + float(stem_len)
        du.add_line(
            float(x_mm),
            float(y_mm),
            x2,
            float(y_mm),
            color=self._notation_color,
            width_mm=max(0.05, float(stem_w)),
            tags=['preview_stem'],
        )

    def _layout_value(self, name: str, default):
        src = self._layout
        if isinstance(src, dict):
            return src.get(name, default)
        return getattr(src, name, default)

    def selected_notehead(self) -> str:
        return str(self.combo.currentData(QtCore.Qt.ItemDataRole.UserRole) or "auto")

    @classmethod
    def get_notehead(
        cls,
        *,
        note: Any,
        layout,
        semitone_space_mm: float,
        notation_color: tuple[float, float, float, float],
        paper_color: tuple[float, float, float, float],
        default_black_above: bool,
        choices: Optional[list[tuple[str, str]]] = None,
        show_stem: bool = True,
        outline_width_mm_override: float | None = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> tuple[str, bool]:
        dlg = cls(
            note=note,
            layout=layout,
            semitone_space_mm=semitone_space_mm,
            notation_color=notation_color,
            paper_color=paper_color,
            default_black_above=default_black_above,
            choices=choices,
            show_stem=show_stem,
            outline_width_mm_override=outline_width_mm_override,
            parent=parent,
        )
        if dlg.exec() == int(QtWidgets.QDialog.DialogCode.Accepted):
            return dlg.selected_notehead(), True
        return normalize_notehead_literal(getattr(note, "notehead", "auto")), False
