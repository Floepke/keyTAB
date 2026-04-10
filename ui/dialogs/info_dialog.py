from __future__ import annotations

from dataclasses import fields
from PySide6 import QtCore, QtWidgets

from file_model.info import Info
from file_model.analysis import Analysis
from file_model.SCORE import SCORE, MetaData


class InfoDialog(QtWidgets.QDialog):
    def __init__(self, score: SCORE, parent=None) -> None:
        super().__init__(parent)
        self.setWindowFlag(QtCore.Qt.WindowType.WindowMinMaxButtonsHint, True)
        self.setSizeGripEnabled(True)
        self.setWindowFlags(self.windowFlags())
        self.setWindowTitle(self.tr("Titles, info & analysis"))
        self.setModal(False)
        self.resize(768, 768)
        self._score = score

        layout = QtWidgets.QVBoxLayout(self)

        info_group = QtWidgets.QGroupBox(self.tr("Info"), self)
        info_form = QtWidgets.QFormLayout(info_group)
        info_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self._title_edit = QtWidgets.QLineEdit(self)
        self._composer_edit = QtWidgets.QLineEdit(self)
        self._copyright_edit = QtWidgets.QLineEdit(self)
        self._arranger_edit = QtWidgets.QLineEdit(self)
        self._lyricist_edit = QtWidgets.QLineEdit(self)
        self._comment_edit = QtWidgets.QPlainTextEdit(self)
        self._comment_edit.setMinimumHeight(90)

        self._copyright_btn = QtWidgets.QPushButton("©", self)
        self._copyright_btn.setFixedWidth(28)
        self._copyright_btn.setFixedHeight(self._copyright_edit.sizeHint().height())
        self._copyright_btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self._copyright_btn.setToolTip(self.tr("Insert © symbol at cursor"))
        self._copyright_btn.clicked.connect(self._insert_copyright_symbol)
        copyright_row = QtWidgets.QWidget(self)
        copyright_hl = QtWidgets.QHBoxLayout(copyright_row)
        copyright_hl.setContentsMargins(0, 0, 0, 0)
        copyright_hl.setSpacing(4)
        copyright_hl.addWidget(self._copyright_edit)
        copyright_hl.addWidget(self._copyright_btn)

        info_form.addRow(self.tr("Title:"), self._title_edit)
        info_form.addRow(self.tr("Composer:"), self._composer_edit)
        info_form.addRow(self.tr("Copyright:"), copyright_row)
        info_form.addRow(self.tr("Arranger:"), self._arranger_edit)
        info_form.addRow(self.tr("Lyricist:"), self._lyricist_edit)
        info_form.addRow(self.tr("Comment:"), self._comment_edit)
        layout.addWidget(info_group, stretch=1)

        meta_group = QtWidgets.QGroupBox(self.tr("Meta data"), self)
        meta_form = QtWidgets.QFormLayout(meta_group)
        meta_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self._meta_labels: dict[str, QtWidgets.QLabel] = {}
        meta_label_map = {
            "description": self.tr("Description:"),
            "extension": self.tr("Extension:"),
            "format": self.tr("Format:"),
            "creation_timestamp": self.tr("Creation timestamp:"),
            "modification_timestamp": self.tr("Modification timestamp:"),
        }
        for f in fields(MetaData):
            label = QtWidgets.QLabel(self)
            label.setText("")
            key = str(f.name)
            meta_form.addRow(meta_label_map.get(key, self.tr("Metadata:")), label)
            self._meta_labels[key] = label
        layout.addWidget(meta_group)

        analysis_group = QtWidgets.QGroupBox(self.tr("Analysis"), self)
        analysis_form = QtWidgets.QFormLayout(analysis_group)
        analysis_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self._analysis_labels: dict[str, QtWidgets.QLabel] = {}
        for key, label_text in (
            ("notes", self.tr("Notes:")),
            ("grace_notes", self.tr("Grace notes:")),
            ("pages", self.tr("Pages:")),
            ("measures", self.tr("Measures:")),
            ("lines", self.tr("Lines:")),
        ):
            label = QtWidgets.QLabel(self)
            label.setText("")
            analysis_form.addRow(label_text, label)
            self._analysis_labels[key] = label
        layout.addWidget(analysis_group)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._load_from_score()

    def _load_from_score(self) -> None:
        meta = getattr(self._score, "meta_data", None)
        for key, label in self._meta_labels.items():
            value = ""
            if meta is not None:
                value = str(getattr(meta, key, "") or "")
            label.setText(value or self.tr("(not set)"))

        info = getattr(self._score, "info", None) or Info()
        self._title_edit.setText(str(getattr(info, "title", "") or ""))
        self._composer_edit.setText(str(getattr(info, "composer", "") or ""))
        self._copyright_edit.setText(str(getattr(info, "copyright", "") or ""))
        self._arranger_edit.setText(str(getattr(info, "arranger", "") or ""))
        self._lyricist_edit.setText(str(getattr(info, "lyricist", "") or ""))
        self._comment_edit.setPlainText(str(getattr(info, "comment", "") or ""))

        analysis_obj = getattr(self._score, "analysis", None)
        try:
            analysis_snapshot = Analysis.compute(
                self._score,
                lines_count=getattr(analysis_obj, "lines", None),
                pages_count=getattr(analysis_obj, "pages", None),
            )
        except Exception:
            analysis_snapshot = Analysis()
        try:
            self._score.analysis = analysis_snapshot
        except Exception:
            pass
        for key, label in self._analysis_labels.items():
            try:
                value = getattr(analysis_snapshot, key, None)
            except Exception:
                value = None
            if key == "pages" and (value is None or value <= 0):
                text = self.tr("Not engraved yet")
            else:
                text = str(value if value is not None else 0)
            label.setText(text)

    def apply_to_score(self) -> None:
        info = getattr(self._score, "info", None)
        if info is None:
            info = Info()
            self._score.info = info
        info.title = str(self._title_edit.text())
        info.composer = str(self._composer_edit.text())
        info.copyright = str(self._copyright_edit.text())
        info.arranger = str(self._arranger_edit.text())
        info.lyricist = str(self._lyricist_edit.text())
        info.comment = str(self._comment_edit.toPlainText())

    def _insert_copyright_symbol(self) -> None:
        self._copyright_edit.insert("©")
        self._copyright_edit.setFocus()
