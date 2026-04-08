from __future__ import annotations

from pathlib import Path
from PySide6 import QtCore, QtGui, QtWidgets
from icons.icons import get_qicon


class AboutDialog(QtWidgets.QDialog):
    """Shows app licensing and third-party attributions."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("About keyTAB"))
        self.setModal(False)
        self.setMinimumWidth(768)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(10)

        header_container = QtWidgets.QWidget(self)
        header_layout = QtWidgets.QHBoxLayout(header_container)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)

        header = QtWidgets.QLabel(
            "<h2 style='margin-bottom: 10px;'>keyTAB</h2>"
            f"<div style='margin-bottom: 10px;font-size: 10px;'>{self.tr('keyTAB is a long-running passion project. With modern tooling I can focus on the hard part: shaping a clearer way to read and engrave Klavarskribo music.')}</div>"
            f"<div style='margin-bottom: 10px;font-size: 10px;'>{self.tr('Built on Klavarskribo notation, keyTAB turns MIDI into readable plots. Music flows top-to-bottom on a vertical timeline over a customizable, time-signature-aware grid.')}</div>"
            f"<div style='margin-bottom: 10px;font-size: 10px;'>{self.tr('Stave lines map directly to the black piano keys: black noteheads sit on black key lines, white noteheads land between lines. Pitch reads like the piano keyboard\u2014no key signatures, sharps/flats, clef changes, or other detours.')}</div>"
            f"<div style='margin-bottom: 10px;font-size: 10px;'>{self.tr('I hope keyTAB helps musicians, composers, and curious listeners visualize and refine this MIDI style notation with clarity. Feedback is always welcome.')}</div>"
            f"<div style='margin-bottom: 10px;font-size: 10px;'>{self.tr('Have fun exploring your MIDI with keyTAB!')}</div>"
            "<div style='margin-bottom: 0; font-size: 12px;'>Philip Bergwerf</div>"
        )
        header.setTextFormat(QtCore.Qt.TextFormat.RichText)
        header.setWordWrap(True)
        header.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)

        logo_lbl = QtWidgets.QLabel(self)
        try:
            icon = get_qicon('keyTAB', size=(256, 256))
            if icon is not None:
                pm = icon.pixmap(256, 256)
                if not pm.isNull():
                    logo_lbl.setPixmap(pm)
        except Exception:
            pass
        logo_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignTop)

        header_layout.addWidget(header, 1)
        header_layout.addWidget(logo_lbl, 0, QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignTop)
        layout.addWidget(header_container)

        body = QtWidgets.QLabel(self._credits_html())
        body.setTextFormat(QtCore.Qt.TextFormat.RichText)
        body.setOpenExternalLinks(True)
        body.setWordWrap(True)
        layout.addWidget(body)

        btns = QtWidgets.QDialogButtonBox(self)
        btns.setStandardButtons(QtWidgets.QDialogButtonBox.StandardButton.Close)

        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _credits_html(self) -> str:
        return (
            f'<p><b style="font-size: 10px;">{self.tr("Project license:")} </b>MIT License.</p>'
            f'<p><b style="font-size: 10px;">{self.tr("Credits and third-party components:")}</b></p>'
            '<ul style="margin-top: 4px;font-size: 10px;">'
            f'<li>{self.tr("User-provided SoundFont (.sf2/.sf3).")}</li>'
            f'<li>{self.tr("FluidSynth \u2014 LGPL-2.1-or-later.")}</li>'
            f'<li>{self.tr("PySide6 / Qt \u2014 LGPL-3.0.")}</li>'
            f'<li>{self.tr("pretty_midi, mido, python-rtmidi, numpy \u2014 permissive licenses.")}</li>'
            '</ul>'
            '<p style="margin-top: 0px;font-size: 10px;"> </p>'
        )
