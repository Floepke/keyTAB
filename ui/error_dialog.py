from __future__ import annotations

from PySide6 import QtCore, QtWidgets

def _tr(text: str) -> str:
    return QtCore.QCoreApplication.translate("ErrorDialog", text)


def show_error_dialog(
    parent,
    title: str,
    text: str,
    *,
    details: str = "",
    informative_text: str = "",
    icon: QtWidgets.QMessageBox.Icon = QtWidgets.QMessageBox.Icon.Critical,
) -> None:
    """Show an error dialog with an optional copy-log action.

    This is intended as the shared path for user-facing error popups.
    """
    class _StickyCopyMessageBox(QtWidgets.QMessageBox):
        """Keep the dialog open when the copy-action button is clicked."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._copy_btn = None

        def done(self, result: int) -> None:
            if self._copy_btn is not None and self.clickedButton() is self._copy_btn:
                return
            super().done(result)

    msg = _StickyCopyMessageBox(parent)
    msg.setIcon(icon)
    msg.setWindowTitle(str(title or _tr("Error")))
    msg.setText(str(text or _tr("An error occurred.")))
    if informative_text:
        msg.setInformativeText(str(informative_text))
    detail_text = str(details or "").strip()
    if detail_text:
        msg.setDetailedText(detail_text)
    try:
        msg.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            | QtCore.Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
    except Exception:
        pass

    ok_btn = msg.addButton(QtWidgets.QMessageBox.StandardButton.Ok)
    copy_btn = None
    if detail_text or text:
        copy_btn = msg.addButton(_tr("Copy Error Log"), QtWidgets.QMessageBox.ButtonRole.ActionRole)
        msg._copy_btn = copy_btn

    if copy_btn is not None:
        payload = detail_text or str(text or "")

        def _copy_error_log() -> None:
            clipboard = QtWidgets.QApplication.clipboard()
            try:
                clipboard.setText(payload)
            except Exception:
                pass

        copy_btn.clicked.connect(_copy_error_log)

    if isinstance(ok_btn, QtWidgets.QPushButton):
        msg.setDefaultButton(ok_btn)
    msg.exec()
