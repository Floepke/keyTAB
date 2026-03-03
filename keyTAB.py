import argparse
import os
import shutil
import subprocess
import sys
import multiprocessing as mp
from pathlib import Path
from PySide6 import QtCore, QtWidgets, QtGui
from ui.main_window import MainWindow
from ui.style import Style
from settings_manager import get_preferences
from appdata_manager import get_appdata_manager
from icons.icons import get_qicon
from fonts import install_default_ui_font
from utils.file_associations import extract_document_paths

APP_NAME = "keyTAB"
MIME_TYPE_KEYTAB = "application/x-keytab"
MIME_TYPES_MIDI = ["audio/midi", "audio/x-midi"]
MIME_TYPES_MUSICXML = [
    "application/vnd.recordare.musicxml+xml",
    "application/vnd.recordare.musicxml",
]


class KeyTabApplication(QtWidgets.QApplication):
    fileRequested = QtCore.Signal(str)

    def event(self, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.Type.FileOpen:
            try:
                file_path = event.file()
            except AttributeError:
                file_path = None
            if file_path:
                self.fileRequested.emit(file_path)
            return True
        return super().event(event)


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install desktop entry and MIME types (Linux only).",
    )
    return parser.parse_known_args(argv)


def _write_desktop_entry(appimage_path: Path, icon_path: Path | None) -> None:
    desktop_dir = Path.home() / ".local" / "share" / "applications"
    desktop_dir.mkdir(parents=True, exist_ok=True)
    desktop_path = desktop_dir / f"{APP_NAME}.desktop"

    icon_value = APP_NAME
    if icon_path and icon_path.exists():
        icon_dir = Path.home() / ".local" / "share" / "icons" / "hicolor" / "256x256" / "apps"
        icon_dir.mkdir(parents=True, exist_ok=True)
        target_icon = icon_dir / f"{APP_NAME}.png"
        shutil.copy2(icon_path, target_icon)
        icon_value = str(target_icon)

    desktop_path.write_text(
        "[Desktop Entry]\n"
        f"Name={APP_NAME}\n"
        "Comment=Professional MIDI engraving to clear, readable Klavarskribo-style notation.\n"
        f"Exec=\"{appimage_path}\" %f\n"
        f"Icon={icon_value}\n"
        "Type=Application\n"
        "Categories=AudioVideo;Audio;Music;\n"
        f"MimeType={MIME_TYPE_KEYTAB};"
        f"{';'.join(MIME_TYPES_MIDI)};"
        f"{';'.join(MIME_TYPES_MUSICXML)};\n"
        "Terminal=false\n",
        encoding="utf-8",
    )


def _write_mime_package() -> None:
    mime_dir = Path.home() / ".local" / "share" / "mime" / "packages"
    mime_dir.mkdir(parents=True, exist_ok=True)
    mime_path = mime_dir / f"{APP_NAME}.xml"
    mime_path.write_text(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<mime-info xmlns=\"http://www.freedesktop.org/standards/shared-mime-info\">\n"
        f"  <mime-type type=\"{MIME_TYPE_KEYTAB}\">\n"
        "    <comment>keyTAB score</comment>\n"
        "    <glob pattern=\"*.piano\"/>\n"
        "  </mime-type>\n"
        "  <mime-type type=\"application/vnd.recordare.musicxml+xml\">\n"
        "    <comment>MusicXML score</comment>\n"
        "    <glob pattern=\"*.musicxml\"/>\n"
        "  </mime-type>\n"
        "  <mime-type type=\"application/vnd.recordare.musicxml\">\n"
        "    <comment>Compressed MusicXML score</comment>\n"
        "    <glob pattern=\"*.mxl\"/>\n"
        "  </mime-type>\n"
        "</mime-info>\n",
        encoding="utf-8",
    )


def _update_xdg_databases() -> None:
    mime_db = Path.home() / ".local" / "share" / "mime"
    apps_dir = Path.home() / ".local" / "share" / "applications"

    update_mime = shutil.which("update-mime-database")
    if update_mime:
        subprocess.run([update_mime, str(mime_db)], check=False)

    update_desktop = shutil.which("update-desktop-database")
    if update_desktop:
        subprocess.run([update_desktop, str(apps_dir)], check=False)


def _find_appimage_icon() -> Path | None:
    appdir = os.environ.get("APPDIR")
    if not appdir:
        return None
    candidates = [
        Path(appdir) / "usr" / "share" / "icons" / "hicolor" / "256x256" / "apps" / f"{APP_NAME}.png",
        Path(appdir) / f"{APP_NAME}.png",
        Path(appdir) / ".DirIcon",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def install_desktop_integration() -> None:
    if not sys.platform.startswith("linux"):
        print("--install is supported on Linux only.")
        return

    appimage_path = os.environ.get("APPIMAGE") or sys.argv[0]
    appimage_path = Path(appimage_path).expanduser().resolve()
    if not appimage_path.exists():
        raise SystemExit(f"AppImage not found: {appimage_path}")

    target_dir = Path.home() / ".local" / "share" / APP_NAME
    target_dir.mkdir(parents=True, exist_ok=True)
    target_appimage = target_dir / f"{APP_NAME}.AppImage"
    if target_appimage != appimage_path:
        shutil.copy2(appimage_path, target_appimage)
        appimage_path = target_appimage

    _write_desktop_entry(appimage_path, _find_appimage_icon())
    _write_mime_package()
    _update_xdg_databases()
    print("Installed desktop entry and MIME types.")


def prompt_install_if_needed() -> None:
    if not sys.platform.startswith("linux"):
        return
    if not os.environ.get("APPIMAGE"):
        return

    adm = get_appdata_manager()
    show_prompt = bool(adm.get("show_install_question", True))
    if not show_prompt:
        return

    message = (
        "<b>Install keyTAB for desktop integration?</b><br><br>"
        "This will:<ul>"
        "<li>Add keyTAB to your application menu</li>"
        "<li>Associate .piano, .mid/.midi, and .musicxml/.mxl files with keyTAB</li>"
        "<li>Copy this AppImage to a stable location in your home folder</li>"
        "</ul>"
        "You can remove the integration later by deleting the desktop entry in "
        "~/.local/share/applications and the AppImage in ~/.local/share/keyTAB."
    )

    dialog = QtWidgets.QMessageBox()
    dialog.setIcon(QtWidgets.QMessageBox.Icon.Question)
    dialog.setWindowTitle("Install keyTAB")
    dialog.setTextFormat(QtCore.Qt.TextFormat.RichText)
    dialog.setText(message)
    dont_show_checkbox = QtWidgets.QCheckBox("Don't show again")
    dialog.setCheckBox(dont_show_checkbox)
    install_button = dialog.addButton("Install", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
    dialog.addButton("Not now", QtWidgets.QMessageBox.ButtonRole.RejectRole)
    dialog.exec()

    if dont_show_checkbox.isChecked():
        adm.set("show_install_question", False)
        adm.save()

    if dialog.clickedButton() == install_button:
        adm.set("show_install_question", False)
        adm.save()
        try:
            install_desktop_integration()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                None,
                "Install failed",
                f"Install failed: {exc}",
                "You can still use the AppImage without installing.",
            )


def main(argv: list[str] | None = None):
    if argv is None:
        argv = sys.argv[1:]

    args, qt_args = parse_args(argv)
    initial_documents = extract_document_paths(qt_args)
    if args.install:
        install_desktop_integration()
        return

    # Load settings and apply UI scale before creating QApplication
    preferences = get_preferences()
    ui_scale = float(preferences.get("ui_scale", 1.0))
    
    # Initialize appdata to ensure ~/.keyTAB/appdata.py exists
    get_appdata_manager()

    # Ensure style storage exists in hidden app folder
    try:
        user_root = Path.home() / ".keyTAB"
        user_root.mkdir(parents=True, exist_ok=True)
        (user_root / "pstyle").mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    # Platform-specific DPI handling:
    # - On Linux, use Qt env vars to scale UI.
    # - On macOS, explicitly clear Qt scaling and plugin env to avoid Cocoa issues.
    if sys.platform.startswith("linux"):
        os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
        os.environ["QT_SCALE_FACTOR"] = str(ui_scale)
    # On macOS, force menus to render inside the window instead of the global menu bar
    elif sys.platform == "darwin":
        QtCore.QCoreApplication.setAttribute(
            QtCore.Qt.ApplicationAttribute.AA_DontUseNativeMenuBar, False
        )
    # Create QApplication with argv to ensure proper initialization paths on macOS
    app = KeyTabApplication([sys.argv[0], *qt_args])
    
    # Enforce arrow cursor globally: app never changes the mouse pointer
    QtGui.QGuiApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.CursorShape.ArrowCursor))
    
    # Install and apply embedded UI font (FiraCode-SemiBold) globally
    try:
        install_default_ui_font(app, name='FiraCode-SemiBold', point_size=int(10))
    except Exception:
        # Proceed with default font if installation fails
        pass

    # Set application window icon from icons package
    # Scale window icon slightly smaller for the title bar
    icon = get_qicon('keyTAB', size=(64, 64))
    if icon:
        app.setWindowIcon(icon)
    
    # Apply application palette based on preferences
    theme = str(preferences.get('theme', 'light')).lower()
    sty = Style()
    if theme == 'dark':
        sty.set_dark_theme()
    else:
        sty.set_light_theme()

    win = MainWindow()

    def _handle_file_request(path: str) -> None:
        if not path:
            return
        win.open_documents_from_paths([path], confirm_dirty=True)

    try:
        app.fileRequested.connect(_handle_file_request)
    except Exception:
        pass

    if initial_documents:
        QtCore.QTimer.singleShot(0, lambda: win.open_documents_from_paths(initial_documents, confirm_dirty=False))

    prompt_install_if_needed()
    try:
        win.schedule_edwin_prompt(250)
    except Exception:
        try:
            QtCore.QTimer.singleShot(250, win._maybe_prompt_edwin_install)
        except Exception:
            pass
    
    # Ensure clean shutdown of background threads on app exit
    app.aboutToQuit.connect(win.prepare_close)
    
    # Restore window geometry or start maximized based on appdata
    try:
        adm = get_appdata_manager()
        start_max = bool(adm.get("window_maximized", True))
        if not start_max:
            geom_b64 = str(adm.get("window_geometry", ""))
            try:
                if geom_b64:
                    win.restoreGeometry(QtCore.QByteArray.fromBase64(geom_b64.encode("ascii")))
            except Exception:
                pass
            win.show()
        else:
            win.showMaximized()
    except Exception:
        # Fallback: show maximized
        win.showMaximized()
    app.exec()


if __name__ == "__main__":
    mp.freeze_support()
    main()
