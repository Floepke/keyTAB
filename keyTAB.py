import argparse
import os
import shutil
import subprocess
import sys
import threading
import multiprocessing as mp


def _install_fluidsynth_warning_filter() -> None:
    """Filter known noisy native stderr lines while preserving real errors."""
    ignored_prefixes = (
        b"fluidsynth: warning: ",
        b"(process:",
    )
    ignored_glib_fragments = (
        b"GLib-GObject-CRITICAL",
        b"g_param_spec_enum: assertion ",
        b"validate_pspec_to_install: assertion ",
        b"g_param_spec_ref_sink: assertion ",
        b"g_param_spec_unref: assertion ",
    )

    saved = os.dup(2)
    r_fd, w_fd = os.pipe()
    os.dup2(w_fd, 2)
    os.close(w_fd)

    def _relay() -> None:
        buf = b""
        while True:
            try:
                chunk = os.read(r_fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if line.startswith(ignored_prefixes) and any(
                    fragment in line for fragment in ignored_glib_fragments
                ):
                    continue
                if not line.startswith(b"fluidsynth: warning: "):
                    try:
                        os.write(saved, line + b"\n")
                    except OSError:
                        pass

    threading.Thread(target=_relay, daemon=True).start()


_install_fluidsynth_warning_filter()
from pathlib import Path
from PySide6 import QtCore, QtWidgets, QtGui
from ui.main_window import MainWindow
from ui.style import Style
from settings_manager import get_preferences, set_ui_scale
from appdata_manager import get_appdata_manager
from icons.icons import get_qicon
from fonts import (
    has_installed_embedded_font_file,
    install_default_ui_font,
    install_embedded_font_to_system,
    register_font_from_bytes,
)
from utils.file_associations import extract_document_paths
from utils.multiprocessing_utils import configure_start_method

APP_NAME = "keyTAB"
MIME_TYPE_KEYTAB = "application/x-keytab"
MIME_TYPES_MIDI = ["audio/midi", "audio/x-midi"]
MIME_TYPES_MUSICXML = [
    "application/vnd.recordare.musicxml+xml",
    "application/vnd.recordare.musicxml",
]
SUPPORTED_UI_LANGUAGES = {"en", "nl"}


def _resolve_ui_language(preferences: dict) -> str:
    raw = str(preferences.get("ui_language", "system") or "system").strip().lower()
    if raw == "system":
        raw = str(QtCore.QLocale.system().name() or "en").split("_", 1)[0].lower()
    if raw not in SUPPORTED_UI_LANGUAGES:
        return "en"
    return raw


def _translation_search_paths() -> list[Path]:
    paths: list[Path] = [
        Path(__file__).resolve().parent / "i18n",
        Path(sys.argv[0]).resolve().parent / "i18n",
    ]
    appdir = os.environ.get("APPDIR")
    if appdir:
        paths.append(Path(appdir) / "usr" / "share" / APP_NAME / "i18n")

    seen: set[str] = set()
    unique_paths: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique_paths.append(path)
    return unique_paths


def _install_ui_translator(app: QtWidgets.QApplication, preferences: dict) -> None:
    lang = _resolve_ui_language(preferences)
    if lang == "en":
        return

    translator = QtCore.QTranslator(app)
    file_name = f"keytab_{lang}.qm"
    for base_dir in _translation_search_paths():
        candidate = base_dir / file_name
        if not candidate.exists():
            continue
        if translator.load(str(candidate)):
            app.installTranslator(translator)
            # Keep a strong reference for the lifetime of the app.
            setattr(app, "_ui_translator", translator)
            return


def _install_qt_translator(app: QtWidgets.QApplication, preferences: dict) -> None:
    """Install Qt's built-in translations (standard buttons, dialogs, etc.)."""
    lang = _resolve_ui_language(preferences)
    if lang == "en":
        return

    qt_translator = QtCore.QTranslator(app)
    qt_translation_dirs: list[Path] = []
    try:
        qt_dir = Path(QtCore.QLibraryInfo.path(QtCore.QLibraryInfo.LibraryPath.TranslationsPath))
        qt_translation_dirs.append(qt_dir)
    except Exception:
        pass

    appdir = os.environ.get("APPDIR")
    if appdir:
        qt_translation_dirs.append(Path(appdir) / "usr" / "translations")
        qt_translation_dirs.append(Path(appdir) / "usr" / "share" / "qt6" / "translations")

    # Try common Qt translation catalogs in order.
    for base_name in ("qtbase", "qt"):
        file_name = f"{base_name}_{lang}.qm"
        for directory in qt_translation_dirs:
            candidate = directory / file_name
            if not candidate.exists():
                continue
            if qt_translator.load(str(candidate)):
                app.installTranslator(qt_translator)
                setattr(app, "_qt_translator", qt_translator)
                return


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
        "    <glob pattern=\"*.keytab\"/>\n"
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
        print(QtCore.QCoreApplication.translate("keyTAB", "--install is supported on Linux only."))
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
    print(QtCore.QCoreApplication.translate("keyTAB", "Installed desktop entry and MIME types."))


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
        QtCore.QCoreApplication.translate("keyTAB", "<b>Install keyTAB for desktop integration?</b><br><br>")
        + QtCore.QCoreApplication.translate("keyTAB", "This will:<ul>")
        + QtCore.QCoreApplication.translate("keyTAB", "<li>Add keyTAB to your application menu</li>")
        + QtCore.QCoreApplication.translate("keyTAB", "<li>Associate .keytab/.piano, .mid/.midi, and .musicxml/.mxl files with keyTAB</li>")
        + QtCore.QCoreApplication.translate("keyTAB", "<li>Copy this AppImage to a stable location in your home folder</li>")
        + QtCore.QCoreApplication.translate("keyTAB", "</ul>")
        + QtCore.QCoreApplication.translate("keyTAB", "You can remove the integration later by deleting the desktop entry in ")
        + QtCore.QCoreApplication.translate("keyTAB", "~/.local/share/applications and the AppImage in ~/.local/share/keyTAB.")
    )

    dialog = QtWidgets.QMessageBox()
    dialog.setIcon(QtWidgets.QMessageBox.Icon.Question)
    dialog.setWindowTitle(QtCore.QCoreApplication.translate("keyTAB", "Install keyTAB"))
    dialog.setTextFormat(QtCore.Qt.TextFormat.RichText)
    dialog.setText(message)
    dont_show_checkbox = QtWidgets.QCheckBox(QtCore.QCoreApplication.translate("keyTAB", "Don't show again"))
    dialog.setCheckBox(dont_show_checkbox)
    install_button = dialog.addButton(
        QtCore.QCoreApplication.translate("keyTAB", "Install"),
        QtWidgets.QMessageBox.ButtonRole.AcceptRole,
    )
    dialog.addButton(
        QtCore.QCoreApplication.translate("keyTAB", "Not now"),
        QtWidgets.QMessageBox.ButtonRole.RejectRole,
    )
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
                QtCore.QCoreApplication.translate("keyTAB", "Install failed"),
                QtCore.QCoreApplication.translate("keyTAB", "Install failed: {error}").format(error=exc),
                QtCore.QCoreApplication.translate("keyTAB", "You can still use the AppImage without installing."),
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
    try:
        ui_scale = float(preferences.get("ui_scale", 1.0))
    except Exception:
        ui_scale = 1.0
    ui_scale = max(0.5, min(3.0, ui_scale))
    
    # Initialize appdata to ensure ~/.keyTAB/appdata.py exists
    get_appdata_manager()

    # Ensure style storage exists in hidden app folder
    user_root = Path.home() / ".keyTAB"
    user_root.mkdir(parents=True, exist_ok=True)
    (user_root / "pstyle").mkdir(parents=True, exist_ok=True)

    # Store ui_scale for widget construction (macOS/Linux bypass QT_SCALE_FACTOR).
    set_ui_scale(ui_scale)

    # On Windows use QT_SCALE_FACTOR for full Qt widget scaling.
    # On macOS and Linux QT_SCALE_FACTOR causes unwanted rendering artefacts at
    # non-1.0 scales (known Qt issue), so we fake the scale via font size and
    # icon/button sizes instead (handled per-widget via get_ui_scale()).
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    if sys.platform.startswith("win"):
        os.environ["QT_SCALE_FACTOR"] = str(ui_scale)
    else:
        # Leave QT_SCALE_FACTOR at default (1) to avoid artefacts.
        os.environ.pop("QT_SCALE_FACTOR", None)
    
    # Ensure high DPI scaling rounding is disabled to allow for smooth scaling at fractional ui_scale values.
    QtGui.QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
            QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    # Platform-specific startup handling.
    # On macOS, keep the in-window (non-native) menu bar in all window states.
    if sys.platform == "darwin":
        # Force Fusion style in-process to avoid accidental fallback to native style
        # from shell/launchd environment overrides.
        os.environ["QT_STYLE_OVERRIDE"] = "Fusion"
        os.environ.pop("QT_QPA_PLATFORMTHEME", None)
    else:
        # On Windows/Linux, disable system platform theme that may override our palette
        # This is especially important for bundled apps where system theme can cause
        # unreadable menus or broken colors
        os.environ["QT_STYLE_OVERRIDE"] = "Fusion"
        os.environ["QT_QPA_PLATFORMTHEME"] = ""
    # Create QApplication with argv to ensure proper initialization paths on macOS
    app = KeyTabApplication([sys.argv[0], *qt_args])
    _install_ui_translator(app, preferences)
    _install_qt_translator(app, preferences)

    # Always register embedded engraving fonts for in-process use.
    # On Windows, Cairo can require the user font store to see the font.
    try:
        for font_name in ("Edwin", "LelandText"):
            register_font_from_bytes(font_name)
            if sys.platform.startswith("win") and not has_installed_embedded_font_file(font_name):
                install_embedded_font_to_system(font_name)
        if sys.platform.startswith("win") and not has_installed_embedded_font_file("FiraCode-SemiBold"):
            install_embedded_font_to_system("FiraCode-SemiBold")
    except Exception:
        pass

    # Suppress unhelpful Qt platform warning about grabMouse on non-popup windows.
    # grabMouse() is used to track mouse releases outside the widget; this warning
    # is harmless on Wayland/xcb where it isn't fully supported.
    _original_handler = QtCore.qInstallMessageHandler(None)
    def _qt_message_handler(msg_type, context, message):
        if "This plugin supports grabbing the mouse only for popup windows" in message:
            return
        if _original_handler is not None:
            _original_handler(msg_type, context, message)
        else:
            print(message, file=sys.stderr)
    QtCore.qInstallMessageHandler(_qt_message_handler)

    # Belt-and-suspenders: explicitly pick Fusion when available on macOS.
    if sys.platform == "darwin":
        styles = {str(s).lower(): str(s) for s in QtWidgets.QStyleFactory.keys()}
        fusion_name = styles.get("fusion")
        if fusion_name:
            QtWidgets.QApplication.setStyle(fusion_name)
    
    # Enforce arrow cursor globally: app never changes the mouse pointer
    QtGui.QGuiApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.CursorShape.ArrowCursor))
    
    # Force the font via stylesheet as a fallback if Qt ignores the app font
    app.setStyleSheet(app.styleSheet() + "\n* { font-family: 'Fira Code'; }\n")

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

    # Install and apply embedded UI font (FiraCode-SemiBold) globally AFTER palette/style reset.
    # On macOS/Linux scale the font point size to fake the ui_scale (no QT_SCALE_FACTOR used).
    _font_pt = int(round(10 * ui_scale)) if not sys.platform.startswith('win') else 10
    install_default_ui_font(app, name='FiraCode-SemiBold', point_size=_font_pt)
    # Fallback stylesheet to force family if Qt ignores app font
    app.setStyleSheet(app.styleSheet() + "\n* { font-family: 'Fira Code'; }\n")

    win = MainWindow()

    def _handle_file_request(path: str) -> None:
        if not path:
            return
        win.open_documents_from_paths([path], confirm_dirty=True)

    app.fileRequested.connect(_handle_file_request)

    if initial_documents:
        QtCore.QTimer.singleShot(0, lambda: win.open_documents_from_paths(initial_documents, confirm_dirty=False))

    prompt_install_if_needed()
    win.schedule_fonts_install_prompt(250)

    # Ensure clean shutdown of background threads on app exit
    app.aboutToQuit.connect(win.prepare_close)

    win.restore_window_state_from_appdata()

    # PySide on macOS can crash in QApplication teardown when Python finalizes.
    # Force a fast, clean termination after the event loop exits to skip Qt/PySide
    # atexit cleanup (state is already persisted in prepare_close above).
    exit_code = app.exec()
    while QtGui.QGuiApplication.overrideCursor() is not None:
        QtGui.QGuiApplication.restoreOverrideCursor()
    os._exit(int(exit_code))


if __name__ == "__main__":
    mp.freeze_support()
    configure_start_method()
    main()
