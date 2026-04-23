from __future__ import annotations

import base64
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

try:
    from PySide6.QtCore import QByteArray
    from PySide6.QtGui import QFontDatabase, QFont
    from PySide6.QtWidgets import QApplication
except Exception:
    QByteArray = None
    QFontDatabase = None
    QFont = None
    QApplication = None

# Lazy import of generated base64 mapping
try:
    from .fonts_byte64 import FONTS, FONT_ALIASES, FONT_GROUPS  # type: ignore
except Exception:
    FONTS = {}
    FONT_ALIASES = {}
    FONT_GROUPS = {}


_EMBEDDED_FONT_NAMES: set[str] = set()
_REGISTERED_FONT_CACHE: dict[str, Optional[str]] = {}
_FONT_EXTS = ('.ttf', '.otf', '.ttc', '.otc')


def _font_members(name: str) -> list[str]:
    group = FONT_GROUPS.get(name)
    if group:
        return [str(member) for member in group]
    alias = FONT_ALIASES.get(name)
    if alias:
        return [str(alias)]
    return [str(name)]


def _font_file_targets(name: str) -> list[str]:
    targets: list[str] = []
    seen: set[str] = set()
    for candidate in [name, *_font_members(name)]:
        text = str(candidate)
        if text and text not in seen:
            seen.add(text)
            targets.append(text)
    return targets


def _check_system_font_family(family: str) -> bool:
    normalized = _normalize_font_name(family)
    if not normalized:
        return False
    if _font_file_exists(family):
        return True
    if QFontDatabase is None:
        return False
    try:
        families = set(QFontDatabase.families())
    except Exception:
        return False
    if family not in families:
        return False
    return normalized not in _EMBEDDED_FONT_NAMES


def _normalize_font_name(name: str) -> str:
    return str(name or '').strip().lower()


def _decoded_font_bytes(name: str) -> Optional[bytes]:
    b64 = FONTS.get(name)
    if not b64:
        return None
    try:
        return base64.b64decode(b64)
    except Exception:
        return None


def _guess_font_extension(data: bytes) -> str:
    if data.startswith(b'OTTO'):
        return '.otf'
    if data.startswith(b'\x00\x01\x00\x00') or data.startswith(b'true'):
        return '.ttf'
    return '.ttf'


def _user_font_dir() -> Path:
    home = Path.home()
    if sys.platform.startswith('win'):
        base = Path(os.environ.get('LOCALAPPDATA', home / 'AppData/Local'))
        return base / 'Microsoft/Windows/Fonts'
    if sys.platform == 'darwin':
        return home / 'Library/Fonts'
    return home / '.local/share/fonts'


def _candidate_font_dirs() -> list[Path]:
    dirs: list[Path] = []
    user_dir = _user_font_dir()
    dirs.append(user_dir)
    if sys.platform.startswith('win'):
        dirs.append(Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts')
    elif sys.platform == 'darwin':
        dirs.extend([Path('/Library/Fonts'), Path('/System/Library/Fonts')])
    else:
        dirs.extend([
            Path.home() / '.fonts',
            Path('/usr/local/share/fonts'),
            Path('/usr/share/fonts'),
        ])
    return dirs


def _font_file_exists(family: str) -> bool:
    targets: set[str] = set()
    for target_name in _font_file_targets(family):
        normalized = _normalize_font_name(target_name)
        title_case = normalized.title() if normalized else target_name.title()
        targets.update({target_name, normalized, title_case})
    for directory in _candidate_font_dirs():
        for target in list(targets):
            if not target:
                continue
            for ext in _FONT_EXTS:
                path = directory / f"{target}{ext}"
                if path.exists():
                    return True
    return False


def _refresh_system_font_cache(installed_path: Path) -> None:
    try:
        if sys.platform.startswith('linux'):
            subprocess.run(['fc-cache', '-f', str(installed_path.parent)], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif sys.platform.startswith('win'):
            import ctypes

            try:
                fr_private = 0x10
                ctypes.windll.gdi32.AddFontResourceExW(str(installed_path), fr_private, 0)
                HWND_BROADCAST = 0xFFFF
                WM_FONTCHANGE = 0x001D
                ctypes.windll.user32.SendMessageTimeoutW(HWND_BROADCAST, WM_FONTCHANGE, 0, 0, 0, 1000, None)
            except Exception:
                pass
        # macOS picks up fonts automatically from ~/Library/Fonts
    except Exception:
        pass


def has_system_font(family: str) -> bool:
    names = _font_file_targets(family)
    if len(names) > 1:
        return all(_check_system_font_family(name) for name in names)
    return _check_system_font_family(str(family))


def has_installed_embedded_font_file(name: str) -> bool:
    """Return True if the embedded font file is present in the user font directory.

    This is a pragmatic persistence check used by startup install prompts.
    On Windows, newly added per-user fonts are not always immediately visible
    via QFontDatabase in a fresh process.
    """
    dest_dir = _user_font_dir()
    if not dest_dir.exists():
        return False

    members = _font_members(name)
    for member in members:
        data = _decoded_font_bytes(member)
        if not data:
            return False
        ext = _guess_font_extension(data)
        primary = dest_dir / f"{member}{ext}"
        if primary.exists():
            continue
        found = False
        for candidate_ext in _FONT_EXTS:
            if (dest_dir / f"{member}{candidate_ext}").exists():
                found = True
                break
        if not found:
            return False
    return True


def install_embedded_font_to_system(name: str) -> tuple[bool, str]:
    dest_dir = _user_font_dir()
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        return False, f"Cannot create font dir: {exc}"
    installed_paths: list[str] = []
    for member in _font_members(name):
        data = _decoded_font_bytes(member)
        if not data:
            return False, f"No embedded font named {member}."
        ext = _guess_font_extension(data)
        target = dest_dir / f"{member}{ext}"
        try:
            if target.exists():
                try:
                    if target.read_bytes() == data:
                        _refresh_system_font_cache(target)
                        installed_paths.append(str(target))
                        continue
                except Exception:
                    pass
            target.write_bytes(data)
        except Exception as exc:
            return False, f"Failed to write font {member}: {exc}"
        _refresh_system_font_cache(target)
        installed_paths.append(str(target))
    return True, ', '.join(installed_paths)


def register_font_from_bytes(name: str) -> Optional[str]:
    """Register the embedded font by `name` and return the primary family name.

    Returns None if registration fails or PySide6 is unavailable.
    """
    if QFontDatabase is None:
        return None
    cache_key = _normalize_font_name(name)
    if cache_key in _REGISTERED_FONT_CACHE:
        return _REGISTERED_FONT_CACHE[cache_key]
    try:
        resolved: Optional[str] = None
        members = _font_members(name)
        for member in members:
            raw = _decoded_font_bytes(member)
            if raw is None:
                continue
            if QByteArray is not None:
                data = QByteArray(raw)
            else:
                data = raw
            fid = QFontDatabase.addApplicationFontFromData(data)
            if fid < 0:
                continue
            fams = [str(f) for f in QFontDatabase.applicationFontFamilies(fid)]
            _EMBEDDED_FONT_NAMES.add(_normalize_font_name(member))
            for fam in fams:
                _EMBEDDED_FONT_NAMES.add(_normalize_font_name(fam))
            if resolved is None and fams:
                resolved = fams[0]
        if resolved is None:
            _REGISTERED_FONT_CACHE[cache_key] = None
            return None
        _EMBEDDED_FONT_NAMES.add(cache_key)
        _REGISTERED_FONT_CACHE[cache_key] = resolved
        return resolved
    except Exception:
        _REGISTERED_FONT_CACHE[cache_key] = None
        return None


def resolve_font_family(family: str, fallback_family: str = 'Edwin') -> str:
    """Resolve a usable font family name.

    - Prefer the requested system font if available.
    - Otherwise, register the embedded fallback font and use it if available.
    - As a last resort, return the original family string.
    """
    if QFontDatabase is None:
        return family
    try:
        families = set(QFontDatabase.families())
        if family in families:
            return family
    except Exception:
        pass
    try:
        fallback = register_font_from_bytes(fallback_family)
        if fallback:
            return fallback
    except Exception:
        pass
    return family


def install_default_ui_font(app: Optional[QApplication] = None, name: str = 'FiraCode-SemiBold', point_size: int = 11) -> bool:
    """Install the embedded font and set it as the QApplication default.

    - Tries to register the font from embedded base64 (fonts_byte64.py).
    - If embedded font is missing, tries to use system-installed font by name.
    - Returns True if the app font was set; False otherwise.
    """
    if QApplication is None:
        return False
    if app is None:
        app = QApplication.instance()
    if app is None:
        return False

    family = register_font_from_bytes(name)

    # On macOS, Qt sometimes ignores in-memory fonts for UI widgets.
    # Install the embedded font to the user font dir and register from the file as a fallback.
    if not family:
        ok, path = install_embedded_font_to_system(name)
        if ok and QFontDatabase is not None:
            try:
                fid = QFontDatabase.addApplicationFont(path)
                if fid >= 0:
                    fams = [str(f) for f in QFontDatabase.applicationFontFamilies(fid)]
                    if fams:
                        family = fams[0]
            except Exception:
                pass

    # Try a list of likely family names/aliases for Fira Code
    candidates = []
    if family:
        candidates.append(family)
    candidates.extend([
        name,
        'Fira Code',
        'FiraCode',
        'Fira Code SemiBold',
        'FiraCode-SemiBold',
    ])

    try:
        for fam in candidates:
            if not fam:
                continue
            f = QFont(str(fam), point_size)
            # Prefer semi-bold weight when available
            try:
                f.setWeight(QFont.Weight.DemiBold)
            except Exception:
                pass
            if f and f.family():
                app.setFont(f)
                return True
        # Last resort: let Qt pick default font with size
        f = QFont()
        f.setPointSize(point_size)
        app.setFont(f)
    except Exception:
        return False
    return False
